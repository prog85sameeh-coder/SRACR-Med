from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import independent_leakage as il

PRIMARY_CKPT = {
    "chest": "chest_xray_resnet18.pt",
    "retinal": "retinal_11class_resnet18.pt",
    "split": "fundus_4class_resnet18.pt",
}

IMAGENET_MEAN = il.IMAGENET_MEAN
IMAGENET_STD = il.IMAGENET_STD


def load_primary_model(alias: str, model_dir: Path, device: str):
    path = model_dir / PRIMARY_CKPT[alias]
    if not path.exists():
        raise FileNotFoundError(
            f"Primary checkpoint not found: {path}\n"
            "Extract SRACR_Med_R1C11_Model_Checkpoints.zip and pass --primary-model-dir."
        )
    ck = torch.load(path, map_location=device)
    classes = list(ck["class_names"])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = ck.get("model_state_dict", ck.get("state_dict"))
    if state is None:
        raise KeyError(f"No model_state_dict/state_dict in {path}")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, int(ck.get("image_size", 224))


def model_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def cam_map_native(model, image: Image.Image, native_h: int, native_w: int, image_size: int, device: str) -> np.ndarray:
    """Class Activation Map for the model-predicted class using the final ResNet layer4 feature map."""
    holder: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        holder["feat"] = out.detach()

    h = model.layer4.register_forward_hook(hook)
    try:
        x = model_transform(image_size)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        target = int(logits.argmax(1).item())
        feat = holder["feat"][0]  # C,H,W
        w = model.fc.weight[target].detach().view(-1, 1, 1)
        cam = torch.relu((feat * w).sum(dim=0, keepdim=True).unsqueeze(0))  # 1,1,h,w
        maxv = float(cam.max().item())
        if maxv > 1e-12:
            cam = cam / maxv
        else:
            cam = torch.zeros_like(cam)
        cam = F.interpolate(cam, size=(native_h, native_w), mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy().astype(np.float32)
    finally:
        h.remove()


def log_edge_map_native(image: Image.Image, native_h: int, native_w: int, image_size: int = 224, sigma: float = 1.0) -> np.ndarray:
    """
    Protocol-normalized LoG ROI proxy inspired by recent selective medical-image ROI encryption.
    It ranks tiles by mean absolute Laplacian-of-Gaussian response. This reproduces the ROI-selection
    principle under a common fixed budget, not the original paper's chaotic cipher or exact binary ROI mask.
    """
    im = image.resize((image_size, image_size), Image.BILINEAR).convert("L")
    gray = np.asarray(im, dtype=np.float32) / 255.0
    log_resp = np.abs(ndimage.gaussian_laplace(gray, sigma=sigma)).astype(np.float32)
    # Robust normalization only affects scale, not rank.
    p99 = float(np.quantile(log_resp, 0.99)) if log_resp.size else 0.0
    if p99 > 1e-12:
        log_resp = np.clip(log_resp / p99, 0.0, 1.0)
    else:
        log_resp.fill(0.0)
    native = Image.fromarray(np.uint8(np.clip(log_resp * 255.0, 0, 255))).resize((native_w, native_h), Image.BILINEAR)
    return np.asarray(native, dtype=np.float32) / 255.0


def tile_mean_scores(map2d: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    scores = np.empty(len(boxes), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        patch = map2d[y0:y1, x0:x1]
        scores[i] = float(patch.mean()) if patch.size else 0.0
    return scores


def evaluate_alias(args, alias: str):
    il.set_seed(args.seed)
    index = pd.read_csv(args.dataset_index)
    source = il.SOURCES[alias]
    test = index[(index["dataset_source"] == source) & (index["final_split"] == "test")].copy()
    if test.empty:
        raise RuntimeError(f"No test rows for {alias}")

    cache_path = Path(args.cache_dir) / il.CACHE_GLOB[alias]
    cache = il.load_cache(cache_path)
    records = cache["records"]
    tile_size = int(cache.get("tile_size", 64))
    rec_by_path = {str(r["image_path"]): r for r in records}

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    attacker_path = Path(args.attacker_dir) / f"{alias}_mobilenetv3small_independent.pt"
    if not attacker_path.exists():
        raise FileNotFoundError(
            f"Independent attacker checkpoint not found: {attacker_path}\n"
            "Use the attacker_models directory produced for Reviewer 1 Comment 1."
        )
    attacker, attack_ck = il.load_attacker(attacker_path, device)
    attacker_classes = list(attack_ck["classes"])
    attacker_class_to_idx = {c: i for i, c in enumerate(attacker_classes)}
    attack_tf = model_transform(int(attack_ck.get("image_size", 224)))

    primary, primary_classes, primary_image_size = load_primary_model(alias, Path(args.primary_model_dir), device)

    methods = [
        "sracr_saliency",
        "log_roi_inspired",
        "cam_priority",
        "content_variance",
        "center_roi",
        "random",
    ]
    budgets = [float(x) for x in args.budgets]
    rows = []

    for pos, row in test.reset_index(drop=True).iterrows():
        csv_path = str(row["image_path"])
        path = il.resolve_path(csv_path, args.dataset_root)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}\nOriginal CSV path: {csv_path}")
        rec = rec_by_path[csv_path]

        with Image.open(path) as im:
            image = im.convert("RGB")
            arr = np.asarray(image).copy()
        native_h, native_w = arr.shape[:2]
        areas, boxes = il.tile_geometry(native_w, native_h, tile_size)
        total_area = int(areas.sum())

        sracr_scores = np.asarray(rec["tile_scores"], dtype=np.float32)
        if len(sracr_scores) != len(boxes):
            raise RuntimeError(f"Tile-count mismatch for {path}: cache={len(sracr_scores)}, geometry={len(boxes)}")

        content_scores = il.content_variance_scores(arr, boxes)
        center_scores = il.center_scores(boxes, native_w, native_h)
        log_map = log_edge_map_native(image, native_h, native_w, image_size=args.log_image_size, sigma=args.log_sigma)
        log_scores = tile_mean_scores(log_map, boxes)
        cam_map = cam_map_native(primary, image, native_h, native_w, primary_image_size, device)
        cam_scores = tile_mean_scores(cam_map, boxes)

        score_map = {
            "sracr_saliency": sracr_scores,
            "log_roi_inspired": log_scores,
            "cam_priority": cam_scores,
            "content_variance": content_scores,
            "center_roi": center_scores,
        }

        conditions = [("unprotected", 0.0, np.empty(0, dtype=np.int64), 0.0, 0.0)]
        sracr_order = il.stable_desc_order(sracr_scores)
        for budget in budgets:
            nominal_target = int(round(budget * total_area))
            sracr_sel = il.choose_prefix_for_target_area(sracr_order, areas, nominal_target)
            matched_target_area = int(areas[sracr_sel].sum())
            actual = matched_target_area / total_area
            conditions.append(("sracr_saliency", budget, sracr_sel, actual, actual - budget))

            for method in ["log_roi_inspired", "cam_priority", "content_variance", "center_roi"]:
                order = il.stable_desc_order(score_map[method])
                sel = il.choose_prefix_for_target_area(order, areas, matched_target_area)
                frac = float(areas[sel].sum() / total_area) if len(sel) else 0.0
                conditions.append((method, budget, sel, frac, frac - actual))

            order = il.random_order(len(areas), args.seed, csv_path, budget)
            sel = il.choose_prefix_for_target_area(order, areas, matched_target_area)
            frac = float(areas[sel].sum() / total_area) if len(sel) else 0.0
            conditions.append(("random", budget, sel, frac, frac - actual))

        tensors = []
        for method, budget, selected, _, _ in conditions:
            masked = Image.fromarray(arr) if len(selected) == 0 else il.mask_tiles(arr, selected, boxes, fill=args.mask_value)
            tensors.append(attack_tf(masked))
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            probs = torch.softmax(attacker(batch), dim=1).cpu().numpy()

        y = attacker_class_to_idx[str(row["label"])]
        for j, (method, budget, selected, protected_frac, budget_error) in enumerate(conditions):
            p = probs[j]
            pred = int(p.argmax())
            rows.append({
                "dataset": alias,
                "dataset_source": source,
                "image_index": pos,
                "image_path": csv_path,
                "true_label": str(row["label"]),
                "method": method,
                "nominal_budget": budget,
                "protected_pixel_fraction": protected_frac,
                "budget_match_error": budget_error,
                "n_protected_tiles": int(len(selected)),
                "n_total_tiles": int(len(areas)),
                "true_idx": y,
                "pred_idx": pred,
                "pred_label": attacker_classes[pred],
                "correct": int(pred == y),
                "confidence": float(p[pred]),
                "true_probability": float(p[y]),
                "nll": float(-math.log(max(float(p[y]), 1e-12))),
                "predictive_entropy": il.entropy_from_probs(p),
            })
        if (pos + 1) % 25 == 0 or pos + 1 == len(test):
            print(f"[{alias}] R2C2 baseline evaluation {pos+1}/{len(test)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_df = pd.DataFrame(rows)
    image_df.to_csv(out_dir / f"{alias}_r2c2_recent_baselines_image_level.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for (method, budget), g in image_df.groupby(["method", "nominal_budget"], sort=True):
        y = g["true_idx"].to_numpy(int)
        pred = g["pred_idx"].to_numpy(int)
        acc = float(np.mean(y == pred))
        mf1 = float(il.f1_score(y, pred, average="macro", zero_division=0))
        bal = float(il.balanced_accuracy_score(y, pred))
        acc_lo, acc_hi = il.bootstrap_metric_ci(y, pred, "accuracy", args.bootstrap, args.seed + 100)
        f1_lo, f1_hi = il.bootstrap_metric_ci(y, pred, "macro_f1", args.bootstrap, args.seed + 200)
        tp = g["true_probability"].to_numpy(float)
        nll = g["nll"].to_numpy(float)
        tp_lo, tp_hi = il.bootstrap_mean_ci(tp, args.bootstrap, args.seed + 300)
        nll_lo, nll_hi = il.bootstrap_mean_ci(nll, args.bootstrap, args.seed + 400)
        summary_rows.append({
            "dataset": alias,
            "method": method,
            "nominal_budget": float(budget),
            "n_images": len(g),
            "protected_pixel_fraction_mean": float(g["protected_pixel_fraction"].mean()),
            "max_abs_budget_match_error": float(np.abs(g["budget_match_error"]).max()),
            "attacker_accuracy": acc,
            "attacker_accuracy_ci95_low": acc_lo,
            "attacker_accuracy_ci95_high": acc_hi,
            "attacker_macro_f1": mf1,
            "attacker_macro_f1_ci95_low": f1_lo,
            "attacker_macro_f1_ci95_high": f1_hi,
            "attacker_balanced_accuracy": bal,
            "attacker_true_probability_mean": float(tp.mean()),
            "attacker_true_probability_ci95_low": tp_lo,
            "attacker_true_probability_ci95_high": tp_hi,
            "attacker_nll_mean": float(nll.mean()),
            "attacker_nll_ci95_low": nll_lo,
            "attacker_nll_ci95_high": nll_hi,
            "attacker_entropy_mean": float(g["predictive_entropy"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{alias}_r2c2_recent_baselines_summary.csv", index=False, encoding="utf-8-sig")

    # Paired SRACR vs every alternative under the same image/budget.
    pairs = []
    alternatives = ["log_roi_inspired", "cam_priority", "content_variance", "center_roi", "random"]
    for budget in budgets:
        sr = image_df[(image_df.method == "sracr_saliency") & np.isclose(image_df.nominal_budget, budget)].sort_values("image_index")
        for alt in alternatives:
            aa = image_df[(image_df.method == alt) & np.isclose(image_df.nominal_budget, budget)].sort_values("image_index")
            diffs = {
                "correctness_alt_minus_sracr": aa.correct.to_numpy(float) - sr.correct.to_numpy(float),
                "true_probability_alt_minus_sracr": aa.true_probability.to_numpy(float) - sr.true_probability.to_numpy(float),
                "nll_sracr_minus_alt": sr.nll.to_numpy(float) - aa.nll.to_numpy(float),
            }
            for metric, diff in diffs.items():
                lo, hi = il.bootstrap_mean_ci(diff, args.bootstrap, args.seed + 500)
                pval = il.signflip_p(diff, args.permutations, args.seed + 600)
                sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
                dz = float(diff.mean() / sd) if sd > 0 else 0.0
                pairs.append({
                    "dataset": alias,
                    "nominal_budget": budget,
                    "comparison": f"sracr_saliency_vs_{alt}",
                    "alternative": alt,
                    "metric": metric,
                    "n_pairs": len(diff),
                    "mean_difference_positive_favors_sracr_privacy": float(diff.mean()),
                    "difference_ci95_low": lo,
                    "difference_ci95_high": hi,
                    "paired_effect_dz": dz,
                    "permutation_p": pval,
                })
    pair_df = pd.DataFrame(pairs)
    pair_df["holm_p"] = np.nan
    for metric, ix in pair_df.groupby("metric").groups.items():
        pair_df.loc[ix, "holm_p"] = il.holm_adjust(pair_df.loc[ix, "permutation_p"].values)
    pair_df["significant_holm_0_05"] = (pair_df["holm_p"] < 0.05).astype(int)
    pair_df.to_csv(out_dir / f"{alias}_r2c2_recent_baselines_paired_tests.csv", index=False, encoding="utf-8-sig")

    meta = {
        "dataset": alias,
        "test_images": int(len(test)),
        "methods": methods,
        "budgets": budgets,
        "common_attacker": "independent MobileNetV3-Small",
        "primary_model_for_CAM": PRIMARY_CKPT[alias],
        "cam_target_rule": "argmax class on the original unmasked primary-classifier input",
        "log_roi_baseline": {
            "name": "protocol-normalized LoG ROI-inspired selector",
            "reference_family": "Zhang et al., Complex & Intelligent Systems, 2024, DOI 10.1007/s40747-023-01258-2",
            "implementation": "absolute Laplacian-of-Gaussian response, sigma fixed by --log-sigma, tile priority = mean response",
            "claim_boundary": "reproduces the LoG ROI-selection principle under a common tile budget; not the original chaotic cipher/binary-boundary implementation"
        },
        "budget_matching": "SRACR defines actual per-image protected-pixel target; all alternatives match that target by tile-area prefix.",
        "tile_size_native_pixels": tile_size,
        "mask_value_rgb": [args.mask_value]*3,
        "bootstrap_resamples": args.bootstrap,
        "paired_signflip_permutations": args.permutations,
        "seed": args.seed,
    }
    with open(out_dir / f"{alias}_r2c2_recent_baselines_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def combine(args):
    out = Path(args.output_dir)
    summaries=[]; images=[]; pairs=[]
    for alias in il.SOURCES:
        for coll, suffix in [
            (summaries,"summary"), (images,"image_level"), (pairs,"paired_tests")
        ]:
            p=out/f"{alias}_r2c2_recent_baselines_{suffix}.csv"
            if p.exists(): coll.append(pd.read_csv(p))
    if summaries:
        s=pd.concat(summaries,ignore_index=True)
        s.to_csv(out/"R2C2_ALL_DATASETS_recent_baselines_summary.csv",index=False,encoding="utf-8-sig")
        num=[c for c in s.columns if c not in {"dataset","method"} and pd.api.types.is_numeric_dtype(s[c])]
        macro=s.groupby(["method","nominal_budget"],as_index=False)[num].mean(numeric_only=True)
        macro.to_csv(out/"R2C2_CROSS_DATASET_MACRO_recent_baselines.csv",index=False,encoding="utf-8-sig")
    if images:
        pd.concat(images,ignore_index=True).to_csv(out/"R2C2_ALL_DATASETS_recent_baselines_image_level.csv",index=False,encoding="utf-8-sig")
    if pairs:
        pd.concat(pairs,ignore_index=True).to_csv(out/"R2C2_ALL_DATASETS_recent_baselines_paired_tests.csv",index=False,encoding="utf-8-sig")
    print("Combined R2C2 outputs written.")


def main():
    ap=argparse.ArgumentParser(description="Reviewer 2 Comment 2 same-condition literature-positioned selective-protection baselines")
    ap.add_argument("--source",choices=["chest","retinal","split","all"],default="all")
    ap.add_argument("--dataset-index",default="dataset_index.csv")
    ap.add_argument("--dataset-root",default=None)
    ap.add_argument("--cache-dir",default="feature_cache")
    ap.add_argument("--attacker-dir",default="attacker_models")
    ap.add_argument("--primary-model-dir",default="primary_models")
    ap.add_argument("--output-dir",default="outputs_r2c2")
    ap.add_argument("--budgets",nargs="+",type=float,default=[0.25,0.50,0.75])
    ap.add_argument("--mask-value",type=int,default=127)
    ap.add_argument("--bootstrap",type=int,default=2000)
    ap.add_argument("--permutations",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--log-sigma",type=float,default=1.0)
    ap.add_argument("--log-image-size",type=int,default=224)
    ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args()

    aliases=list(il.SOURCES) if args.source=="all" else [args.source]
    for alias in aliases:
        evaluate_alias(args,alias)
    combine(args)


if __name__=="__main__":
    main()
