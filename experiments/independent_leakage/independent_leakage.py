from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SOURCES = {
    "chest": "Detection of Pneumonia Disease using Chest Radiogr",
    "retinal": "Retinal Dataset",
    "split": "split",
}
PRIMARY_ACCURACY = {"chest": 0.9221, "retinal": 0.8750, "split": 0.9146}
CACHE_GLOB = {
    "chest": "chest_all_seed42_chest_xray_resnet18.pkl.gz",
    "retinal": "retinal_all_seed42_retinal_11class_resnet18.pkl.gz",
    "split": "split_all_seed42_fundus_4class_resnet18.pkl.gz",
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def resolve_path(raw: str, dataset_root: Optional[str] = None) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    if dataset_root:
        # Re-map the tail after the original SRACR-Med directory.
        win = PureWindowsPath(str(raw))
        parts = list(win.parts)
        lower = [x.lower() for x in parts]
        try:
            i = lower.index("sracr-med")
            rel = parts[i + 1:]
            candidate = Path(dataset_root).joinpath(*rel)
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    return p


class IndexedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: Sequence[str], image_size: int, train: bool,
                 dataset_root: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.dataset_root = dataset_root
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = resolve_path(str(row["image_path"]), self.dataset_root)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}\nOriginal CSV path: {row['image_path']}")
        with Image.open(path) as im:
            image = im.convert("RGB")
        y = self.class_to_idx[str(row["label"])]
        return self.tf(image), y, str(row["image_path"])


def build_attacker(num_classes: int, pretrained: bool = True):
    used_pretrained = False
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            used_pretrained = True
        except Exception as exc:
            print(f"WARNING: ImageNet weights could not be loaded ({exc}). Falling back to random initialization.")
            model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model, used_pretrained


def class_weights(df: pd.DataFrame, classes: Sequence[str]) -> torch.Tensor:
    counts = df["label"].astype(str).value_counts().reindex(classes).fillna(0).values.astype(np.float32)
    counts[counts <= 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def predict_loader(model, loader, device):
    model.eval()
    ys, ps, paths = [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            ys.extend(y.numpy().tolist())
            ps.append(probs.cpu().numpy())
            paths.extend(list(p))
    return np.asarray(ys), np.concatenate(ps, axis=0), paths


def metrics_from_probs(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, probs, labels=list(range(probs.shape[1])))),
    }


def train_attacker(args, alias: str):
    set_seed(args.seed)
    index = pd.read_csv(args.dataset_index)
    source = SOURCES[alias]
    d = index[(index["dataset_source"] == source) & (index["valid_image"] == True)].copy()  # noqa: E712
    classes = sorted(d["label"].astype(str).unique().tolist())
    tr = d[d["final_split"] == "train"].copy()
    va = d[d["final_split"] == "val"].copy()
    te = d[d["final_split"] == "test"].copy()
    print(f"[{alias}] classes={len(classes)} train={len(tr)} val={len(va)} test={len(te)}")

    train_ds = IndexedDataset(tr, classes, args.image_size, True, args.dataset_root)
    val_ds = IndexedDataset(va, classes, args.image_size, False, args.dataset_root)
    test_ds = IndexedDataset(te, classes, args.image_size, False, args.dataset_root)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, used_pretrained = build_attacker(len(classes), pretrained=not args.no_pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr, classes).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    out_dir = Path(args.attacker_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{alias}_mobilenetv3small_independent.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for bi, (x, y, _) in enumerate(train_ld, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(y)
            n += len(y)
            if bi % 25 == 0 or bi == len(train_ld):
                print(f"[{alias}] epoch {epoch}/{args.epochs} batch {bi}/{len(train_ld)}", flush=True)
        yv, pv, _ = predict_loader(model, val_ld, device)
        vf1 = f1_score(yv, pv.argmax(1), average="macro", zero_division=0)
        sched.step(vf1)
        row = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "val_macro_f1": float(vf1), "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        print(f"[{alias}] epoch={epoch} loss={row['train_loss']:.5f} val_macro_f1={vf1:.5f}")
        if vf1 > best_f1:
            best_f1 = float(vf1)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "image_size": args.image_size,
                "architecture": "mobilenet_v3_small",
                "independent_from_primary_saliency_model": True,
                "pretrained_used": used_pretrained,
                "seed": args.seed,
                "best_val_macro_f1": best_f1,
            }, ckpt)

    saved = torch.load(ckpt, map_location=device)
    model, _ = build_attacker(len(saved["classes"]), pretrained=False)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    yt, pt, paths = predict_loader(model, test_ld, device)
    m = metrics_from_probs(yt, pt)
    m["best_val_macro_f1"] = float(saved["best_val_macro_f1"])
    m["pretrained_used"] = bool(saved.get("pretrained_used", False))
    m["primary_classifier_accuracy_reference"] = PRIMARY_ACCURACY[alias]
    m["accuracy_gap_vs_primary_pp"] = 100.0 * (PRIMARY_ACCURACY[alias] - m["accuracy"])
    quality_ok = m["accuracy"] >= max(0.50, PRIMARY_ACCURACY[alias] - 0.12)
    m["quality_gate_pass"] = bool(quality_ok)
    report = {"dataset": alias, "classes": classes, "history": history, "test_metrics": m, "checkpoint": str(ckpt)}
    with open(out_dir / f"{alias}_attacker_training.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pd.DataFrame({
        "image_path": paths,
        "true_idx": yt,
        "pred_idx": pt.argmax(1),
        "confidence": pt.max(1),
        "true_probability": pt[np.arange(len(yt)), yt],
    }).to_csv(out_dir / f"{alias}_attacker_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(m, indent=2))
    if not quality_ok:
        print("WARNING: independent attacker quality gate did not pass. Do not use leakage conclusions until attacker performance is improved.")


def load_attacker(path: Path, device: str):
    ck = torch.load(path, map_location=device)
    if ck.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Expected mobilenet_v3_small independent attacker checkpoint")
    model, _ = build_attacker(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def load_cache(path: Path):
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def tile_geometry(width: int, height: int, tile_size: int):
    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    areas = []
    boxes = []
    for r in range(rows):
        y0, y1 = r * tile_size, min(height, (r + 1) * tile_size)
        for c in range(cols):
            x0, x1 = c * tile_size, min(width, (c + 1) * tile_size)
            boxes.append((x0, y0, x1, y1))
            areas.append((x1 - x0) * (y1 - y0))
    return np.asarray(areas, dtype=np.int64), boxes


def stable_desc_order(scores: np.ndarray) -> np.ndarray:
    ids = np.arange(len(scores))
    # Primary key = descending score, secondary = ascending tile id.
    return np.lexsort((ids, -np.nan_to_num(scores, nan=-np.inf)))


def choose_prefix_for_target_area(order: np.ndarray, areas: np.ndarray, target_area: int) -> np.ndarray:
    if target_area <= 0:
        return np.empty(0, dtype=np.int64)
    total = int(areas.sum())
    if target_area >= total:
        return np.arange(len(areas), dtype=np.int64)
    cs = np.cumsum(areas[order])
    k = int(np.searchsorted(cs, target_area, side="left"))
    candidates = []
    if k < len(order): candidates.append(k)
    if k > 0: candidates.append(k - 1)
    best = min(candidates, key=lambda j: abs(int(cs[j]) - target_area))
    return order[:best + 1]


def content_variance_scores(arr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    gray = arr.astype(np.float32).mean(axis=2)
    out = np.empty(len(boxes), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        out[i] = float(gray[y0:y1, x0:x1].var())
    return out


def center_scores(boxes, width, height):
    cx, cy = width / 2.0, height / 2.0
    maxd = math.sqrt(cx * cx + cy * cy) + 1e-12
    vals = []
    for x0, y0, x1, y1 in boxes:
        tx, ty = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        vals.append(1.0 - math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) / maxd)
    return np.asarray(vals, dtype=np.float32)


def random_order(n: int, seed: int, image_path: str, budget: float) -> np.ndarray:
    key = f"{seed}|{image_path}|{budget:.6f}".encode("utf-8")
    h = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(h)
    return rng.permutation(n)


def mask_tiles(image_arr: np.ndarray, selected: np.ndarray, boxes, fill: int = 127) -> Image.Image:
    out = image_arr.copy()
    for idx in selected:
        x0, y0, x1, y1 = boxes[int(idx)]
        out[y0:y1, x0:x1, :] = fill
    return Image.fromarray(out)


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(max(len(p), 2)))


def bootstrap_metric_ci(y_true, y_pred, metric: str, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        if metric == "accuracy":
            vals[b] = np.mean(y_true[ix] == y_pred[ix])
        elif metric == "macro_f1":
            vals[b] = f1_score(y_true[ix], y_pred[ix], average="macro", zero_division=0)
        else:
            raise ValueError(metric)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        means[i] = float(np.mean(values[ix]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signflip_p(diff: np.ndarray, n_perm: int, seed: int):
    diff = np.asarray(diff, dtype=np.float64)
    obs = abs(float(diff.mean()))
    if np.allclose(diff, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    count = 1
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff))
        if abs(float(np.mean(diff * signs))) >= obs - 1e-15:
            count += 1
    return count / (n_perm + 1)


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(1.0, running)
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out


def evaluate_alias(args, alias: str):
    set_seed(args.seed)
    index = pd.read_csv(args.dataset_index)
    source = SOURCES[alias]
    test = index[(index["dataset_source"] == source) & (index["final_split"] == "test")].copy()
    if test.empty:
        raise RuntimeError(f"No test rows for {alias}")

    cache_path = Path(args.cache_dir) / CACHE_GLOB[alias]
    cache = load_cache(cache_path)
    records = cache["records"]
    tile_size = int(cache.get("tile_size", 64))
    rec_by_path = {str(r["image_path"]): r for r in records}
    missing = [p for p in test["image_path"].astype(str) if p not in rec_by_path]
    if missing:
        raise RuntimeError(f"{len(missing)} test paths missing from cache; first: {missing[0]}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt_path = Path(args.attacker_dir) / f"{alias}_mobilenetv3small_independent.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Independent attacker checkpoint not found: {ckpt_path}\nRun train-attacker first.")
    model, attack_ck = load_attacker(ckpt_path, device)
    classes = attack_ck["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    tf = transforms.Compose([
        transforms.Resize((int(attack_ck.get("image_size", 224)), int(attack_ck.get("image_size", 224)))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    methods = ["sracr_saliency", "content_variance", "center_roi", "random"]
    budgets = [float(x) for x in args.budgets]
    rows = []

    for pos, row in test.reset_index(drop=True).iterrows():
        original_csv_path = str(row["image_path"])
        path = resolve_path(original_csv_path, args.dataset_root)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}\nOriginal CSV path: {original_csv_path}")
        rec = rec_by_path[original_csv_path]
        with Image.open(path) as im:
            image = im.convert("RGB")
            arr = np.asarray(image).copy()
        height, width = arr.shape[:2]
        areas, boxes = tile_geometry(width, height, tile_size)
        sracr_scores = np.asarray(rec["tile_scores"], dtype=np.float32)
        if len(sracr_scores) != len(boxes):
            raise RuntimeError(f"Tile-count mismatch for {path}: cache={len(sracr_scores)}, geometry={len(boxes)}")
        content_scores = content_variance_scores(arr, boxes)
        ctr_scores = center_scores(boxes, width, height)
        score_map = {
            "sracr_saliency": sracr_scores,
            "content_variance": content_scores,
            "center_roi": ctr_scores,
        }

        conditions = [("unprotected", 0.0, np.empty(0, dtype=np.int64), 0.0, 0.0)]
        total_area = int(areas.sum())
        sracr_order = stable_desc_order(sracr_scores)
        for budget in budgets:
            # Define the actual per-image budget using the SRACR ordering, then match all alternatives to that pixel area.
            nominal_target = int(round(budget * total_area))
            sracr_sel = choose_prefix_for_target_area(sracr_order, areas, nominal_target)
            matched_target_area = int(areas[sracr_sel].sum())
            actual = matched_target_area / total_area
            conditions.append(("sracr_saliency", budget, sracr_sel, actual, actual - budget))
            for method in ["content_variance", "center_roi"]:
                order = stable_desc_order(score_map[method])
                sel = choose_prefix_for_target_area(order, areas, matched_target_area)
                frac = float(areas[sel].sum() / total_area) if len(sel) else 0.0
                conditions.append((method, budget, sel, frac, frac - actual))
            order = random_order(len(areas), args.seed, original_csv_path, budget)
            sel = choose_prefix_for_target_area(order, areas, matched_target_area)
            frac = float(areas[sel].sum() / total_area) if len(sel) else 0.0
            conditions.append(("random", budget, sel, frac, frac - actual))

        if args.include_full_mask:
            all_sel = np.arange(len(areas), dtype=np.int64)
            conditions.append(("full_mask", 1.0, all_sel, 1.0, 0.0))

        tensors = []
        for method, budget, selected, _, _ in conditions:
            if len(selected) == 0:
                masked = Image.fromarray(arr)
            else:
                masked = mask_tiles(arr, selected, boxes, fill=args.mask_value)
            tensors.append(tf(masked))
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()

        y = class_to_idx[str(row["label"])]
        for j, (method, budget, selected, protected_frac, budget_error) in enumerate(conditions):
            p = probs[j]
            pred = int(p.argmax())
            rows.append({
                "dataset": alias,
                "dataset_source": source,
                "image_index": pos,
                "image_path": original_csv_path,
                "true_label": str(row["label"]),
                "true_idx": y,
                "method": method,
                "nominal_budget": budget,
                "protected_pixel_fraction": protected_frac,
                "budget_match_error": budget_error,
                "n_protected_tiles": int(len(selected)),
                "n_total_tiles": int(len(areas)),
                "pred_idx": pred,
                "pred_label": classes[pred],
                "correct": int(pred == y),
                "confidence": float(p[pred]),
                "true_probability": float(p[y]),
                "nll": float(-math.log(max(float(p[y]), 1e-12))),
                "predictive_entropy": entropy_from_probs(p),
            })
        if (pos + 1) % 25 == 0 or pos + 1 == len(test):
            print(f"[{alias}] leakage evaluation {pos+1}/{len(test)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_df = pd.DataFrame(rows)
    image_df.to_csv(out_dir / f"{alias}_independent_leakage_image_level.csv", index=False, encoding="utf-8-sig")

    # Summary with bootstrap CIs.
    summary_rows = []
    for (method, budget), g in image_df.groupby(["method", "nominal_budget"], sort=True):
        y = g["true_idx"].to_numpy(int)
        pred = g["pred_idx"].to_numpy(int)
        acc = float(np.mean(y == pred))
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        bal = float(balanced_accuracy_score(y, pred))
        acc_lo, acc_hi = bootstrap_metric_ci(y, pred, "accuracy", args.bootstrap, args.seed + 100)
        f1_lo, f1_hi = bootstrap_metric_ci(y, pred, "macro_f1", args.bootstrap, args.seed + 200)
        tp = g["true_probability"].to_numpy(float)
        nll = g["nll"].to_numpy(float)
        tp_lo, tp_hi = bootstrap_mean_ci(tp, args.bootstrap, args.seed + 300)
        nll_lo, nll_hi = bootstrap_mean_ci(nll, args.bootstrap, args.seed + 400)
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
    base_acc = float(summary[(summary.method == "unprotected") & (summary.nominal_budget == 0.0)]["attacker_accuracy"].iloc[0])
    base_f1 = float(summary[(summary.method == "unprotected") & (summary.nominal_budget == 0.0)]["attacker_macro_f1"].iloc[0])
    summary["accuracy_drop_from_unprotected_pp"] = 100.0 * (base_acc - summary["attacker_accuracy"])
    summary["macro_f1_drop_from_unprotected_pp"] = 100.0 * (base_f1 - summary["attacker_macro_f1"])
    summary.to_csv(out_dir / f"{alias}_independent_leakage_summary.csv", index=False, encoding="utf-8-sig")

    # Paired SRACR-vs-alternative comparisons under exactly matched nominal budgets.
    pairs = []
    alternatives = ["random", "content_variance", "center_roi"]
    for budget in budgets:
        sr = image_df[(image_df.method == "sracr_saliency") & np.isclose(image_df.nominal_budget, budget)].sort_values("image_index")
        for alt in alternatives:
            aa = image_df[(image_df.method == alt) & np.isclose(image_df.nominal_budget, budget)].sort_values("image_index")
            if len(sr) != len(aa) or not np.array_equal(sr.image_index.values, aa.image_index.values):
                raise RuntimeError("Paired rows misaligned")
            diffs = {
                # Positive values favor SRACR privacy: alternative leaks more / performs better.
                "correctness_alt_minus_sracr": aa.correct.to_numpy(float) - sr.correct.to_numpy(float),
                "true_probability_alt_minus_sracr": aa.true_probability.to_numpy(float) - sr.true_probability.to_numpy(float),
                # Positive means SRACR produces larger attacker NLL than alternative.
                "nll_sracr_minus_alt": sr.nll.to_numpy(float) - aa.nll.to_numpy(float),
            }
            for metric, diff in diffs.items():
                lo, hi = bootstrap_mean_ci(diff, args.bootstrap, args.seed + 500)
                p = signflip_p(diff, args.permutations, args.seed + 600)
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
                    "permutation_p": p,
                })
    pair_df = pd.DataFrame(pairs)
    pair_df["holm_p"] = np.nan
    for metric, ix in pair_df.groupby("metric").groups.items():
        pair_df.loc[ix, "holm_p"] = holm_adjust(pair_df.loc[ix, "permutation_p"].values)
    pair_df["significant_holm_0_05"] = (pair_df["holm_p"] < 0.05).astype(int)
    pair_df.to_csv(out_dir / f"{alias}_independent_leakage_paired_tests.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "dataset": alias,
        "test_images": int(len(test)),
        "independent_attacker": "MobileNetV3-Small",
        "attacker_checkpoint": str(ckpt_path),
        "attacker_independent_from_saliency_generator": True,
        "mask_value_rgb": [args.mask_value] * 3,
        "tile_size_pixels_native": tile_size,
        "budgets": budgets,
        "methods": methods,
        "budget_matching": "SRACR defines the per-image protected-pixel target at each nominal budget; every alternative selects a score-ranked prefix whose cumulative tile area is closest to the same target.",
        "selection_tie_rule": "descending score, then ascending tile ID",
        "random_seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "signflip_permutations": args.permutations,
        "scientific_interpretation": "Lower independent-attacker accuracy/macro-F1/true-class probability and higher NLL indicate less diagnostically recoverable information in unprotected regions under this attacker-view model. PMC is not used as the validation target.",
    }
    with open(out_dir / f"{alias}_independent_leakage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[{alias}] wrote independent leakage outputs to {out_dir}")


def combine_outputs(args):
    out = Path(args.output_dir)
    summaries, images, pairs = [], [], []
    for alias in SOURCES:
        for coll, suffix in [(summaries, "summary"), (images, "image_level"), (pairs, "paired_tests")]:
            p = out / f"{alias}_independent_leakage_{suffix}.csv"
            if p.exists(): coll.append(pd.read_csv(p))
    if summaries:
        s = pd.concat(summaries, ignore_index=True)
        s.to_csv(out / "ALL_DATASETS_independent_leakage_summary.csv", index=False, encoding="utf-8-sig")
        # Cross-dataset macro means at each method/budget.
        num_cols = [c for c in s.columns if c not in {"dataset", "method"} and pd.api.types.is_numeric_dtype(s[c])]
        macro = s.groupby(["method", "nominal_budget"], as_index=False)[num_cols].mean(numeric_only=True)
        macro.to_csv(out / "CROSS_DATASET_MACRO_independent_leakage.csv", index=False, encoding="utf-8-sig")
    if images:
        pd.concat(images, ignore_index=True).to_csv(out / "ALL_DATASETS_independent_leakage_image_level.csv", index=False, encoding="utf-8-sig")
    if pairs:
        p = pd.concat(pairs, ignore_index=True)
        p.to_csv(out / "ALL_DATASETS_independent_leakage_paired_tests.csv", index=False, encoding="utf-8-sig")
    print("Combined outputs written.")


def main():
    ap = argparse.ArgumentParser(description="Reviewer-1 Comment-1: independent residual diagnostic leakage evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-index", default="dataset_index.csv")
    common.add_argument("--dataset-root", default=None, help="Optional replacement root if CSV absolute paths have moved")
    common.add_argument("--attacker-dir", default="attacker_models")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--cpu", action="store_true")

    p = sub.add_parser("train-attacker", parents=[common])
    p.add_argument("--source", choices=["chest", "retinal", "split", "all"], required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-pretrained", action="store_true")

    p = sub.add_parser("evaluate", parents=[common])
    p.add_argument("--source", choices=["chest", "retinal", "split", "all"], required=True)
    p.add_argument("--cache-dir", default="feature_cache")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    p.add_argument("--mask-value", type=int, default=127)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--include-full-mask", action="store_true")

    p = sub.add_parser("combine")
    p.add_argument("--output-dir", default="outputs")

    args = ap.parse_args()
    if args.cmd == "train-attacker":
        aliases = list(SOURCES) if args.source == "all" else [args.source]
        for alias in aliases:
            train_attacker(args, alias)
    elif args.cmd == "evaluate":
        aliases = list(SOURCES) if args.source == "all" else [args.source]
        for alias in aliases:
            evaluate_alias(args, alias)
        combine_outputs(args)
    else:
        combine_outputs(args)


if __name__ == "__main__":
    main()
