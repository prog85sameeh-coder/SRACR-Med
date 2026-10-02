from pathlib import Path
import json
import math
import time
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from .adaptive_policy import AdaptiveCoverageController, global_risk_score
from .crypto_benchmark import tile_scores, select_top, select_random, priority_metrics, encrypt_selected_tiles
from .model import load_checkpoint
from .saliency import GradCAM, saliency_stability

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def normalize_tensor(x):
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def raw_tensor(image, size, device):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tf(image).unsqueeze(0).to(device)


def normalized_entropy(probs):
    p = probs.detach().float().clamp_min(1e-8)
    p = p / p.sum().clamp_min(1e-8)
    h = -(p * p.log()).sum().item()
    return float(np.clip(h / math.log(max(int(p.numel()), 2)), 0, 1))


def clinical_prior(class_name, cfg):
    classes = cfg.get("classes", {})
    if class_name in classes:
        return float(classes[class_name])
    lower = {str(k).lower(): float(v) for k, v in classes.items()}
    return float(lower.get(str(class_name).lower(), cfg.get("default", 0.65)))


def _predict(model, image, size, device):
    x = raw_tensor(image, size, device)
    with torch.no_grad():
        probs = torch.softmax(model(normalize_tensor(x)), dim=1)[0]
    return probs


def analyze_priority(model, image, image_size, device, saliency_cfg):
    raw = raw_tensor(image, image_size, device)
    x = normalize_tensor(raw)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax().item())
        uncertainty = normalized_entropy(probs)
    cam_engine = GradCAM(model, model.layer4[-1].conv2)
    try:
        cam, _, _ = cam_engine(x, target_index=pred)
        reliability = saliency_stability(
            model, cam_engine, raw, cam, normalize_tensor,
            n_perturb=int(saliency_cfg.get("perturbations", 4)),
            noise_std=float(saliency_cfg.get("noise_std", 0.02)),
            brightness_delta=float(saliency_cfg.get("brightness_delta", 0.05)),
            contrast_delta=float(saliency_cfg.get("contrast_delta", 0.05)),
        )
    finally:
        cam_engine.remove()
    reliable = np.clip(cam.detach().cpu().numpy().astype(np.float32) * reliability, 0, 1)
    return pred, probs.detach().cpu().numpy(), uncertainty, reliability, reliable


def benchmark(cfg, index_df, dataset_source, model_path, out_dir, samples=20, threat=0.40,
              seed=42, device="cpu", run_utility_inference=True):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model, meta = load_checkpoint(model_path, device=device)
    classes = meta["class_names"]
    image_size = int(meta.get("image_size", 224))
    tile_size = int(cfg.get("crypto", {}).get("tile_size", 64))
    fixed_saliency_coverage = float(cfg.get("benchmark", {}).get("saliency_fixed_coverage", 0.50))
    risk_cfg = cfg.get("global_risk", {})
    ctrl_cfg = cfg.get("controller", {})
    sal_cfg = cfg.get("saliency", {})
    clinical_cfg = cfg.get("clinical_prior", {})

    controller = AdaptiveCoverageController(
        ctrl_cfg.get("thresholds", [0.30, 0.50, 0.70]),
        ctrl_cfg.get("hysteresis", 0.05),
        ctrl_cfg.get("coverage_by_level", {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}),
    )

    subset = index_df[(index_df["dataset_source"] == dataset_source) & (index_df["final_split"] == "test")].copy()
    if subset.empty:
        raise ValueError("No test images for requested source")
    rng = np.random.default_rng(seed)
    n = min(int(samples), len(subset))
    selected_rows = subset.iloc[rng.choice(len(subset), size=n, replace=False)]

    rows = []
    decisions = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SRACR-Med v7.1 Adaptive-Coverage Benchmark")
    print("=" * 72)
    print("Source : %s" % dataset_source)
    print("Samples: %d" % n)
    print("Threat : %.3f" % float(threat))
    print("Device : %s" % device)
    print()

    for j, (_, r) in enumerate(selected_rows.iterrows(), start=1):
        path = Path(r["image_path"])
        print("[%d/%d] %s" % (j, n, path.name), flush=True)
        with Image.open(path) as im:
            image = im.convert("RGB")
        original = np.asarray(image, dtype=np.uint8)
        pred, probs, uncertainty, reliability, reliable_saliency = analyze_priority(
            model, image, image_size, device, sal_cfg
        )
        pred_class = classes[pred]
        clinical = clinical_prior(pred_class, clinical_cfg)
        global_risk, components = global_risk_score(
            reliable_saliency, clinical, threat, uncertainty, risk_cfg.get("weights", {})
        )
        prev_level, target_level, controller_level, sracr_coverage, rotate = controller.update(global_risk, threat)
        decisions.append({
            "image_path": str(path), "predicted_class": pred_class,
            "global_risk": global_risk, "previous_level": prev_level,
            "target_level": target_level, "controller_level": controller_level,
            "target_coverage": sracr_coverage, "key_rotation": rotate,
            **components,
        })

        scores = tile_scores(reliable_saliency, original.shape[:2], tile_size)
        n_tiles = len(scores)
        selections = {
            "full_aes": set(idx for idx, _ in scores),
            "random_matched": select_random(n_tiles, sracr_coverage, rng),
            "saliency_fixed": select_top(scores, fixed_saliency_coverage),
            "sracr_adaptive": select_top(scores, sracr_coverage),
        }

        original_probs = probs
        key = None
        for method, selected in selections.items():
            protected, mass, recall, efficiency = priority_metrics(scores, selected)
            crypt = encrypt_selected_tiles(image, selected, tile_size, key=key)
            recovered = crypt.pop("recovered")
            lossless = bool(np.array_equal(recovered, original))
            if run_utility_inference:
                rec_image = Image.fromarray(recovered, mode="RGB")
                rec_probs = _predict(model, rec_image, image_size, device).detach().cpu().numpy()
                rec_pred = int(rec_probs.argmax())
                pred_preserved = int(rec_pred == pred)
                prob_l1 = float(np.abs(rec_probs - original_probs).sum())
            else:
                pred_preserved = int(lossless)
                prob_l1 = 0.0 if lossless else float("nan")
            rows.append({
                "image_path": str(path),
                "true_label": str(r.get("label", "")),
                "predicted_class": pred_class,
                "confidence": float(original_probs[pred]),
                "uncertainty": float(uncertainty),
                "saliency_reliability": float(reliability),
                "clinical_score": float(clinical),
                "threat": float(threat),
                "saliency_burden": float(components["saliency_burden"]),
                "global_risk": float(global_risk),
                "target_level": int(target_level),
                "controller_level": int(controller_level),
                "target_coverage": float(sracr_coverage),
                "key_rotation": int(rotate),
                "method": method,
                "protected_fraction": protected,
                "priority_mass_coverage": mass,
                "top20_priority_recall": recall,
                "priority_coverage_per_protected_fraction": efficiency,
                "encryption_ms": crypt["encryption_ms"],
                "decryption_ms": crypt["decryption_ms"],
                "encrypted_payload_bytes": crypt["encrypted_payload_bytes"],
                "metadata_overhead_est_bytes": crypt["metadata_overhead_est_bytes"],
                "encrypted_tiles": crypt["encrypted_tiles"],
                "total_tiles": crypt["total_tiles"],
                "lossless": int(lossless),
                "prediction_preserved": pred_preserved,
                "probability_l1": prob_l1,
            })

    detail = pd.DataFrame(rows)
    decision_df = pd.DataFrame(decisions)
    detail.to_csv(out_dir / "benchmark_detail.csv", index=False, encoding="utf-8-sig")
    decision_df.to_csv(out_dir / "policy_decisions.csv", index=False, encoding="utf-8-sig")

    summary = detail.groupby("method", sort=False).agg(
        images=("image_path", "count"),
        protected_fraction_mean=("protected_fraction", "mean"),
        priority_mass_coverage_mean=("priority_mass_coverage", "mean"),
        top20_priority_recall_mean=("top20_priority_recall", "mean"),
        encryption_ms_mean=("encryption_ms", "mean"),
        encryption_ms_std=("encryption_ms", "std"),
        decryption_ms_mean=("decryption_ms", "mean"),
        encrypted_payload_bytes_mean=("encrypted_payload_bytes", "mean"),
        lossless_rate=("lossless", "mean"),
        prediction_preservation_rate=("prediction_preserved", "mean"),
        probability_l1_mean=("probability_l1", "mean"),
        priority_coverage_per_protected_fraction=("priority_coverage_per_protected_fraction", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")

    policy_counts = decision_df.groupby("controller_level").agg(
        images=("image_path", "count"),
        global_risk_mean=("global_risk", "mean"),
        target_coverage_mean=("target_coverage", "mean"),
    ).reset_index()
    policy_counts.to_csv(out_dir / "policy_level_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "benchmark_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "7.1",
            "source": dataset_source,
            "samples": n,
            "threat": float(threat),
            "device": device,
            "model": str(model_path),
            "saliency_fixed_coverage": fixed_saliency_coverage,
            "note": "Reliable saliency determines WHERE; global cyber-clinical risk determines HOW MUCH.",
        }, f, ensure_ascii=False, indent=2)

    print("\n=== BENCHMARK SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== POLICY LEVELS ===")
    print(policy_counts.to_string(index=False))
    print("\nSaved to: %s" % out_dir)
    return summary, detail, decision_df


def threat_sweep(cfg, index_df, dataset_source, model_path, out_dir, threats, samples=20,
                 seed=42, device="cpu"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    # Use same seed and sample set for every threat value.
    for threat in threats:
        tdir = out_dir / ("threat_%0.2f" % float(threat))
        _, detail, decisions = benchmark(
            cfg, index_df, dataset_source, model_path, tdir,
            samples=samples, threat=float(threat), seed=seed, device=device,
            run_utility_inference=False,
        )
        sracr = detail[detail["method"] == "sracr_adaptive"]
        all_rows.append({
            "threat": float(threat),
            "images": len(decisions),
            "global_risk_mean": float(decisions["global_risk"].mean()),
            "controller_level_mean": float(decisions["controller_level"].mean()),
            "protected_fraction_mean": float(sracr["protected_fraction"].mean()),
            "priority_mass_coverage_mean": float(sracr["priority_mass_coverage"].mean()),
            "encryption_ms_mean": float(sracr["encryption_ms"].mean()),
            "critical_rate": float((decisions["controller_level"] == 4).mean()),
            "key_rotation_rate": float(decisions["key_rotation"].mean()),
        })
    sweep = pd.DataFrame(all_rows)
    sweep.to_csv(out_dir / "threat_sweep_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== THREAT SWEEP ===")
    print(sweep.to_string(index=False))
    print("\nSaved to: %s" % (out_dir / "threat_sweep_summary.csv"))
    return sweep
