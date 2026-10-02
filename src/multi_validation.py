from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .adaptive_policy import AdaptiveCoverageController, global_risk_score
from .benchmark import analyze_priority, clinical_prior, _predict
from .crypto_benchmark import tile_scores, select_top, select_random, priority_metrics, encrypt_selected_tiles
from .model import load_checkpoint

ALIASES = {
    "chest": "Detection of Pneumonia Disease using Chest Radiogr",
    "retinal": "Retinal Dataset",
    "split": "split",
}


def bootstrap_ci(values, rng, reps=1000, alpha=0.05):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    means = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _sample_rows(index_df, source, samples, seed):
    subset = index_df[(index_df["dataset_source"] == source) & (index_df["final_split"] == "test")].copy()
    if subset.empty:
        raise ValueError("No test images for %s" % source)
    if samples is None or int(samples) <= 0 or int(samples) >= len(subset):
        return subset.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return subset.iloc[rng.choice(len(subset), size=int(samples), replace=False)].reset_index(drop=True)


def precompute_features(cfg, index_df, source, model_path, samples, seed, device):
    model, meta = load_checkpoint(model_path, device=device)
    classes = meta["class_names"]
    image_size = int(meta.get("image_size", 224))
    sal_cfg = cfg.get("saliency", {})
    clinical_cfg = cfg.get("clinical_prior", {})
    selected = _sample_rows(index_df, source, samples, seed)
    features = []
    print("Precomputing model/saliency features once for %s (%d images)..." % (source, len(selected)))
    for i, r in selected.iterrows():
        path = Path(r["image_path"])
        print("  [%d/%d] %s" % (i + 1, len(selected), path.name), flush=True)
        with Image.open(path) as im:
            image = im.convert("RGB")
        pred, probs, uncertainty, reliability, reliable_saliency = analyze_priority(model, image, image_size, device, sal_cfg)
        pred_class = classes[pred]
        features.append({
            "image_path": str(path),
            "true_label": str(r.get("label", "")),
            "predicted_class": pred_class,
            "predicted_index": int(pred),
            "probs": probs,
            "uncertainty": float(uncertainty),
            "reliability": float(reliability),
            "clinical": float(clinical_prior(pred_class, clinical_cfg)),
            "saliency": reliable_saliency,
        })
    return model, meta, features


def _crypto_once(image, scores, selected, tile_size, model, image_size, device, original_probs, pred_idx, utility):
    protected, mass, recall, efficiency = priority_metrics(scores, selected)
    crypt = encrypt_selected_tiles(image, selected, tile_size)
    recovered = crypt.pop("recovered")
    original = np.asarray(image, dtype=np.uint8)
    lossless = int(np.array_equal(recovered, original))
    if utility:
        rec_probs = _predict(model, Image.fromarray(recovered, mode="RGB"), image_size, device).detach().cpu().numpy()
        pred_preserved = int(int(rec_probs.argmax()) == int(pred_idx))
        prob_l1 = float(np.abs(rec_probs - original_probs).sum())
    else:
        pred_preserved = lossless
        prob_l1 = 0.0 if lossless else float("nan")
    return {
        "protected_fraction": protected,
        "priority_mass_coverage": mass,
        "top20_priority_recall": recall,
        "priority_coverage_per_protected_fraction": efficiency,
        "encryption_ms": crypt["encryption_ms"],
        "decryption_ms": crypt["decryption_ms"],
        "encrypted_payload_bytes": crypt["encrypted_payload_bytes"],
        "encrypted_tiles": crypt["encrypted_tiles"],
        "total_tiles": crypt["total_tiles"],
        "lossless": lossless,
        "prediction_preserved": pred_preserved,
        "probability_l1": prob_l1,
    }


def validate_dataset(cfg, index_df, source, model_path, out_dir, threats, samples=0, seed=42,
                     device="cpu", utility=True, bootstrap_reps=1000):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, meta, features = precompute_features(cfg, index_df, source, model_path, samples, seed, device)
    image_size = int(meta.get("image_size", 224))
    tile_size = int(cfg.get("crypto", {}).get("tile_size", 64))
    fixed_coverage = float(cfg.get("benchmark", {}).get("saliency_fixed_coverage", 0.50))
    risk_weights = cfg.get("global_risk", {}).get("weights", {})
    ctrl_cfg = cfg.get("controller", {})
    rng = np.random.default_rng(seed)
    all_detail = []
    sweep_rows = []

    for threat in [float(t) for t in threats]:
        controller = AdaptiveCoverageController(
            ctrl_cfg.get("thresholds", [0.30, 0.50, 0.70]),
            ctrl_cfg.get("hysteresis", 0.05),
            ctrl_cfg.get("coverage_by_level", {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}),
        )
        print("\n%s | threat=%.2f" % (source, threat))
        threat_rows = []
        decision_rows = []
        for i, feat in enumerate(features):
            path = Path(feat["image_path"])
            with Image.open(path) as im:
                image = im.convert("RGB")
            original = np.asarray(image, dtype=np.uint8)
            g, comp = global_risk_score(feat["saliency"], feat["clinical"], threat, feat["uncertainty"], risk_weights)
            prev, target, level, coverage, rotate = controller.update(g, threat)
            scores = tile_scores(feat["saliency"], original.shape[:2], tile_size)
            n_tiles = len(scores)
            selections = {
                "full_aes": set(idx for idx, _ in scores),
                "random_matched": select_random(n_tiles, coverage, rng),
                "saliency_fixed": select_top(scores, fixed_coverage),
                "sracr_adaptive": select_top(scores, coverage),
            }
            decision_rows.append({
                "dataset_source": source, "threat": threat, "image_path": str(path),
                "predicted_class": feat["predicted_class"], "uncertainty": feat["uncertainty"],
                "saliency_reliability": feat["reliability"], "clinical_score": feat["clinical"],
                "saliency_burden": comp["saliency_burden"], "global_risk": g,
                "previous_level": prev, "target_level": target, "controller_level": level,
                "target_coverage": coverage, "key_rotation": int(rotate),
            })
            for method, selected in selections.items():
                m = _crypto_once(image, scores, selected, tile_size, model, image_size, device,
                                 feat["probs"], feat["predicted_index"], utility)
                threat_rows.append({
                    "dataset_source": source, "threat": threat, "image_path": str(path),
                    "true_label": feat["true_label"], "predicted_class": feat["predicted_class"],
                    "uncertainty": feat["uncertainty"], "saliency_reliability": feat["reliability"],
                    "clinical_score": feat["clinical"], "global_risk": g, "controller_level": level,
                    "target_coverage": coverage, "key_rotation": int(rotate), "method": method, **m,
                })
        tdf = pd.DataFrame(threat_rows)
        ddf = pd.DataFrame(decision_rows)
        tdir = out_dir / ("threat_%0.2f" % threat); tdir.mkdir(parents=True, exist_ok=True)
        tdf.to_csv(tdir / "benchmark_detail.csv", index=False, encoding="utf-8-sig")
        ddf.to_csv(tdir / "policy_decisions.csv", index=False, encoding="utf-8-sig")
        all_detail.append(tdf)

        sr = tdf[tdf["method"] == "sracr_adaptive"]
        sweep_rows.append({
            "dataset_source": source, "threat": threat, "images": len(ddf),
            "global_risk_mean": float(ddf["global_risk"].mean()),
            "controller_level_mean": float(ddf["controller_level"].mean()),
            "protected_fraction_mean": float(sr["protected_fraction"].mean()),
            "priority_mass_coverage_mean": float(sr["priority_mass_coverage"].mean()),
            "top20_priority_recall_mean": float(sr["top20_priority_recall"].mean()),
            "encryption_ms_mean": float(sr["encryption_ms"].mean()),
            "encrypted_payload_bytes_mean": float(sr["encrypted_payload_bytes"].mean()),
            "critical_rate": float((ddf["controller_level"] == 4).mean()),
            "key_rotation_rate": float(ddf["key_rotation"].mean()),
            "lossless_rate": float(sr["lossless"].mean()),
            "prediction_preservation_rate": float(sr["prediction_preserved"].mean()),
        })

    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(out_dir / "all_threats_detail.csv", index=False, encoding="utf-8-sig")
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out_dir / "threat_sweep_summary.csv", index=False, encoding="utf-8-sig")

    # Method summary with image-level bootstrap CIs pooled across threats.
    ci_rng = np.random.default_rng(seed + 991)
    summary_rows = []
    metrics = ["protected_fraction", "priority_mass_coverage", "top20_priority_recall", "encryption_ms", "decryption_ms", "encrypted_payload_bytes"]
    for (threat, method), gdf in detail.groupby(["threat", "method"], sort=False):
        row = {"dataset_source": source, "threat": threat, "method": method, "images": len(gdf)}
        for metric in metrics:
            vals = gdf[metric].astype(float).values
            lo, hi = bootstrap_ci(vals, ci_rng, reps=bootstrap_reps)
            row[metric + "_mean"] = float(np.nanmean(vals))
            row[metric + "_ci95_low"] = lo
            row[metric + "_ci95_high"] = hi
        row["lossless_rate"] = float(gdf["lossless"].mean())
        row["prediction_preservation_rate"] = float(gdf["prediction_preserved"].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "method_summary_ci95.csv", index=False, encoding="utf-8-sig")
    return sweep, summary, detail


def validate_all(cfg, index_df, model_paths, output_root, threats, samples=0, seed=42,
                 device="cpu", utility=True, bootstrap_reps=1000):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    sweeps, summaries = [], []
    for alias in ("chest", "retinal", "split"):
        source = ALIASES[alias]
        if alias not in model_paths:
            raise FileNotFoundError("Missing model for %s" % alias)
        out = output_root / alias
        sweep, summary, _ = validate_dataset(
            cfg, index_df, source, model_paths[alias], out, threats, samples=samples,
            seed=seed, device=device, utility=utility, bootstrap_reps=bootstrap_reps,
        )
        sweeps.append(sweep); summaries.append(summary)
    all_sweep = pd.concat(sweeps, ignore_index=True)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_sweep.to_csv(output_root / "ALL_DATASETS_threat_sweep.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(output_root / "ALL_DATASETS_method_summary_ci95.csv", index=False, encoding="utf-8-sig")

    cross = all_sweep.groupby("threat").agg(
        datasets=("dataset_source", "nunique"),
        global_risk_mean=("global_risk_mean", "mean"),
        controller_level_mean=("controller_level_mean", "mean"),
        protected_fraction_mean=("protected_fraction_mean", "mean"),
        priority_mass_coverage_mean=("priority_mass_coverage_mean", "mean"),
        encryption_ms_mean=("encryption_ms_mean", "mean"),
        critical_rate=("critical_rate", "mean"),
        key_rotation_rate=("key_rotation_rate", "mean"),
    ).reset_index()
    cross.to_csv(output_root / "CROSS_DATASET_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== CROSS-DATASET SRACR SUMMARY ===")
    print(cross.to_string(index=False))
    print("\nSaved to: %s" % output_root)
    return all_sweep, all_summary, cross
