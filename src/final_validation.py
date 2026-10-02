from __future__ import annotations

from pathlib import Path
import gzip
import math
import pickle
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .adaptive_policy import AdaptiveCoverageController, saliency_burden
from .benchmark import clinical_prior, normalize_tensor, normalized_entropy, raw_tensor
from .crypto_benchmark import tile_scores, tile_slices, select_top, select_random, priority_metrics
from .model import load_checkpoint
from .saliency import GradCAM, saliency_stability
from .multi_validation import ALIASES, bootstrap_ci, _sample_rows


ABLATIONS = (
    "full_sracr",
    "no_reliability",
    "no_saliency",
    "no_clinical",
    "no_threat",
    "no_uncertainty",
    "no_controller",
    "random_matched",
    "full_aes",
)

COMPONENT_KEYS = ("saliency", "clinical", "threat", "uncertainty")


def _base_weights(cfg):
    src = cfg.get("global_risk", {}).get("weights", {})
    w = {
        "saliency": float(src.get("saliency", 0.30)),
        "clinical": float(src.get("clinical", 0.25)),
        "threat": float(src.get("threat", 0.30)),
        "uncertainty": float(src.get("uncertainty", 0.15)),
    }
    total = sum(w.values())
    if total <= 0 or any(v < 0 for v in w.values()):
        raise ValueError("global-risk weights must be non-negative with positive sum")
    return {k: v / total for k, v in w.items()}


def _replacement_values(records, cfg):
    """Dataset-specific neutral replacements used by component ablations.

    The original SRACR weights are kept unchanged. Removing a signal means
    replacing the varying signal with a neutral/reference value, rather than
    deleting its weight and rescaling the remaining components. This preserves
    the global-risk scale and isolates the information carried by that signal.
    """
    fv = cfg.get("final_validation", {})
    nominal_threat = float(fv.get("nominal_threat", 0.40))
    return {
        "saliency": float(np.mean([r["reliable_saliency_burden"] for r in records])),
        "clinical": float(np.mean([r["clinical"] for r in records])),
        "uncertainty": float(np.mean([r["uncertainty"] for r in records])),
        "threat": float(np.clip(nominal_threat, 0.0, 1.0)),
    }


def _risk_from_components(sal_burden, clinical, threat, uncertainty, weights):
    vals = {
        "saliency": float(np.clip(sal_burden, 0, 1)),
        "clinical": float(np.clip(clinical, 0, 1)),
        "threat": float(np.clip(threat, 0, 1)),
        "uncertainty": float(np.clip(uncertainty, 0, 1)),
    }
    return float(np.clip(sum(float(weights[k]) * vals[k] for k in COMPONENT_KEYS), 0, 1))


def _analyze_components(model, image, image_size, device, saliency_cfg):
    raw = raw_tensor(image, image_size, device)
    x = normalize_tensor(raw)
    with torch.no_grad():
        probs_t = torch.softmax(model(x), dim=1)[0]
        pred = int(probs_t.argmax().item())
        uncertainty = normalized_entropy(probs_t)
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
    raw_cam = np.clip(cam.detach().cpu().numpy().astype(np.float32), 0, 1)
    reliable = np.clip(raw_cam * float(reliability), 0, 1)
    return {
        "predicted_index": pred,
        "probs": probs_t.detach().cpu().numpy().astype(np.float32),
        "uncertainty": float(uncertainty),
        "reliability": float(reliability),
        "raw_saliency_burden": float(saliency_burden(raw_cam)),
        "reliable_saliency_burden": float(saliency_burden(reliable)),
        "raw_saliency": raw_cam,
        "reliable_saliency": reliable,
    }


def _cache_path(cache_root, alias, samples, seed, model_path):
    sample_tag = "all" if int(samples) <= 0 else str(int(samples))
    model_tag = Path(model_path).stem.replace(" ", "_")[:80]
    return Path(cache_root) / ("%s_%s_seed%d_%s.pkl.gz" % (alias, sample_tag, int(seed), model_tag))


def _seed_feature_precomputation(seed):
    """Fix RNG state before stochastic saliency perturbations."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def precompute_or_load(cfg, index_df, alias, model_path, cache_root, samples, seed, device, rebuild=False):
    source = ALIASES[alias]
    cache_root = Path(cache_root); cache_root.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_root, alias, samples, seed, model_path)
    if cp.exists() and not rebuild:
        print("Loading cached features: %s" % cp, flush=True)
        with gzip.open(cp, "rb") as f:
            payload = pickle.load(f)
        return payload

    _seed_feature_precomputation(seed)
    model, meta = load_checkpoint(model_path, device=device)
    classes = meta["class_names"]
    image_size = int(meta.get("image_size", 224))
    tile_size = int(cfg.get("crypto", {}).get("tile_size", 64))
    sal_cfg = cfg.get("saliency", {})
    clinical_cfg = cfg.get("clinical_prior", {})
    selected = _sample_rows(index_df, source, samples, seed)
    records = []
    print("Precomputing final-validation features for %s (%d images)..." % (source, len(selected)), flush=True)
    for i, r in selected.iterrows():
        path = Path(r["image_path"])
        print("  [%d/%d] %s" % (i + 1, len(selected), path.name), flush=True)
        with Image.open(path) as im:
            image = im.convert("RGB")
        h, w = np.asarray(image).shape[:2]
        comp = _analyze_components(model, image, image_size, device, sal_cfg)
        pred_class = classes[comp["predicted_index"]]
        raw_scores = tile_scores(comp.pop("raw_saliency"), (h, w), tile_size)
        scores = tile_scores(comp.pop("reliable_saliency"), (h, w), tile_size)
        score_ids = np.asarray([idx for idx, _ in scores], dtype=np.int32)
        score_vals = np.asarray([v for _, v in scores], dtype=np.float32)
        raw_score_vals = np.asarray([v for _, v in raw_scores], dtype=np.float32)
        tile_plain_bytes = []
        for idx, y0, y1, x0, x1 in tile_slices(h, w, tile_size):
            tile_plain_bytes.append(int((y1-y0) * (x1-x0) * 3))
        records.append({
            "image_path": str(path),
            "true_label": str(r.get("label", "")),
            "predicted_class": pred_class,
            "clinical": float(clinical_prior(pred_class, clinical_cfg)),
            "height": int(h), "width": int(w),
            "tile_ids": score_ids,
            "tile_scores": score_vals,
            "raw_tile_scores": raw_score_vals,
            "tile_plain_bytes": np.asarray(tile_plain_bytes, dtype=np.int64),
            **comp,
        })
    payload = {
        "version": "7.3.1",
        "alias": alias,
        "source": source,
        "model": str(model_path),
        "image_size": image_size,
        "tile_size": tile_size,
        "samples": len(records),
        "seed": int(seed),
        "records": records,
    }
    with gzip.open(cp, "wb", compresslevel=4) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved feature cache: %s" % cp, flush=True)
    return payload


def _top_selected(scores, coverage):
    pairs = [(int(i), float(v)) for i, v in zip(scores["ids"], scores["values"])]
    return select_top(pairs, coverage)


def _priority_metrics_arrays(ids, vals, selected):
    return priority_metrics([(int(i), float(v)) for i, v in zip(ids, vals)], selected)


def _payload_bytes(record, selected):
    # cryptography AESGCM.encrypt returns ciphertext with a 16-byte GCM tag.
    plain = record["tile_plain_bytes"]
    return int(sum(int(plain[i]) + 16 for i in selected))


def _variant_risk(variant, rec, threat, base_weights, replacements):
    sal_burden = rec["reliable_saliency_burden"]
    clinical = rec["clinical"]
    threat_value = float(threat)
    uncertainty = rec["uncertainty"]
    controller_threat = float(threat)

    if variant == "no_reliability":
        # Q_S -> 1: use raw Grad-CAM burden instead of S_R = S * Q_S.
        sal_burden = rec["raw_saliency_burden"]
    elif variant == "no_saliency":
        # Remove image-specific saliency information while preserving risk scale.
        sal_burden = replacements["saliency"]
    elif variant == "no_clinical":
        # Replace image/class-specific clinical prior with dataset mean.
        clinical = replacements["clinical"]
    elif variant == "no_threat":
        # Freeze cyber-threat information at a nominal state for every scenario.
        threat_value = replacements["threat"]
        controller_threat = replacements["threat"]
    elif variant == "no_uncertainty":
        # Replace image-specific predictive uncertainty with dataset mean.
        uncertainty = replacements["uncertainty"]

    g = _risk_from_components(sal_burden, clinical, threat_value, uncertainty, base_weights)
    return g, controller_threat


def _evaluate_sequence(cfg, records, threat, sequence_seed, variants, base_weights, replacements):
    ctrl_cfg = cfg.get("controller", {})
    fixed_cov = float(cfg.get("benchmark", {}).get("saliency_fixed_coverage", 0.50))
    controllers = {}
    for v in variants:
        if v not in ("no_controller", "full_aes"):
            controllers[v] = AdaptiveCoverageController(
                ctrl_cfg.get("thresholds", [0.30, 0.50, 0.70]),
                ctrl_cfg.get("hysteresis", 0.05),
                ctrl_cfg.get("coverage_by_level", {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}),
            )
    rng_order = np.random.default_rng(sequence_seed)
    order = rng_order.permutation(len(records))
    rows = []

    # full SRACR coverage is needed for matched-random baseline.
    full_cov_by_image = {}
    full_level_by_image = {}
    full_risk_by_image = {}
    full_rotate_by_image = {}
    for pos in order:
        rec = records[int(pos)]
        g, cthreat = _variant_risk("full_sracr", rec, threat, base_weights, replacements)
        prev, target, level, cov, rotate = controllers["full_sracr"].update(g, cthreat)
        full_cov_by_image[int(pos)] = float(cov)
        full_level_by_image[int(pos)] = int(level)
        full_risk_by_image[int(pos)] = float(g)
        full_rotate_by_image[int(pos)] = int(rotate)

    # reset full controller because the generic loop below must reproduce the same sequence.
    controllers["full_sracr"] = AdaptiveCoverageController(
        ctrl_cfg.get("thresholds", [0.30, 0.50, 0.70]),
        ctrl_cfg.get("hysteresis", 0.05),
        ctrl_cfg.get("coverage_by_level", {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}),
    )

    for pos in order:
        idx = int(pos)
        rec = records[idx]
        ids = rec["tile_ids"]
        vals = rec["tile_scores"]
        n_tiles = len(ids)
        pairs = {"ids": ids, "values": vals}
        for variant in variants:
            if variant == "full_aes":
                g = full_risk_by_image[idx]
                level = 4
                cov = 1.0
                rotate = int(float(threat) >= 0.90)
                selected = set(int(x) for x in ids)
            elif variant == "no_controller":
                g, _ = _variant_risk("full_sracr", rec, threat, base_weights, replacements)
                level = 0
                cov = fixed_cov
                rotate = 0
                selected = _top_selected(pairs, cov)
            elif variant == "random_matched":
                g = full_risk_by_image[idx]
                level = full_level_by_image[idx]
                cov = full_cov_by_image[idx]
                rotate = full_rotate_by_image[idx]
                # deterministic per image/sequence, independent from iteration order
                rr = np.random.default_rng(int(sequence_seed) * 1000003 + idx * 97 + 17)
                selected = select_random(n_tiles, cov, rr)
            else:
                g, cthreat = _variant_risk(variant, rec, threat, base_weights, replacements)
                prev, target, level, cov, rotate = controllers[variant].update(g, cthreat)
                if variant == "no_saliency":
                    rr = np.random.default_rng(int(sequence_seed) * 1000003 + idx * 97 + 31)
                    selected = select_random(n_tiles, cov, rr)
                elif variant == "no_reliability":
                    raw_pairs = {"ids": ids, "values": rec["raw_tile_scores"]}
                    selected = _top_selected(raw_pairs, cov)
                else:
                    selected = _top_selected(pairs, cov)

            protected, mass, recall, efficiency = _priority_metrics_arrays(ids, vals, selected)
            rows.append({
                "image_index": idx,
                "image_path": rec["image_path"],
                "true_label": rec["true_label"],
                "predicted_class": rec["predicted_class"],
                "sequence_seed": int(sequence_seed),
                "threat": float(threat),
                "variant": variant,
                "global_risk": float(g),
                "controller_level": int(level),
                "target_coverage": float(cov),
                "key_rotation": int(rotate),
                "protected_fraction": float(protected),
                "priority_mass_coverage": float(mass),
                "top20_priority_recall": float(recall),
                "priority_efficiency": float(efficiency),
                "encrypted_payload_bytes_est": _payload_bytes(rec, selected),
                "saliency_reliability": float(rec["reliability"]),
                "uncertainty": float(rec["uncertainty"]),
                "clinical_score": float(rec["clinical"]),
            })
    return pd.DataFrame(rows)


def _aggregate_sequences(detail):
    metrics = [
        "global_risk", "controller_level", "target_coverage", "key_rotation",
        "protected_fraction", "priority_mass_coverage", "top20_priority_recall",
        "priority_efficiency", "encrypted_payload_bytes_est",
    ]
    fixed = ["image_path", "true_label", "predicted_class", "saliency_reliability", "uncertainty", "clinical_score"]
    agg = {m: "mean" for m in metrics}
    agg.update({c: "first" for c in fixed})
    return detail.groupby(["threat", "variant", "image_index"], as_index=False).agg(agg)


def _effect_dz(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return 0.0
    sd = float(np.std(d, ddof=1))
    return 0.0 if sd <= 1e-15 else float(np.mean(d) / sd)


def _signflip_permutation_p(diff, rng, reps=5000, exact_max_n=15):
    """Two-sided paired sign-flip permutation test on the mean difference.

    Zero differences are removed. For small effective sample sizes an exact
    enumeration is used; otherwise a Monte-Carlo sign-flip test is used with
    the standard +1 correction.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 1e-15]
    n = int(len(d))
    if n == 0:
        return 1.0, 0, 0
    obs = abs(float(np.mean(d)))
    if n <= int(exact_max_n):
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signs = np.ones(n, dtype=float)
            for j in range(n):
                if (mask >> j) & 1:
                    signs[j] = -1.0
            stat = abs(float(np.mean(signs * d)))
            if stat >= obs - 1e-15:
                extreme += 1
        return float(extreme / total), n, total

    reps = max(1000, int(reps))
    extreme = 0
    # Chunk to avoid large allocations for full held-out test sets.
    remaining = reps
    while remaining > 0:
        k = min(2000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, n), replace=True)
        stats = np.abs(np.mean(signs * d[None, :], axis=1))
        extreme += int(np.sum(stats >= obs - 1e-15))
        remaining -= k
    return float((extreme + 1) / (reps + 1)), n, reps


def _holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def _summaries(dataset_alias, source, image_level, seed, bootstrap_reps):
    rng = np.random.default_rng(seed + 7300)
    summary_rows = []
    metrics = ["global_risk", "controller_level", "protected_fraction", "priority_mass_coverage",
               "top20_priority_recall", "priority_efficiency", "encrypted_payload_bytes_est", "key_rotation"]
    for (threat, variant), g in image_level.groupby(["threat", "variant"], sort=False):
        row = {"dataset": dataset_alias, "dataset_source": source, "threat": float(threat),
               "variant": variant, "images": int(len(g))}
        for metric in metrics:
            vals = g[metric].astype(float).values
            lo, hi = bootstrap_ci(vals, rng, reps=bootstrap_reps)
            row[metric + "_mean"] = float(np.nanmean(vals))
            row[metric + "_ci95_low"] = lo
            row[metric + "_ci95_high"] = hi
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _paired_tests(dataset_alias, source, image_level, seed, bootstrap_reps, permutation_reps):
    metrics = ["protected_fraction", "priority_mass_coverage", "top20_priority_recall",
               "priority_efficiency", "encrypted_payload_bytes_est", "controller_level", "global_risk"]
    compare_variants = [v for v in ABLATIONS if v != "full_sracr"]
    rows = []
    rng = np.random.default_rng(seed + 7311)
    for threat in sorted(image_level["threat"].unique()):
        base = image_level[(image_level["threat"] == threat) & (image_level["variant"] == "full_sracr")]
        for variant in compare_variants:
            alt = image_level[(image_level["threat"] == threat) & (image_level["variant"] == variant)]
            merged = base.merge(alt, on="image_index", suffixes=("_full", "_alt"))
            for metric in metrics:
                diff = merged[metric + "_full"].astype(float).values - merged[metric + "_alt"].astype(float).values
                lo, hi = bootstrap_ci(diff, rng, reps=bootstrap_reps)
                permutation_p, n_effective, permutations_used = _signflip_permutation_p(
                    diff, rng, reps=permutation_reps
                )
                rows.append({
                    "dataset": dataset_alias,
                    "dataset_source": source,
                    "threat": float(threat),
                    "comparison": "full_sracr-vs-" + variant,
                    "variant": variant,
                    "metric": metric,
                    "n_pairs": int(len(diff)),
                    "n_nonzero_pairs": int(n_effective),
                    "full_mean": float(merged[metric + "_full"].mean()),
                    "alternative_mean": float(merged[metric + "_alt"].mean()),
                    "mean_difference_full_minus_alt": float(np.mean(diff)),
                    "difference_ci95_low": lo,
                    "difference_ci95_high": hi,
                    "permutation_p": permutation_p,
                    "permutations_used": int(permutations_used),
                    "paired_effect_dz": _effect_dz(diff),
                })
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["holm_p"] = np.nan
        # Correct the family of ablation comparisons within each dataset/threat/metric.
        for _, idx in tests.groupby(["dataset", "threat", "metric"]).groups.items():
            idx = list(idx)
            tests.loc[idx, "holm_p"] = _holm_adjust(tests.loc[idx, "permutation_p"].values)
        tests["significant_holm_0_05"] = (tests["holm_p"] < 0.05).astype(int)
    return tests


def validate_dataset_final(cfg, index_df, alias, model_path, output_root, threats, samples=0,
                           seed=42, device="cpu", bootstrap_reps=2000, permutation_reps=5000,
                           sequence_reps=5, rebuild_cache=False):
    source = ALIASES[alias]
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root.parent / "feature_cache"
    payload = precompute_or_load(cfg, index_df, alias, model_path, cache_root, samples, seed, device, rebuild_cache)
    records = payload["records"]
    base_weights = _base_weights(cfg)
    replacements = _replacement_values(records, cfg)
    all_seq = []
    for threat in [float(t) for t in threats]:
        print("\n%s | FINAL ABLATION | threat=%.2f" % (source, threat), flush=True)
        for rep in range(int(sequence_reps)):
            seq_seed = int(seed) + rep * 1009 + int(round(threat * 1000)) * 17
            print("  sequence %d/%d seed=%d" % (rep + 1, sequence_reps, seq_seed), flush=True)
            all_seq.append(_evaluate_sequence(cfg, records, threat, seq_seed, ABLATIONS, base_weights, replacements))
    seq_detail = pd.concat(all_seq, ignore_index=True)
    image_level = _aggregate_sequences(seq_detail)
    summary = _summaries(alias, source, image_level, seed, bootstrap_reps)
    tests = _paired_tests(alias, source, image_level, seed, bootstrap_reps, permutation_reps)

    seq_detail.to_csv(output_root / "sequence_level_detail.csv", index=False, encoding="utf-8-sig")
    image_level.to_csv(output_root / "image_level_ablation.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "ablation_summary_ci95.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(output_root / "paired_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**replacements, **{("weight_" + k): v for k, v in base_weights.items()}}]).to_csv(
        output_root / "ablation_replacement_values.csv", index=False, encoding="utf-8-sig"
    )
    return summary, tests, image_level


def final_validate_all(cfg, index_df, model_paths, output_root, threats, samples=0, seed=42,
                       device="cpu", bootstrap_reps=2000, permutation_reps=5000,
                       sequence_reps=5, rebuild_cache=False):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    summaries, tests_all, images_all = [], [], []
    for alias in ("chest", "retinal", "split"):
        print("\n" + "=" * 76)
        print("FINAL ABLATION: %s" % alias)
        print("=" * 76)
        summary, tests, image_level = validate_dataset_final(
            cfg, index_df, alias, model_paths[alias], output_root / alias, threats,
            samples=samples, seed=seed, device=device, bootstrap_reps=bootstrap_reps,
            permutation_reps=permutation_reps, sequence_reps=sequence_reps, rebuild_cache=rebuild_cache,
        )
        summaries.append(summary); tests_all.append(tests)
        tmp = image_level.copy(); tmp["dataset"] = alias; images_all.append(tmp)

    summary_all = pd.concat(summaries, ignore_index=True)
    tests_all_df = pd.concat(tests_all, ignore_index=True)
    image_all = pd.concat(images_all, ignore_index=True)

    expected_variants = set(ABLATIONS)
    for alias in ("chest", "retinal", "split"):
        present = set(summary_all.loc[summary_all["dataset"] == alias, "variant"].unique())
        missing = sorted(expected_variants - present)
        if missing:
            raise RuntimeError("Missing ablation variants for %s: %s" % (alias, missing))

    summary_all.to_csv(output_root / "ALL_DATASETS_ablation_summary_ci95.csv", index=False, encoding="utf-8-sig")
    tests_all_df.to_csv(output_root / "ALL_DATASETS_paired_tests.csv", index=False, encoding="utf-8-sig")
    image_all.to_csv(output_root / "ALL_DATASETS_image_level_ablation.csv", index=False, encoding="utf-8-sig")

    # Macro-dataset means: each dataset contributes equally regardless of test-set size.
    macro = summary_all.groupby(["threat", "variant"], as_index=False).agg(
        datasets=("dataset", "nunique"),
        protected_fraction_mean=("protected_fraction_mean", "mean"),
        priority_mass_coverage_mean=("priority_mass_coverage_mean", "mean"),
        top20_priority_recall_mean=("top20_priority_recall_mean", "mean"),
        priority_efficiency_mean=("priority_efficiency_mean", "mean"),
        controller_level_mean=("controller_level_mean", "mean"),
        global_risk_mean=("global_risk_mean", "mean"),
        key_rotation_rate=("key_rotation_mean", "mean"),
        encrypted_payload_bytes_est_mean=("encrypted_payload_bytes_est_mean", "mean"),
    )
    macro.to_csv(output_root / "CROSS_DATASET_MACRO_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # Publication-oriented table: full SRACR plus each ablation, macro averaged.
    pub = macro[["threat", "variant", "protected_fraction_mean", "priority_mass_coverage_mean",
                 "top20_priority_recall_mean", "priority_efficiency_mean", "controller_level_mean",
                 "key_rotation_rate"]].copy()
    pub.to_csv(output_root / "PUBLICATION_TABLE_ablation.csv", index=False, encoding="utf-8-sig")

    # Significant paired findings only, useful for manuscript drafting.
    sig = tests_all_df[(tests_all_df["significant_holm_0_05"] == 1) &
                       (tests_all_df["metric"].isin(["priority_mass_coverage", "priority_efficiency", "protected_fraction"]))].copy()
    sig.to_csv(output_root / "SIGNIFICANT_PAIRED_FINDINGS.csv", index=False, encoding="utf-8-sig")

    # Threat responsiveness: paired per-image change between the lowest and highest threat.
    threat_values = sorted(float(x) for x in image_all["threat"].unique())
    responsiveness_rows = []
    responsiveness_tests = []
    if len(threat_values) >= 2:
        t_low, t_high = threat_values[0], threat_values[-1]
        resp_metrics = ["global_risk", "protected_fraction", "controller_level", "key_rotation"]
        rrng = np.random.default_rng(seed + 7391)
        for (dataset, variant), g in image_all.groupby(["dataset", "variant"], sort=False):
            lo = g[g["threat"] == t_low].set_index("image_index")
            hi = g[g["threat"] == t_high].set_index("image_index")
            common = lo.index.intersection(hi.index)
            row = {"dataset": dataset, "variant": variant, "threat_low": t_low, "threat_high": t_high,
                   "images": int(len(common))}
            for metric in resp_metrics:
                d = hi.loc[common, metric].astype(float).values - lo.loc[common, metric].astype(float).values
                ci_lo, ci_hi = bootstrap_ci(d, rrng, reps=bootstrap_reps)
                row[metric + "_delta_mean"] = float(np.mean(d))
                row[metric + "_delta_ci95_low"] = ci_lo
                row[metric + "_delta_ci95_high"] = ci_hi
            responsiveness_rows.append(row)

        responsiveness = pd.DataFrame(responsiveness_rows)
        responsiveness.to_csv(output_root / "THREAT_RESPONSIVENESS.csv", index=False, encoding="utf-8-sig")

        # Compare full SRACR responsiveness with no_threat and no_controller.
        for dataset in responsiveness["dataset"].unique():
            for alt_variant in ("no_threat", "no_controller"):
                base_g = image_all[(image_all["dataset"] == dataset) & (image_all["variant"] == "full_sracr")]
                alt_g = image_all[(image_all["dataset"] == dataset) & (image_all["variant"] == alt_variant)]
                for metric in resp_metrics:
                    def deltas(gx):
                        a = gx[gx["threat"] == t_low].set_index("image_index")
                        b = gx[gx["threat"] == t_high].set_index("image_index")
                        common2 = a.index.intersection(b.index)
                        return pd.Series(b.loc[common2, metric].astype(float).values - a.loc[common2, metric].astype(float).values, index=common2)
                    db = deltas(base_g); da = deltas(alt_g)
                    common3 = db.index.intersection(da.index)
                    diff = db.loc[common3].values - da.loc[common3].values
                    ci_lo, ci_hi = bootstrap_ci(diff, rrng, reps=bootstrap_reps)
                    pp, nz, used = _signflip_permutation_p(diff, rrng, reps=permutation_reps)
                    responsiveness_tests.append({
                        "dataset": dataset, "comparison": "full_sracr-vs-" + alt_variant,
                        "metric": metric, "n_pairs": int(len(diff)), "n_nonzero_pairs": int(nz),
                        "mean_difference_in_threat_response": float(np.mean(diff)),
                        "difference_ci95_low": ci_lo, "difference_ci95_high": ci_hi,
                        "permutation_p": pp, "permutations_used": int(used),
                        "paired_effect_dz": _effect_dz(diff),
                    })
        rt = pd.DataFrame(responsiveness_tests)
        if not rt.empty:
            rt["holm_p"] = np.nan
            for _, idx in rt.groupby(["dataset", "metric"]).groups.items():
                idx = list(idx)
                rt.loc[idx, "holm_p"] = _holm_adjust(rt.loc[idx, "permutation_p"].values)
            rt["significant_holm_0_05"] = (rt["holm_p"] < 0.05).astype(int)
            rt.to_csv(output_root / "THREAT_RESPONSIVENESS_PAIRED_TESTS.csv", index=False, encoding="utf-8-sig")

    print("\n=== CROSS-DATASET MACRO ABLATION SUMMARY ===")
    show = macro[macro["variant"].isin(["full_sracr", "no_reliability", "no_saliency", "no_clinical", "no_threat", "no_uncertainty", "no_controller", "random_matched", "full_aes"])]
    print(show.to_string(index=False))
    print("\nSaved to: %s" % output_root)
    return summary_all, tests_all_df, macro
