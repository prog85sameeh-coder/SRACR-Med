# SRACR-Med v7.3.1 — Statistical Fix

v7.3.1 keeps the SRACR-Med architecture frozen and fixes the final ablation/statistical protocol discovered during the v7.3 manuscript run.

## What changed from v7.3

### 1. Replacement ablations instead of weight renormalization

The original risk weights are kept fixed. A removed information source is replaced with a neutral/reference value so the global-risk scale does not change merely because a component was ablated.

- `full_sracr`: complete method.
- `no_reliability`: sets the reliability adjustment effectively to `Q_S = 1`; raw Grad-CAM is used for both saliency burden and spatial tile ranking.
- `no_saliency`: replaces image-specific saliency burden with the dataset mean and uses deterministic random spatial selection.
- `no_clinical`: replaces the image/class-specific clinical criticality prior with the dataset mean clinical prior.
- `no_threat`: freezes threat at the nominal state `T = 0.40` in both global risk and the controller, so it cannot adapt to the externally swept threat scenario.
- `no_uncertainty`: replaces image-specific predictive uncertainty with the dataset mean uncertainty.
- `no_controller`: fixed 50% reliable-saliency-prioritized coverage and no key rotation.
- `random_matched`: random tile selection with the same per-image coverage chosen by full SRACR.
- `full_aes`: 100% protection reference.

The replacement values and the unchanged component weights are saved for every dataset in `ablation_replacement_values.csv`.

### 2. Paired sign-flip permutation tests

The primary paired significance test is now a two-sided paired sign-flip permutation test on the mean difference. Small effective samples use exact enumeration automatically; larger samples use Monte-Carlo permutations. This avoids the repeated small-sample normal-approximation warnings seen with Wilcoxon in v7.3.

The protocol also retains:

- 95% paired bootstrap confidence intervals for mean differences.
- Holm correction within each dataset × threat × metric family.
- Paired standardized effect size `dz`.
- One observation per held-out image after averaging randomized sequence repetitions, preventing pseudo-replication.

### 3. Threat-responsiveness analysis

`THREAT_RESPONSIVENESS.csv` measures the per-image change from the lowest to highest tested threat for:

- global risk,
- protected fraction,
- controller level,
- key rotation.

`THREAT_RESPONSIVENESS_PAIRED_TESTS.csv` directly compares full SRACR threat response with `no_threat` and `no_controller` using paired bootstrap CIs, sign-flip tests, Holm correction, and effect sizes.

### 4. Ablation integrity guard

The final run now fails if any expected ablation variant is missing from any dataset. `no_reliability` is also included in the printed cross-dataset summary.

## Bootstrap from your completed v7.3 run

Use your v7.3 directory so the three trained models, leakage-safe dataset index, and compatible Grad-CAM/saliency caches are copied:

```bat
C:\Python39\python.exe C:\Users\Sameeh\Downloads\SRACR_Med_v7_3_1\main.py bootstrap --from-dir C:\Users\Sameeh\Downloads\SRACR_Med_v7_3
```

If the v7.3 feature cache is present, v7.3.1 reuses it; the expensive Grad-CAM/reliability stage does not need to be recomputed.

## Quick validation

```bat
C:\Python39\python.exe C:\Users\Sameeh\Downloads\SRACR_Med_v7_3_1\main.py final-validate --samples 100 --threats 0.10 0.40 0.70 0.95 --sequence-reps 5 --bootstrap-reps 2000 --permutation-reps 5000 --cpu
```

## Final full-test manuscript run

```bat
C:\Python39\python.exe C:\Users\Sameeh\Downloads\SRACR_Med_v7_3_1\main.py final-validate --samples 0 --threats 0.10 0.40 0.70 0.95 --sequence-reps 5 --bootstrap-reps 5000 --permutation-reps 10000 --cpu
```

Use `--rebuild-cache` only if the trained model, preprocessing, Grad-CAM implementation, perturbation protocol, or dataset split has changed.

## Main outputs

Under `outputs/final_validation/`:

- `ALL_DATASETS_ablation_summary_ci95.csv`
- `ALL_DATASETS_paired_tests.csv`
- `ALL_DATASETS_image_level_ablation.csv`
- `CROSS_DATASET_MACRO_ablation_summary.csv`
- `PUBLICATION_TABLE_ablation.csv`
- `SIGNIFICANT_PAIRED_FINDINGS.csv`
- `THREAT_RESPONSIVENESS.csv`
- `THREAT_RESPONSIVENESS_PAIRED_TESTS.csv`

Each dataset directory also contains:

- `sequence_level_detail.csv`
- `image_level_ablation.csv`
- `ablation_summary_ci95.csv`
- `paired_tests.csv`
- `ablation_replacement_values.csv`

## Interpretation warning

The clinical criticality values remain configurable security-policy priors, not clinically validated disease-severity constants. The threat values are controlled security scenarios unless the system is connected to a real threat detector/IDS. Selective encryption below 100% coverage is an efficiency/security trade-off and must not be described as full-image confidentiality.


## v7.3.2 reproducibility patch
The published numerical analyses remain tied to the archived v7.3.1 feature caches.
This patch explicitly fixes Python/NumPy/PyTorch RNG state to the command seed before stochastic saliency-feature recomputation, so future cache rebuilds are deterministic on the documented CPU software stack. It does not silently replace the archived manuscript results.
