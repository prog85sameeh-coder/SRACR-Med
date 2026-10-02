# Reviewer 2 Comment 4 — Controlled Distribution-Shift Validation

This package performs a controlled held-out stress test of SRACR-Med without claiming external-site or patient-level validation.

## Why this experiment
Reviewer 2 noted that saliency and uncertainty may change under distribution shift. The study has no new multicenter cohort and the source datasets do not provide consistent authoritative patient IDs. The defensible route is therefore a controlled shift experiment on the exact held-out test sets, while retaining the patient-independence limitation.

## Controlled shifts
The script evaluates the clean held-out image and three deterministic shifts that are **not the same perturbations used to compute saliency reliability Q_S**:

1. `gaussian_blur_r1p5` — Gaussian blur radius 1.5.
2. `jpeg_q40` — JPEG encode/decode at quality 40, subsampling 2.
3. `downsample50_bilinear` — native image downsampled to 50% width/height and restored with bilinear interpolation.

The internal Q_S perturbations remain Gaussian noise, +0.05 brightness, +5% contrast, and a 2-pixel spatial roll.

## What is measured
For every test image and shift:
- diagnostic accuracy, macro-F1, balanced accuracy;
- prediction agreement with the clean image;
- normalized predictive uncertainty;
- Grad-CAM saliency reliability;
- reliable-saliency burden;
- tile-score Spearman correlation versus the archived clean map;
- top-50%-tile Jaccard overlap versus the archived clean map.

The stateful controller is then replayed with the same four threat states and the same five stream seeds used in the manuscript. The output includes:
- global-risk shift;
- controller-level agreement with clean;
- absolute level change;
- protected fraction;
- monotonicity of protection with increasing T.

## Exact Windows command for the current study machine
From this folder run:

```bat
C:\Python39\python.exe r2c4_controlled_shift.py --source all --cpu --dataset-root "F:\نهى ترقية\SRACR-Med" --primary-model-dir "C:\Users\Sameeh\Downloads\SRACR_Med_v5\SRACR_Med\models" --feature-cache-dir "feature_cache" --output-dir "outputs_r2c4" --bootstrap 2000 --seed 42
```

## Files to return
Please upload these files from `outputs_r2c4`:

- `R2C4_ALL_DATASETS_controlled_shift_summary.csv`
- `R2C4_CROSS_DATASET_MACRO_shift_summary.csv`
- `R2C4_ALL_DATASETS_controlled_shift_controller_summary.csv`
- `R2C4_CROSS_DATASET_MACRO_controller_summary.csv`
- `R2C4_controller_monotonicity_checks.csv`
- `R2C4_ALL_DATASETS_controlled_shift_image_level.csv`
- `R2C4_controlled_shift_protocol.json`

The two large controller-detail files are useful for audit but are not required for the first interpretation unless a discrepancy appears.

## Claim boundary
This is a controlled distribution-shift experiment on internal held-out data. It is **not** a multicenter validation, an external-site validation, or proof of patient-level independence. Those limitations must remain explicit in the manuscript.
