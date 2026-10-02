# Reviewer 1 – Comment 1: Independent Residual Diagnostic Leakage Experiment

This package addresses the reviewer criticism that **PMC is coupled to the same saliency representation used for tile selection**. PMC is therefore not used here as the independent validation target.

## Independent endpoint

An **independent MobileNetV3-Small attacker classifier** is trained using the original train/validation split. It is not used to generate Grad-CAM, saliency reliability, the risk score, or the tile-selection policy.

For each held-out test image, protected tiles are removed from the attacker's view by replacing them with a neutral RGB value of 127. The independent attacker then tries to recover the ground-truth diagnostic class from the remaining visible regions.

Lower residual attacker accuracy, macro-F1, and true-class probability (and higher NLL) mean less diagnostically recoverable information under this explicitly defined attacker model.

## Spatial policies compared at matched encryption budget

At nominal protected-pixel budgets 25%, 50%, and 75%:

- `sracr_saliency`: the existing reliability-weighted saliency tile order from the frozen feature cache.
- `content_variance`: image-content baseline ranking tiles by local grayscale variance.
- `center_roi`: fixed ROI baseline ranking tiles by proximity to the image center.
- `random`: deterministic matched-random tile ordering.

For each image and budget, SRACR defines the actual protected-pixel target after tile-boundary rounding. Every competing method selects a prefix whose cumulative pixel area is closest to **that same per-image target**. The output reports the residual budget-matching error.

**Important:** because the current saliency reliability term is a scalar multiplier per image, it does not change the within-image ranking of Grad-CAM tiles. Therefore, under an externally fixed encryption budget, the SRACR spatial order is the same as the Grad-CAM/saliency order. The controller contribution is evaluated separately through threat-responsive budget adaptation. This experiment evaluates whether that frozen model-derived spatial order removes more independently recoverable diagnostic information than non-model baselines.

## Why the raw images are still needed

The supplied `feature_cache` contains the real test-time tile priorities, but it does not contain the image pixels. The script therefore opens the original images through the paths already stored in `dataset_index.csv` (the `F:\...` paths on the study computer).

## Step 1 — Train the independent attacker models

From Windows CMD, inside this folder:

```bat
C:\Python39\python.exe independent_leakage.py train-attacker --source chest --cpu
C:\Python39\python.exe independent_leakage.py train-attacker --source retinal --cpu
C:\Python39\python.exe independent_leakage.py train-attacker --source split --cpu
```

ImageNet pretrained weights are requested by default. If torchvision cannot download them, the script prints a warning and falls back to random initialization. For a publication-quality endpoint, pretrained initialization is preferred if available.

The script evaluates each independent attacker on the **unmasked held-out test set** and writes a quality gate. If attacker performance is too weak relative to the original study classifier, do not interpret the leakage experiment until the attacker is improved.

## Step 2 — Run independent leakage evaluation

```bat
C:\Python39\python.exe independent_leakage.py evaluate --source all --cpu --budgets 0.25 0.50 0.75 --bootstrap 2000 --permutations 10000 --include-full-mask
```

## Outputs to send back

Please send these files after the run:

- `outputs/ALL_DATASETS_independent_leakage_summary.csv`
- `outputs/CROSS_DATASET_MACRO_independent_leakage.csv`
- `outputs/ALL_DATASETS_independent_leakage_paired_tests.csv`
- `attacker_models/chest_attacker_training.json`
- `attacker_models/retinal_attacker_training.json`
- `attacker_models/split_attacker_training.json`

The image-level CSV is retained as the reproducibility artifact but does not need to be pasted into the manuscript.

## Statistical interpretation

For each policy/budget, the package reports bootstrap 95% CIs for attacker accuracy, macro-F1, true-class probability, and NLL. SRACR is compared pairwise with random, content-variance, and center-ROI policies on the same images, using paired sign-flip tests and Holm correction.

No result is assumed in advance. If an alternative policy produces lower residual attacker performance, the manuscript must report that result rather than claiming SRACR superiority.
