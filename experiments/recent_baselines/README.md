# Reviewer 2 – Comment 2: Same-condition recent/representative baselines

This package extends the independent residual-diagnostic-leakage protocol already used for Reviewer 1 Comment 1/10.
It adds two spatial selectors while preserving the same test images, 64x64 native-pixel tile grid, matched 25/50/75% protected-pixel budgets, neutral-mask attacker view, independent MobileNetV3-Small attacker, bootstrap CIs, and paired sign-flip tests:

1. **LoG ROI-inspired selector (`log_roi_inspired`)** — a protocol-normalized reproduction of the ROI-selection principle used by Zhang et al., *Complex & Intelligent Systems* (2024), DOI 10.1007/s40747-023-01258-2. The original paper uses Laplacian-of-Gaussian edge detection followed by boundary tracking/filling and a chaotic cipher. Here, to isolate *selection* under the same AES-GCM/budget protocol, tiles are ranked by mean absolute LoG response (sigma=1.0 on a standardized 224x224 grayscale view). This is deliberately labeled "inspired" rather than an exact reproduction of the original cipher/ROI binary-mask implementation.
2. **CAM priority (`cam_priority`)** — an alternative model-derived priority map from the exact study ResNet-18 checkpoint. The target is the primary model's argmax class on the unmasked image. Standard Class Activation Mapping is computed from the final `layer4` feature tensor and the target-class FC weights, resized to native image size, and aggregated by the same 64x64 tiles.

Existing same-condition controls are retained: `content_variance`, `center_roi`, and deterministic `random`, plus the published `sracr_saliency` ordering from the archived feature cache.

## Why YOLOv7/UNet3+ are not directly retrained here
Recent selective medical-image methods include YOLOv7-based ROI selection (Priyanka et al., FGCS 2024) and UNet3+-based ROI selection in DeepENC (IEEE TCE 2024). Those are task-specific learned ROI detectors/segmenters and require ROI/lesion annotations or detector-training targets that are not available consistently for the three study datasets. Training an unvalidated surrogate detector would not be a faithful same-condition reproduction. The revised comparison therefore includes a directly reproducible recent ROI-selection principle (LoG) and an alternative model-derived map (CAM), while documenting the non-reproducibility constraint for annotation-dependent YOLO/UNet baselines.

## Required files
The package already contains:
- `dataset_index.csv`
- `feature_cache/` with the three archived SRACR feature caches
- `independent_leakage.py`
- `r2c2_recent_baselines.py`

You must provide two directories:

### A. Independent attacker checkpoints
Copy the three checkpoints created in the earlier R1C1 experiment into:

```
attacker_models/
  chest_mobilenetv3small_independent.pt
  retinal_mobilenetv3small_independent.pt
  split_mobilenetv3small_independent.pt
```

If they are still in your previous `R1C1_IndependentLeakage\attacker_models` directory, you can pass that directory with `--attacker-dir` instead of copying them.

### B. Primary ResNet-18 checkpoints
Extract the previously supplied `SRACR_Med_R1C11_Model_Checkpoints.zip` and either copy the files into:

```
primary_models/
  chest_xray_resnet18.pt
  retinal_11class_resnet18.pt
  fundus_4class_resnet18.pt
```

or pass the extracted directory with `--primary-model-dir`.

## Windows CPU command
From inside this package directory:

```bat
C:\Python39\python.exe r2c2_recent_baselines.py --source all --cpu --dataset-root "F:\نهى ترقية\SRACR-Med" --attacker-dir "PATH_TO_R1C1_ATTACKER_MODELS" --primary-model-dir "PATH_TO_EXTRACTED_R1C11_CHECKPOINTS" --budgets 0.25 0.50 0.75 --bootstrap 2000 --permutations 10000
```

If `dataset_index.csv` paths still resolve directly on your machine, `--dataset-root` may be omitted.

## Send back these outputs
Please upload:

- `outputs_r2c2/R2C2_ALL_DATASETS_recent_baselines_summary.csv`
- `outputs_r2c2/R2C2_CROSS_DATASET_MACRO_recent_baselines.csv`
- `outputs_r2c2/R2C2_ALL_DATASETS_recent_baselines_paired_tests.csv`
- `outputs_r2c2/R2C2_ALL_DATASETS_recent_baselines_image_level.csv`

The manuscript and reviewer response should be updated only after these real outputs are inspected.
