# SRACR-Med

Reproducibility repository for:

**From Saliency to Cyber Risk: A Self-Governing Risk-Adaptive Encryption Framework for Secure Medical Imaging**

Repository: https://github.com/prog85sameeh-coder/SRACR-Med

## Scope

**Terminology.** In this repository and manuscript, *self-governing* means automatic stateful execution of the protection policy after externally supplied threat state and pre-specified engineering priors are available. It does **not** mean autonomous threat detection, live-IDS inference, or clinically calibrated cyber-risk estimation.

SRACR-Med combines ResNet-18 diagnostic classification, Grad-CAM spatial prioritization, perturbation-based saliency reliability, predictive uncertainty, engineering clinical-criticality priors, externally supplied synthetic threat states, a stateful hysteretic controller, and AES-256-GCM.

Important claim boundaries:

- The current evidence is **proof-of-concept validation** of controller response and selective-protection behavior.
- The global score is an **engineering control index**, not a calibrated cyber-risk probability.
- The threat variable is a **synthetic externally supplied input**, not a live IDS output.
- PMC is an **internal policy-concentration metric**, not an independent security endpoint.
- Selective modes with `rho < 1` do **not** provide full-image confidentiality.
- The production recommendation is an always-on full-file AEAD envelope, with SRACR used as an additional orchestration/access-control layer.

## Repository structure

```text
main.py
configs/
src/
experiments/
  independent_leakage/
  recent_baselines/
  controlled_shift/
reproducibility/
  config/
  data_index/
  documentation/
results/
paper/
release_assets/
```

## Installation

Python 3.9 was used for the publication workflow.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Raw medical images are not redistributed.

Datasets:

1. Chest: *Detection of Pneumonia Disease using Chest Radiograph Image Dataset*, Mendeley Data V1, DOI `10.17632/kpg5yz77gj.1`.
2. Retinal-11: *A Multi-Class Retinal Fundus Image Dataset for Deep Learning-Based Ocular Disease Diagnosis*, Mendeley Data V1, DOI `10.17632/z227d626vm.1`; the study uses the supplied 5,500-image augmented directory.
3. Fundus-4: *Cataract DR normal glaucoma fundus images dataset*, Kaggle 2022, slug `drskprabhakar/cataract-dr-normal-glaucoma-fundus-images-dataset`.

The exact 10,216-row split/group manifest is:

`reproducibility/data_index/exact_split_manifest.csv`

Because consistent authoritative patient/study IDs were not available for every source image, the study guarantees exact/verified-near-duplicate group independence, not patient-level independence.

## Implemented controller

```text
G = 0.30*A(S_R) + 0.25*C + 0.30*T + 0.15*U
```

Thresholds: `0.30 / 0.50 / 0.70`

Hysteresis: `0.05`

Override: `T >= 0.90 -> Level 4`

Protection levels: `25% / 50% / 75% / 100%`

Exact C assignments, seeds, controller stream orders, saliency settings, and threat-model configuration are under `reproducibility/config/`.

## Saliency reproducibility

Grad-CAM target layer: `ResNet-18 layer4[-1].conv2`.

Reliability perturbations:

- Gaussian noise `sigma=0.02`
- brightness `+0.05`
- contrast `+5%`
- circular roll `(+2, -2)` pixels

The target class is the unperturbed argmax and is fixed across reliability perturbations. Model input is 224×224. Native-image tile size is 64×64, with partial edge tiles retained.

The archived v7.3 caches are the numerical source for the manuscript results. v7.3.2 explicitly fixes Python/NumPy/PyTorch RNG state before fresh stochastic cache rebuilding.

## Independent security endpoint

Cross-dataset macro attacker accuracy:

| Condition | Accuracy |
|---|---:|
| Unprotected | 86.25% |
| SRACR 25% | 52.50% |
| SRACR 50% | 40.92% |
| SRACR 75% | 30.95% |
| 100% neutral-mask sanity | 24.93% |

The full-mask condition is a classifier/prior-bias sanity floor, not a cryptographic-security measurement.

## Same-condition baselines

The same images, splits, tile geometry, budgets, independent attacker, and evaluation metrics were used for:

- SRACR / Grad-CAM
- matched-random
- fixed center ROI
- grayscale content variance
- CAM
- LoG-ROI-inspired selection

No selector is claimed to be universally superior.

## Controlled shift experiment

Internal stress tests:

- Gaussian blur radius 1.5
- JPEG quality 40
- 50% downsample + bilinear restoration

This is a controlled internal shift test, not external-site/multicenter/patient-level validation.

## Large files

Do not commit model checkpoints, feature caches, or raw medical images to normal Git history.

See `release_assets/README.md`.

## Citation

Edit `CITATION.cff.template` with the actual author list, rename it to `CITATION.cff`, then add the Zenodo DOI after the first archived release.

## License

MIT License. Update the copyright holder if your institution or author group requires different wording.
