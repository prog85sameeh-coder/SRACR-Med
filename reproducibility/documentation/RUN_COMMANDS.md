# Reproduction Commands

Reference execution environment for the reported study: Windows, Python 3.9, CPU-only.

## Main full-test validation using archived v7.3 feature caches
From `code/main_pipeline/`:

```bat
C:\Python39\python.exe main.py final-validate --samples 0 --threats 0.10 0.40 0.70 0.95 --sequence-reps 5 --bootstrap-reps 5000 --permutation-reps 10000 --cpu
```

For exact audit of the published numbers, place/copy the supplied `archived_feature_cache/*.pkl.gz` into the cache location expected by the pipeline rather than rebuilding saliency features.

## Fresh deterministic feature-cache rebuild
The v7.3.2 reproducibility patch explicitly seeds Python, NumPy, and PyTorch before stochastic saliency computation. Use `--rebuild-cache` only when intentionally rebuilding the saliency features; the resulting fresh deterministic cache is conceptually equivalent but must not be silently substituted when auditing the archived v7.3 numeric outputs.

## Independent attacker training
From `code/independent_leakage/`:

```bat
C:\Python39\python.exe independent_leakage.py train-attacker --source chest --cpu
C:\Python39\python.exe independent_leakage.py train-attacker --source retinal --cpu
C:\Python39\python.exe independent_leakage.py train-attacker --source split --cpu
```

## Independent residual-leakage / same-condition baselines
```bat
C:\Python39\python.exe independent_leakage.py evaluate --source all --cpu --budgets 0.25 0.50 0.75 --bootstrap 2000 --permutations 10000 --include-full-mask
```

## Dataset relocation
If the local absolute paths differ from those in the historical `dataset_index.csv`, use the split manifest (`data_index/exact_split_manifest.csv`) to reconstruct the same split from public-source downloads. Validate file identity with SHA-256 before running.
