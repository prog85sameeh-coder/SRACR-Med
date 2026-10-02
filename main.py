from pathlib import Path
import argparse
import shutil
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import benchmark, threat_sweep
from src.config import load_config, project_path
from src.multi_validation import validate_all, ALIASES
from src.final_validation import final_validate_all
from src.train_eval import train_dataset

MODEL_NAMES = {
    "chest": "chest_xray",
    "retinal": "retinal_11class",
    "split": "fundus_4class",
}


def resolve_source(df, value):
    source = ALIASES.get(value, value)
    available = df["dataset_source"].dropna().unique().tolist()
    if source not in available:
        raise ValueError("Unknown source %r. Available: %s" % (source, available))
    return source


def load_index(cfg):
    p = project_path(cfg, "output_dir", "outputs") / "dataset_index.csv"
    if not p.exists():
        raise FileNotFoundError("Dataset index not found: %s\nUse bootstrap --from-dir <v7.1 path> or copy dataset_index.csv into outputs." % p)
    return pd.read_csv(p), p


def find_model(cfg, model_name=None, model=None):
    if model:
        p = Path(model).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    model_dir = project_path(cfg, "model_dir", "models")
    files = sorted(model_dir.glob("*.pt"))
    if model_name:
        files = [p for p in files if model_name.lower() in p.stem.lower()]
    if not files:
        raise FileNotFoundError("No matching .pt model found in %s for %r" % (model_dir, model_name))
    return files[0]


def bootstrap_cmd(args, cfg):
    src = Path(args.from_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    out = project_path(cfg, "output_dir", "outputs")
    models = project_path(cfg, "model_dir", "models")
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("dataset_index.csv", "dataset_summary.json"):
        candidate = src / "outputs" / name
        if candidate.exists():
            shutil.copy2(candidate, out / name); copied.append(str(out / name))
    for p in (src / "models").glob("*.pt") if (src / "models").exists() else []:
        shutil.copy2(p, models / p.name); copied.append(str(models / p.name))

    # Reuse expensive v7.3 saliency/Grad-CAM feature caches when available.
    src_cache = src / "outputs" / "final_validation" / "feature_cache"
    dst_cache = out / "final_validation" / "feature_cache"
    if src_cache.exists():
        dst_cache.mkdir(parents=True, exist_ok=True)
        for cp in src_cache.glob("*.pkl.gz"):
            shutil.copy2(cp, dst_cache / cp.name)
            copied.append(str(dst_cache / cp.name))
    if not copied:
        raise RuntimeError("No dataset index or models found under %s" % src)
    print("Imported assets:")
    for p in copied: print("  %s" % p)


def train_cmd(args, cfg):
    df, _ = load_index(cfg)
    source = resolve_source(df, args.source)
    alias = next((k for k, v in ALIASES.items() if v == source), args.source)
    name = args.name or MODEL_NAMES.get(alias, str(alias))
    tr_cfg = cfg.get("train", {})
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train_dataset(
        df[df["dataset_source"] == source].copy(), name,
        project_path(cfg, "training_dir", "outputs/training"),
        project_path(cfg, "model_dir", "models"),
        image_size=int(tr_cfg.get("image_size", 224)),
        batch_size=int(tr_cfg.get("batch_size", 32)),
        epochs=int(args.epochs or tr_cfg.get("epochs", 8)),
        lr=float(tr_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        seed=int(args.seed),
        pretrained=not args.no_pretrained,
        device=device,
    )


def train_all_cmd(args, cfg):
    df, _ = load_index(cfg)
    model_dir = project_path(cfg, "model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for alias in ("chest", "retinal", "split"):
        name = MODEL_NAMES[alias]
        existing = list(model_dir.glob("*%s*.pt" % name))
        if existing and args.skip_existing:
            print("Skipping %s: %s" % (alias, existing[0]))
            continue
        ns = argparse.Namespace(source=alias, name=name, epochs=args.epochs, seed=args.seed,
                                no_pretrained=args.no_pretrained, cpu=args.cpu)
        train_cmd(ns, cfg)


def benchmark_cmd(args, cfg):
    df, index_path = load_index(cfg)
    source = resolve_source(df, args.source)
    model = find_model(cfg, args.model_name, args.model)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    out = project_path(cfg, "benchmark_dir", "outputs/benchmark") / source.replace(" ", "_")
    print("Index  : %s" % index_path); print("Model  : %s" % model)
    benchmark(cfg, df, source, model, out, samples=args.samples, threat=args.threat,
              seed=args.seed, device=device, run_utility_inference=not args.skip_utility_inference)


def sweep_cmd(args, cfg):
    df, index_path = load_index(cfg)
    source = resolve_source(df, args.source)
    model = find_model(cfg, args.model_name, args.model)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    out = project_path(cfg, "sweep_dir", "outputs/threat_sweep") / source.replace(" ", "_")
    print("Index  : %s" % index_path); print("Model  : %s" % model)
    threat_sweep(cfg, df, source, model, out, args.threats, samples=args.samples,
                 seed=args.seed, device=device)


def validate_all_cmd(args, cfg):
    df, index_path = load_index(cfg)
    model_paths = {}
    missing = []
    for alias, name in MODEL_NAMES.items():
        try:
            model_paths[alias] = find_model(cfg, name, None)
        except FileNotFoundError:
            missing.append((alias, name))
    if missing:
        print("Missing model checkpoints:")
        for alias, name in missing:
            print("  %-7s -> expected model containing '%s'" % (alias, name))
        print("\nTrain them with: python main.py train-all --skip-existing --cpu")
        raise FileNotFoundError("Cannot validate all datasets until all three models are available.")
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    out = project_path(cfg, "validation_dir", "outputs/multi_dataset_validation")
    print("=" * 76)
    print("SRACR-Med v7.2 MULTI-DATASET VALIDATION")
    print("=" * 76)
    print("Index  : %s" % index_path)
    print("Device : %s" % device)
    print("Samples: %s per dataset" % ("ALL TEST" if args.samples <= 0 else args.samples))
    print("Threats: %s" % args.threats)
    for k, v in model_paths.items(): print("Model %-7s: %s" % (k, v))
    validate_all(
        cfg, df, model_paths, out, args.threats, samples=args.samples, seed=args.seed,
        device=device, utility=not args.skip_utility_inference,
        bootstrap_reps=args.bootstrap_reps,
    )



def final_validate_cmd(args, cfg):
    df, index_path = load_index(cfg)
    model_paths = {}
    missing = []
    for alias, name in MODEL_NAMES.items():
        try:
            model_paths[alias] = find_model(cfg, name, None)
        except FileNotFoundError:
            missing.append((alias, name))
    if missing:
        for alias, name in missing:
            print("Missing %-7s model containing %r" % (alias, name))
        raise FileNotFoundError("All three trained checkpoints are required for final validation.")
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    out = project_path(cfg, "final_validation_dir", "outputs/final_validation")
    print("=" * 76)
    print("SRACR-Med v7.3.1 FINAL ABLATION & STATISTICAL VALIDATION")
    print("=" * 76)
    print("Index        : %s" % index_path)
    print("Device       : %s" % device)
    print("Samples      : %s per dataset" % ("ALL TEST" if args.samples <= 0 else args.samples))
    print("Threats      : %s" % args.threats)
    print("Sequence reps: %d" % args.sequence_reps)
    print("Bootstrap    : %d" % args.bootstrap_reps)
    print("Permutation  : %d" % args.permutation_reps)
    final_validate_all(
        cfg, df, model_paths, out, args.threats, samples=args.samples, seed=args.seed,
        device=device, bootstrap_reps=args.bootstrap_reps, permutation_reps=args.permutation_reps,
        sequence_reps=args.sequence_reps, rebuild_cache=args.rebuild_cache,
    )


def parser():
    p = argparse.ArgumentParser(description="SRACR-Med v7.3.1 Final Ablation & Statistical Validation")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="copy dataset index/models and compatible feature cache from v7.3/v7.2")
    b.add_argument("--from-dir", required=True); b.set_defaults(fn=bootstrap_cmd)

    t = sub.add_parser("train", help="train one dataset classifier")
    t.add_argument("--source", required=True, choices=["chest", "retinal", "split"])
    t.add_argument("--name", default=None); t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--cpu", action="store_true"); t.set_defaults(fn=train_cmd)

    t = sub.add_parser("train-all", help="train missing classifiers for all three datasets")
    t.add_argument("--epochs", type=int, default=None); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--skip-existing", action="store_true"); t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--cpu", action="store_true"); t.set_defaults(fn=train_all_cmd)

    b = sub.add_parser("benchmark", help="v7.1-compatible single-dataset benchmark")
    b.add_argument("--source", required=True); b.add_argument("--model-name", default=None); b.add_argument("--model", default=None)
    b.add_argument("--samples", type=int, default=20); b.add_argument("--threat", type=float, default=0.40)
    b.add_argument("--seed", type=int, default=42); b.add_argument("--cpu", action="store_true")
    b.add_argument("--skip-utility-inference", action="store_true"); b.set_defaults(fn=benchmark_cmd)

    s = sub.add_parser("sweep", help="v7.1-compatible threat sweep on one dataset")
    s.add_argument("--source", required=True); s.add_argument("--model-name", default=None); s.add_argument("--model", default=None)
    s.add_argument("--samples", type=int, default=20); s.add_argument("--threats", type=float, nargs="+", default=[0.10, 0.40, 0.70, 0.95])
    s.add_argument("--seed", type=int, default=42); s.add_argument("--cpu", action="store_true"); s.set_defaults(fn=sweep_cmd)

    v = sub.add_parser("validate-all", help="validate SRACR-Med across all three datasets")
    v.add_argument("--samples", type=int, default=100, help="per dataset; 0 means full test set")
    v.add_argument("--threats", type=float, nargs="+", default=[0.10, 0.40, 0.70, 0.95])
    v.add_argument("--seed", type=int, default=42); v.add_argument("--bootstrap-reps", type=int, default=1000)
    v.add_argument("--cpu", action="store_true"); v.add_argument("--skip-utility-inference", action="store_true")
    v.set_defaults(fn=validate_all_cmd)

    f = sub.add_parser("final-validate", help="final multi-dataset ablation + paired statistical validation")
    f.add_argument("--samples", type=int, default=100, help="per dataset; 0 means full test set")
    f.add_argument("--threats", type=float, nargs="+", default=[0.10, 0.40, 0.70, 0.95])
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--bootstrap-reps", type=int, default=2000)
    f.add_argument("--permutation-reps", type=int, default=5000, help="Monte-Carlo sign-flip permutations; exact enumeration is used automatically for small effective n")
    f.add_argument("--sequence-reps", type=int, default=5, help="randomized stream orders for hysteresis robustness")
    f.add_argument("--rebuild-cache", action="store_true")
    f.add_argument("--cpu", action="store_true")
    f.set_defaults(fn=final_validate_cmd)
    return p


def main():
    args = parser().parse_args()
    cfg = load_config(args.config, ROOT)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
