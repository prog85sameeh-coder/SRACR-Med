from pathlib import Path
import yaml


def load_config(path=None, root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    cfg_path = Path(path).expanduser().resolve() if path else root / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(root)
    return cfg


def project_path(cfg, key, default):
    p = Path(cfg.get("paths", {}).get(key, default))
    if not p.is_absolute():
        p = Path(cfg["_project_root"]) / p
    return p.resolve()
