"""YAML configuration loading with CLI overrides and repository-relative path resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = REPO_ROOT / "assets"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict:
    cfg: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    if overrides:
        cfg = deep_update(cfg, {k: v for k, v in overrides.items() if v is not None})
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("data_dir", str(DEFAULT_ASSETS))
    cfg.setdefault("output_dir", str(DEFAULT_OUTPUTS))
    return cfg


def resolve_path(value: str | Path, base: str | Path) -> Path:
    """Resolve ``value`` against ``base`` unless already absolute (never Colab Drive paths)."""
    p = Path(value)
    if str(p).startswith("/content/drive"):
        raise ValueError(f"Colab Drive path {p} is not valid locally; use a path under assets/")
    return p if p.is_absolute() else Path(base) / p
