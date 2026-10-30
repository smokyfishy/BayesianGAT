"""Small shared helpers: seeding, device selection, safe artifact loading, JSON output."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("dmd_nfatc4")


class LegacyMethodWarning(UserWarning):
    """Raised when a code path reproduces a known-invalid original methodology."""


def warn_legacy(message: str) -> None:
    warnings.warn(message, LegacyMethodWarning, stacklevel=2)
    log.warning("LEGACY: %s", message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "cpu") -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint with ``weights_only=True`` (no arbitrary unpickling)."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise TypeError(f"{path} is not a plain state_dict of tensors")
    return obj


def load_pyg_data(path: str | Path, trust_pickle: bool = False):
    """Load a pickled PyG ``Data`` object.

    First tries ``weights_only=True`` with PyG's storage classes allow-listed. If that
    fails, full unpickling is only performed when ``trust_pickle=True`` -- an explicit
    trust decision by the caller (the supplied NFATC4 graph comes from the author's
    own Drive, but the code must not silently unpickle arbitrary files).
    """
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    safe = [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
    try:
        with torch.serialization.safe_globals(safe):
            return torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on file contents
        if not trust_pickle:
            raise PermissionError(
                f"{path} needs full unpickling ({type(exc).__name__}: {exc}). "
                "Re-run with --trust-pickle only if you trust the file's origin."
            ) from exc
        log.warning("Unpickling %s with weights_only=False (explicitly trusted).", path)
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")
    return path
