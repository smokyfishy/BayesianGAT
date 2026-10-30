"""Train/validation/test splitting and leakage audits (DMD.ipynb cell 65)."""

from __future__ import annotations

import numpy as np
import torch


def stratified_split(
    y: np.ndarray, seed: int = 42, train: float = 0.70, val: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class shuffled 70/15/15 split -- the exact procedure of cell 65.

    Legacy mode applies it to the *augmented* label vector (synthetic nodes included);
    corrected mode applies it to original nodes only, before any augmentation.
    """
    rng = np.random.default_rng(seed)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for cls in (1, 0):  # positives first, as in the notebook
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr, n_va = int(train * len(idx)), int(val * len(idx))
        parts.append((idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]))
    return tuple(np.concatenate([p[s] for p in parts]) for s in range(3))  # type: ignore[return-value]


def index_to_mask(idx: np.ndarray, n: int) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.as_tensor(idx, dtype=torch.long)] = True
    return m


def assert_disjoint(*splits: np.ndarray) -> None:
    seen: set[int] = set()
    for s in splits:
        s_set = set(np.asarray(s).tolist())
        if seen & s_set:
            raise AssertionError(f"splits overlap on {len(seen & s_set)} nodes")
        seen |= s_set


def audit_augmentation_leakage(parents: np.ndarray, n_original: int, train_idx, val_idx, test_idx) -> dict:
    """Quantify how synthetic nodes and their parents straddle the split.

    A synthetic node in val/test is evaluated against a label it inherited from its
    parents; if either parent is in train, the evaluation sees an interpolated copy
    of a training example (the core leakage in the original protocol).
    """
    train_set, val_set, test_set = (set(np.asarray(s).tolist()) for s in (train_idx, val_idx, test_idx))
    out = {}
    for name, eval_set in (("val", val_set), ("test", test_set)):
        syn = [s for s in eval_set if s >= n_original]
        leaked = [s for s in syn if parents[s - n_original][0] in train_set or parents[s - n_original][1] in train_set]
        orig_pos_parents_eval = {int(p) for s in syn for p in parents[s - n_original]}
        out[name] = {
            "n_total": len(eval_set),
            "n_synthetic": len(syn),
            "frac_synthetic": len(syn) / max(len(eval_set), 1),
            "n_synthetic_with_parent_in_train": len(leaked),
            "n_distinct_original_parents": len(orig_pos_parents_eval),
        }
    syn_train = [s for s in train_set if s >= n_original]
    out["train"] = {"n_total": len(train_set), "n_synthetic": len(syn_train)}
    return out


def cross_split_edge_counts(edge_index: torch.Tensor, split_of: np.ndarray) -> dict:
    """Count edges by (split of src, split of dst); ``split_of`` holds labels per node."""
    src = split_of[edge_index[0].cpu().numpy()]
    dst = split_of[edge_index[1].cpu().numpy()]
    keys, counts = np.unique(np.char.add(np.char.add(src.astype(str), "->"), dst.astype(str)), return_counts=True)
    return {str(k): int(c) for k, c in zip(keys, counts)}
