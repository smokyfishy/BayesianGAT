"""GraphSMOTE-style minority oversampling (DMD.ipynb cell 63).

``graphsmote_legacy`` reproduces the original algorithm, which ran on the *whole*
graph before any split. Parent indices are additionally recorded (the original did
not keep them) so the resulting leakage can be audited.

``graphsmote_train_only`` is the corrected variant: only training positives are
interpolated, partners are drawn only from training positives, synthetic nodes are
training-only, and by default they only connect to training-node neighbours of
their parent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.utils import coalesce, remove_self_loops


@dataclass
class AugmentedGraph:
    X: np.ndarray
    y: np.ndarray
    edge_index: torch.Tensor
    parents: np.ndarray  # (n_synthetic, 2): original indices (i, j) interpolated
    n_original: int

    @property
    def n_synthetic(self) -> int:
        return self.X.shape[0] - self.n_original

    @property
    def is_synthetic(self) -> np.ndarray:
        flag = np.zeros(self.X.shape[0], bool)
        flag[self.n_original :] = True
        return flag


def _adjacency(edge_index: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    return adj


def _interpolate(
    X: np.ndarray,
    minority: np.ndarray,
    n_new: int,
    k: int,
    rng_choice,
    rng_rand,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(minority) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X[minority])
    feats, parents = [], []
    for _ in range(n_new):
        i = rng_choice(minority)
        _, neigh = nbrs.kneighbors(X[i].reshape(1, -1))
        neigh = neigh.flatten()
        neigh = neigh[neigh != np.where(minority == i)[0][0]]
        j = i if len(neigh) == 0 else minority[rng_choice(neigh)]
        lam = rng_rand()
        feats.append(X[i] + lam * (X[j] - X[i]))
        parents.append((i, j))
    return np.asarray(feats, dtype=X.dtype), np.asarray(parents, dtype=np.int64).reshape(-1, 2)


def _finish(X, y, edge_np, new_feats, new_edges, parents, n) -> AugmentedGraph:
    X_aug = np.vstack([X, new_feats]) if len(new_feats) else X.copy()
    y_aug = np.concatenate([y, np.ones(len(new_feats), dtype=y.dtype)])
    if new_edges:
        edge_np = np.concatenate([edge_np, np.asarray(new_edges, dtype=np.int64).T], axis=1)
    ei = torch.tensor(edge_np, dtype=torch.long)
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, num_nodes=X_aug.shape[0])
    return AugmentedGraph(X_aug, y_aug, ei, parents, n)


def graphsmote_legacy(
    X: np.ndarray, y: np.ndarray, edge_index: torch.Tensor, ratio: float = 0.25, k: int = 5, seed: int | None = None
) -> AugmentedGraph:
    """Original cell-63 algorithm (augment the full graph; positives -> ratio * negatives).

    The notebook used the global ``np.random`` state without a seed; pass ``seed`` to
    make a run repeatable (it seeds ``np.random``, matching the original call pattern).
    """
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    edge_np = edge_index.cpu().numpy()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_new = max(0, int(ratio * n_neg) - n_pos)
    if n_new == 0:
        return AugmentedGraph(X.copy(), y.copy(), edge_index.clone(), np.zeros((0, 2), np.int64), n)
    minority = np.where(y == 1)[0]
    feats, parents = _interpolate(X, minority, n_new, k, np.random.choice, np.random.rand)
    adj = _adjacency(edge_np, n)
    new_edges = []
    for s, (i, _j) in enumerate(parents):
        new_id = n + s
        for nbr in adj[i]:
            new_edges += [(new_id, nbr), (nbr, new_id)]
    return _finish(X, y, edge_np, feats, new_edges, parents, n)


def graphsmote_train_only(
    X: np.ndarray,
    y: np.ndarray,
    edge_index: torch.Tensor,
    train_idx: np.ndarray,
    ratio: float = 0.25,
    k: int = 5,
    synthetic_edges: str = "train_neighbors",
    seed: int = 0,
) -> tuple[AugmentedGraph, np.ndarray]:
    """Corrected oversampling restricted to the training split.

    Returns the augmented graph and the new training index set (original training
    nodes + all synthetic nodes). ``synthetic_edges``:
      * ``train_neighbors`` -- connect to the parent's neighbours that are training nodes
      * ``none``            -- synthetic nodes are isolated (features-only)
      * ``all_neighbors``   -- legacy wiring (exposes eval nodes to synthetic copies)
    """
    if synthetic_edges not in {"train_neighbors", "none", "all_neighbors"}:
        raise ValueError(f"unknown synthetic_edges={synthetic_edges!r}")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    train_idx = np.asarray(train_idx)
    edge_np = edge_index.cpu().numpy()
    y_train = y[train_idx]
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    n_new = max(0, int(ratio * n_neg) - n_pos)
    minority = train_idx[y_train == 1]
    if n_new == 0 or len(minority) < 2:
        return AugmentedGraph(X.copy(), y.copy(), edge_index.clone(), np.zeros((0, 2), np.int64), n), train_idx
    feats, parents = _interpolate(X, minority, n_new, k, rng.choice, rng.random)
    train_set = set(train_idx.tolist())
    adj = _adjacency(edge_np, n)
    new_edges = []
    if synthetic_edges != "none":
        for s, (i, _j) in enumerate(parents):
            new_id = n + s
            for nbr in adj[i]:
                if synthetic_edges == "all_neighbors" or nbr in train_set:
                    new_edges += [(new_id, nbr), (nbr, new_id)]
    aug = _finish(X, y, edge_np, feats, new_edges, parents, n)
    new_train = np.concatenate([train_idx, np.arange(n, n + len(feats))])
    return aug, new_train
