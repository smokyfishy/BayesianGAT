"""Gene prioritisation (DMD.ipynb cells 73, 75-79, 81).

* :func:`zscore_ranking` -- cell 73/79: MC-dropout mean P(positive) for every
  original gene, z-scored against the non-positive background, known positives
  excluded, sorted descending.
* :func:`random_walk_with_restart` -- cell 75: RWR (restart 0.3) seeded uniformly on
  known positives. Sparse implementation (the notebook built a dense N x N matrix).
* :func:`neighbor_masking_effects` -- cell 81: zero one neighbour's features at a time
  and record the % change in a target gene's MC probability.

A high rank means "the model scores this unlabeled gene like the curated positives"
(or "is close to them in the network" for RWR). It is a hypothesis-generating
prioritisation, not evidence of causality or of a role in DMD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def zscore_ranking(genes: list[str], mean_prob: np.ndarray, std_prob: np.ndarray, positives) -> pd.DataFrame:
    pos = set(positives)
    bg = np.array([g not in pos for g in genes])
    bg_mean, bg_std = mean_prob[bg].mean(), mean_prob[bg].std() + 1e-9
    df = pd.DataFrame({"node_index": np.arange(len(genes)), "gene": genes, "mean_prob": mean_prob, "std_prob": std_prob})[bg]
    df["z_score"] = (df["mean_prob"] - bg_mean) / bg_std
    df = df.sort_values("z_score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["percentile"] = 100.0 * (1 - (df["rank"] - 1) / len(df))
    return df


def random_walk_with_restart(
    edge_index: np.ndarray, n: int, seed_idx: list[int], restart: float = 0.3, tol: float = 1e-8, max_iter: int = 1000
) -> tuple[np.ndarray, int, bool]:
    """Column-normalised RWR identical in update rule to cell 75."""
    if not seed_idx:
        raise ValueError("no seed nodes")
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)  # undirected, unweighted
    col = np.asarray(A.sum(axis=0)).ravel()
    col[col == 0] = 1.0
    A_norm = A @ sp.diags(1.0 / col)
    s = np.zeros(n)
    s[seed_idx] = 1.0 / len(seed_idx)
    p = s.copy()
    for it in range(1, max_iter + 1):
        p_next = (1 - restart) * (A_norm @ p) + restart * s
        if np.abs(p_next - p).sum() < tol:
            return p_next, it, True
        p = p_next
    return p, max_iter, False


def rwr_table(genes: list[str], scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"gene": genes, "rwr_score": scores})
    df["rank"] = df["rwr_score"].rank(method="dense", ascending=False).astype(int)
    df = df.sort_values("rank", kind="mergesort").reset_index(drop=True)
    df["percentile"] = 100.0 * (1 - (df["rank"] - 1) / len(df))
    return df


def gene_report(df: pd.DataFrame, gene: str, total_key: str = "rank") -> dict:
    row = df[df["gene"] == gene]
    if row.empty:
        return {"gene": gene, "present": False, "note": "absent from ranked set (known positive or not in graph)"}
    r = row.iloc[0].to_dict()
    r.update({"present": True, "n_ranked": int(len(df))})
    return r


def rank_from_checkpoint(
    node_table, ppi_edges_tsv, checkpoint, hparams: dict, positives, knn_k: int | None = 10, mc_samples: int = 50, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Score every original gene with a saved GAT and rank unlabeled genes (cell 73/79).

    Graph: PPI + all-node kNN rebuilt over the original genes. The notebook ranked on
    the *augmented* graph (synthetic nodes present); exact reproduction would need
    ``augmented_X_y.pt`` / ``augmented_edge_index.pt``, which are not available here.
    """
    from ..data.gene_graph import knn_edge_index, load_node_table_sparse, load_ppi_gene_edges, merge_edges, ppi_edge_index
    from ..models.gat import enable_sparse_input
    from ..models.legacy_gat import EnhancedBayesianGAT
    from ..training.train_gat import _to_torch_x
    from ..training.uncertainty import mc_dropout_gat
    from ..utils import load_state_dict, set_seed

    set_seed(seed)
    genes, X, blocks = load_node_table_sparse(node_table)
    sd = load_state_dict(checkpoint)
    expected = sd["input_mlp.0.weight"].shape[1]
    if X.shape[1] != expected:
        raise ValueError(f"node table yields {X.shape[1]} features but checkpoint expects {expected}")
    ei, _ = ppi_edge_index(load_ppi_gene_edges(ppi_edges_tsv), genes)
    if knn_k:
        ei = merge_edges(ei, knn_edge_index(X, knn_k), num_nodes=len(genes))
    model = EnhancedBayesianGAT(X.shape[1], **hparams)
    model.load_state_dict(sd, strict=True)
    enable_sparse_input(model)
    mean, std = mc_dropout_gat(model, _to_torch_x(X, "cpu"), ei, mc_samples=mc_samples)
    info = {
        "n_genes": len(genes),
        "n_features": int(X.shape[1]),
        "n_edges": int(ei.shape[1]),
        "feature_blocks": blocks.diagnostics,
    }
    return zscore_ranking(genes, mean, std, positives), info


@torch.no_grad()
def neighbor_masking_effects(model, x, edge_index, target_idx: int, neighbors: list[int], mc_samples: int = 50):
    def mc_prob(xi):
        model.train()
        vals = [torch.softmax(model(xi, edge_index) / model.temperature, dim=1)[target_idx, 1] for _ in range(mc_samples)]
        model.eval()
        return torch.stack(vals).mean().item()

    base = mc_prob(x) + 1e-9
    out = []
    for nb in neighbors:
        xm = x.clone()
        xm[nb] = 0.0
        out.append((nb, (base - mc_prob(xm)) / base * 100.0))
    return base, out
