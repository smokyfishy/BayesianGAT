"""Gene-level graph assembly (DMD.ipynb cells 41-43, 60-62).

Node table convention (cell 42): one row per gene symbol with columns prefixed
``TPM_`` (expression), ``GO_`` (multi-hot GO terms) and ``ESMdim_`` (mean-pooled
ESM-2 150M embedding). The original code located blocks *by position* (first 6 =
expression, last 640 = ESM). Here blocks are located *by name* and the positional
assumption is checked and reported, never silently trusted.

Feature order fed to the legacy model is ``expr + esm + go`` (cell 60), which differs
from the on-disk column order ``expr + go + esm``; the checkpoint depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import coalesce, remove_self_loops

from .expression import l2_normalize_rows, log1p_zscore

EXPR_PREFIX, GO_PREFIX, ESM_PREFIX = "TPM_", "GO_", "ESMdim_"


@dataclass
class FeatureBlocks:
    expr_cols: list[str]
    go_cols: list[str]
    esm_cols: list[str]
    diagnostics: dict = field(default_factory=dict)

    @property
    def legacy_order(self) -> list[str]:
        return self.expr_cols + self.esm_cols + self.go_cols

    @property
    def n_features(self) -> int:
        return len(self.expr_cols) + len(self.go_cols) + len(self.esm_cols)


def parse_feature_blocks(columns: list[str], expected_expr: int | None = 6, expected_esm: int | None = 640) -> FeatureBlocks:
    """Locate feature blocks by column-name prefix and audit the legacy positional rule."""
    columns = [str(c) for c in columns]
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate column names in node table")
    expr = [c for c in columns if c.startswith(EXPR_PREFIX)]
    go = [c for c in columns if c.startswith(GO_PREFIX)]
    esm = [c for c in columns if c.startswith(ESM_PREFIX)]
    other = [c for c in columns if c not in set(expr) | set(go) | set(esm)]
    if other:
        raise ValueError(f"{len(other)} columns have no recognised prefix, e.g. {other[:5]}")
    diag = {
        "n_expr": len(expr),
        "n_go": len(go),
        "n_esm": len(esm),
        "expected_expr": expected_expr,
        "expected_esm": expected_esm,
        # The legacy code used columns[:6] / columns[-640:] / columns[6:-640].
        "positional_rule_matches_names": (
            columns[: len(expr)] == expr
            and columns[len(columns) - len(esm) :] == esm
            and columns[len(expr) : len(columns) - len(esm)] == go
        ),
    }
    if expected_expr is not None and len(expr) != expected_expr:
        diag["warning_expr"] = f"expected {expected_expr} expression columns, found {len(expr)}"
    if expected_esm is not None and len(esm) != expected_esm:
        diag["warning_esm"] = f"expected {expected_esm} ESM columns, found {len(esm)}"
    return FeatureBlocks(expr, go, esm, diag)


def build_feature_matrix(node_df: pd.DataFrame, blocks: FeatureBlocks, fit_rows: np.ndarray | None = None) -> np.ndarray:
    """Normalise blocks as in cell 60 and return ``[expr | esm | go]`` (float32).

    expression: log1p + z-score (fit on ``fit_rows`` if given, else all nodes = legacy)
    ESM:        per-row L2 normalisation (row-local, no fitting involved)
    GO:         binary, unchanged
    """
    expr = log1p_zscore(node_df[blocks.expr_cols].to_numpy(float), fit_rows)
    esm = l2_normalize_rows(node_df[blocks.esm_cols].to_numpy(float))
    go = node_df[blocks.go_cols].to_numpy(np.float32)
    return np.hstack([expr, esm, go]).astype(np.float32)


def load_node_table_sparse(path: str | Path, fit_rows: np.ndarray | None = None):
    """Memory-aware loader for the real ``node_features.parquet`` (~18.5k x 61k).

    A dense float32 copy would need ~4.5 GB; the GO block is binary and very sparse,
    so it is read in column chunks into CSR. Returns ``(genes, X_csr, blocks)`` with X in
    legacy ``expr | esm | go`` order.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.pandas_metadata or {}
    index_cols = [c for c in meta.get("index_columns", []) if isinstance(c, str)]
    names = [n for n in pf.schema_arrow.names if n not in index_cols and not n.startswith("__index_level")]
    blocks = parse_feature_blocks(names)
    idx_name = index_cols[0] if index_cols else None
    head = pq.read_table(path, columns=blocks.expr_cols + blocks.esm_cols + ([idx_name] if idx_name else []))
    head_df = head.to_pandas()
    genes = list(head_df.index)
    expr = log1p_zscore(head_df[blocks.expr_cols].to_numpy(float), fit_rows)
    esm = l2_normalize_rows(head_df[blocks.esm_cols].to_numpy(float))
    go_parts = []
    for start in range(0, len(blocks.go_cols), 2000):
        cols = blocks.go_cols[start : start + 2000]
        chunk = pq.read_table(path, columns=cols).to_pandas().to_numpy(np.float32)
        go_parts.append(sp.csr_matrix(chunk))
    X = sp.hstack([sp.csr_matrix(expr.astype(np.float32)), sp.csr_matrix(esm.astype(np.float32))] + go_parts)
    return genes, X.tocsr(), blocks


def load_ppi_gene_edges(path: str | Path) -> pd.DataFrame:
    """Load ``ppi_edges_genes.tsv`` (gene1, gene2, combined_score) with validation."""
    df = pd.read_csv(path, sep="\t", header=0, names=["gene1", "gene2", "combined_score"])
    if df[["gene1", "gene2"]].isna().any().any():
        raise ValueError("missing gene symbols in PPI edges")
    s = df["combined_score"]
    if s.isna().any() or (s < 0).any() or (s > 1).any():
        raise ValueError("combined_score must lie in [0, 1] (STRING API scale)")
    return df


def audit_ppi_edges(df: pd.DataFrame) -> dict:
    canon = pd.DataFrame({"a": df[["gene1", "gene2"]].min(axis=1), "b": df[["gene1", "gene2"]].max(axis=1)})
    loops = df["gene1"] == df["gene2"]
    return {
        "n_rows": len(df),
        "n_self_loops": int(loops.sum()),
        "n_duplicate_undirected": int(canon.duplicated().sum()),
        "canonical_orientation_gene1_lt_gene2_excl_self_loops": bool((df.loc[~loops, "gene1"] < df.loc[~loops, "gene2"]).all()),
        "frac_score_below_0.4": float((df["combined_score"] < 0.4).mean()),
        "n_genes": int(pd.unique(df[["gene1", "gene2"]].values.ravel()).size),
        "score_min": float(df["combined_score"].min()),
        "score_max": float(df["combined_score"].max()),
        "score_median": float(df["combined_score"].median()),
    }


def ppi_edge_index(edges: pd.DataFrame, genes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected edge_index (both directions) + aligned combined_score weights (cell 60)."""
    gene2idx = {g: i for i, g in enumerate(genes)}
    if len(gene2idx) != len(genes):
        raise ValueError("gene list contains duplicates")
    mask = edges["gene1"].isin(gene2idx) & edges["gene2"].isin(gene2idx)
    e = edges.loc[mask]
    i = e["gene1"].map(gene2idx).to_numpy()
    j = e["gene2"].map(gene2idx).to_numpy()
    w = e["combined_score"].to_numpy(np.float32)
    ei = torch.tensor(np.stack([np.concatenate([i, j]), np.concatenate([j, i])]), dtype=torch.long)
    ew = torch.tensor(np.concatenate([w, w]), dtype=torch.float32)
    ei, ew = remove_self_loops(ei, ew)
    ei, ew = coalesce(ei, ew, num_nodes=len(genes), reduce="max")
    return ei, ew


def knn_edge_index(X: np.ndarray, k: int = 10, rows: np.ndarray | None = None) -> torch.Tensor:
    """Symmetrised Euclidean kNN edges (cell 62).

    ``rows=None`` reproduces the legacy all-node kNN. Passing ``rows`` restricts both
    queries and candidate neighbours to that subset (used by the inductive protocol so
    held-out genes do not shape the training graph). Indices are returned in the
    global node numbering.
    """
    from sklearn.neighbors import NearestNeighbors

    sub = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    Xs = X[sub]
    if sp.issparse(Xs):
        Xs = Xs.toarray()
    k_eff = min(k, len(sub) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)
    _, ind = nbrs.kneighbors(Xs)
    src = np.repeat(np.arange(len(sub)), k_eff)
    dst = ind[:, 1 : k_eff + 1].reshape(-1)
    src, dst = sub[src], sub[dst]
    ei = torch.tensor(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long)
    ei, _ = remove_self_loops(ei)
    return coalesce(ei, num_nodes=X.shape[0])


def merge_edges(*edge_indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ei = torch.cat(list(edge_indices), dim=1)
    ei, _ = remove_self_loops(ei)
    return coalesce(ei, num_nodes=num_nodes)


def labels_from_gene_list(genes: list[str], positives: list[str]) -> tuple[np.ndarray, dict]:
    pos = set(positives)
    y = np.array([1 if g in pos else 0 for g in genes], dtype=np.int64)
    present = sorted(pos & set(genes))
    return y, {
        "n_nodes": len(genes),
        "n_positive_labels_listed": len(pos),
        "n_positive_present": len(present),
        "positives_missing_from_graph": sorted(pos - set(genes)),
        "n_unlabeled": int((y == 0).sum()),
    }


def make_synthetic_gene_table(
    n_genes: int = 300, n_pos: int = 24, n_go: int = 40, n_esm: int = 16, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tiny **SYNTHETIC** fixture shaped like the real node table + PPI edge list.

    Positives get a shifted ESM block, more of a few GO terms and denser mutual edges,
    so a working pipeline should rank them above chance. This is a software test
    fixture only and carries no biological meaning.
    """
    rng = np.random.default_rng(seed)
    genes = [f"SYN{i:04d}" for i in range(n_genes)]
    pos_idx = rng.choice(n_genes, n_pos, replace=False)
    is_pos = np.zeros(n_genes, bool)
    is_pos[pos_idx] = True
    tpm = rng.gamma(2.0, 10.0, size=(n_genes, 6)) * np.where(is_pos[:, None], 2.0, 1.0)
    go_p = np.full((n_genes, n_go), 0.05)
    go_p[is_pos, :5] = 0.6
    go = (rng.random((n_genes, n_go)) < go_p).astype(np.int8)
    esm = rng.normal(size=(n_genes, n_esm))
    esm[is_pos, :4] += 1.5
    df = pd.concat(
        [
            pd.DataFrame(tpm, columns=[f"{EXPR_PREFIX}S{i}_quant" for i in range(6)]),
            pd.DataFrame(go, columns=[f"{GO_PREFIX}GO:{i:07d}" for i in range(n_go)]),
            pd.DataFrame(esm, columns=[f"{ESM_PREFIX}{i}" for i in range(n_esm)]),
        ],
        axis=1,
    )
    df.index = pd.Index(genes, name="gene_symbol")
    rows = []
    for _ in range(n_genes * 3):
        a, b = rng.choice(n_genes, 2, replace=False)
        rows.append((genes[a], genes[b], float(rng.uniform(0.15, 0.99))))
    for _ in range(n_pos * 3):
        a, b = rng.choice(pos_idx, 2, replace=False)
        rows.append((genes[a], genes[b], float(rng.uniform(0.4, 0.99))))
    edges = pd.DataFrame(rows, columns=["gene1", "gene2", "combined_score"])
    return df, edges, [genes[i] for i in sorted(pos_idx)]
