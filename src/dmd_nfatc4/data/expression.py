"""RNA-seq TPM processing (DMD.ipynb cells 18-22 and 60; NFATC4.ipynb cell 3).

The raw Salmon ``quant.sf`` files and the GENCODE v39 GTF are not in this repository.
This module provides the transcript->gene collapse used originally (for re-running
from raw quantifications) and the feature normalisation used by the GAT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# NFATC4.ipynb cell 2 sample groups (Salmon run accessions).
CONTROL_RUNS = ["SRR16104020", "SRR3192550", "SRR7942570"]
DMD_RUNS = ["SRR4240749", "SRR4240760", "SRR4240869"]


def load_gene_tpm(path: str | Path) -> pd.DataFrame:
    """Load the protein-coding gene-level TPM matrix (genes x samples) and validate it."""
    df = pd.read_parquet(path)
    if not df.index.is_unique:
        raise ValueError("gene symbols in TPM matrix are not unique")
    if df.isna().any().any():
        raise ValueError("TPM matrix contains missing values")
    if (df.values < 0).any():
        raise ValueError("TPM matrix contains negative values")
    return df


def sample_group(column: str) -> str | None:
    run = column.replace("_quant", "")
    if run in CONTROL_RUNS:
        return "control"
    if run in DMD_RUNS:
        return "DMD"
    return None


def gene_log2_fold_change(tpm: pd.DataFrame, gene: str, pseudo: float = 1e-5) -> dict:
    """Group-mean log2 fold change (DMD / control) as in NFATC4.ipynb cell 3.

    The original summed TPM over Salmon transcripts whose *name contains* the gene
    string; the supplied matrix is already collapsed to gene level via GENCODE tx2gene,
    so values can differ slightly (e.g. if a read-through transcript matched).
    """
    groups = {c: sample_group(c) for c in tpm.columns}
    ctrl = [c for c, g in groups.items() if g == "control"]
    dmd = [c for c, g in groups.items() if g == "DMD"]
    if not ctrl or not dmd:
        raise ValueError(f"could not identify control/DMD columns in {list(tpm.columns)}")
    row = tpm.loc[gene]
    mean_ctrl, mean_dmd = float(row[ctrl].mean()), float(row[dmd].mean())
    return {
        "gene": gene,
        "control_tpm": row[ctrl].to_dict(),
        "dmd_tpm": row[dmd].to_dict(),
        "mean_control": mean_ctrl,
        "mean_dmd": mean_dmd,
        "log2_fc": float(np.log2((mean_dmd + pseudo) / (mean_ctrl + pseudo))),
        "n_control": len(ctrl),
        "n_dmd": len(dmd),
    }


def collapse_transcripts_to_genes(expr_tx: pd.DataFrame, tx2gene: pd.Series) -> pd.DataFrame:
    """Strip Salmon/GENCODE transcript decorations and versions, then sum TPM per gene."""
    ids = expr_tx.index.to_series().astype(str).str.split("|").str[0].str.split(".").str[0]
    expr_tx = expr_tx.copy()
    expr_tx.index = ids
    expr_tx = expr_tx.groupby(level=0).sum()
    t2g = tx2gene.copy()
    t2g.index = t2g.index.astype(str).str.split(".").str[0]
    common = expr_tx.index.intersection(t2g.index)
    out = expr_tx.loc[common]
    return out.groupby(t2g.loc[common].values).sum()


def log1p_zscore(values: np.ndarray, fit_rows: np.ndarray | None = None) -> np.ndarray:
    """log1p then per-column z-score.

    ``fit_rows=None`` reproduces the legacy behaviour (statistics over all nodes).
    Passing a boolean/index array fits mean/std on those rows only (corrected mode).
    """
    logged = np.log1p(values.astype(float))
    ref = logged if fit_rows is None else logged[fit_rows]
    mean, std = ref.mean(axis=0), ref.std(axis=0)
    return (logged - mean) / (std + 1e-8)


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)
