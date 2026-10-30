"""Contiguous high-score residue windows (NFATC4.ipynb cells 36 and 41).

Pipeline: Gaussian smoothing (sigma=2) -> keep residues strictly above the 93rd
percentile -> connected runs (``scipy.ndimage.label``) -> discard runs < 5 residues.
Output carries both the 0-based array index (what the notebook printed as
``Start``/``End``) and the 1-based UniProt position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, label


def smooth(signal: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return gaussian_filter1d(np.asarray(signal, dtype=float), sigma=sigma)


def find_clusters(
    joint_raw: np.ndarray, sigma: float = 2.0, percentile: float = 93.0, min_len: int = 5
) -> tuple[pd.DataFrame, np.ndarray, float]:
    score = smooth(joint_raw, sigma)
    threshold = float(np.percentile(score, percentile))
    labels, _ = label((score > threshold).astype(int))
    rows = []
    for cid in np.unique(labels):
        if cid == 0:
            continue
        pos = np.where(labels == cid)[0]
        if len(pos) < min_len:
            continue
        rows.append(
            {
                "ClusterID": int(cid),
                "Start_index0": int(pos[0]),
                "End_index0": int(pos[-1]),
                "Start_uniprot": int(pos[0]) + 1,
                "End_uniprot": int(pos[-1]) + 1,
                "Length": int(pos[-1] - pos[0] + 1),
                "AvgJointScore": float(score[pos].mean()),
                "MaxJointScore": float(score[pos].max()),
            }
        )
    cols = ["ClusterID", "Start_index0", "End_index0", "Start_uniprot", "End_uniprot", "Length", "AvgJointScore", "MaxJointScore"]
    df = pd.DataFrame(rows, columns=cols).sort_values("AvgJointScore", ascending=False).reset_index(drop=True)
    return df, score, threshold


def per_residue_table(saliency, std_pred, delta_hidden, smoothed, rsa=None) -> pd.DataFrame:
    n = len(smoothed)
    df = pd.DataFrame(
        {
            "index0": np.arange(n),
            "uniprot_pos": np.arange(1, n + 1),
            "Saliency": saliency,
            "Uncertainty": std_pred,
            "DeltaHidden": delta_hidden,
            "JointScore": smoothed,
        }
    )
    if rsa is not None:
        df["RSA"] = rsa
    return df
