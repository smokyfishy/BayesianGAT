"""Reproduction of the *simulated* NFATC4 'DMD mutation' vector (NFATC4.ipynb cell 4).

What the original did: if the group-mean NFATC4 log2 fold change (DMD vs control,
three bulk RNA-seq runs each) was below -1, every residue in three hand-picked
0-based index ranges was flagged 1. No variant calling, no patient mutation data
and no per-residue evidence was involved. The flags are therefore a **simulation
input**, not observed pathogenic mutations, and zeros are not confirmed benign.

Index note: ``flags[start:end+1] = 1`` with 0-based ``(40, 95)`` marks UniProt residues
41-96 (the original PyMOL script indeed selects ``resi 41-96``), even though the plot
in cell 6 labels the axis "Residue Index (1-based)".
"""

from __future__ import annotations

import numpy as np

SIMULATION_DOMAINS_0BASED = [(40, 95), (200, 280), (700, 750)]
SIMULATION_DOMAIN_LABELS = {  # labels used in cell 6
    (40, 95): "DNA-binding",
    (200, 280): "Calcineurin-binding",
    (700, 750): "Transactivation",
}
LOG2FC_THRESHOLD = -1.0


def simulate_mutation_flags(log2_fc: float, n_residues: int = 902) -> np.ndarray:
    flags = np.zeros(n_residues)
    if log2_fc < LOG2FC_THRESHOLD:
        for start, end in SIMULATION_DOMAINS_0BASED:
            start, end = max(0, start), min(n_residues - 1, end)
            flags[start : end + 1] = 1
    return flags


def describe_flags(flags: np.ndarray) -> dict:
    idx = np.where(flags == 1)[0]
    runs = []
    if idx.size:
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        runs = [
            {"index_start": int(s), "index_end": int(e), "uniprot_start": int(s) + 1, "uniprot_end": int(e) + 1}
            for s, e in zip(starts, ends)
        ]
    return {"n_flagged": int(idx.size), "n_residues": int(flags.size), "runs": runs}
