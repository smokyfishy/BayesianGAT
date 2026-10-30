"""Structural annotations: relative solvent accessibility (RSA) and AlphaFold pLDDT.

The original RSA vector (cell 39) came from DSSP via Biopython (default Sander
max-ASA normalisation). DSSP is often unavailable on Windows, so this module also
offers a Shrake-Rupley RSA via Biopython (same Sander max-ASA table). The two
methods differ numerically; :func:`compare_rsa` reports their agreement rather than
assuming equality.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Sander & Rost (1994) maximum accessible surface areas (A^2), Biopython "Sander" table.
MAX_ASA_SANDER = {
    "ALA": 106.0,
    "ARG": 248.0,
    "ASN": 157.0,
    "ASP": 163.0,
    "CYS": 135.0,
    "GLN": 198.0,
    "GLU": 194.0,
    "GLY": 84.0,
    "HIS": 184.0,
    "ILE": 169.0,
    "LEU": 164.0,
    "LYS": 205.0,
    "MET": 188.0,
    "PHE": 197.0,
    "PRO": 136.0,
    "SER": 130.0,
    "THR": 142.0,
    "TRP": 227.0,
    "TYR": 222.0,
    "VAL": 142.0,
}


def rsa_shrake_rupley(pdb_path: str | Path, chain: str = "A") -> np.ndarray:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = structure[0]
    ShrakeRupley().compute(model, level="R")
    vals = []
    for res in model[chain]:
        if "CA" not in res:
            continue
        vals.append(min(1.0, res.sasa / MAX_ASA_SANDER.get(res.get_resname(), np.nan)))
    return np.asarray(vals, dtype=float)


def rsa_dssp(pdb_path: str | Path) -> np.ndarray | None:
    exe = shutil.which("mkdssp") or shutil.which("dssp")
    if exe is None:
        return None
    from Bio.PDB import DSSP, PDBParser

    model = PDBParser(QUIET=True).get_structure("s", str(pdb_path))[0]
    dssp = DSSP(model, str(pdb_path), dssp=exe)
    return np.asarray([dssp[k][3] for k in dssp.keys()], dtype=float)


def compare_rsa(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"comparable": False, "shapes": [list(a.shape), list(b.shape)]}
    rho, p = spearmanr(a, b)
    return {
        "comparable": True,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "exposed_agreement_at_0.25": float(np.mean((a > 0.25) == (b > 0.25))),
    }


def plddt_from_residues(residues: list[dict]) -> np.ndarray:
    return np.asarray([r["plddt"] for r in residues], dtype=float)
