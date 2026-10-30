"""NFATC4 residue contact graph (NFATC4.ipynb cell 20).

Original construction: Ca coordinates of chain A of the AlphaFold v4 model, an edge
for every pair with ``0 < d(Ca_i, Ca_j) < 8 A`` (both directions, no self loops), and
node features ``x = [ESM-2 650M layer-33 embedding (1280) | simulated mutation flag]``.
Because the flag is also the training target (cell 25: ``y = x[:, -1]``), the legacy
feature builder leaks the label; :func:`build_residue_features` refuses to do that in
corrected mode and :func:`assert_no_label_leakage` guards every corrected training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

LEGACY_CUTOFF_A = 8.0


def parse_pdb_ca(path: str | Path, chain: str = "A") -> list[dict]:
    """Minimal fixed-column PDB parser returning one record per residue with a CA atom.

    Keeps the first alternate location for each CA and records insertion codes and
    pLDDT (AlphaFold stores it in the B-factor column).
    """
    residues, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[21] != chain:
            continue
        key = (int(line[22:26]), line[26])
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resseq": int(line[22:26]),
                "icode": line[26],
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "plddt": float(line[60:66]) if line[60:66].strip() else float("nan"),
            }
        )
    if not residues:
        raise ValueError(f"no CA atoms for chain {chain!r} in {path}")
    return residues


def ca_coordinates(residues: list[dict]) -> np.ndarray:
    return np.asarray([r["xyz"] for r in residues], dtype=np.float32)


def contact_edge_index(coords: np.ndarray, cutoff: float = LEGACY_CUTOFF_A) -> torch.Tensor:
    """Directed edge list for ``0 < ||x_i - x_j|| < cutoff`` (identical to cell 20)."""
    diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
    dist = np.sqrt((diff**2).sum(-1))
    i, j = np.where((dist < cutoff) & (dist > 0.0))
    return torch.tensor(np.stack([i, j]), dtype=torch.long)


def build_residue_features(esm: np.ndarray, mutation_flags: np.ndarray | None = None, mode: str = "corrected") -> torch.Tensor:
    """Legacy: ``[ESM | flag]`` (1281 dims, leaks the target). Corrected: ESM only."""
    esm_t = torch.tensor(esm, dtype=torch.float32)
    if mode == "corrected":
        return esm_t
    if mode != "legacy":
        raise ValueError(f"mode must be 'legacy' or 'corrected', got {mode!r}")
    if mutation_flags is None:
        raise ValueError("legacy features require the mutation flag vector")
    return torch.cat([esm_t, torch.tensor(mutation_flags, dtype=torch.float32).reshape(-1, 1)], dim=1)


class LabelLeakageError(AssertionError):
    pass


def assert_no_label_leakage(x: torch.Tensor, y: torch.Tensor, atol: float = 0.0) -> None:
    """Fail if any input column equals the target (or its complement) on every node."""
    y = y.reshape(-1).to(x.dtype)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y have different numbers of nodes")
    same = torch.isclose(x, y[:, None], atol=atol, rtol=0).all(dim=0)
    comp = torch.isclose(x, 1 - y[:, None], atol=atol, rtol=0).all(dim=0)
    bad = torch.nonzero(same | comp).flatten().tolist()
    if bad:
        raise LabelLeakageError(f"input feature column(s) {bad} reproduce the target exactly")


def build_residue_graph(pdb_path, esm_path, mutation_path=None, mode="corrected", cutoff=LEGACY_CUTOFF_A):
    """Assemble a PyG ``Data`` object from raw assets (no pickled graph needed)."""
    from torch_geometric.data import Data

    residues = parse_pdb_ca(pdb_path)
    coords = ca_coordinates(residues)
    esm = np.load(esm_path, allow_pickle=False)
    flags = np.load(mutation_path, allow_pickle=False) if mutation_path is not None else None
    if esm.shape[0] != len(residues):
        raise ValueError(f"ESM rows ({esm.shape[0]}) != PDB residues with CA ({len(residues)})")
    if flags is not None and flags.shape[0] != len(residues):
        raise ValueError("mutation vector length does not match the structure")
    data = Data(
        x=build_residue_features(esm, flags, mode),
        edge_index=contact_edge_index(coords, cutoff),
        pos=torch.tensor(coords),
        name="NFATC4",
    )
    if flags is not None:
        data.y_simulated = torch.tensor(flags, dtype=torch.float32)
    data.resseq = torch.tensor([r["resseq"] for r in residues])
    return data


def compare_graphs(a, b) -> dict:
    """Compare a rebuilt graph with the saved one (shapes, features, edge sets)."""
    ea = {tuple(e) for e in a.edge_index.t().tolist()}
    eb = {tuple(e) for e in b.edge_index.t().tolist()}
    out = {
        "x_shape": [list(a.x.shape), list(b.x.shape)],
        "x_max_abs_diff": float((a.x - b.x).abs().max()) if a.x.shape == b.x.shape else None,
        "n_edges": [len(ea), len(eb)],
        "edge_sets_identical": ea == eb,
    }
    if getattr(a, "pos", None) is not None and getattr(b, "pos", None) is not None:
        out["pos_max_abs_diff"] = float((a.pos - b.pos).abs().max())
    return out
