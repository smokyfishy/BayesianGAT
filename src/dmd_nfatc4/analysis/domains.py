"""NFATC4 domain tables used in the original notebook -- all *historical*.

The notebook used three mutually inconsistent domain definitions, none with a
citation beyond "GO/domain knowledge" or "from UniProt Q14934". They are kept here
verbatim for reproducibility, **not** as an authoritative annotation. Before any
biological interpretation, re-derive domains from a cited canonical source (e.g. the
UniProt Q14934 feature table / InterPro) and record the release used.
"""

from __future__ import annotations

import pandas as pd

# Cell 4/6: ranges used to simulate the mutation vector (0-based array indices).
SIMULATION_DOMAINS = {"DNA-binding": (40, 95), "Calcineurin-binding": (200, 280), "Transactivation": (700, 750)}

# Cells 23/27/40/41: "canonical_domains" (numbers used directly as 0-based slice
# bounds in cells 23/27 but described as residue numbers; ambiguous in the original).
CANONICAL_DOMAINS_CELL41 = {
    "Transactivation A": (1, 100),
    "Calcineurin-binding": (110, 140),
    "DNA-binding (RHR)": (140, 260),
    "Regulatory": (260, 400),
    "Transactivation B": (650, 900),
}

# Cell 44: "Manually curated NFATC4 functional domains (from UniProt Q14934)".
# Unverified against a UniProt release; note it places the RHD at 330-600, which
# contradicts the cell-41 table (140-260).
CURATED_DOMAINS_CELL44 = {
    "Transactivation Domain (TAD)": (1, 300),
    "Calcineurin Docking (PxIxIT)": (250, 270),
    "Nuclear Localization Signal (NLS)": (265, 290),
    "Rel Homology DNA-binding (RHD)": (330, 600),
    "Nuclear Export Signal (NES)": (610, 650),
}

# Cells 43/44/54: candidate windows hard-coded after inspecting cell-41 output.
HISTORICAL_CANDIDATE_WINDOWS = [(36, 44), (88, 99), (104, 115), (195, 203), (278, 283), (696, 704), (747, 753)]
HISTORICAL_WINDOWS_NOTE = (
    "Hard-coded in NFATC4.ipynb cells 43/44/54 from an earlier cluster printout. The cluster "
    "table prints 0-based array indices, while cell 43 passes them to py3Dmol as 1-based 'resi'. "
    "Whether they denote 0-based indices or UniProt numbers is resolved empirically by "
    "`analyze-nfatc4 --from-saved-arrays` (see docs/RESULTS.md). They are exploratory outputs of a "
    "model trained on leaked, simulated labels, not validated DMD-specific sites."
)


def distance_to_simulation_boundaries(start_uniprot: int, end_uniprot: int) -> int:
    """Residues between a 1-based window and the nearest edge of a simulated flag range."""
    bounds = [b + 1 for s, e in SIMULATION_DOMAINS.values() for b in (s, e)]
    return min(0 if start_uniprot <= b <= end_uniprot else min(abs(start_uniprot - b), abs(end_uniprot - b)) for b in bounds)


def overlaps(s1: int, e1: int, s2: int, e2: int) -> bool:
    return max(s1, s2) <= min(e1, e2)


def distance_to_domains(start: int, end: int, domains) -> int:
    """Cell 41: gap (in residues) to the nearest domain; 0 if overlapping."""
    ranges = domains.values() if isinstance(domains, dict) else domains
    dists = []
    for ds, de in ranges:
        if end < ds:
            dists.append(ds - end)
        elif start > de:
            dists.append(start - de)
        else:
            dists.append(0)
    return min(dists)


def annotate_windows(windows: list[tuple[int, int]], domains: dict) -> pd.DataFrame:
    rows = []
    for k, (s, e) in enumerate(windows, start=1):
        hit = [name for name, (ds, de) in domains.items() if overlaps(s, e, ds, de)]
        rows.append({"Window": k, "Start": s, "End": e, "Length": e - s + 1, "OverlappingDomains": ", ".join(hit) or "None"})
    return pd.DataFrame(rows)


def compare_windows(recomputed: list[tuple[int, int]], historical=HISTORICAL_CANDIDATE_WINDOWS) -> dict:
    rec, hist = set(map(tuple, recomputed)), set(map(tuple, historical))
    shifted = {(s + 1, e + 1) for s, e in rec}
    return {
        "exact_match_as_index0": sorted(rec & hist),
        "exact_match_if_historical_were_uniprot": sorted(shifted & hist),
        "historical_not_recovered": sorted(hist - rec),
        "recomputed_not_in_historical": sorted(rec - hist),
    }
