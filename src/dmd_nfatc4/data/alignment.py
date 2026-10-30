"""Sequence <-> structure alignment and residue numbering.

Convention used everywhere in this package:

* ``index`` (0-based) -- position in arrays such as ESM embeddings or saved vectors
* ``uniprot_pos`` (1-based) -- UniProt Q14934 residue number

They are only interchangeable (``uniprot_pos = index + 1``) after
:func:`check_sequence_structure_alignment` has confirmed that the PDB chain covers
the full UniProt sequence contiguously with no gaps, insertion codes or mismatches.
"""

from __future__ import annotations

from pathlib import Path

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
}


def read_fasta(path: str | Path) -> tuple[str, str]:
    header, seq = "", []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header:
                break  # first record only
            header = line[1:].strip()
        elif line.strip():
            seq.append(line.strip())
    return header, "".join(seq)


def check_sequence_structure_alignment(fasta_seq: str, residues: list[dict]) -> dict:
    """Validate that PDB residues map 1:1 onto the UniProt sequence."""
    pdb_seq = "".join(THREE_TO_ONE.get(r["resname"], "X") for r in residues)
    resseq = [r["resseq"] for r in residues]
    icodes = [r["icode"] for r in residues if r["icode"].strip()]
    contiguous = resseq == list(range(resseq[0], resseq[0] + len(resseq))) if resseq else False
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(fasta_seq, pdb_seq)) if a != b]
    report = {
        "fasta_length": len(fasta_seq),
        "pdb_residues_with_ca": len(residues),
        "first_resseq": resseq[0] if resseq else None,
        "last_resseq": resseq[-1] if resseq else None,
        "contiguous_numbering": contiguous,
        "insertion_codes": len(icodes),
        "n_mismatches": len(mismatches),
        "mismatches_first10": mismatches[:10],
        "identical": pdb_seq == fasta_seq,
    }
    report["index_equals_uniprot_minus_1"] = bool(
        report["identical"] and contiguous and report["first_resseq"] == 1 and not icodes
    )
    return report


def index_to_uniprot(index: int) -> int:
    return int(index) + 1


def uniprot_to_index(pos: int) -> int:
    return int(pos) - 1
