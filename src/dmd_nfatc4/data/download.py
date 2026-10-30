"""Documented data sources and *opt-in* downloads.

Nothing here runs automatically. ``print_data_instructions`` lists where each input
comes from; ``download_public`` fetches only the small public NFATC4 files (UniProt
FASTA, AlphaFold model, QuickGO annotations) and only when called explicitly.
The large private Drive artifacts (node_features.parquet ~111 MB, augmented_X_y.pt
~5.6 GB, ...) are never fetched by code: download them manually (see SOURCES.md).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

PUBLIC_SOURCES = {
    "nfatc4/Q14934.fasta": "https://rest.uniprot.org/uniprotkb/Q14934.fasta",
    "nfatc4/AF-Q14934-F1-model_v4.pdb": "https://alphafold.ebi.ac.uk/files/AF-Q14934-F1-model_v4.pdb",
}

UPSTREAM_PIPELINE = {
    "GENCODE v39 transcripts / GTF": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/",
    "SRA bulk RNA-seq runs (Salmon quant)": "SRR4240749, SRR4240760, SRR4240869 (DMD); "
    "SRR16104020, SRR3192550, SRR7942570 (control)",
    "STRING v12 API (species 9606, required_score 400)": "https://string-db.org/api/tsv/network",
    "GO annotations (goa_human.gaf.gz)": "http://current.geneontology.org/annotations/goa_human.gaf.gz",
    "ESM-2 150M (gene-level) / 650M (NFATC4 residues)": "https://github.com/facebookresearch/esm",
    "QuickGO NFATC4 annotations": "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch",
}

PRIVATE_DRIVE_ARTIFACTS = {
    "dmd/node_features.parquet (~111 MB)": "https://drive.google.com/file/d/1J8tTW14Af1ubB-HKGFICywsq7XUk0WKJ/view",
    "dmd/augmented_X_y.pt (~5.64 GB)": "https://drive.google.com/file/d/1sOURAVHnxP7q4k0abKKHXdapVPc-tnXz/view",
    "dmd/augmented_edge_index.pt (~24 MB)": "https://drive.google.com/file/d/1NI9xZ2JAlUYfohRRs4AyGryN5nQMkz9R/view",
    "dmd/augmented_node_features.parquet (~125 MB)": "https://drive.google.com/file/d/1plHSeYJZqESroGaugNsDs_9pxm6AOvVl/view",
    "dmd/esm2_150M_proj.npy (~193 MB)": "https://drive.google.com/file/d/1j1i04EE2PCfpOSkZguMGLi9LX1xZOtyU/view",
    "dmd/go_matrix.parquet (~39 MB)": "https://drive.google.com/file/d/17xB1n8HNBbQxCZU3hy53mK6aDmN3Bqom/view",
}


def print_data_instructions(assets_dir: Path) -> str:
    lines = [f"Place files under: {assets_dir}", "", "Public NFATC4 inputs (opt-in download):"]
    lines += [f"  {k:<40} {v}" for k, v in PUBLIC_SOURCES.items()]
    lines += ["", "Private Drive artifacts (download manually, then place as shown):"]
    lines += [f"  assets/{k:<44} {v}" for k, v in PRIVATE_DRIVE_ARTIFACTS.items()]
    lines += ["", "Upstream sources used by the original notebooks:"]
    lines += [f"  {k:<48} {v}" for k, v in UPSTREAM_PIPELINE.items()]
    text = "\n".join(lines)
    print(text)
    return text


def download_public(assets_dir: Path, overwrite: bool = False) -> list[Path]:
    written = []
    for rel, url in PUBLIC_SOURCES.items():
        dest = Path(assets_dir) / rel
        if dest.exists() and not overwrite:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URLs
            dest.write_bytes(resp.read())
        written.append(dest)
    return written
