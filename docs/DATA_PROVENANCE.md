# Data provenance

All data lives under `assets/`, which is **git-ignored**. Nothing in this table is
committed. SHA-256 digests of the starter files are in `FILE_SHA256SUMS.txt`
(`python -m dmd_nfatc4.cli verify-hashes`). All 23 starter files matched on 2026-10-30; the tracked manifest lists the 20 data, notebook and doc files (`.gitignore` was intentionally extended, and two starter-only instruction files are not part of this repository).

## Present locally (starter archive)

| File | Produced by | Upstream source | Verified shape / content |
|---|---|---|---|
| `dmd/expression_gene_tpm_protein_coding.parquet` | DMD.ipynb cells 18–22 (Salmon → GENCODE v39 tx2gene → symbol → protein-coding filter) | SRA runs SRR4240749/760/869 (DMD), SRR16104020, SRR3192550, SRR7942570 (control) | 19 922 genes × 6 samples, unique symbols, no NaN/negatives |
| `dmd/ppi_edges.tsv` | cell 36 (STRING API `network`, species 9606, `required_score=400`) | STRING | 248 851 UniProt-pair rows |
| `dmd/ppi_edges_genes.tsv` | cell 43 (UniProt → ENSG → symbol, restricted to node-table genes, canonicalised) | STRING | 238 264 rows over 18 088 genes; **12 self-loops** (paralog IDs that collapse onto one symbol), 0 duplicates; scores span 0.000–0.999 even though `required_score=400` was requested (the `network` endpoint also returns partner–partner edges) |
| `dmd/best_enhanced_bayesian_gat.pth` | cell 70/71 | — | plain state_dict, 29 tensors, input dim 61 333, no `lin_edge` |
| `nfatc4/AF-Q14934-F1-model_v4.pdb` | NFATC4.ipynb cell 13 | AlphaFold DB v4 | chain A, residues 1–902, contiguous, no insertion codes, sequence identical to UniProt Q14934; mean pLDDT 59.0, 57.8% of residues < 50 |
| `nfatc4/Q14934.fasta` | cell 12 | UniProt REST | 902 aa (SV=2) |
| `nfatc4/Q14934_esm.npy` | cell 16 (ESM-2 650M, layer 33, BOS/EOS removed) | fair-esm | (902, 1280) float32 |
| `nfatc4/Q14934_go.tsv` | cell 17 | QuickGO | not used by the models |
| `nfatc4/simulated_dmd_mutation_vector.npy` | cell 4 | **simulation** | (902,) float64, 188 ones at 0-based 40–95, 200–280, 700–750; reproduced bit-for-bit by `simulate-mutations` |
| `nfatc4/nfatc4_graph.pt` | cell 20 | — | PyG `Data` (x 902×1281, 5 412 edges). Loads with `weights_only=True` and allow-listed PyG classes; identical to the graph rebuilt from PDB + ESM + vector |
| `nfatc4/bayesian_gvp_model.pt` | cell 26 | — | plain state_dict, input dim 1281 |
| `nfatc4/saliency_vector.npy` | cell 33 | — | (902,) float32; = 2 × single-pass saliency (gradient accumulation) |
| `nfatc4/std_pred_vector.npy` | cell 34 | — | (902,) float32; one stochastic MC draw (50 passes) |
| `nfatc4/delta_hidden_vector.npy` | cell 35 | — | (902,) float32; computed with dropout active |
| `nfatc4/rsa_vector.npy` | cell 39 (DSSP via Biopython) | — | (902,) float64 in [0, 1] |
| `nfatc4/nfatc4_mutation_highlight.pml` | not in the notebook | — | selects resi 41–96, 201–281, 701–751 (= flag index + 1); contains a Colab path |

## Missing (needed for full-scale reproduction)

| File | Size | Needed for | Source |
|---|---|---|---|
| `dmd/node_features.parquet` | ~111 MB | any GAT run with GO + ESM features; `rank-genes` with the original checkpoint | Drive link in `SOURCES.md` |
| `dmd/augmented_X_y.pt`, `augmented_edge_index.pt` | ~5.6 GB, ~24 MB | bit-exact reproduction of the original training graph and ranking | `SOURCES.md` |
| `dmd/esm2_150M_proj.npy`, `go_matrix.parquet` | ~193 MB, ~39 MB | rebuilding the node table from parts | `SOURCES.md` |
| Salmon `quant.sf` files, GENCODE v39 GTF, `ensembl_gene_to_uniprot.tsv` | — | re-running expression processing from raw data | SRA / GENCODE / UniProt (cells 8–28) |
| Residue-level DMD labels with provenance | — | any valid NFATC4 residue classification metric | none exist in this project |

Run `python -m dmd_nfatc4.cli data-instructions` for placement paths. Code never downloads
the private Drive artifacts. `--download-public` fetches only the UniProt FASTA and the
AlphaFold model.

## Licences and sensitive data

* No patient-level or identifiable data is used. The SRA runs are public bulk RNA-seq.
* Check before redistributing derived data: STRING (CC BY 4.0), AlphaFold DB (CC BY 4.0),
  UniProt (CC BY 4.0), GO annotations (CC BY 4.0), GENCODE (open), and the terms of the
  individual SRA studies. This repository redistributes **code only**.
