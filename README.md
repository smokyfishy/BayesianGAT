# DMD Bayesian GAT and NFATC4 residue-graph analysis

A reproducible, modular rebuild of a two-stage graph machine-learning project on
**Duchenne muscular dystrophy (DMD)**:

1. **Gene level.** A GATv2 network with MC-dropout uncertainty (`EnhancedBayesianGAT`)
   on a STRING protein–protein interaction graph with RNA-seq, ESM-2 and Gene Ontology
   node features. It ranks genes by similarity to a curated list of DMD-associated genes.
   NFATC4 came out as one candidate of interest.
2. **Residue level.** A message-passing network on the AlphaFold structure of NFATC4
   (UniProt Q14934), with saliency, MC-dropout uncertainty, hidden-state perturbation,
   solvent accessibility and contiguous-window detection.

The original work was done in two Google Colab notebooks, kept unchanged in
[`reference_notebooks/`](reference_notebooks/). This repository makes that work
**reproducible and auditable**. Re-running it showed that several of the original
results are methodologically invalid as evidence. The repository therefore keeps two
separate tracks:

| Track | Purpose |
|---|---|
| **legacy** | Re-implements the original code exactly. Both original checkpoints load with `strict=True`, and the original residue analysis reproduces exactly from the saved arrays. Known flaws are kept and flagged at runtime. |
| **corrected** | Same model families with leakage-free protocols: split before augmentation, training-only oversampling, label-free inputs, held-out evaluation, PU framing, repeated seeds. |

> **Not for clinical use.** Nothing here is a diagnostic or therapeutic tool. Gene
> rankings and residue windows are hypotheses produced by models with known limitations.
> They have not been validated experimentally.

---

## Contents

- [Key findings of the reproducibility audit](#key-findings-of-the-reproducibility-audit)
- [Method overview](#method-overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Command reference](#command-reference)
- [Project structure](#project-structure)
- [Model architectures](#model-architectures)
- [Data provenance](#data-provenance)
- [Limitations](#limitations)
- [Citation](#citation)

## Key findings of the reproducibility audit

Every statement below comes from running the code in this repository on the supplied
assets. Numbers are in [`docs/RESULTS.md`](docs/RESULTS.md).

**Gene-level GAT**
- Synthetic minority nodes were created **before** the train/val/test split. Synthetic
  interpolations of training positives therefore end up in val/test (`leakage_audit` in
  every `train-gat` report). The originally reported test metrics (macro-F1 ≈ 0.99,
  ROC-AUC ≈ 0.998) come from this protocol. They have **not** been reproduced here,
  because the 111 MB node-feature table is not available locally, and they would not be
  valid generalisation estimates even if they were.
- "Negatives" are simply all genes not on a 79-gene curated list. That list mixes DMD
  pathway genes with generic inflammation/apoptosis genes, and three of its entries are
  not protein-coding. Class 0 means **unlabeled**, not "not DMD-related".
- `use_edge_confidence=True` had no effect: no edge attribute was ever passed. The saved
  checkpoint has no edge-projection weights.

**NFATC4 residue model**
- The training target was an **input feature** (`x = [ESM | flag]`, `y = x[:, -1]`). The
  saved model scores 99.4% accuracy on all residues because it reads the answer from its
  input.
- The target is **simulated**, not observed. Three hand-picked residue ranges were
  flagged because one group-level NFATC4 fold change crossed −1. The supplied TPM data
  gives log2FC = −1.04, and that value is driven by one of the three control samples.
- The model named "GVP" is a scalar MLP message-passing network. It is **not** a
  geometric vector perceptron and is not rotation-equivariant.
- **Six of the seven candidate windows contain a boundary of the hand-picked simulation
  ranges** (UniProt 41, 96, 201, 281, 701, 751), and the seventh lies 9 residues from one.
  The windows mark where the leaked label switches from 0 to 1. They are artefacts of the
  simulation design, not evidence of functional sites.
- The seven historical candidate windows (36–44, 88–99, …) reproduce **exactly** from the
  saved arrays. They are **0-based indices**, so the UniProt residues are 37–45, 89–100, ….
  Most of them lie in low-confidence (pLDDT < 50), highly exposed regions of the
  AlphaFold model. Re-drawing the MC-dropout terms recovers only 3–5 of the 7 windows
  exactly.
- The saved saliency vector is exactly 2× a single backward pass (gradient accumulation
  across notebook cells). The saved delta-hidden vector was computed with dropout active.

See [`docs/LEGACY_VS_CORRECTED.md`](docs/LEGACY_VS_CORRECTED.md) for the full comparison.

## Method overview

```
                         ┌─────────────────────── Stage A: gene graph ───────────────────────┐
 SRA bulk RNA-seq ─ Salmon ─ GENCODE v39 ─▶ gene TPM (6 samples) ─┐
 UniProt seqs ─ ESM-2 150M (mean-pooled, 640-d) ──────────────────┼─▶ node table (genes × [TPM|ESM|GO])
 GO annotations (multi-hot) ──────────────────────────────────────┘            │
 STRING v12 (score ≥ 400 request) ─▶ PPI edges ──┬──▶ graph = PPI (+ kNN on features)
                                                 │            │
             curated DMD list ─▶ weak labels (1 = curated, 0 = unlabeled)
                                                 ▼            ▼
              legacy:    GraphSMOTE(all) ─▶ split ─▶ EnhancedBayesianGAT ─▶ "test" metrics (leaky)
              corrected: split ─▶ GraphSMOTE(train only) ─▶ GAT ─▶ held-out original genes
                                                 │
                                MC dropout ─▶ z-score ranking of unlabeled genes (e.g. NFATC4)
                         └───────────────────────────────────────────────────────────────────┘
                         ┌────────────────────── Stage B: NFATC4 residues ───────────────────┐
 AlphaFold AF-Q14934-F1 v4 ─▶ Cα coordinates ─▶ 8 Å contact graph (902 nodes, 5 412 edges)
 ESM-2 650M layer 33 (902 × 1280) ─▶ node features
 simulated flags (fold-change gate + hand-picked ranges)
              legacy:    x=[ESM|flag], y=flag ─▶ "BayesianGVP_GNN" ─▶ saliency × MC-std × Δhidden
                         ─▶ smooth σ=2 ─▶ >93rd pct ─▶ windows ≥5 aa ─▶ RSA / domain annotation
              corrected: x=ESM only; external labels required, otherwise NOT EVALUATED
                         └───────────────────────────────────────────────────────────────────┘
```

## Installation

Tested on Windows 11 with Python 3.12.10, CPU only. Linux and macOS should work the same
way.

```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1                      # Linux/macOS: source .venv/bin/activate

# 1) PyTorch for your hardware
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
#    CUDA 12.1 instead: --index-url https://download.pytorch.org/whl/cu121

# 2) Everything else, then this package
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch Geometric is installed as the pure-Python `torch_geometric` wheel. The models
only use `GATv2Conv` and `MessagePassing`, so the optional `torch_scatter` and
`torch_sparse` extensions are **not** required. The notebook's
`pip install scipy==1.9.3 --force-reinstall` is deliberately not reproduced: it served
an old ProDy build in an unused cell and does not install on Python 3.12.

### Data

Data is **not** in Git. Put the starter archive's `assets/` folder at the repository
root:

```
assets/dmd/{best_enhanced_bayesian_gat.pth, expression_gene_tpm_protein_coding.parquet, ppi_edges.tsv, ppi_edges_genes.tsv}
assets/nfatc4/{AF-Q14934-F1-model_v4.pdb, Q14934.fasta, Q14934_esm.npy, *.npy, nfatc4_graph.pt, bayesian_gvp_model.pt, ...}
```

`python -m dmd_nfatc4.cli data-instructions` lists every source. For full-scale GAT runs
you also need `assets/dmd/node_features.parquet` (~111 MB), which you have to download
yourself from the Drive link in [`SOURCES.md`](SOURCES.md).

## Quickstart

```powershell
python -m pytest                                              # 54 tests (asset tests skip if assets/ is absent)
python -m dmd_nfatc4.cli verify-hashes                        # SHA-256 check of the starter files
python -m dmd_nfatc4.cli inspect-checkpoint                   # strict-load both original checkpoints
python -m dmd_nfatc4.cli demo                                 # SYNTHETIC end-to-end legacy vs corrected (<1 min)
python -m dmd_nfatc4.cli analyze-nfatc4 --from-saved-arrays --recompute
python -m dmd_nfatc4.cli visualize-nfatc4 --format png pml html
```

Outputs go to `outputs/` (git-ignored), with a JSON report for every command.

## Command reference

All commands accept `--data-dir`, `--output-dir`, `--seed`, `--device cpu|cuda`, and
`--help`. Training commands also accept `--config`.

| Command | What it does |
|---|---|
| `verify-hashes` | Checks files against `FILE_SHA256SUMS.txt` |
| `inspect-assets` | Array shapes, checkpoint sizes, PPI audit, sequence↔structure alignment, list of missing inputs |
| `inspect-checkpoint [--which gat\|nfatc4\|both]` | Strict state-dict load into the legacy classes, key/shape diff, forward smoke test |
| `data-instructions [--download-public]` | Data sources; opt-in download of the public UniProt FASTA and AlphaFold model only |
| `simulate-mutations` | Re-derives the simulated NFATC4 vector from the TPM matrix and compares it with the saved one |
| `build-residue-graph --mode legacy\|corrected [--compare-saved] [--trust-pickle]` | Rebuilds the 8 Å graph from PDB + ESM and diffs it against the saved PyG graph |
| `analyze-nfatc4 --from-saved-arrays [--recompute --stability N]` | Joint score, windows (0- and 1-based), RSA, pLDDT, domain overlap, figures; `--recompute` re-derives the signals from the checkpoint |
| `visualize-nfatc4 --format png pml html` | PyMOL script and standalone py3Dmol HTML with 1-based residue selections |
| `train-residue --config configs/nfatc4_{legacy,corrected,simulation_check}.yaml` | Legacy (leaky) retraining, corrected run (NOT EVALUATED without labels), or explicit simulated-target check |
| `train-gat --config configs/gat_*.yaml [--seeds 0 1 2] [--epochs N]` | Gene GAT under the legacy or corrected protocol; writes `model.pth`, `model_meta.json`, `predictions.csv`, `report.json` |
| `rank-genes --predictions <run>/predictions.csv --top-k 25 --gene NFATC4` | z-score prioritisation of unlabeled genes (cells 73/79) |
| `rank-genes --checkpoint ... [--node-table ...]` | Same with a saved checkpoint (needs `node_features.parquet`) |
| `rwr-genes --gene NFATC4 NFATC2` | Random walk with restart from the curated seeds on the STRING graph |
| `demo` | End-to-end run on a **synthetic** fixture, showing the leakage audit |

Example: corrected GAT on the data available locally (real STRING PPI + real TPM; GO/ESM
missing), three seeds, followed by gene ranking:

```powershell
python -m dmd_nfatc4.cli train-gat --config configs/gat_corrected_tpm_only.yaml --seeds 0 1 2
python -m dmd_nfatc4.cli rank-genes --predictions outputs/gat_corrected_tpm_only/seed0/predictions.csv --top-k 25 --gene NFATC4
```

## Project structure

```
├── configs/                      YAML configs (legacy / corrected / TPM-only / synthetic / NFATC4)
├── docs/
│   ├── REPRODUCIBILITY_AUDIT.md  audit checklist (from the starter archive)
│   ├── LEGACY_VS_CORRECTED.md    side-by-side protocol comparison
│   ├── DATA_PROVENANCE.md        every input, its origin, verified shape, what is missing
│   └── RESULTS.md                historical vs observed vs not-evaluated results
├── reference_notebooks/          original Colab notebooks (outputs stripped, unmodified)
├── scripts/                      verify_source_hashes.py, inspect_assets.py, smoke_test.py
├── src/dmd_nfatc4/
│   ├── cli.py                    all commands
│   ├── config.py, utils.py       config loading, seeding, safe torch loading
│   ├── data/
│   │   ├── expression.py         TPM loading, fold change, tx→gene collapse, scaling
│   │   ├── gene_graph.py         node table parsing (by name), PPI/kNN edges, synthetic fixture
│   │   ├── labels.py             curated DMD list + manifest (weak positives)
│   │   ├── augmentation.py       GraphSMOTE: legacy (whole graph) and training-only
│   │   ├── splits.py             stratified splits, leakage audits
│   │   ├── residue_graph.py      PDB parsing, 8 Å contact graph, label-leakage guard
│   │   ├── alignment.py          UniProt ↔ PDB numbering checks
│   │   ├── simulation.py         simulated mutation vector (cell 4)
│   │   └── download.py           documented sources, opt-in public downloads
│   ├── models/
│   │   ├── legacy_gat.py         EnhancedBayesianGAT (verbatim)
│   │   ├── legacy_residue_mpnn.py GVP / MultiHeadGVPConv / BayesianGVP_GNN (verbatim, honest docs)
│   │   ├── gat.py                factory, EdgeAwareBayesianGAT, sparse-input support
│   │   └── residue_model.py      ResidueScalarMPNN (label-free inputs)
│   ├── training/                 train_gat.py, train_residue.py, evaluate.py, uncertainty.py
│   └── analysis/                 gene_ranking, residue_saliency, clusters, domains, structure, visualization
└── tests/                        54 pytest tests (synthetic fixtures + asset-backed checks)
```

`assets/`, `data/`, `models/` and `outputs/` are git-ignored, as are all `*.pt`, `*.pth`,
`*.npy` and `*.parquet` files.

## Model architectures

**`EnhancedBayesianGAT`** (DMD.ipynb cell 68). The input passes through
`Linear(F→h) → ReLU → LayerNorm`. Then two GATv2 blocks (h → h·H₁ → h·H₂, ELU, residual
when dimensions match, LayerNorm, dropout) feed an output GATv2 layer to 2 classes with
LayerNorm, followed by a learned temperature. The checkpoint uses F = 61 333
(6 TPM + 640 ESM + 60 687 GO), h = 32, H₁ = H₂ = 2, dropout 0.5. "Bayesian" means MC
dropout at inference (50 passes). That is an approximation to epistemic uncertainty, not a
weight posterior. The real node table is very sparse, so `enable_sparse_input` runs the
first layer as a sparse matmul without changing checkpoint keys.

**`BayesianGVP_GNN`**, honest alias `LegacyScalarMPNN` (NFATC4.ipynb cell 24). The input
goes through LayerNorm, then three `MultiHeadGVPConv` blocks (four 2-layer MLP "heads"
averaged per message, sum aggregation, linear residual, LayerNorm; 1281 → 256 → 256 → 256),
then a 256 → 128 → 1 MLP. It works on scalars only, so the name "GVP" is historical.
`ResidueScalarMPNN` is the same network restricted to label-free inputs.

## Data provenance

See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md). In short: public SRA bulk
RNA-seq (3 DMD, 3 control runs), GENCODE v39, STRING, UniProt, GO, ESM-2, and AlphaFold DB
v4. No patient-identifiable data is used, and no data is redistributed here.

## Limitations

- **No full-scale GAT rerun.** Without `node_features.parquet`, the original GAT cannot be
  retrained or used for ranking locally. The checkpoint was verified structurally (strict
  load + forward pass) but not on its real inputs.
- **Weak, small label set.** 76 of the 79 curated genes are present in the graph, and
  everything else is unlabeled. Held-out test sets contain about 12 positives, so metrics
  have wide variance. Report repeated seeds.
- **Tiny expression cohort.** Three DMD and three control bulk RNA-seq runs from different
  studies, selected by smallest read count (cell 8), with no batch correction. The six TPM
  features carry study and batch effects.
- **Transductive caveat.** Even the inductive protocol uses one fixed PPI graph and one
  gene universe. It says nothing about unseen cohorts or patients.
- **NFATC4 has no valid supervised evaluation.** No observed residue-level DMD labels
  exist, so corrected residue classification is reported as NOT EVALUATED.
- **Residue windows are exploratory.** They come from a leaky model trained on simulated
  labels, fall mostly in low-pLDDT regions, and depend on MC-dropout randomness.
- **Domain annotations** in the original notebook are uncited and mutually inconsistent.
- The fpocket, ProDy ANM and docking cells (NFATC4.ipynb 45–57) depend on external
  binaries and were not ported.

## Citation

The source work is unpublished, so there is no formal citation yet. To refer to this
repository:

```
Karthik, A. DMD Bayesian GAT and NFATC4 residue-graph analysis (reproducible rebuild).
GitHub: https://github.com/smokyfishy/BayesianGAT
```

Please also cite the resources the pipelines depend on: STRING, UniProt, AlphaFold DB,
ESM-2, Gene Ontology, GENCODE, PyTorch Geometric, and GATv2 (Brody et al., 2022).
