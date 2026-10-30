# Results

Status labels:

* **HISTORICAL**: reported by the original work; not re-derived here.
* **OBSERVED**: produced by this repository on 2026-10-30 (Windows 11, Python 3.12.10,
  CPU, torch 2.5.1, torch_geometric 2.6.1). Logs and JSON reports go to `outputs/`.
* **NOT EVALUATED**: cannot be computed validly with the data available.

## 1. Gene-level GAT

### 1a. Original-scale model (61 333 features)

| Quantity | Value | Status |
|---|---|---|
| Test macro-F1 ≈ 0.9936, ROC-AUC ≈ 0.9981 | as reported by the original work | **HISTORICAL**. The protocol put synthetic copies of training positives into the test set (§1c). These numbers have not been reproduced and are not valid generalisation estimates. |
| Checkpoint `best_enhanced_bayesian_gat.pth` | strict load OK into `EnhancedBayesianGAT(61333, h=32, H1=H2=2, dropout=0.5)`; learned temperature 0.788; no edge-projection weights | **OBSERVED** |
| Retraining / ranking on the real 61 333-feature table | needs `node_features.parquet` (~111 MB, not supplied) | **NOT EVALUATED (missing data)** |
| NFATC4 GAT rank/percentile from the original notebook | no saved cell outputs | **NOT EVALUATED** |

### 1b. Corrected protocol on locally available data (OBSERVED)

`configs/gat_corrected_tpm_only.yaml`: the real STRING graph (238 252 undirected
gene–gene edges after removing 12 self-loops) and the real 6-sample TPM features, **without
GO and ESM**. The genes are the 19 922 protein-coding genes of the TPM matrix. 76 of the
79 curated genes are present (DMDP1, DMDP2 and XIST are not protein-coding). Inductive
protocol, training-only GraphSMOTE, scaler fitted on training genes, model selection on
val PR-AUC, 100 epochs. Test split per seed: 2 990 original genes, 12 positives
(prevalence 0.004).

| Seed | ROC-AUC (eval) | PR-AUC (eval) | ROC-AUC (MC mean) | PR-AUC (MC mean) | Median rank of 12 test positives | Recall@100 | Best epoch |
|---|---|---|---|---|---|---|---|
| 0 | 0.915 | 0.070 | 0.951 | 0.111 | 70 / 2990 | 0.58 | 64 |
| 1 | 0.665 | 0.010 | 0.257 | 0.003 | 2268 / 2990 | 0.00 | 1 (training collapsed) |
| 2 | 0.948 | 0.052 | 0.959 | 0.102 | 60 / 2990 | 0.67 | 72 |
| **mean ± sd (eval)** | **0.843 ± 0.126** | **0.044 ± 0.025** | | | | | |

**Baseline.** Ranking the same test genes by STRING degree alone gives
ROC-AUC 0.837 and PR-AUC 0.039 on average (`scripts/degree_baseline.py`). The curated list
is dominated by well-studied hub genes, so most of the separability comes from network
centrality. The GAT beats the degree baseline on 2 of 3 seeds and fails on the third.
With 12 positives per test set, intervals are wide.

**Interpretation.** This is positive-vs-**unlabeled** separability with reduced features.
It does not validate the original model and is not evidence of biological discovery.

**Gene ranking** (`rank-genes`, unlabeled genes, z-score of MC mean). Seed 0's top 10 are
MYBPC2, MYOZ2, RPL3L, ACTN2, MYOT, ICAM2, MYPN, NRAP, TNNC1, VASN (muscle and sarcomere
genes, as expected from muscle-only TPM features). **NFATC4** ranks 376 / 19 846
(98.1th percentile) for seed 0, 17 718 (collapsed seed 1) and 759 (96.2th percentile) for
seed 2. NFATC2 ranks 748 for seed 0. These ranks are unstable across seeds.

### 1c. Legacy protocol on the same data (OBSERVED, for contrast)

`configs/gat_legacy_tpm_only.yaml`: PPI + all-gene kNN (k=10), GraphSMOTE on the whole
graph (4 885 synthetic nodes, 24 807 total), then a 70/15/15 split of the augmented nodes.

| Seed | Test as in notebook (n=3 723, 745 positives): ROC-AUC / macro-F1 | Original test genes only (10 positives): ROC-AUC / PR-AUC | Synthetic nodes in test (with a parent in train) |
|---|---|---|---|
| 0 | 0.945 / 0.909 | 0.901 / 0.121 | 735 (643) |
| 1 | 0.655 / 0.444 | 0.616 / 0.015 | 735 (658) |
| 2 | 0.946 / 0.872 | 0.936 / 0.170 | 735 (644) |

**98.7% of the "test positives" in the legacy protocol are synthetic** interpolations, and
about 88% of them were built from at least one training positive. With only 6 features the
inflation is moderate here. It would be expected to be larger with the 61 333-dimensional
original features, but that has not been measured.

### 1d. Random walk with restart (OBSERVED)

`rwr-genes`: STRING PPI only, 19 922 genes, 76 seeds, restart 0.3, converged.
NFATC4 ranks **2 198 / 19 922 (89.0th percentile)**. NFATC2 ranks 1 011 (94.9th
percentile). The top 20 are all seed genes (FN1, COL1A1, TGFB1, DMD, IL1B, …). The
notebook ran RWR on PPI + kNN over the 18 479-gene node table, so its values differ and
were not saved.

## 2. NFATC4 residue analysis

### 2a. Legacy model (label leakage)

| Quantity | Value | Status |
|---|---|---|
| "NFATC4 accuracy ≈ 98.3%" | as reported by the original work | **HISTORICAL**, methodologically invalid: the label is an input feature |
| Saved checkpoint, accuracy on all 902 residues | 0.994 | **OBSERVED** (leaky; meaningless as evidence) |
| Legacy retraining (seed 42, random 80/20 split) | val accuracy 0.994, ROC-AUC 0.9996 (181 residues, 37 positives) | **OBSERVED** (leaky; meaningless as evidence) |
| Rebuilt graph vs saved `nfatc4_graph.pt` | identical x (902×1281), edges (5 412) and coordinates | **OBSERVED** |
| Simulated vector re-derived from TPM | identical to saved; NFATC4 log2FC = −1.037, driven by one control sample (21.4 TPM vs 1.23 and 0.33) | **OBSERVED** |

### 2b. Corrected residue evaluation

| Run | Result | Status |
|---|---|---|
| `nfatc4_corrected.yaml` (no external labels) | — | **NOT EVALUATED**: no observed residue-level DMD labels exist |
| `nfatc4_simulation_check.yaml`: ESM-only inputs, leave-one-contiguous-block-out (6 blocks), inductive edges, **simulated** target | pooled out-of-fold ROC-AUC **0.384**, PR-AUC **0.162** (random baseline 0.208), sensitivity 0.00 | **OBSERVED**. Once the leaked column is removed and whole sequence blocks are held out, the simulated ranges cannot be recovered at all (below chance). The original near-perfect accuracy came entirely from the leak. |

### 2c. Reanalysis of the saved arrays (OBSERVED)

`analyze-nfatc4 --from-saved-arrays --recompute --stability 5`:

| Window (0-based, as printed) | UniProt residues | Length | Avg smoothed joint score | Avg RSA (DSSP) | Avg pLDDT | Distance to a simulated-range boundary |
|---|---|---|---|---|---|---|
| 36–44 | 37–45 | 9 | 489.3 | 0.89 | 36.8 | 0 |
| 195–203 | 196–204 | 9 | 396.7 | 0.67 | 72.0 | 0 |
| 88–99 | 89–100 | 12 | 357.2 | 0.76 | 45.9 | 0 |
| 104–115 | 105–116 | 12 | 321.2 | 0.76 | 48.1 | 9 |
| 696–704 | 697–705 | 9 | 287.9 | 0.88 | 39.9 | 0 |
| 747–753 | 748–754 | 7 | 151.1 | 0.92 | 41.9 | 0 |
| 278–283 | 279–284 | 6 | 104.5 | 0.88 | 35.8 | 0 |

Findings:

1. All seven historical windows are recovered **exactly**, and only under the 0-based
   interpretation. The original 3D rendering (cell 43) and the domain-overlap table
   (cell 44) treated them as 1-based, so they were shifted by one residue.
2. **Six of seven windows contain an edge of a hand-picked simulated range** (first/last
   flagged residues 41, 96, 201, 281, 701, 751). The joint score peaks where the leaked
   label switches from 0 to 1. The windows are artefacts of the simulation design.
3. Six of seven windows lie in low-confidence AlphaFold regions (pLDDT < 50) that are
   highly exposed (RSA 0.76–0.92), consistent with disordered segments. 57.8% of NFATC4
   residues have pLDDT < 50.
4. The saved saliency equals **2.000×** a single backward pass (max relative error
   2.6e-6): `x.grad` accumulated across notebook cells 32 and 33. Rankings are unaffected.
5. The saved delta-hidden matches a dropout-active recomputation (Spearman ρ 0.77–0.79
   over 5 draws) far better than eval mode (ρ 0.31). Cell 34 left the model in train mode.
6. The MC-dropout std is noisy: two independent 50-pass runs correlate at ρ ≈ 0.32, and
   recomputations vs the saved vector at ρ 0.35–0.40. Across 5 fresh draws, **3–5 of the 7**
   windows are recovered exactly, although every historical window overlaps a recomputed one.
7. RSA: Shrake–Rupley (Biopython, Sander max-ASA) vs the saved DSSP RSA gives
   Spearman ρ = 0.979, mean |Δ| = 0.026. DSSP was not installed locally.

## 3. Software validation (OBSERVED)

| Check | Result |
|---|---|
| `python -m pytest` | 54 passed (asset-backed tests included) |
| `ruff check src tests scripts` | all checks passed |
| `verify-hashes` | 20/20 manifest entries match (all 23 starter files matched before `.gitignore` was extended) |
| `inspect-checkpoint` | both original checkpoints: strict load OK, no missing/unexpected keys, forward pass OK |
| `demo` (synthetic fixture) | runs end-to-end on CPU; legacy test set contains synthetic nodes with training parents, corrected has none. **Synthetic, no biological meaning.** |
