# Legacy vs corrected methodology

The original 2025 Colab notebooks (`reference_notebooks/`) are preserved unchanged.
This repository re-implements them twice:

* **legacy** reproduces what the original code actually did, so the saved checkpoints
  load and the original analysis can be re-run. Known problems are kept on purpose
  and flagged at runtime with a `LegacyMethodWarning`.
* **corrected** uses the same model families under evaluation protocols without the
  leakage problems.

Never compare a legacy metric with a corrected metric as though they measured the same
thing.

## A. DMD gene-level Bayesian GAT

| Aspect | Original code (cell) | Legacy mode | Corrected mode |
|---|---|---|---|
| Positive labels | 79 curated symbols; all other genes = 0 (61) | same | same list, but class 0 is called **unlabeled** and metrics are framed as positive-vs-unlabeled (PU) |
| Feature blocks | located by position: first 6 / last 640 columns (60) | located by **name prefix**; the positional rule is checked and reported | same |
| Model input order | `expr + esm + go` (60) | same (the checkpoint depends on it) | same |
| Expression scaling | log1p + z-score over **all** genes (60) | same | z-score fitted on **training genes** (`features.scaler_fit`) |
| Graph | STRING PPI + Euclidean kNN (k=10) over all genes (62) | same | `transductive` (same graph, loss on train only) or `inductive` (training graph induced on train genes; kNN among train genes) |
| Oversampling | GraphSMOTE on the whole graph *before* splitting, positives to 25% of negatives (63) | same; parents recorded for auditing | GraphSMOTE on **training positives only**, partners from training positives only, synthetic nodes are training-only and connect only to training neighbours |
| Split | per-class 70/15/15 over **augmented** nodes, `default_rng(42)` (65) | same | same procedure over **original** genes before augmentation; repeatable over seeds |
| Edge confidence | `use_edge_confidence=True`, but no edge attribute was ever passed and GATv2 had no `edge_dim` (68–73) | identical (inert) | optional `EdgeAwareBayesianGAT` that actually puts STRING score into attention (`edge_dim=1`) |
| Temperature | learned scalar trained with the classifier loss (68, 70) | same | same; calibration is checked with ECE on held-out genes |
| Model selection | best val F1 at argmax, val includes synthetic nodes (70) | same | best val **PR-AUC** on original val genes |
| Test metrics | on augmented test nodes, including synthetic copies (72) | reported, labelled "as in notebook (incl. synthetic)", plus original-test-only metrics | original held-out genes only, plus ranking metrics (recall@k, positive ranks), single-class guard on AUC |
| "Bayesian" | MC dropout, 50 passes (73) | same | same, described as an approximation |
| Gene ranking | z-score of MC mean vs non-positive background, positives excluded (73/79), on the augmented graph | `rank-genes` (rebuilt PPI+kNN graph; exact reproduction needs the augmented tensors) | `rank-genes --predictions` on a corrected run |
| RWR | dense N×N matrix, restart 0.3, on PPI+kNN (75) | `rwr-genes` (sparse, PPI-only, since the kNN features are missing) | — |

**Why augment-then-split leaks.** A synthetic node is `x_i + λ(x_j − x_i)` for two
positives *i, j*, and it inherits every edge of *i*. After a random split, a test-set
synthetic node is an interpolated copy of training positives, with nearly the same
neighbourhood. The model is scored on recognising its own training data. `train-gat`
prints a `leakage_audit` that counts these nodes. In corrected runs, the audit must
show zero synthetic nodes in val/test, and the code asserts that.

## B. NFATC4 residue model

| Aspect | Original code (cell) | Legacy mode | Corrected mode |
|---|---|---|---|
| Target | `mutation_flags` **simulated** from one group fold change (4) | same (`simulate-mutations` reproduces it bit-for-bit) | external, provenance-tracked labels required; otherwise **NOT EVALUATED**. Opt-in `nfatc4_simulation_check.yaml` labels its output `SIMULATED_TARGET` |
| Inputs | `x = [ESM | flag]` (20) and `y = x[:, -1]` (25): **label leakage** | same | ESM only; `assert_no_label_leakage` checked before every run; regression test included |
| Model name | `BayesianGVP_GNN` / `GVP` (24) | class names kept for checkpoint loading; alias `LegacyScalarMPNN` | `ResidueScalarMPNN` (same scalar MPNN, honestly named) |
| Split | random 80/20 over residues (25) | same | leave-one-contiguous-block-out (6 blocks) with **inductive** edge removal, because neighbouring residues are strongly autocorrelated |
| Saliency | grad of `sum(logits)`, summed `|grad|` over all 1281 inputs (32/33) | same; the saved vector equals **2×** a single pass because `x.grad` accumulated across cells 32 and 33 | — |
| Delta-hidden | `block3(block2(block1(x)))` with flag set to 1, skipping `input_norm`/GELU (35), computed while dropout was still active from cell 34 | same; `--recompute` shows the dropout-active version matches the saved array (ρ≈0.77) far better than eval mode (ρ≈0.31) | — |
| Clusters | smooth σ=2, >93rd percentile, runs ≥5 (36/41) | same | — |
| Candidate windows | hard-coded 36–44, 88–99, … (43/44/54) | recomputed from saved arrays; they are **0-based indices**, i.e. UniProt 37–45, 89–100, …; 6 of 7 contain an edge of a simulated flag range (`DistToSimulationBoundary`) | — |
| Domain tables | three inconsistent, uncited tables (4/6, 23/41, 44) | kept as historical constants with caveats | must be re-derived from a cited UniProt/InterPro release |
| RSA | DSSP (39) | saved DSSP vector; Shrake–Rupley fallback (ρ=0.979 vs saved) | — |

### What the "GVP" model really is

`GVP` in cell 24 is `Linear → LayerNorm → GELU → Linear → LayerNorm → GELU` on scalar
features. `MultiHeadGVPConv` averages four of those MLPs over each neighbour's features
and sums the messages. There are no vector channels, no use of directions or coordinates
inside the network, and no rotation equivariance. 3D structure only enters through which
Cα pairs are within 8 Å. A real geometric vector perceptron (Jing et al., 2021) would be
a **new** experiment with its own protocol. It is not implemented here and should not be
described as a reproduction.

### The simulated target rests on one sample

`simulate-mutations` recomputes the gate from the supplied TPM matrix: log2FC = −1.037,
just past the −1 threshold. The control mean (7.65 TPM) is driven by one run
(SRR3192550, 21.4 TPM). The other two controls (1.23, 0.33) are lower than every DMD
sample (3.06–4.24). With n = 3 vs 3 and no statistical test, this gate cannot support a
claim that NFATC4 is down-regulated in DMD. Even if it could, the flagged residue ranges
were chosen by hand and are not mutations.
