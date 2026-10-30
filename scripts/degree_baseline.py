"""Compare a train-gat run against a PPI-degree-only baseline on the same held-out genes.

Usage: python scripts/degree_baseline.py outputs/gat_corrected_tpm_only [assets/dmd/ppi_edges_genes.tsv]

Curated DMD genes are mostly well-studied hub genes, so node degree alone separates them
from unlabeled genes surprisingly well. Any GAT result should be read relative to this.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main(run_dir: Path, edges_path: Path) -> None:
    e = pd.read_csv(edges_path, sep="\t")
    e = e[e.gene1 != e.gene2]
    degree = pd.concat([e.gene1, e.gene2]).value_counts()
    rows = []
    for pred_csv in sorted(run_dir.glob("seed*/predictions.csv")):
        p = pd.read_csv(pred_csv)
        t = p[p.split == "test"]
        d = t.gene.map(degree).fillna(0)
        rows.append(
            {
                "seed": pred_csv.parent.name,
                "n_test": len(t),
                "n_pos": int(t.label.sum()),
                "degree_roc_auc": roc_auc_score(t.label, d),
                "degree_pr_auc": average_precision_score(t.label, d),
                "gat_mc_mean_roc_auc": roc_auc_score(t.label, t.mc_mean),
                "gat_mc_mean_pr_auc": average_precision_score(t.label, t.mc_mean),
                "prevalence": t.label.mean(),
            }
        )
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    print("\nmean:\n" + df.drop(columns="seed").mean().round(4).to_string())
    df.to_csv(run_dir / "degree_baseline.csv", index=False)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    edges = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "assets/dmd/ppi_edges_genes.tsv"
    main(run, edges)
