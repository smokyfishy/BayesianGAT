import numpy as np
import torch

from dmd_nfatc4.analysis.gene_ranking import random_walk_with_restart, rank_from_checkpoint, zscore_ranking
from dmd_nfatc4.data.gene_graph import parse_feature_blocks
from dmd_nfatc4.models import EnhancedBayesianGAT


def test_zscore_ranking_excludes_positives():
    df = zscore_ranking(["A", "B", "C", "D"], np.array([0.9, 0.8, 0.1, 0.5]), np.zeros(4), ["A"])
    assert "A" not in df.gene.tolist() and df.gene.tolist()[0] == "B" and df["rank"].tolist() == [1, 2, 3]


def test_rwr_is_probability_and_favours_seed_neighbourhood():
    ei = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    p, _, conv = random_walk_with_restart(ei, 5, [0])
    assert conv and abs(p.sum() - 1) < 1e-6
    assert p[1] > p[3] and p[3] == 0


def test_rank_from_checkpoint_end_to_end(synthetic_gene_data, tmp_path):
    df, edges, pos = synthetic_gene_data
    node_fp = tmp_path / "node_features.parquet"
    df.to_parquet(node_fp)
    edge_fp = tmp_path / "edges.tsv"
    edges.to_csv(edge_fp, sep="\t", index=False)
    b = parse_feature_blocks(list(df.columns), expected_expr=6, expected_esm=8)
    hp = dict(hidden_channels=8, out_channels=2, heads_1=2, heads_2=2, dropout=0.5, use_edge_confidence=True)
    torch.save(EnhancedBayesianGAT(b.n_features, **hp).state_dict(), tmp_path / "m.pth")
    ranked, info = rank_from_checkpoint(node_fp, edge_fp, tmp_path / "m.pth", hp, pos, knn_k=3, mc_samples=3)
    assert info["n_genes"] == len(df) and len(ranked) == len(df) - len(pos)
    assert not set(pos) & set(ranked.gene)
