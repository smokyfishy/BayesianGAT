import json

import pytest

from dmd_nfatc4.cli import build_parser, main
from dmd_nfatc4.config import REPO_ROOT


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for cmd in ("train-gat", "rank-genes", "analyze-nfatc4", "visualize-nfatc4", "inspect-checkpoint"):
        assert cmd in out


def test_train_gat_synthetic_smoke(tmp_path, capsys):
    cfg = str(REPO_ROOT / "configs" / "gat_synthetic_demo.yaml")
    rc = main(["train-gat", "--config", cfg, "--output-dir", str(tmp_path), "--epochs", "3"])
    assert rc == 0
    run = json.loads(capsys.readouterr().out)["runs"][0]
    assert run["leakage_audit"]["test"]["n_synthetic"] == 0
    assert (tmp_path / "gat_synthetic_demo" / "seed42" / "model.pth").exists()


def test_rank_genes_blocked_without_node_table(tmp_path, capsys):
    rc = main(["rank-genes", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    assert rc == 2 and "BLOCKED" in capsys.readouterr().out
