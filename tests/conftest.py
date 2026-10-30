from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"


@pytest.fixture(scope="session")
def assets() -> Path:
    if not (ASSETS / "nfatc4" / "AF-Q14934-F1-model_v4.pdb").exists():
        pytest.skip("local assets/ not present (git-ignored); see SOURCES.md")
    return ASSETS


@pytest.fixture(scope="session")
def synthetic_gene_data():
    from dmd_nfatc4.data.gene_graph import make_synthetic_gene_table

    return make_synthetic_gene_table(n_genes=120, n_pos=12, n_go=10, n_esm=8, seed=1)
