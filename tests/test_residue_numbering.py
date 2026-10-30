import numpy as np

from dmd_nfatc4.analysis.visualization import write_pml
from dmd_nfatc4.data.alignment import check_sequence_structure_alignment, index_to_uniprot, read_fasta, uniprot_to_index
from dmd_nfatc4.data.residue_graph import parse_pdb_ca
from dmd_nfatc4.data.simulation import describe_flags, simulate_mutation_flags


def test_index_uniprot_roundtrip():
    assert index_to_uniprot(0) == 1 and uniprot_to_index(902) == 901


def test_simulation_runs_in_uniprot_numbering():
    d = describe_flags(simulate_mutation_flags(-2.0))
    assert d["n_flagged"] == 188
    assert [(r["uniprot_start"], r["uniprot_end"]) for r in d["runs"]] == [(41, 96), (201, 281), (701, 751)]
    assert simulate_mutation_flags(0.5).sum() == 0  # fold-change gate not met -> no flags


def test_pml_uses_1based_ranges(tmp_path):
    p = write_pml(tmp_path / "x.pdb", [(37, 45)], tmp_path / "w.pml")
    assert "resi 37-45" in p.read_text()


def test_nfatc4_structure_sequence_alignment(assets):
    _, seq = read_fasta(assets / "nfatc4/Q14934.fasta")
    rep = check_sequence_structure_alignment(seq, parse_pdb_ca(assets / "nfatc4/AF-Q14934-F1-model_v4.pdb"))
    assert rep["fasta_length"] == 902 and rep["index_equals_uniprot_minus_1"]


def test_saved_vector_matches_simulation_and_fold_change(assets):
    from dmd_nfatc4.data.expression import gene_log2_fold_change, load_gene_tpm

    fc = gene_log2_fold_change(load_gene_tpm(assets / "dmd/expression_gene_tpm_protein_coding.parquet"), "NFATC4")
    assert fc["n_control"] == 3 and fc["n_dmd"] == 3
    saved = np.load(assets / "nfatc4/simulated_dmd_mutation_vector.npy")
    np.testing.assert_array_equal(simulate_mutation_flags(fc["log2_fc"]), saved)


def test_original_pml_selection_is_index_plus_one(assets):
    txt = (assets / "nfatc4/nfatc4_mutation_highlight.pml").read_text()
    sel = [ln for ln in txt.splitlines() if ln.startswith("select mutated_residues")][0]
    resi = [int(v) for v in sel.split("resi ")[1].split("+")]
    flags = np.load(assets / "nfatc4/simulated_dmd_mutation_vector.npy")
    assert resi == (np.where(flags == 1)[0] + 1).tolist()
