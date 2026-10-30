import numpy as np

from dmd_nfatc4.analysis.clusters import find_clusters
from dmd_nfatc4.analysis.domains import HISTORICAL_CANDIDATE_WINDOWS, compare_windows, distance_to_domains
from dmd_nfatc4.analysis.residue_saliency import joint_score_raw


def test_bumps_become_clusters_and_short_runs_dropped():
    s = np.ones(300)
    s[50:70] = 50.0
    s[200] = 400.0
    df, _, _ = find_clusters(s, sigma=2, percentile=93, min_len=5)
    assert ((df.Start_index0 <= 55) & (df.End_index0 >= 65)).any()
    assert (df.Length >= 5).all()
    assert (df.Start_uniprot == df.Start_index0 + 1).all()


def test_distance_to_domains():
    assert distance_to_domains(10, 20, [(30, 40)]) == 10
    assert distance_to_domains(35, 50, [(30, 40)]) == 0
    assert distance_to_domains(50, 60, {"a": (30, 40)}) == 10


def test_saved_arrays_reproduce_historical_windows_as_0based(assets):
    A = assets / "nfatc4"
    raw = joint_score_raw(*(np.load(A / f) for f in ("saliency_vector.npy", "std_pred_vector.npy", "delta_hidden_vector.npy")))
    df, _, _ = find_clusters(raw)
    cmp = compare_windows(list(zip(df.Start_index0, df.End_index0)))
    assert sorted(cmp["exact_match_as_index0"]) == sorted(HISTORICAL_CANDIDATE_WINDOWS)
    assert cmp["exact_match_if_historical_were_uniprot"] == []


def test_windows_sit_on_simulation_boundaries():
    from dmd_nfatc4.analysis.domains import distance_to_simulation_boundaries

    d = [distance_to_simulation_boundaries(s + 1, e + 1) for s, e in HISTORICAL_CANDIDATE_WINDOWS]
    assert d == [0, 0, 9, 0, 0, 0, 0]
