import pytest

from scripts.analyze_user_sessions import compute_sus_stats, run_statistical_tests


def test_latency_percentiles_exclude_missing_observations():
    from scripts.analyze_user_sessions import compute_timing_stats
    values = [{"system": "hybrid", "system_latency_ms": 1000},
              {"system": "hybrid", "system_latency_ms": 3000}, {"system": "hybrid"}]
    result = compute_timing_stats([{"timestamps": values}])["hybrid"]
    assert result["system_latency_n"] == 2
    assert result["system_latency_p50_ms"] == 2000
    assert result["system_latency_p95_ms"] == pytest.approx(2900)
    assert "total_n" not in result

def test_valid_zero_sus_is_included_but_missing_is_not():
    result = compute_sus_stats([{"sus_score": 0}, {"sus_score": 100}, {}])
    assert result["n"] == 2
    assert result["mean"] == 50


def test_pairs_follow_participant_identity_with_missing_systems():
    sessions = []
    for pid, values in [("P01", {"hybrid": 1}), ("P02", {"lexical": 5}),
                        ("P03", {"hybrid": 2, "lexical": 3}),
                        ("P04", {"hybrid": 3, "lexical": 4}),
                        ("P05", {"hybrid": 4, "lexical": 5})]:
        sessions.append({"participant_id": pid, "ratings": [
            {"system": system, "utility_rating": score} for system, score in values.items()]})
    result = run_statistical_tests(sessions)["hybrid_vs_lexical"]
    assert result["n"] == 3
    assert result["participant_ids"] == ["P03", "P04", "P05"]
    assert result["mean_a"] == 3
    assert result["mean_b"] == 4
    assert result["cohens_d"] is None  # nonzero constant difference: undefined d_z
    assert "wilcoxon_pvalue_bh" in result


def test_duplicate_participants_are_rejected():
    with pytest.raises(ValueError, match="Duplicate participant"):
        run_statistical_tests([{"participant_id": "P01"}, {"participant_id": "P01"}])


def test_zero_differences_are_explicit():
    sessions = [{"participant_id": f"P{i}", "ratings": [
        {"system": s, "utility_rating": 3} for s in ("hybrid", "lexical", "semantic")
    ]} for i in range(3)]
    result = run_statistical_tests(sessions)["hybrid_vs_lexical"]
    assert result["wilcoxon_pvalue"] == 1
    assert result["wilcoxon_pvalue_bh"] == 1
    assert not result["significant_005"]
