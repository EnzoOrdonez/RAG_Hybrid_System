import csv
import json

import numpy as np
import pytest

from scripts import compute_descriptive_cis as descriptive


def test_wilson_and_exact_binomial_match_known_values():
    assert descriptive.wilson_interval(16, 50) == pytest.approx(
        (0.207582, 0.458103), abs=1e-6
    )
    assert descriptive.wilson_interval(11, 20) == pytest.approx(
        (0.342085, 0.741802), abs=1e-6
    )
    assert descriptive.exact_two_sided_binomial(11, 14) == pytest.approx(
        0.057373046875
    )
    assert descriptive.exact_two_sided_binomial(11, 16) == pytest.approx(
        0.210113525390625
    )


def test_retest_matrix_reconstructs_original_only_from_derived_disagreements():
    retest = {
        "mapeo": {"C-01": "A-001", "C-02": "A-002", "C-03": "A-003", "C-04": "A-004"},
        "juicios": {
            "C-01": {"juicio": "correcto"},
            "C-02": {"juicio": "incorrecto"},
            "C-03": {"juicio": "dudoso"},
            "C-04": {"juicio": "incorrecto"},
        },
    }
    derived = {
        "n": 4,
        "discordantes": [
            {"tandaC": "C-02", "ref": "A-002", "gold": "correcto", "retest": "incorrecto"}
        ],
    }

    result = descriptive.analyze_retest(retest, derived)

    assert result["matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert result["agreement_count"] == 3
    assert result["raw_agreement"] == pytest.approx(0.75)
    assert result["cohen_kappa"] == pytest.approx(0.6363636364)


def _write_semicolon_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def test_evidence_set_sensitivity_uses_paired_stage_a_references(tmp_path):
    stage_a = tmp_path / "a.csv"
    stage_b = tmp_path / "b.csv"
    _write_semicolon_csv(
        stage_a,
        ["idx", "juicio_humano"],
        [
            {"idx": "1", "juicio_humano": "correcto"},
            {"idx": "2", "juicio_humano": "correcto"},
            {"idx": "3", "juicio_humano": "incorrecto"},
            {"idx": "4", "juicio_humano": "dudoso"},
            {"idx": "5", "juicio_humano": "incorrecto"},
        ],
    )
    _write_semicolon_csv(
        stage_b,
        ["stage_a_idx", "juicio_humano"],
        [
            {"stage_a_idx": "1", "juicio_humano": "correcto"},
            {"stage_a_idx": "2", "juicio_humano": "dudoso"},
            {"stage_a_idx": "3", "juicio_humano": "correcto"},
            {"stage_a_idx": "4", "juicio_humano": "incorrecto"},
            {"stage_a_idx": "5", "juicio_humano": "incorrecto"},
        ],
    )

    result = descriptive.analyze_evidence_set(stage_a, stage_b)

    assert result["matrix"] == [[1, 0, 1], [1, 1, 0], [0, 1, 0]]
    assert result["n"] == 5
    assert result["flip_count"] == 3
    assert result["binary_correct_rest"] == {
        "a_correct_b_correct": 1,
        "a_correct_b_other": 1,
        "a_other_b_correct": 1,
        "a_other_b_other": 2,
    }
    assert result["mcnemar_exact_p"] == pytest.approx(1.0)
    assert result["all_flips_towards_correct"] == 1
    assert result["all_flips_other_direction"] == 2


def test_taxonomy_summary_reuses_ht_weights_from_calibration_analyzer(tmp_path):
    taxonomy = tmp_path / "taxonomy.csv"
    with taxonomy.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stratum", "stratum_size", "inclusion_prob", "human_verdict"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"stratum": "s1", "stratum_size": 4, "inclusion_prob": 0.5, "human_verdict": "correcto"},
                {"stratum": "s1", "stratum_size": 4, "inclusion_prob": 0.5, "human_verdict": "incorrecto"},
                {"stratum": "s2", "stratum_size": 6, "inclusion_prob": 1 / 3, "human_verdict": "correcto"},
                {"stratum": "s2", "stratum_size": 6, "inclusion_prob": 1 / 3, "human_verdict": "dudoso"},
            ]
        )

    result = descriptive.analyze_taxonomy(taxonomy, expected_population=10)

    assert result["counts"] == {"correcto": 2, "incorrecto": 1, "dudoso": 1}
    assert result["ht_rates"] == pytest.approx(
        {"correcto": 0.5, "incorrecto": 0.2, "dudoso": 0.3}
    )
    assert result["kish_n_eff"] == pytest.approx(100 / 26)


def test_threshold_sensitivity_counts_equal_scores_as_unsupported(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    np.save(scores_dir / "q1.npy", np.array([[0.4, 0.2], [0.5, 0.1], [0.61, 0.1]]))
    np.save(scores_dir / "q2.npy", np.array([[0.39, 0.2], [0.6, 0.1]]))
    index = {
        "q1": {"claims": ["a", "b", "c"], "shape": [3, 2]},
        "q2": {"claims": ["d", "e"], "shape": [2, 2]},
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = descriptive.analyze_thresholds(index_path, scores_dir)

    assert result == {
        "n_queries": 2,
        "n_claims": 5,
        "comparator": "best_over_pool <= tau",
        "counts": {"0.4": 2, "0.5": 3, "0.6": 4},
    }


def test_threshold_sensitivity_rejects_missing_matrix(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"q1": {"claims": ["a"], "shape": [1, 1]}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="matriz ausente"):
        descriptive.analyze_thresholds(index_path, tmp_path / "scores")


def _make_cli_fixture(tmp_path, *, include_matrix=True):
    audit = tmp_path / "audit"
    exp18 = tmp_path / "exp18"
    audit.mkdir()
    (exp18 / "selection_scores_v2").mkdir(parents=True)
    _write_semicolon_csv(
        audit / "claim_audit_sample_v4.csv",
        ["idx", "juicio_humano"],
        [{"idx": "1", "juicio_humano": "correcto"}],
    )
    _write_semicolon_csv(
        audit / "claim_audit_sample_v4_stageB.csv",
        ["stage_a_idx", "juicio_humano"],
        [{"stage_a_idx": "1", "juicio_humano": "correcto"}],
    )
    (audit / "gold_v4_tandaC_enzo_2026-08-29.json").write_text(
        json.dumps(
            {
                "mapeo": {"C-01": "A-001", "C-02": "A-002"},
                "juicios": {
                    "C-01": {"juicio": "correcto"},
                    "C-02": {"juicio": "incorrecto"},
                },
            }
        ),
        encoding="utf-8",
    )
    (audit / "gold_v4_tandaC_resultado.json").write_text(
        json.dumps({"n": 2, "discordantes": []}), encoding="utf-8"
    )
    with (audit / "unsupported_claims_sample_v2.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stratum", "stratum_size", "inclusion_prob", "human_verdict"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stratum": "only",
                "stratum_size": 759,
                "inclusion_prob": 1 / 759,
                "human_verdict": "correcto",
            }
        )
    index = {"q1": {"claims": ["a"], "shape": [1, 1]}}
    (exp18 / "selection_scores_v2_index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    if include_matrix:
        np.save(exp18 / "selection_scores_v2" / "q1.npy", np.array([[0.5]]))
    return audit, exp18


def test_cli_writes_json_and_markdown_after_all_inputs_validate(tmp_path):
    audit, exp18 = _make_cli_fixture(tmp_path)
    output_json = tmp_path / "result.json"
    output_md = tmp_path / "result.md"

    assert descriptive.main(
        [
            "--audit-dir", str(audit),
            "--exp18-dir", str(exp18),
            "--output-json", str(output_json),
            "--output-md", str(output_md),
        ]
    ) == 0

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["metadata"]["descriptive_only"] is True
    assert result["threshold_sensitivity"]["counts"] == {"0.4": 0, "0.5": 1, "0.6": 1}
    assert "fuera de las familias BH" in output_md.read_text(encoding="utf-8")


def test_cli_creates_no_output_when_an_input_is_invalid(tmp_path):
    audit, exp18 = _make_cli_fixture(tmp_path, include_matrix=False)
    output_json = tmp_path / "result.json"
    output_md = tmp_path / "result.md"

    with pytest.raises(SystemExit):
        descriptive.main(
            [
                "--audit-dir", str(audit),
                "--exp18-dir", str(exp18),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
            ]
        )

    assert not output_json.exists()
    assert not output_md.exists()
