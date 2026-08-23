"""Synthetic-only tests for the unsupported-taxonomy calibration analyzer."""

import csv

import pytest

from scripts import analyze_taxonomy_calibration as calibration


def _row(stratum, size, probability, verdict, retest=None):
    row = {
        "stratum": stratum,
        "stratum_size": str(size),
        "inclusion_prob": str(probability),
        "human_verdict": verdict,
    }
    if retest is not None:
        row["human_verdict_retest"] = retest
    return row


def test_horvitz_thompson_and_kish_with_known_weights():
    rows = [
        _row("large", 80, 2 / 80, "correcto"),
        _row("large", 80, 2 / 80, "incorrecto"),
        _row("small", 20, 2 / 20, "correcto"),
        _row("small", 20, 2 / 20, "correcto"),
    ]

    result = calibration.analyze_rows(rows, expected_population=100)

    assert result["ht_totals"] == pytest.approx(
        {"correcto": 60.0, "incorrecto": 40.0, "dudoso": 0.0}
    )
    assert result["ht_rates"] == pytest.approx(
        {"correcto": 0.6, "incorrecto": 0.4, "dudoso": 0.0}
    )
    assert result["kish_n_eff"] == pytest.approx(10000 / 3400)
    assert result["strata"]["large"]["rates"]["correcto"] == pytest.approx(0.5)


def test_cohen_kappa_matches_hand_calculation():
    first = ["correcto", "correcto", "incorrecto", "incorrecto"]
    second = ["correcto", "incorrecto", "incorrecto", "incorrecto"]

    raw, kappa = calibration.cohen_kappa(first, second)

    assert raw == pytest.approx(0.75)
    assert kappa == pytest.approx(0.5)


def test_more_than_twenty_percent_blank_fails_loudly():
    rows = [
        _row("one", 5, 1.0, "correcto"),
        _row("one", 5, 1.0, "incorrecto"),
        _row("one", 5, 1.0, "dudoso"),
        _row("one", 5, 1.0, ""),
        _row("one", 5, 1.0, ""),
    ]

    with pytest.raises(ValueError, match="el humano aún no termina.*2/5"):
        calibration.analyze_rows(rows, expected_population=5)


def test_unexpected_vocabulary_is_reported_with_count():
    rows = [
        _row("one", 3, 1.0, "correcto"),
        _row("one", 3, 1.0, "síntesis-legítima"),
        _row("one", 3, 1.0, "síntesis-legítima"),
    ]

    with pytest.raises(ValueError) as caught:
        calibration.analyze_rows(rows, expected_population=3)

    message = str(caught.value)
    assert "síntesis-legítima" in message
    assert ": 2" in message
    assert "correcto, incorrecto, dudoso" in message


def test_optional_retest_column_is_analyzed_when_present():
    rows = [
        _row("one", 4, 1.0, "correcto", "correcto"),
        _row("one", 4, 1.0, "correcto", "incorrecto"),
        _row("one", 4, 1.0, "incorrecto", "incorrecto"),
        _row("one", 4, 1.0, "incorrecto", "incorrecto"),
    ]

    result = calibration.analyze_rows(rows, expected_population=4)

    assert result["retest"] == pytest.approx(
        {"n_pairs": 4, "raw_agreement": 0.75, "cohen_kappa": 0.5}
    )


def test_missing_retest_column_is_declared_without_failure():
    result = calibration.analyze_rows(
        [_row("one", 1, 1.0, "correcto")], expected_population=1
    )

    assert result["retest_column_present"] is False
    assert result["retest"] is None


def test_synthetic_csv_is_read_and_report_is_guarded(tmp_path, monkeypatch):
    input_path = tmp_path / "synthetic.csv"
    output_path = tmp_path / "report.md"
    rows = [
        _row("large", 80, 2 / 80, "correcto"),
        _row("large", 80, 2 / 80, "incorrecto"),
        _row("small", 20, 2 / 20, "correcto"),
        _row("small", 20, 2 / 20, "correcto"),
    ]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    guarded = []

    def guard_spy(path):
        guarded.append(path)
        return path

    monkeypatch.setattr(calibration, "guard_write", guard_spy)
    result = calibration.write_report(input_path, output_path, expected_population=100)

    assert guarded == [output_path]
    assert result["ht_totals"]["correcto"] == pytest.approx(60.0)
    assert "n efectivo de Kish" in output_path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSING to overwrite"):
        calibration.write_report(input_path, output_path, expected_population=100)
