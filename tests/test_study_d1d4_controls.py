import json
from datetime import datetime, timezone

import pytest

from scripts.review_study_language import annotate
from src.ui.components.study_protocol import ROOT, draw_study_configuration, verify_draw


def test_draw_is_reproducible_sealed_and_has_approved_reserves(tmp_path):
    config = ROOT / "config/study.example.json"
    assignments = ROOT / "config/study_assignments.example.csv"
    first = draw_study_configuration(config, assignments, tmp_path / "first")
    second = draw_study_configuration(config, assignments, tmp_path / "second")
    assert first["fingerprint"] == second["fingerprint"]
    rows = first["assignments"]
    assert [
        (rows[f"P{i:02d}"]["cell"], rows[f"P{i:02d}"]["profile"]) for i in range(21, 25)
    ] == [
        (1, "with_experience"),
        (2, "without_experience"),
        (3, "with_experience"),
        (4, "without_experience"),
    ]
    (tmp_path / "first" / "study.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_draw(tmp_path / "first")


def test_review_records_operator_time_and_never_changes_export(tmp_path):
    export = tmp_path / "full_session.json"
    export.write_text(
        json.dumps({"attempts": [{"analysis_role": "free_query", "attempt_id": "a"}]}),
        encoding="utf-8",
    )
    before = export.read_bytes()
    result = annotate(
        export,
        "english",
        tmp_path / "review.json",
        "OP01",
        now=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    assert result["reviewer_id"] == "OP01" and result["annotated_at"].endswith("+00:00")
    assert export.read_bytes() == before
    with pytest.raises(ValueError):
        annotate(export, "english", tmp_path / "other.json", "")
