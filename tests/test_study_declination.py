"""Study metadata reuses the historical v2 rule without changing presentation."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import compute_faithfulness_metrics as historical
from src.evaluation.study_analysis import analyze
from src.ui.components import study_service as service
from src.ui.components.session_storage import SessionConflict
from tests.test_study_analysis import sample
from tests.test_study_sessions import active as active, response


SAMPLES = [
    ("The documentation does not mention this service.", "pure_decline"),
    ("Documented details. " * 30 + "There is no information about quotas.", "hedged_partial"),
    ("Use the documented CLI command.\n\nKeep this formatting.  ", "answered"),
]


def test_historical_and_study_share_the_same_classifier_object():
    from src.evaluation import decline_classifier as canonical
    from src.ui.components import study_sessions

    assert study_sessions.classify_response is historical.classify_response
    assert canonical.classify_response is historical.classify_response
    assert canonical.OPENING_WINDOW == 300
    assert len(canonical._refusal_markers()) == 28
    assert canonical.CLASSIFIER_VERSION == "faithfulness_v2_28_patterns_300_chars"


@pytest.mark.parametrize("text,expected", SAMPLES)
@pytest.mark.parametrize("role", ["tasks", "free_query"])
@pytest.mark.parametrize("block", [0, 1])
def test_success_records_class_without_editing_or_allowing_retry(active, text, expected, role, block):
    store, session, token = active
    session.familiarization_done()
    session.data.update(stage=role, block_index=block)
    session.save()
    service.answer(session, lambda _: SimpleNamespace(query=lambda q: response(text)), "free")
    assert session.pending["status"] == "success"
    assert session.pending["error"] is None
    assert session.pending["answer"] == text
    assert session.pending["decline_class"] == expected
    assert session.pending["decline_classifier_version"] == "faithfulness_v2_28_patterns_300_chars"
    with pytest.raises(SessionConflict):
        session.begin("free")
    loaded = store.admit(token)
    assert loaded.pending == session.pending
    loaded.shown()
    loaded.acknowledge()
    assert len(loaded.data["attempts"]) == 1


def test_completed_export_has_class_on_all_eight_answers(active):
    from tests.test_study_sessions import finish

    _, session, _ = active
    finish(session)
    path = session.export()
    original = path.read_bytes()
    row = json.loads(original)
    assert len(row["attempts"]) == 8
    assert all(a["decline_class"] == "answered" for a in row["attempts"])
    assert all(a["decline_classifier_version"] for a in row["attempts"])
    assert session.export().read_bytes() == original


def test_technical_error_is_not_a_declination(active):
    _, session, _ = active
    session.familiarization_done()
    service.answer(session, lambda _: SimpleNamespace(query=lambda q: response("")))
    assert session.pending["status"] == "error"
    assert session.pending["decline_class"] is None
    assert session.pending["decline_classifier_version"] is None


def test_analysis_separates_classes_tasks_free_queries_and_f4_without_imputation():
    from src.evaluation.decline_classifier import CLASSIFIER_VERSION

    rows = sample(2)
    for row in rows:
        row["attempts"] = [dict(status="acknowledged", condition="hybrid", query_id=f"q00{i+1}",
            analysis_role="tasks", answer=text, decline_class=cls,
            decline_classifier_version=CLASSIFIER_VERSION) for i, (text, cls) in enumerate(SAMPLES)]
        row["attempts"] += [dict(status="error", condition="hybrid", analysis_role="tasks"),
            dict(status="acknowledged", condition="no_rag", analysis_role="free_query", query_id=None)]
    original = copy.deepcopy(rows)
    result = analyze(rows)
    d = result["declination_descriptive"]["conditions"]
    assert d["hybrid"]["F4"] == [4, 4]
    assert d["hybrid"]["tasks"]["counts"] == dict(pure_decline=2, hedged_partial=2, answered=2)
    assert d["hybrid"]["tasks"]["proportions"]["pure_decline"] == pytest.approx(1/3)
    assert d["hybrid"]["by_task"]["q001"]["proportions"]["pure_decline"] == 1
    assert d["no_rag"]["free_queries"]["missing_metadata"] == 2
    assert d["no_rag"]["free_queries"]["classified_denominator"] == 0
    assert d["no_rag"]["free_queries"]["proportions"]["answered"] is None
    assert result["technical_errors_in_included_sessions"] == 2
    assert set(result["contrasts"]) == {"SUS", "F", "U"}
    assert rows == original
    rows[0]["attempts"][0]["decline_classifier_version"] = "unknown"
    with pytest.raises(ValueError, match="classifier"):
        analyze(rows)


@pytest.mark.parametrize("purpose", ["smoke", "rehearsal", "pilot", "technical"])
def test_non_study_declinations_never_enter_analysis(purpose):
    rows = sample(1)
    rows[0]["purpose"] = purpose
    result = analyze(rows)
    assert not result["included"]
    assert result["declination_descriptive"]["conditions"]["hybrid"]["tasks"]["classified_denominator"] == 0
