import pytest

from scripts.study_task_evidence import literal_search, reseal, validate_evidence
from src.ui.components.study_protocol import ROOT, digest, draw_study_configuration


def test_reviewed_reseal_preserves_assignments_and_binds_evidence(
    tmp_path, monkeypatch
):
    import json
    from scripts import study_task_evidence as task
    from src.ui.components.study_protocol import verify_draw

    original = tmp_path / "original"
    protocol = draw_study_configuration(
        ROOT / "config/study.example.json",
        ROOT / "config/study_assignments.example.csv",
        original,
    )
    chunk = json.loads(
        (ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json").read_text(
            encoding="utf-8"
        )
    )
    official = {
        p: next(
            (k, v) for k, v in chunk.items() if v["cloud_provider"] == p and v["text"]
        )
        for p in ("aws", "azure")
    }
    evidence = dict(
        method="literal-corpus-review-v1",
        corpus_sha256=digest(
            ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
        ),
        tasks={},
    )
    for qid in protocol["config"]["tasks"]["T1"] + protocol["config"]["tasks"]["T2"]:
        evidence["tasks"][qid] = dict(
            verdict="DIRECT_ANSWER",
            rationale="Synthetic validation of seal mechanics only, not semantic coverage",
            template=protocol["queries"][qid]["query_type"],
            fragments=[
                dict(chunk_id=official[p][0], excerpt=official[p][1]["text"])
                for p in protocol["queries"][qid]["cloud_providers"]
            ],
        )
    destination = tmp_path / "new"
    updated = task.reseal(original, destination, protocol["config"]["tasks"], evidence)
    assert updated["assignments"] == protocol["assignments"]
    assert updated["config"]["labels"] == protocol["config"]["labels"]
    assert (original / "assignments.csv").read_bytes() == (
        destination / "assignments.csv"
    ).read_bytes()
    (destination / "task_evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="Task evidence changed"):
        verify_draw(destination)


def test_literal_search_uses_only_text_and_preserves_provider():
    chunks = {
        "a": dict(
            text="Quota is 10",
            cloud_provider="aws",
            service_name="ECS",
            url_source="https://docs.aws.amazon.com/ecs/",
        ),
        "b": dict(
            text="No answer",
            cloud_provider="aws",
            service_name="Quota",
            url_source="https://docs.aws.amazon.com/ecs/",
        ),
    }
    assert [x["chunk_id"] for x in literal_search(chunks, ["quota"], "aws")] == ["a"]
    assert not literal_search(chunks, ["quota"], "azure")
    with pytest.raises(ValueError):
        literal_search(chunks, [""])


def test_unapproved_tasks_cannot_create_new_seal(tmp_path):
    original = tmp_path / "original"
    old = draw_study_configuration(
        ROOT / "config/study.example.json",
        ROOT / "config/study_assignments.example.csv",
        original,
    )
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="exactly all six"):
        reseal(original, destination, old["config"]["tasks"], {"tasks": {}})
    assert not destination.exists()
    assert before == {p.name: p.read_bytes() for p in original.iterdir()}


def test_unsupported_direct_answer_does_not_pass_on_a_declared_boolean(tmp_path):
    original = tmp_path / "original"
    protocol = draw_study_configuration(
        ROOT / "config/study.example.json",
        ROOT / "config/study_assignments.example.csv",
        original,
    )
    ids = protocol["config"]["tasks"]["T1"] + protocol["config"]["tasks"]["T2"]
    evidence = dict(
        method="literal-corpus-review-v1",
        corpus_sha256=digest(
            ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
        ),
        tasks={
            qid: dict(verdict="DIRECT_ANSWER", rationale="declared only", fragments=[])
            for qid in ids
        },
    )
    with pytest.raises(ValueError, match="direct answer"):
        validate_evidence({}, protocol, evidence)
