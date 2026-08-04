"""Whatever the deployed pipeline generates with, it must be RECORDED, not assumed.

Found 2026-08-04, and it is the biggest divergence in the phase: `SURVEY_DEPLOY` -- the config
the Streamlit demo and the SUS/Likert survey run on -- generates with
`llama3.1:8b-instruct-q4_K_M`, while exp15 Tier A, exp16, exp17 and exp18 are ALL
`granite4.1:8b`. Every faithfulness number in the phase (the 0.4638 HHEM anchor, exp17's
balancing effect, exp18's three verdicts) is granite. The survey would ship a generator with
no faithfulness evidence behind it, and the first latency run compared its timings against
exp18's without noticing they were different models.

Which model the survey runs is Enzo's decision, not a bug to fix here, so these tests do NOT
assert that the two agree. They assert the divergence cannot be INVISIBLE again: the config
carries the model, and any artifact that times the deployment path records which generator
produced the numbers.

Run: pytest tests/test_deployment_generator.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "experiments" / "results"
LATENCY_SCRIPT = PROJECT_ROOT / "scripts" / "measure_survey_config_latency.py"


def _summer_models():
    out = set()
    for p in sorted(RESULTS.glob("exp1[5-8]*/results.json")):
        for c in json.loads(p.read_text(encoding="utf-8"))["configs"].values():
            if c.get("model"):
                out.add(c["model"])
    return out


def test_the_deployment_config_names_its_generator():
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    assert SURVEY_DEPLOY.llm_model, "the deployed config must say what it generates with"


@pytest.mark.needs_artifacts
def test_the_survey_runs_the_generator_the_evidence_was_measured_on():
    """Enzo's decision, 2026-08-04 (ledger entry 24), pinned so it cannot drift back.

    Derived from the committed results.json, never from a constant here: if the phase ever
    re-measures on a different generator, this follows it instead of going stale.
    """
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    models = _summer_models()
    if not models:
        pytest.skip("summer results not present")
    assert {SURVEY_DEPLOY.llm_model} == models, (
        f"the survey would generate with {SURVEY_DEPLOY.llm_model} while every summer number "
        f"comes from {sorted(models)}. That is defect #12 coming back: participants would rate "
        f"a system with no faithfulness evidence behind it.")


def test_the_paper_record_config_is_not_edited_to_match():
    """PROPOSED_HYBRID is the record of what was submitted to LACCI, not a knob.

    The submission describes Llama 3.1 8B Q4 over 200 queries. Aligning it with the survey
    would make the two agree by falsifying the record instead of by making a decision.
    """
    from src.pipeline.pipeline_config import PROPOSED_HYBRID, get_config
    assert PROPOSED_HYBRID.llm_model.startswith("llama3.1"), (
        "PROPOSED_HYBRID no longer carries the LACCI-submitted generator")
    assert get_config("hybrid").llm_model == PROPOSED_HYBRID.llm_model, (
        "the experimental registry must keep returning the measured system")


@pytest.mark.needs_artifacts
def test_the_summer_evidence_is_one_single_generator():
    """If the evidence itself were mixed, 'the measured model' would be meaningless."""
    models = _summer_models()
    if not models:
        pytest.skip("summer results not present")
    assert len(models) == 1, f"summer evidence spans several generators: {sorted(models)}"


@pytest.mark.needs_artifacts
def test_a_divergence_between_deployed_and_measured_generator_is_declared_not_silent():
    """The decision is Enzo's; being unable to SEE it is not a decision at all."""
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    models = _summer_models()
    if not models:
        pytest.skip("summer results not present")
    if {SURVEY_DEPLOY.llm_model} == models:
        return  # aligned; nothing to declare
    src = LATENCY_SCRIPT.read_text(encoding="utf-8")
    assert "llm_model_of_summer_evidence" in src and "model_divergence" in src, (
        f"the deployed generator ({SURVEY_DEPLOY.llm_model}) differs from the one behind every "
        f"summer number ({sorted(models)}), and the deployment-path measurement does not record "
        f"it. Timings and any faithfulness claim would silently mix two models.")


def test_the_latency_artifact_records_the_generator():
    """Derived from the config at write time, never hardcoded into the report.

    Checked on executable tokens only: the prose explains WHICH model was found and why that
    mattered, and a guard that fires on its own explanation teaches people to delete it.
    """
    from conftest import code_only
    src = LATENCY_SCRIPT.read_text(encoding="utf-8")
    assert "SURVEY_DEPLOY.llm_model" in src, \
        "the latency artifact must record the generator it measured"
    assert "llama3.1" not in code_only(src), \
        "the model name is hardcoded in code; derive it from the config"
