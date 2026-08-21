"""exp21 — the hosted-equivalence harness, driven against a mock endpoint.

Seccion de Claude Code — 2026-08-21 08:10 (hora local).

This gate decides whether the SUS/Likert survey may run on rented hardware, and it is the first
thing in this project that can spend money and can talk to a machine that is not Enzo's. So the
properties pinned here are the ones whose failure would be expensive or silent:

  1. No endpoint may be hardcoded. A URL or a token in the repo is a leak, and the harness must
     refuse to run rather than fall back to a default.
  2. The endpoint must never reach an artifact. Only a hash prefix is recorded, so two runs are
     comparable without publishing where they ran.
  3. The digest gate must fire BEFORE generating. Discovering after 194 hosted queries that the
     weights differed means having paid for a number that means nothing.
  4. The arm schema must match what the local scorers discover, or the run is unscoreable.

Nothing here touches a network: the endpoint is injected.

Run: pytest tests/test_exp21_harness.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def harness():
    path = PROJECT_ROOT / "scripts" / "run_exp21_hosted_equivalence.py"
    assert path.exists(), "scripts/run_exp21_hosted_equivalence.py missing"
    spec = importlib.util.spec_from_file_location("exp21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _show(digest="sha256:abc", quant="Q4_K_M", params="8.0B"):
    """Shape of an Ollama /api/show payload, with the identity fields split across `details`."""
    return {"digest": digest, "details": {"quantization_level": quant, "parameter_size": params}}


# ------------------------------------------------------------------ 1. the endpoint is a secret
def test_no_endpoint_is_hardcoded_anywhere_in_the_harness(harness):
    """A committed URL or token is a leak; only localhost may appear, for the local arm."""
    from conftest import code_only
    src = code_only((PROJECT_ROOT / "scripts" / "run_exp21_hosted_equivalence.py")
                    .read_text(encoding="utf-8"))
    for pattern in ("runpod", "vast.ai", "ngrok", "bearer ", "sk-", "api_key", "apikey"):
        assert pattern not in src.lower(), f"{pattern!r} reachable in executable code"


def test_the_hosted_stage_refuses_to_run_without_the_env_var(harness, monkeypatch):
    """No default endpoint, on purpose: a fallback host is how one gets committed."""
    monkeypatch.delenv(harness.HOST_ENV, raising=False)
    with pytest.raises(SystemExit) as e:
        harness.resolve_endpoint("hosted", None)
    assert harness.HOST_ENV in str(e.value)


def test_the_local_stage_needs_no_env_var(harness, monkeypatch):
    monkeypatch.delenv(harness.HOST_ENV, raising=False)
    ep, host = harness.resolve_endpoint("local", None)
    assert host == harness.LOCAL_HOST


def test_the_endpoint_is_recorded_only_as_a_fingerprint(harness):
    host = "http://203.0.113.7:11434"
    fp = harness.host_fingerprint(host)
    assert host not in fp and "203.0.113" not in fp
    assert len(fp) == 16
    assert fp == harness.host_fingerprint(host), "same host must give the same fingerprint"
    assert fp != harness.host_fingerprint(host + "/x")


def test_the_token_is_not_stored_on_the_client_in_the_clear_attribute_name(harness):
    ep = harness.OllamaEndpoint("http://h", token="secret-token")
    assert "secret-token" not in json.dumps(
        {k: str(v) for k, v in vars(ep).items() if not k.startswith("_")}), \
        "public attributes must not carry the token"


# ------------------------------------------------------------------ 2. the digest gate
def test_matching_models_pass_the_digest_gate(harness):
    ok, report = harness.digest_gate(_show(), _show())
    assert ok and report["passed"] and report["mismatched_fields"] == []


@pytest.mark.parametrize("field,other", [
    ("digest", _show(digest="sha256:def")),
    ("quantization_level", _show(quant="Q8_0")),
    ("parameter_size", _show(params="70B")),
])
def test_any_identity_difference_fails_the_gate(harness, field, other):
    """Same tag, different weights is exactly the failure that would waste the whole spend."""
    ok, report = harness.digest_gate(_show(), other)
    assert not ok
    assert field in report["mismatched_fields"]


def test_the_gate_reads_fields_from_details_as_well_as_the_top_level(harness):
    """Ollama splits these across the payload; a gate that only looked at the top level would
    silently compare None to None and pass everything."""
    ident = harness.model_identity(_show(quant="Q4_K_M"))
    assert ident["quantization_level"] == "Q4_K_M"
    assert ident["digest"] == "sha256:abc"


def test_a_missing_field_on_one_side_is_a_mismatch_not_a_pass(harness):
    ok, report = harness.digest_gate(_show(), {"digest": "sha256:abc"})
    assert not ok, "None == None must not be read as agreement"


# ------------------------------------------------------------------ 3. arm schema and parsing
def test_the_anchor_arm_is_named_baseline(harness):
    """verify_summer_offline.py finds the anchor as the arm named `baseline*`."""
    assert harness.ARM_FOR_STAGE["local"].startswith("baseline")
    assert harness.ARM_FOR_STAGE["hosted"] == "hosted"


def test_answer_extraction_survives_an_empty_or_odd_payload(harness):
    assert harness.answer_of({"message": {"content": "hi"}}) == "hi"
    assert harness.answer_of({}) == ""
    assert harness.answer_of({"message": {}}) == ""
    assert harness.answer_of(None) == ""


def test_results_doc_carries_the_arm_schema_and_no_endpoint(harness, tmp_path, monkeypatch):
    label = "granite4.1-8b"
    for arm in ("baseline_local", "hosted"):
        (tmp_path / f"checkpoint__{label}__{arm}.json").write_text(json.dumps({
            "config_name": f"{arm} | {label}", "completed_ids": ["q001"],
            "results": [{"query_id": "q001", "scenario": arm, "answer": "a",
                         "retrieved_ids": ["c1"], "tokens": {"input": 5, "output": 2}}]}),
            encoding="utf-8")

    class Args:
        stage = "hosted"
        local_source = "fresh"
    host = "http://203.0.113.7:11434"
    doc = harness.write_results(tmp_path, label, {}, ["q001"], None, host, Args())

    assert set(doc["configs"]) == {f"baseline_local | {label}", f"hosted | {label}"}
    for cfg in doc["configs"].values():
        assert "scenario" in cfg
    blob = json.dumps(doc)
    assert host not in blob and "203.0.113" not in blob, "the endpoint must never be serialised"
    assert doc["endpoint_fingerprint"]["hosted"] == harness.host_fingerprint(host)


def test_the_band_and_family_are_declared_in_the_artifact(harness, tmp_path):
    class Args:
        stage = "local"
        local_source = "fresh"
    doc = harness.write_results(tmp_path, "granite4.1-8b", {}, [], None,
                                harness.LOCAL_HOST, Args())
    assert "0.081" in doc["equivalence_note"]
    assert "p_BH == p_raw" in doc["bh_family_note"]
    assert "bit-identity" in doc["equivalence_note"], \
        "the artifact must say equality is NOT the criterion"


def test_verifiers_are_declared_to_stay_local(harness, tmp_path):
    """No verifier may run on rented hardware: the instrument must not move with the host."""
    class Args:
        stage = "local"
        local_source = "fresh"
    doc = harness.write_results(tmp_path, "granite4.1-8b", {}, [], None,
                                harness.LOCAL_HOST, Args())
    assert "locally" in doc["scoring_note"]
