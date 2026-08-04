"""The exp19 selector must never touch a verifier that scores it.

If the thing that PICKS the evidence is the same thing that later JUDGES whether the evidence
grounds the answer, the experiment measures its own preferences. exp19's whole design rests on
keeping those separate: selection runs on ms-marco-MiniLM-L-12-v2 (the production reranker), so
NLI-small, NLI-base and HHEM all stay clean evaluators.

Two further separations this pins:
  * `bge-reranker-large` is the INDEPENDENT retrieval oracle (Flag 17). Using it inside the
    method would burn it, and every later "independent oracle" claim with it.
  * the selector conditions on the DRAFT's claims. Conditioning on the final answer's claims
    would be the circularity the selection bound already is, and the bound is not an estimator.

Source-level, on purpose. A runtime check would only fire on the path a test happens to walk;
the constraint is about what the module is ALLOWED to reach at all.

Run: pytest tests/test_selector_hygiene.py -v
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SELECTOR_SCRIPTS = ["compute_exp19a_selector_probe.py"]

# Anything that produces a faithfulness label or score. If the selector can reach one of these,
# the arm it produces cannot be scored by that verifier without circularity.
VERIFIER_TOKENS = [
    "load_hhem", "HHEM", "hhem-2.1", "vectara",
    "decide_nli_status", "nli-deberta", "nli_probs", "HallucinationDetector",
    "rescore_grounding", "rescore_nli",
]
ORACLE_TOKENS = ["bge-reranker-large", "bge_reranker_large"]


def _sources():
    out = []
    for name in SELECTOR_SCRIPTS:
        p = PROJECT_ROOT / "scripts" / name
        if p.exists():
            out.append((name, p.read_text(encoding="utf-8")))
    return out


def _code_only(src):
    """Executable tokens only — every string literal and comment removed.

    Prose may legitimately NAME what the code refuses to reach ("not bge-reranker-large: that
    stays the independent oracle"), and a guard that fires on its own documentation trains
    people to weaken it. What matters is what the module can actually call.
    """
    import io
    import tokenize
    # Py3.12+ splits f-strings into FSTRING_START/MIDDLE/END, so filtering tokenize.STRING alone
    # leaves the literal text of every f-string behind. Match by NAME to stay version-tolerant.
    drop = {"STRING", "COMMENT", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"}
    kept = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tokenize.tok_name.get(tok.type) in drop:
                continue
            kept.append(tok.string)
    except tokenize.TokenError:  # pragma: no cover - malformed source is its own failure
        return src
    return " ".join(kept)


def test_the_selector_scripts_exist():
    assert _sources(), f"none of {SELECTOR_SCRIPTS} found — did a rename orphan this guard?"


@pytest.mark.parametrize("name,token", [(n, t) for n, _ in _sources() for t in VERIFIER_TOKENS])
def test_no_faithfulness_verifier_in_the_selection_path(name, token):
    src = _code_only(dict(_sources())[name])
    assert token.lower() not in src.lower(), (
        f"{name} reaches `{token}` outside its docstring. If the verifier that scores exp19 also "
        f"selects its evidence, the result is instrument circularity, not a finding.")


@pytest.mark.parametrize("name,token", [(n, t) for n, _ in _sources() for t in ORACLE_TOKENS])
def test_the_independent_oracle_is_not_burned(name, token):
    src = _code_only(dict(_sources())[name])
    assert token.lower() not in src.lower(), (
        f"{name} uses `{token}`, the independent retrieval oracle (Flag 17). Putting it inside "
        f"the method destroys its independence for every later comparison.")


@pytest.mark.parametrize("name,src", _sources())
def test_the_selector_uses_the_production_reranker(name, src):
    assert "ms-marco" in src, f"{name} does not name the reranker it selects with"


@pytest.mark.parametrize("name,src", _sources())
def test_the_probe_declares_its_gate_before_running(name, src):
    """A gate chosen after seeing the numbers is not a gate."""
    low = src.lower()
    assert "declared before running" in low or "before running" in low, \
        f"{name} does not declare its decision rule up front"
    assert "fail" in low and "pass" in low, f"{name} does not state both gate outcomes"


@pytest.mark.parametrize("name,src", _sources())
def test_the_circular_metric_is_labelled_as_such(name, src):
    """Fixed-answer faithfulness is computed against the claims the baseline already wrote."""
    assert "CIRCULAR" in src, f"{name} reports a fixed-answer metric without labelling it circular"
    assert "no BH family" in src or "not_in_any_family" in src, \
        f"{name} must state that the probe enters no BH family"
