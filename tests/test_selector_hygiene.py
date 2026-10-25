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

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Every module allowed to CHOOSE evidence for exp19. The verifier/oracle bans below apply to
# all of them, because the constraint is about what a selector may reach, not about which
# experiment happens to call it.
#
# Split added by Claude Code, 2026-08-21 15:10: exp19b's selector produces a SCORED arm, so it
# needs these bans more than exp19a's probe did. But two of the assertions here are specific to
# a probe -- "declares its gate" and "the circular metric is labelled" -- and exp19b is neither
# gated nor circular: it enters a BH family and its metric compares two real answers. Applying
# probe assertions to it would have forced false prose into the runner, which is how a guard
# starts teaching people to lie to it. The bans stayed universal; only the probe-shaped
# assertions were narrowed.
SELECTOR_SCRIPTS = ["compute_exp19a_selector_probe.py", "select_exp19b_evidence.py"]

# Selectors that are also PROBES: they exist to answer a pre-declared gate with a deliberately
# circular fixed-answer metric. Those two properties are what the last two tests check.
PROBE_SCRIPTS = ["compute_exp19a_selector_probe.py"]

# Anything that produces a faithfulness label or score. If the selector can reach one of these,
# the arm it produces cannot be scored by that verifier without circularity.
VERIFIER_TOKENS = [
    "load_hhem", "HHEM", "hhem-2.1", "vectara",
    "decide_nli_status", "nli-deberta", "nli_probs", "HallucinationDetector",
    "rescore_grounding", "rescore_nli",
]
ORACLE_TOKENS = ["bge-reranker-large", "bge_reranker_large"]


def _read(names):
    out = []
    for name in names:
        p = PROJECT_ROOT / "scripts" / name
        if p.exists():
            out.append((name, p.read_text(encoding="utf-8")))
    return out


def _sources():
    return _read(SELECTOR_SCRIPTS)


def _probe_sources():
    return _read(PROBE_SCRIPTS)


def _code_only(src):
    """Executable tokens only. Shared with the other source-level guard via conftest."""
    from conftest import code_only
    return code_only(src)


def _model_literals(src):
    """Retain executable model IDs, including constants hidden from token-only guards.

    Narrative strings mention evaluators legitimately. Match actual repository IDs,
    not prose, and exclude genuine module/class/function docstrings.
    """
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                docstrings.add(id(node.body[0].value))
    return [node.value.lower() for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings and not any(c.isspace() for c in node.value)]


@pytest.mark.parametrize("name,src", _sources())
def test_selector_model_literals_exclude_verifiers_and_oracle(name, src):
    forbidden = ("nli-deberta", "hhem-2.1", "bge-reranker-large")
    assert not any(token in literal for literal in _model_literals(src) for token in forbidden), name


def test_model_literal_guard_sees_calls_and_assigned_ids_but_not_docstrings():
    source = '\"\"\"cross-encoder/nli-deberta-v3-small\"\"\"\nMODEL = "vectara/hhem-2.1-open"\nCrossEncoder("BAAI/bge-reranker-large")'
    assert _model_literals(source) == ["vectara/hhem-2.1-open", "baai/bge-reranker-large"]


def test_the_selector_scripts_exist():
    found = {n for n, _ in _sources()}
    assert found == set(SELECTOR_SCRIPTS), (
        f"missing {sorted(set(SELECTOR_SCRIPTS) - found)} — a selector that is not on disk is a "
        f"guard that silently covers nothing; did a rename orphan it?")
    assert {n for n, _ in _probe_sources()} == set(PROBE_SCRIPTS)


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


@pytest.mark.parametrize("name,src", _probe_sources())
def test_the_probe_declares_its_gate_before_running(name, src):
    """A gate chosen after seeing the numbers is not a gate."""
    low = src.lower()
    assert "declared before running" in low or "before running" in low, \
        f"{name} does not declare its decision rule up front"
    assert "fail" in low and "pass" in low, f"{name} does not state both gate outcomes"


@pytest.mark.parametrize("name,src", _probe_sources())
def test_the_circular_metric_is_labelled_as_such(name, src):
    """Fixed-answer faithfulness is computed against the claims the baseline already wrote."""
    assert "CIRCULAR" in src, f"{name} reports a fixed-answer metric without labelling it circular"
    assert "no BH family" in src or "not_in_any_family" in src, \
        f"{name} must state that the probe enters no BH family"


# ------------------------------------------------------- the selector, on known-answer cases
# `select_by_claims` returned an EMPTY selection on its first smoke run: `best_so_far` started
# at -inf, so the gain of every chunk was +inf, `np.isfinite` rejected it, and the loop broke
# before choosing anything. Recall came out as exactly 0.0 and the gate read FAIL -- which would
# have killed a valid experiment on a bug. Implausible-but-directional output is the dangerous
# kind, so the selector now has cases whose right answer is known by construction.
@pytest.fixture(scope="module")
def probe():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_t", PROJECT_ROOT / "scripts" / "compute_exp19a_selector_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selector_returns_exactly_k_chunks(probe):
    import numpy as np
    R = np.random.default_rng(0).random((7, 20))
    assert len(probe.select_by_claims(R, 5)) == 5
    assert len(set(probe.select_by_claims(R, 5))) == 5, "no chunk may be picked twice"


def test_selector_never_returns_empty(probe):
    """The bug. Any finite score matrix must yield a selection."""
    import numpy as np
    for R in (np.zeros((3, 10)), np.full((3, 10), -5.0), np.ones((1, 2))):
        assert probe.select_by_claims(R, 5), f"empty selection for shape {R.shape}"


def test_selector_has_diminishing_returns_on_an_already_served_claim(probe):
    """The property that separates claim-level selection from score-ranking, known by hand.

    Claim A is served well by chunks 0 and 1; claim B only by chunk 2, and less strongly. Ranking
    chunks by their score (9 > 8 > 7 — identical to ranking by mean or by sum over claims) takes
    {0, 1} and leaves claim B with nothing. A selector that serves each claim by its own best
    chunk must take {0, 2}, because once claim A is served, chunk 1 adds nothing.

    At k=1 the two rules cannot differ (argmax of a sum is argmax of a mean), so the case needs
    k=2 to be a real test.
    """
    import numpy as np
    R = np.array([[9.0, 8.0, 0.0],      # claim A: chunks 0 and 1
                  [0.0, 0.0, 7.0]])     # claim B: chunk 2 only
    assert set(probe.select_by_claims(R, 2)) == {0, 2}
    assert list(np.argsort(-R.sum(axis=0))[:2]) == [0, 1], \
        "the case must actually distinguish the two rules"


def test_selector_picks_the_single_useful_chunk_first(probe):
    import numpy as np
    R = np.array([[0.1, 0.1, 8.0, 0.1]])
    assert probe.select_by_claims(R, 1) == [2]


def test_fixed_answer_faithfulness_matches_hand_count(probe):
    import numpy as np
    sup = np.array([[True, False, False],
                    [False, False, False],
                    [False, True, False]])
    assert probe.faith_of(sup, [0]) == pytest.approx(1 / 3)
    assert probe.faith_of(sup, [0, 1]) == pytest.approx(2 / 3)
    assert probe.faith_of(sup, [2]) == 0.0
    assert probe.faith_of(sup, []) == 0.0
