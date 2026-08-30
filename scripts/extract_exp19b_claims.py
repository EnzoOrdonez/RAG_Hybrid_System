"""exp19b — claims of the DRAFT, the only thing the selector is allowed to condition on.

Seccion de Claude Code — 2026-08-21 14:30 (hora local).

A separate script for a reason that is structural, not cosmetic. The claim extractor lives in
`src/generation/hallucination_detector.py`, and `tests/test_selector_hygiene.py` forbids the
SELECTOR from reaching anything in that module: if the component that PICKS the evidence can
also reach the component that JUDGES it, the arm measures its own preferences. Splitting the
extraction out means `select_exp19b_evidence.py` is literally unable to import a verifier,
instead of merely promising not to.

What this script does is a text split. It produces no score, no label and no probability: the
extractor is the same deterministic segmenter that Pass N runs before any model is loaded
(`HallucinationDetector(use_nli=False)` loads nothing). What it emits is the draft's claims,
each flagged as a format artifact or not, using the module-level `classify_artifact` rule --
identical to run_exp15_ablation.py::pass_n, so the claims the selector conditions on are
exactly the claims the scorers will later count.

ONLY GENUINE CLAIMS CONDITION THE SELECTOR. Markdown headers, table rows and meta-coverage
sentences are layout, not assertions; steering a pool re-rank with them would optimise the
context for the shape of the answer instead of its content.

Usage: python scripts/extract_exp19b_claims.py [--smoke] [--stage-dir DIR]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes <exp dir>/draft_claims.json
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector, classify_artifact)

EXP_ID = "exp19b_anchored_selector"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
DRAFT_ARM = "baseline_repro"
MODEL_LABEL = "granite4.1-8b"

_DET = None


def _detector():
    """use_nli=False: the segmenter only. No verifier weights are loaded by this script."""
    global _DET
    if _DET is None:
        _DET = HallucinationDetector(use_nli=False)
    return _DET


def split_claims(answer):
    """{claims, artifact, genuine} for one answer — the Pass N rule, not a second copy of it.

    `artifact` is parallel to `claims` so the count of dropped items stays visible: an artifact
    that silently disappears is a claim the reader never learns was there.
    """
    text = answer or ""
    if not text.strip():
        return {"claims": [], "artifact": [], "genuine": []}
    claims = _detector()._extract_claims(text)
    artifact = [bool(classify_artifact(c)) for c in claims]
    genuine = [c for c, a in zip(claims, artifact) if not a]
    return {"claims": claims, "artifact": artifact, "genuine": genuine}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp-dir", default=None,
                    help="override the experiment dir (default: the real one, or _smoke)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else (
        EXP_DIR / "_smoke" if args.smoke else EXP_DIR)
    cpath = exp_dir / f"checkpoint__{MODEL_LABEL}__{DRAFT_ARM}.json"
    if not cpath.exists():
        sys.exit(f"{cpath} missing — run run_exp19b_generation.py --stage draft first")

    results = json.loads(cpath.read_text(encoding="utf-8"))["results"]
    per_query, empty = {}, []
    for r in results:
        out = split_claims(r.get("answer"))
        per_query[r["query_id"]] = out
        if not out["genuine"]:
            empty.append(r["query_id"])

    doc = {
        "experiment_id": EXP_ID, "source_arm": DRAFT_ARM,
        "extractor": "HallucinationDetector._extract_claims + classify_artifact (Pass N rule)",
        "n_queries": len(per_query),
        "n_without_genuine_claims": len(empty), "qids_without_genuine_claims": empty,
        "fallback_rule": ("a draft with no genuine claim gives the selector nothing to "
                          "condition on, so that query keeps the baseline top-5 and "
                          "contributes an exact zero difference; counted as n_fallback"),
        "generated_by": "scripts/extract_exp19b_claims.py", "per_query": per_query,
    }
    (exp_dir / "draft_claims.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    tot = sum(len(v["genuine"]) for v in per_query.values())
    print(f"wrote {exp_dir / 'draft_claims.json'}: {len(per_query)} queries, {tot} genuine "
          f"claims, {len(empty)} queries without any")


if __name__ == "__main__":
    main()
