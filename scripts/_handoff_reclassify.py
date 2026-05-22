"""
Ad-hoc reclassification of exp9_llm_only_no_rag responses into three
categories per external audit Block 1. DOES NOT mutate production code.

Classifies each of the 200 saved responses as:
  A) FABRICATION       : non-decline, total_claims>0; faithfulness=0.0 valid
  B) HONEST_DECLINE    : text matches any of 14 lowercased decline patterns
  C) EXTRACTOR_FAILURE : total_claims==0 AND no decline pattern match
                         (e.g. q196 — list-format response, extractor
                         returned [] because every sentence was filtered
                         by SKIP_PATTERNS)

Outputs experiments/results/exp9_llm_only_no_rag/_handoff/reclassification.json.
"""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "experiments" / "results" / "exp9_llm_only_no_rag" / "results.json"
HANDOFF_DIR = ROOT / "experiments" / "results" / "exp9_llm_only_no_rag" / "_handoff"
OUT_PATH = HANDOFF_DIR / "reclassification.json"

DECLINE_PATTERNS = [
    r"cannot find sufficient information",
    r"i cannot find enough information",
    r"not enough information in the.*context",
    r"the.*context does not.*contain",
    r"no relevant.*information.*found",
    r"i don't have.*information",
    r"based on the available documentation.*cannot",
    r"the provided context does not",
    r"i'm not familiar with",
    r"unfortunately, i don't",
    r"my knowledge.*is limited",
    r"i'm unable to (provide|answer)",
    r"i don't have access to",
    r"i don't have specific information",
]


def matches_any_decline(text_lower: str) -> str:
    for p in DECLINE_PATTERNS:
        if re.search(p, text_lower):
            return p
    return ""


def main() -> int:
    if not RESULTS_PATH.exists():
        print(f"results.json missing: {RESULTS_PATH}")
        return 1
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    qrs = data["configs"]["LLM Only (No RAG)"]["results"]

    fabrication_ids = []
    honest_decline_ids = []
    extractor_failure_ids = []
    honest_decline_matches = {}
    faith_after_excluding_extractor_failures = []

    for q in qrs:
        qid = q["query_id"]
        answer = q.get("answer", "") or ""
        text_lower = answer.lower()
        total_claims = q.get("hallucination_metrics", {}).get("total_claims", 0)
        faith = q.get("hallucination_metrics", {}).get("faithfulness", 0.0)

        decline_pattern = matches_any_decline(text_lower)
        if decline_pattern:
            honest_decline_ids.append(qid)
            honest_decline_matches[qid] = decline_pattern
            faith_after_excluding_extractor_failures.append(faith)
        elif total_claims > 0:
            fabrication_ids.append(qid)
            faith_after_excluding_extractor_failures.append(faith)
        else:
            extractor_failure_ids.append(qid)

    out = {
        "fabrication_count": len(fabrication_ids),
        "honest_decline_count": len(honest_decline_ids),
        "extractor_failure_count": len(extractor_failure_ids),
        "fabrication_ids": fabrication_ids,
        "honest_decline_ids": honest_decline_ids,
        "extractor_failure_ids": extractor_failure_ids,
        "honest_decline_matches": honest_decline_matches,
        "honest_decline_rate_corrected": round(
            len(honest_decline_ids) / len(qrs), 6
        ),
        "faithfulness_mean_after_excluding_extractor_failures": (
            round(statistics.mean(faith_after_excluding_extractor_failures), 6)
            if faith_after_excluding_extractor_failures
            else None
        ),
        "n_total": len(qrs),
        "patterns_applied": DECLINE_PATTERNS,
    }

    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list) or k.endswith("_ids")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
