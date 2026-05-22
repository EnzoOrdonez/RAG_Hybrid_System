"""
Phase 1 no-regression tests for the DECLINE_PATTERNS + _extract_claims
fixes. Reads the saved exp9_llm_only_no_rag/results.json and re-applies
the production code (post-fix) over each saved response.

Test 1: _is_honest_decline over 200 responses → expect 4 matches
        (q081, q097, q106, q117) with the matching pattern reported.

Test 2: _extract_claims over q196's saved answer → expect >0 claims;
        print each claim literally.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.response_formatter import DECLINE_PATTERNS, ResponseFormatter  # noqa: E402
from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402

RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "exp9_llm_only_no_rag"
    / "results.json"
)


def main() -> int:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    qrs = data["configs"]["LLM Only (No RAG)"]["results"]
    formatter = ResponseFormatter()
    detector = HallucinationDetector()

    # ---- Test 1: _is_honest_decline over 200 saved responses ----
    matches = []
    for q in qrs:
        answer = q.get("answer", "") or ""
        if formatter._is_honest_decline(answer):
            text_lower = answer.lower()
            matched_pattern = next(
                (p for p in DECLINE_PATTERNS if re.search(p, text_lower)),
                None,
            )
            matches.append((q["query_id"], matched_pattern))

    print("=" * 60)
    print(f"TEST 1: _is_honest_decline over {len(qrs)} saved responses")
    print(f"  Matches: {len(matches)}")
    for qid, pattern in matches:
        print(f"    {qid}  <-  /{pattern}/")
    expected = {"q081", "q097", "q106", "q117"}
    got = {qid for qid, _ in matches}
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got:      {sorted(got)}")
    print(f"  Test 1: {'PASS' if got == expected else 'FAIL'}")

    # ---- Test 2: _extract_claims over q196 ----
    q196 = next(q for q in qrs if q["query_id"] == "q196")
    answer = q196["answer"]
    print()
    print("=" * 60)
    print("TEST 2: _extract_claims over q196 saved answer")
    print(f"  Answer length: {len(answer)} chars")
    print(f"  Answer first 200 chars:")
    print("    " + repr(answer[:200]))
    claims = detector._extract_claims(answer)
    print(f"  Claims extracted: {len(claims)}")
    for i, c in enumerate(claims, 1):
        print(f"    [{i}] {c!r}")
    print(f"  Test 2: {'PASS' if len(claims) > 0 else 'FAIL'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
