"""Summer-phase iteration subset: n=60, stratified, deterministic (seed 42).

Design (Enzo, 2026-07-22): proportional allocation over the 9 query_type x
difficulty cells of the 194-query set (largest-remainder rounding), sampled
with seed 42 inside each cell. The 25 cross-cloud queries are NOT force-included:
cross-cloud-sensitive arms run data/evaluation/cross_cloud_subset.json as a
separate recorded arm (exp13 pattern), keeping this subset's strata clean.

Subset faithfulness is TRIAGE-grade only (attrition leaves effective n ~27-35);
retrieval metrics carry confirmatory weight at n=60. Scaling any arm to 194 q
is a per-arm Enzo decision.

Output: data/evaluation/summer_subset.json (query ids in canonical file order,
plus the strata table for the ledger).

Usage: python scripts/make_summer_subset.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "summer_subset.json"
N_TARGET = 60
SEED = 42


def main():
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if isinstance(queries, dict):
        queries = queries.get("queries", queries)
    n_total = len(queries)

    # ---- proportional allocation over query_type x difficulty (largest remainder)
    cells = {}
    for q in queries:
        cells.setdefault((q["query_type"], q["difficulty"]), []).append(q)
    quotas = {k: N_TARGET * len(v) / n_total for k, v in cells.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = N_TARGET - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[:remainder]:
        alloc[k] += 1
    assert sum(alloc.values()) == N_TARGET

    # ---- deterministic sample inside each cell
    rng = random.Random(SEED)
    chosen = []
    for k in sorted(alloc):  # sorted -> stable iteration order
        pool = sorted(cells[k], key=lambda q: q["query_id"])
        chosen += rng.sample(pool, alloc[k])
    chosen_ids = {q["query_id"] for q in chosen}

    # canonical run order = test_queries.json file order
    ordered = [q for q in queries if q["query_id"] in chosen_ids]

    strata = {
        "query_type": dict(Counter(q["query_type"] for q in ordered)),
        "difficulty": dict(Counter(q["difficulty"] for q in ordered)),
        "providers": dict(Counter("+".join(sorted(q["cloud_providers"])) for q in ordered)),
        "category": dict(Counter(q["category"] for q in ordered)),
    }
    n_cc = sum(1 for q in ordered if len(q["cloud_providers"]) > 1)

    out = {
        "purpose": "summer-phase ablation iteration subset (triage; ledger entrada 0)",
        "seed": SEED, "n": len(ordered), "n_source": n_total,
        "allocation": {f"{t}/{d}": a for (t, d), a in sorted(alloc.items())},
        "strata": strata,
        "n_cross_cloud_inside": n_cc,
        "note": "cross-cloud-sensitive arms use data/evaluation/cross_cloud_subset.json separately",
        "query_ids": [q["query_id"] for q in ordered],
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH}  n={len(ordered)}")
    print("allocation:", out["allocation"])
    print("strata:", json.dumps(strata, ensure_ascii=False))
    print(f"cross-cloud inside subset: {n_cc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
