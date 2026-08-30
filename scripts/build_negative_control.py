"""Tier 3 · Block B — negative-control pair set (false-contradicted base rate).

Reproduces the h2_variant_eval.json negative control that selected vb_agree
(random-chunk false-contradicted rate 0.62 -> 0.175). A verifier should almost
never label a claim CONTRADICTED (NLI) / GROUNDED (HHEM) against a RANDOM,
topically-unrelated chunk; the rate at which it does is a construct-validity
error rate, independent of any downstream scenario contrast — the anti-p-hacking
selection criterion for Block B.

Builds a FIXED, reproducible set of (claim, [K random chunks]) tuples (seed 42):
for N sampled genuine claims, draw K=5 chunks uniformly from the corpus
EXCLUDING that claim's own retrieved chunk_ids — mirroring the real 5-chunk
scoring so the multi-chunk decision rules (vb_agree's >=2-chunk guard, noisy_or)
behave as in production. Persists claim text + the K chunk ids/text so scoring
is reproducible and auditable. No GPU.

Output: experiments/results/exp15_ablation_nli/negative_control_pairs.json

Usage: python scripts/build_negative_control.py [--n 400]
Env: PYTHONHASHSEED=42
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results/exp15_ablation_nli"
CHUNK_MAP = ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
SEED = 42


def genuine(meta):
    return [c for c, a in zip(meta["claims"], meta["artifact"]) if not a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400,
                    help="number of claims (h2 used 200; 400 tightens the "
                         "base-rate CI to ~+-0.05)")
    ap.add_argument("--k", type=int, default=5,
                    help="random chunks per claim (mirror the real 5-chunk scoring)")
    args = ap.parse_args()
    rng = random.Random(SEED)

    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    all_cids = sorted(chunk_map.keys())  # sorted -> deterministic

    # flat pool of (cfg, qid, claim_idx, claim_text, own_chunk_ids)
    pool = []
    for cfg in sorted(claims):
        for qid in sorted(claims[cfg]):
            meta = claims[cfg][qid]
            gc = genuine(meta)
            own = set(meta["chunk_ids"])
            for i, claim in enumerate(gc):
                pool.append((cfg, qid, i, claim, own))
    rng.shuffle(pool)
    picked = pool[: args.n]

    pairs = []
    for cfg, qid, i, claim, own in picked:
        # K distinct random chunks, none among the claim's own retrieved chunks
        cids = []
        while len(cids) < args.k:
            cid = all_cids[rng.randrange(len(all_cids))]
            if cid not in own and cid not in cids:
                cids.append(cid)
        pairs.append({
            "config": cfg, "query_id": qid, "claim_idx": i, "claim": claim,
            "random_chunk_ids": cids,
            "random_chunk_texts": [chunk_map[c].get("text", "") for c in cids],
        })

    out = {"purpose": "negative control: false-contradicted / false-grounded base rate on random pairs",
           "method": "h2_variant_eval.json reproduction; lower rate = better construct validity",
           "seed": SEED, "n": len(pairs),
           "generated_by": "scripts/build_negative_control.py", "pairs": pairs}
    (OUT / "negative_control_pairs.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote negative_control_pairs.json  n={len(pairs)}")


if __name__ == "__main__":
    main()
