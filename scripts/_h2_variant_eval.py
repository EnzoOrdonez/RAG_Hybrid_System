"""
H2 NLI-guard variant evaluation + synthetic base-rate (ledger N8, sub-gate 2b).

RUNS ON ENZO'S MACHINE (needs torch + sentence-transformers + the NLI model).
NOT executed in the authoring environment (no scientific stack there).

Purpose: produce the trade-off Enzo needs to PICK the H2 contradiction-guard
variant before the definitive v3 re-score. Nothing is written to signed
evidence; output is a console table + a scratch JSON under output/audit/.

Two measurements, all over responses ALREADY SAVED in exp12_matrix:

  (A) Synthetic base-rate. For N sampled genuine claims, score each against 5
      RANDOM (unrelated) chunks. A random chunk is neutral wrt the claim, so
      any "contradicted" verdict is a false positive produced by taking max
      over 5 chances at the 0.7 threshold (the H2 mechanism). Report the
      false-contradiction rate under v0 vs va_margin(δ) vs vb_agree.

  (B) Real-sample re-score. Over the 50-claim audit-sample responses + q085 +
      a stratified random sample, re-extract claims with the CORRECTED
      extractor (format artifacts tagged not_a_claim and excluded), score the
      genuine claims against the response's real retrieved chunks, and apply
      each variant. Report contradicted/supported/unsupported %, the q085
      breakdown, and how many v0-"contradicted" claims flip under each variant.

Decision rule logic is imported from the runtime module so it is identical to
what the v3 re-score will use.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Usage:
  python scripts/_h2_variant_eval.py [--verifier base|small] [--n-sample 200] [--n-baserate 200]
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector,
    classify_artifact,
    decide_nli_status,
)

SEED = 42
MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
EXP12 = PROJECT_ROOT / "experiments/results/exp12_matrix/results.json"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
SAMPLE_CSV = PROJECT_ROOT / "output/audit/claim_audit_sample.csv"
BASE_LOCAL = PROJECT_ROOT / "data/models/nli-deberta-v3-base"

# Variants to compare. (label, variant, margin)
VARIANTS = [
    ("v0 (legacy)", "v0", 0.0),
    ("va_margin d=0.05", "va_margin", 0.05),
    ("va_margin d=0.10", "va_margin", 0.10),
    ("vb_agree (>=2 chunks)", "vb_agree", 0.0),
]


def load_model(verifier):
    from sentence_transformers import CrossEncoder
    import torch
    name = (str(BASE_LOCAL) if verifier == "base" and BASE_LOCAL.exists()
            else "cross-encoder/nli-deberta-v3-base" if verifier == "base"
            else "cross-encoder/nli-deberta-v3-small")
    m = CrossEncoder(name, max_length=512)
    if torch.cuda.is_available():
        m.model.half()  # fp16, documented (N5)
    return m, name


def matrices_for_claim(model, claim, chunk_texts):
    """Return (contr_scores, ent_scores) over the chunks for one claim."""
    pairs = [(t, claim) for t in chunk_texts]
    preds = model.predict(pairs, batch_size=64, show_progress_bar=False, apply_softmax=True)
    contr = [float(p[0]) for p in preds]
    ent = [float(p[1]) for p in preds]
    return contr, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="base", choices=["base", "small"])
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--n-baserate", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(SEED)

    results = json.loads(EXP12.read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    all_chunk_ids = list(chunk_map.keys())
    det = HallucinationDetector(use_nli=False)  # extractor only
    model, model_name = load_model(args.verifier)
    print(f"verifier: {model_name}")

    # ---- pinned sample set: audit-sample responses + q085 + stratified random ----
    pinned = set()
    if SAMPLE_CSV.exists():
        import csv
        with open(SAMPLE_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                pinned.add((r["config"], r["query_id"]))
    # stratified random responses (nli method) across the 12 RAG configs
    strat = []
    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            rows = [r for r in results.get(cname, {}).get("results", [])
                    if (r.get("hallucination_metrics") or {}).get("method") == "nli"]
            rng.shuffle(rows)
            for r in rows[:max(1, args.n_sample // 12)]:
                strat.append((cname, r["query_id"]))
    sample_keys = list(pinned) + [k for k in strat if k not in pinned]

    # index responses for lookup
    by_key = {}
    for cname, c in results.items():
        for r in c["results"]:
            by_key[(cname, r["query_id"])] = r

    # ---- (B) real-sample re-score under each variant ----
    per_variant = {lbl: Counter() for lbl, _, _ in VARIANTS}
    flips = {lbl: 0 for lbl, _, _ in VARIANTS}
    q085 = {lbl: Counter() for lbl, _, _ in VARIANTS}
    n_claims = n_artifacts = 0
    t0 = time.time()
    for i, key in enumerate(sample_keys):
        r = by_key.get(key)
        if not r:
            continue
        chunks = [chunk_map[cid] for cid in r.get("retrieved_ids", []) if cid in chunk_map]
        texts = [c["text"] for c in chunks]
        if not texts:
            continue
        for claim in det._extract_claims(r.get("answer") or ""):
            if classify_artifact(claim):
                n_artifacts += 1
                continue
            n_claims += 1
            contr, ent = matrices_for_claim(model, claim, texts)
            v0_status, _, _ = decide_nli_status(contr, ent)
            for lbl, variant, margin in VARIANTS:
                st, _, _ = decide_nli_status(contr, ent, variant=variant, margin=margin)
                per_variant[lbl][st] += 1
                if v0_status == "contradicted" and st != "contradicted":
                    flips[lbl] += 1
                if key == ("hibrido | granite4.1-8b", "q085"):
                    q085[lbl][st] += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sample_keys)}] {n_claims} claims ({time.time()-t0:.0f}s)")

    # ---- (A) synthetic base-rate: genuine claims vs 5 RANDOM chunks ----
    base = {lbl: 0 for lbl, _, _ in VARIANTS}
    base_n = 0
    pool_claims = []
    for key in sample_keys:
        r = by_key.get(key)
        if not r:
            continue
        for claim in det._extract_claims(r.get("answer") or ""):
            if not classify_artifact(claim):
                pool_claims.append(claim)
    rng.shuffle(pool_claims)
    for claim in pool_claims[:args.n_baserate]:
        rand_texts = [chunk_map[cid]["text"] for cid in rng.sample(all_chunk_ids, 5)]
        contr, ent = matrices_for_claim(model, claim, rand_texts)
        base_n += 1
        for lbl, variant, margin in VARIANTS:
            st, _, _ = decide_nli_status(contr, ent, variant=variant, margin=margin)
            if st == "contradicted":
                base[lbl] += 1

    # ---- report ----
    out = {"verifier": model_name, "n_claims_sample": n_claims,
           "n_artifacts_excluded": n_artifacts, "n_baserate": base_n,
           "variants": {}, "q085": {}, "base_rate_false_contradicted": {}}
    print("\n" + "=" * 80)
    print(f"H2 VARIANT TRADE-OFF  (verifier={args.verifier}, {n_claims} genuine claims, "
          f"{n_artifacts} artifacts excluded)")
    print("=" * 80)
    print(f"{'variant':<24}{'contr%':>8}{'supp%':>8}{'unsup%':>8}{'flips(v0->)':>12}")
    for lbl, _, _ in VARIANTS:
        c = per_variant[lbl]
        tot = sum(c.values()) or 1
        print(f"{lbl:<24}{100*c['contradicted']/tot:>7.1f}%{100*c['supported']/tot:>7.1f}%"
              f"{100*c['unsupported']/tot:>7.1f}%{flips[lbl]:>12}")
        out["variants"][lbl] = {k: c[k] for k in ("contradicted", "supported", "unsupported")}
        out["variants"][lbl]["flips_from_v0_contradicted"] = flips[lbl]
    print(f"\nq085 (28 procedural claims) by variant:")
    for lbl, _, _ in VARIANTS:
        print(f"  {lbl:<24} {dict(q085[lbl])}")
        out["q085"][lbl] = dict(q085[lbl])
    print(f"\nSYNTHETIC BASE-RATE -- false 'contradicted' vs 5 RANDOM chunks "
          f"(n={base_n}; lower is better):")
    for lbl, _, _ in VARIANTS:
        print(f"  {lbl:<24} {100*base[lbl]/(base_n or 1):>5.1f}%")
        out["base_rate_false_contradicted"][lbl] = base[lbl] / (base_n or 1)

    out_path = PROJECT_ROOT / "output/audit/h2_variant_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nNEXT: pick a variant, then run rescore_nli_v3.py --variant <v> [--margin d].")


if __name__ == "__main__":
    main()
