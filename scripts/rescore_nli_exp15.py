"""exp15 Tier 0 — NLI instrument ablation: score once, persist RAW probabilities.

Parameterized descendant of rescore_nli_v3.py (which hardcodes exp12_matrix as
its OUTPUT dir — signed evidence; this script only READS exp12 and writes to
experiments/results/exp15_ablation_nli/). Two additions:

  1. Persists the raw per-(config, query, claim, chunk) softmax probabilities
     (contradiction, entailment, neutral) gzipped, so the whole
     variant x threshold sweep becomes pure-CPU re-aggregation — the GPU pays
     once per verifier.
  2. Emits, from those same probs, the vb_agree@0.7 aggregation in the exact
     rescore_v3 row format, as a cross-check: it must match the signed
     faithfulness_rescore_v3__<verifier>__vb_agree.json row-for-row (same
     extractor, same pooling order, same fp16 batching), validating that the
     scoring path reproduces before any new operating point is trusted.

Outputs (exp15_ablation_nli/):
  claims_extraction.json            claim texts + artifact flags + chunk ids
                                    (verifier-independent; written once)
  nli_probs__<verifier>.json.gz     raw probs [claim][chunk][contr, ent, neut]
  rescore_check__<verifier>__vb_agree.json   v3-format rows for the cross-check

Usage:
  python scripts/rescore_nli_exp15.py --verifier small [--max-queries 2]
  python scripts/rescore_nli_exp15.py --verifier base
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector,
    classify_artifact,
    decide_nli_status,
)

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
ENT_T = 0.7
CONTR_T = 0.7
BASE_LOCAL = PROJECT_ROOT / "data" / "models" / "nli-deberta-v3-base"
SMALL_LOCAL = PROJECT_ROOT / "data" / "models" / "nli-deberta-v3-small"
EXP12_DIR = PROJECT_ROOT / "experiments/results/exp12_matrix"      # READ-ONLY
OUT_DIR = PROJECT_ROOT / "experiments/results/exp15_ablation_nli"  # writes here


def model_path(verifier):
    if verifier == "base":
        return str(BASE_LOCAL) if BASE_LOCAL.exists() else "cross-encoder/nli-deberta-v3-base"
    return str(SMALL_LOCAL) if SMALL_LOCAL.exists() else "cross-encoder/nli-deberta-v3-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="small", choices=["base", "small"])
    ap.add_argument("--max-queries", type=int, default=None,
                    help="smoke mode: limit eligible responses per config; outputs "
                         "get a __smokeN suffix and never touch the full-run files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__smoke{args.max_queries}" if args.max_queries else ""
    probs_path = OUT_DIR / f"nli_probs__{args.verifier}{suffix}.json.gz"
    part_path = OUT_DIR / f"nli_probs__{args.verifier}{suffix}.partial.json.gz"
    claims_path = OUT_DIR / f"claims_extraction{suffix}.json"
    check_path = OUT_DIR / f"rescore_check__{args.verifier}__vb_agree{suffix}.json"

    from sentence_transformers import CrossEncoder
    import torch
    name = model_path(args.verifier)
    model = CrossEncoder(name, max_length=512)
    if torch.cuda.is_available():
        model.model.half()  # fp16, mirrors rescore_nli_v3 (N5)
    det = HallucinationDetector(use_nli=False)  # extractor only

    results = json.loads((EXP12_DIR / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(
        (PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json").read_text(encoding="utf-8"))

    probs_out = {"verifier": name, "verifier_tag": args.verifier,
                 "classes": ["contradiction", "entailment", "neutral"],
                 "source": "experiments/results/exp12_matrix/results.json (read-only)",
                 "pooling": "identical to rescore_nli_v3 (per-config predict, batch 64, fp16)",
                 "generated_by": "scripts/rescore_nli_exp15.py", "configs": {}}
    claims_out = {"generated_by": "scripts/rescore_nli_exp15.py",
                  "extractor": "HallucinationDetector._extract_claims + classify_artifact",
                  "configs": {}}
    check_out = {"verifier": name, "variant": "vb_agree", "margin": 0.0,
                 "purpose": "cross-check vs signed faithfulness_rescore_v3__*__vb_agree.json",
                 "generated_by": "scripts/rescore_nli_exp15.py", "configs": {}}
    if part_path.exists():
        with gzip.open(part_path, "rt", encoding="utf-8") as f:
            prev = json.load(f)
        probs_out["configs"] = prev.get("configs", {})
        print(f"resuming, {len(probs_out['configs'])} configs done", flush=True)
    if claims_path.exists():
        claims_out = json.loads(claims_path.read_text(encoding="utf-8"))
    t0 = time.time()

    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            done = cname in probs_out["configs"] and cname in claims_out["configs"]
            rows = results[cname]["results"]
            if args.max_queries:
                eligible = [r for r in rows
                            if (r.get("hallucination_metrics") or {}).get("method") == "nli"
                            and ((r.get("hallucination_metrics") or {}).get("total_claims") or 0)]
                rows = eligible[:args.max_queries]
            if done:
                continue
            cfg_claims, cfg_probs, cfg_check = {}, {}, {}
            pairs, spans = [], []  # spans: (qid, genuine, k_chunks, n_artifacts, start)
            for r in rows:
                hm = r.get("hallucination_metrics") or {}
                if hm.get("method") != "nli" or not (hm.get("total_claims") or 0):
                    continue
                claims = det._extract_claims(r["answer"])
                artifact = [bool(classify_artifact(c)) for c in claims]
                genuine = [c for c, a in zip(claims, artifact) if not a]
                n_art = len(claims) - len(genuine)
                cids = [cid for cid in r["retrieved_ids"] if cid in chunk_map]
                texts = [chunk_map[cid]["text"] for cid in cids]
                if not texts:
                    continue
                cfg_claims[r["query_id"]] = {"claims": claims, "artifact": artifact,
                                             "chunk_ids": cids}
                if not genuine:
                    cfg_probs[r["query_id"]] = []  # vacuous: no scorable claim
                    cfg_check[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                                "genuine": 0, "supported": 0, "contradicted": 0,
                                                "unsupported": 0, "faithfulness": 1.0}
                    continue
                spans.append((r["query_id"], genuine, len(texts), n_art, len(pairs)))
                pairs.extend((t, cl) for cl in genuine for t in texts)
            preds = (model.predict(pairs, batch_size=64, show_progress_bar=False,
                                   apply_softmax=True) if pairs else [])
            for qid, genuine, k, n_art, start in spans:
                agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
                q_probs = []
                for ci in range(len(genuine)):
                    rows_p = preds[start + ci * k: start + (ci + 1) * k]
                    q_probs.append([[float(p[0]), float(p[1]), float(p[2])] for p in rows_p])
                    contr = [float(p[0]) for p in rows_p]
                    ent = [float(p[1]) for p in rows_p]
                    st, _, _ = decide_nli_status(contr, ent, ENT_T, CONTR_T,
                                                 variant="vb_agree", margin=0.0)
                    agg[st] += 1
                g = len(genuine)
                cfg_probs[qid] = q_probs
                cfg_check[qid] = {"total_claims": g + n_art, "not_a_claim": n_art, "genuine": g,
                                  **agg, "faithfulness": round(agg["supported"] / g, 4)}
            probs_out["configs"][cname] = cfg_probs
            claims_out["configs"][cname] = cfg_claims
            check_out["configs"][cname] = cfg_check
            with gzip.open(part_path, "wt", encoding="utf-8") as f:
                json.dump(probs_out, f)
            claims_path.write_text(json.dumps(claims_out, indent=1), encoding="utf-8")
            print(f"  {cname}: {len(cfg_check)} responses, {len(pairs)} pairs "
                  f"({time.time()-t0:.0f}s)", flush=True)

    probs_out["n_responses"] = sum(len(v) for v in probs_out["configs"].values())
    with gzip.open(probs_path, "wt", encoding="utf-8") as f:
        json.dump(probs_out, f)
    part_path.unlink(missing_ok=True)
    check_out["n_responses"] = sum(len(v) for v in check_out["configs"].values())
    check_path.write_text(json.dumps(check_out, indent=1), encoding="utf-8")
    print(f"wrote {probs_path}\nwrote {check_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
