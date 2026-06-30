"""
v3 faithfulness re-score (ledger N8) — corrected extraction + H2 guard variant.

RUNS ON ENZO'S MACHINE (torch + sentence-transformers + NLI model). Mirror of
rescore_nli_v2.py; never regenerates LLM responses and never overwrites signed
evidence — writes NEW files only.

Two corrections relative to v2 (audit 2026-06-30):
  H1 — format artifacts (classify_artifact: ATX headers, table rows, unbalanced
       ** spans, doc-coverage meta) are tagged not_a_claim and EXCLUDED from the
       faithfulness denominator (supported / genuine_claims).
  H2 — the contradiction decision uses the chosen guard variant via the shared
       decide_nli_status() (so it is byte-identical to the runtime fix once the
       2b sub-gate flips the runtime default).

Run both verifiers for the N5-style robustness check:
  python scripts/rescore_nli_v3.py --verifier base  --variant <v> [--margin d]
  python scripts/rescore_nli_v3.py --verifier small --variant <v> [--margin d]

Output: experiments/results/exp12_matrix/faithfulness_rescore_v3__<verifier>__<variant>.json
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
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
EXP_DIR = PROJECT_ROOT / "experiments/results/exp12_matrix"


def model_path(verifier):
    if verifier == "base":
        return str(BASE_LOCAL) if BASE_LOCAL.exists() else "cross-encoder/nli-deberta-v3-base"
    return "cross-encoder/nli-deberta-v3-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="base", choices=["base", "small"])
    ap.add_argument("--variant", default="v0", choices=["v0", "va_margin", "vb_agree"])
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()
    tag = f"{args.verifier}__{args.variant}" + (f"_d{args.margin}" if args.variant == "va_margin" else "")
    out_path = EXP_DIR / f"faithfulness_rescore_v3__{tag}.json"
    part_path = out_path.with_suffix(".partial.json")

    from sentence_transformers import CrossEncoder
    import torch
    name = model_path(args.verifier)
    model = CrossEncoder(name, max_length=512)
    if torch.cuda.is_available():
        model.model.half()  # fp16, documented (N5)
    det = HallucinationDetector(use_nli=False)  # extractor only

    results = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(
        (PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json").read_text(encoding="utf-8"))

    out = {"verifier": name, "variant": args.variant, "margin": args.margin,
           "procedure": "corrected extraction (not_a_claim excluded) + decide_nli_status guard; "
                        "softmax, thresholds 0.7, max over 5 chunks",
           "generated_by": "scripts/rescore_nli_v3.py", "configs": {}}
    if part_path.exists():
        out["configs"] = json.loads(part_path.read_text(encoding="utf-8")).get("configs", {})
        print(f"resuming, {len(out['configs'])} configs done", flush=True)
    t0 = time.time()

    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            if cname in out["configs"]:
                continue
            rows_out = {}
            # Pool every (chunk, genuine-claim) pair of the config into one predict().
            pairs, spans = [], []  # spans: (qid, genuine_claims, k_chunks, n_artifacts, start)
            for r in results[cname]["results"]:
                hm = r.get("hallucination_metrics") or {}
                if hm.get("method") != "nli" or not (hm.get("total_claims") or 0):
                    continue
                claims = det._extract_claims(r["answer"])
                genuine = [c for c in claims if not classify_artifact(c)]
                n_art = len(claims) - len(genuine)
                texts = [chunk_map[cid]["text"] for cid in r["retrieved_ids"] if cid in chunk_map]
                if not texts:
                    continue
                if not genuine:
                    # all claims were artifacts -> vacuous (no verifiable claim)
                    rows_out[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                               "genuine": 0, "supported": 0, "contradicted": 0,
                                               "unsupported": 0, "faithfulness": 1.0}
                    continue
                spans.append((r["query_id"], genuine, len(texts), n_art, len(pairs)))
                pairs.extend((t, cl) for cl in genuine for t in texts)
            if pairs:
                preds = model.predict(pairs, batch_size=64, show_progress_bar=False, apply_softmax=True)
            for qid, genuine, k, n_art, start in spans:
                agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
                for ci in range(len(genuine)):
                    rows = preds[start + ci * k: start + (ci + 1) * k]
                    contr = [float(p[0]) for p in rows]
                    ent = [float(p[1]) for p in rows]
                    st, _, _ = decide_nli_status(contr, ent, ENT_T, CONTR_T,
                                                 variant=args.variant, margin=args.margin)
                    agg[st] += 1
                g = len(genuine)
                rows_out[qid] = {"total_claims": g + n_art, "not_a_claim": n_art, "genuine": g,
                                 **agg, "faithfulness": round(agg["supported"] / g, 4)}
            out["configs"][cname] = rows_out
            part_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"  {cname}: {len(rows_out)} responses, {len(pairs)} pairs "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out["n_responses"] = sum(len(v) for v in out["configs"].values())
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    part_path.unlink(missing_ok=True)
    print(f"wrote {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
