"""Survey config (k=5 vs k=10) measured on the DEPLOYMENT path, not on frozen contexts.

Why this exists. exp18's `final_top_k_10` arm is the cleanest evidence we have that more chunks
buy coverage and not grounding, but it does NOT transfer to the survey config unchanged:

  - exp18 generated from the FROZEN exp18 pool via run_generation_matrix; `SURVEY_DEPLOY`
    retrieves LIVE, so retrieval time is absent from exp18's numbers entirely;
  - `SURVEY_DEPLOY` carries `balance_cross_cloud_providers=True`, which exp18's arm did not;
  - exp18 measures TOTAL generation time, but the UI streams (`chat_page.py` ->
    `query_stream`), so what a survey participant actually experiences is TIME TO FIRST TOKEN.
    A +30 s change in total is a very different UX claim from a +30 s change in TTFT.

So this script measures, on the real deployment object, per config and per query:
  retrieval_ms · ttft_ms · total_ms · answer chars/words · n retrieved

It does NOT score faithfulness. That stays with the canonical offline scorers on the exp18
artifacts, so the instrument never changes underneath a comparison.

Queries are drawn from the exp18 set with a DECLARED, seeded stratification (routing type x
whether the query truncated at k=10 in exp18), because a latency claim about "the truncated
stratum" is meaningless if the sample missed that stratum -- the defect family this phase keeps
finding. The sampled ids are written into the artifact.

Usage:
  python scripts/measure_survey_config_latency.py --n 16 --k 5,10
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes output/audit/survey_config_latency_<date>.json/.md (checkpointed every query).
"""
import argparse
import json
import statistics as st
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
OUT_DIR = PROJECT_ROOT / "output/audit"
SEED = 42


def stratified_ids(n, seed=SEED):
    """Balanced over (routing type, truncated-at-k10), with the strata recorded."""
    ids_doc = json.loads((EXP_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))
    res = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
    truncated = set(res["observed_truncation"]["final_top_k_10"]["qids"])
    buckets = {}
    for qid, meta in ids_doc["ids"].items():
        key = (meta.get("routing_query_type", "default"), qid in truncated)
        buckets.setdefault(key, []).append(qid)
    rng = np.random.default_rng(seed)
    chosen, strata = [], {}
    per = max(1, n // max(1, len(buckets)))
    for key, qs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        take = min(per, len(qs))
        pick = [qs[i] for i in rng.choice(len(qs), take, replace=False)]
        chosen += pick
        strata[f"{key[0]}|{'trunc' if key[1] else 'no_trunc'}"] = pick
    # top up deterministically if integer division left room
    rest = [q for q in sorted(ids_doc["ids"]) if q not in chosen]
    while len(chosen) < n and rest:
        chosen.append(rest.pop(0))
    return chosen[:n], strata, {q: ids_doc["ids"][q]["question"] for q in chosen[:n]}


def run_one(pipeline, question):
    """One streamed query; TTFT is the wall time until the first token event."""
    t0 = time.perf_counter()
    ttft = None
    retrieval_ms = None
    answer = []
    n_chunks = None
    for kind, value in pipeline.query_stream(question):
        if kind == "token":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            answer.append(value)
        elif kind == "stage" and value == "generation":
            retrieval_ms = (time.perf_counter() - t0) * 1000
        elif kind == "done":
            n_chunks = len(value.get("retrieved_chunks") or [])
    total_ms = (time.perf_counter() - t0) * 1000
    text = "".join(answer)
    return {"ttft_ms": round(ttft, 1) if ttft else None,
            "retrieval_ms": round(retrieval_ms, 1) if retrieval_ms else None,
            "total_ms": round(total_ms, 1), "chars": len(text),
            "words": len(text.split()), "n_chunks": n_chunks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", default="5,10", help="comma-separated final_top_k values")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]

    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    from src.pipeline.rag_pipeline import RAGPipeline, load_hybrid_index

    qids, strata, questions = stratified_ids(args.n)
    print(f"queries: {len(qids)} | estratos: { {k: len(v) for k, v in strata.items()} }", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ck = OUT_DIR / "survey_config_latency.partial.json"
    rows = json.loads(ck.read_text(encoding="utf-8")) if ck.exists() else {}

    index = load_hybrid_index()
    for k in ks:
        cfg = SURVEY_DEPLOY.model_copy(update={"final_top_k": k})
        pipeline = RAGPipeline(config=cfg, hybrid_index=index)
        for i, qid in enumerate(qids, 1):
            key = f"k{k}|{qid}"
            if key in rows:
                continue
            rows[key] = {"qid": qid, "k": k, **run_one(pipeline, questions[qid])}
            ck.write_text(json.dumps(rows), encoding="utf-8")
            print(f"  [k={k} {i}/{len(qids)}] {qid} ttft={rows[key]['ttft_ms']}ms "
                  f"total={rows[key]['total_ms']}ms", flush=True)

    summary = {}
    for k in ks:
        v = [r for r in rows.values() if r["k"] == k]
        def q(field, p):
            xs = sorted(x[field] for x in v if x.get(field) is not None)
            return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None
        summary[f"k={k}"] = {
            "n": len(v),
            "retrieval_ms_p50": q("retrieval_ms", .5),
            "ttft_ms_p50": q("ttft_ms", .5), "ttft_ms_p90": q("ttft_ms", .9),
            "total_ms_p50": q("total_ms", .5), "total_ms_p90": q("total_ms", .9),
            "words_mean": round(st.mean([x["words"] for x in v]), 1) if v else None,
            "n_chunks_mean": round(st.mean([x["n_chunks"] for x in v if x["n_chunks"]]), 2)
            if any(x["n_chunks"] for x in v) else None,
        }

    out = {"generated": str(date.today()), "config": "SURVEY_DEPLOY (prompt_routing + "
           "balance_cross_cloud_providers), live retrieval", "seed": SEED,
           "n_queries": len(qids), "strata": strata, "summary": summary, "rows": rows,
           "why": ("exp18's k=10 arm used FROZEN contexts and no provider balancing, and reports "
                   "TOTAL generation time; the UI streams, so the survey-relevant latency is "
                   "TTFT. These numbers are the deployment-path counterpart."),
           "not_measured": "faithfulness — that stays with the offline scorers on exp18",
           "generated_by": "scripts/measure_survey_config_latency.py"}
    p = OUT_DIR / f"survey_config_latency_{date.today()}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    ck.unlink(missing_ok=True)

    L = [f"# Config de encuestas — latencia en la RUTA DE DESPLIEGUE ({date.today()})", "",
         out["why"], "",
         "| config | n | retrieval p50 | **TTFT p50** | TTFT p90 | total p50 | total p90 | palabras | chunks |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        L.append(f"| {k} | {s['n']} | {s['retrieval_ms_p50']} ms | **{s['ttft_ms_p50']} ms** | "
                 f"{s['ttft_ms_p90']} ms | {s['total_ms_p50']} ms | {s['total_ms_p90']} ms | "
                 f"{s['words_mean']} | {s['n_chunks_mean']} |")
    L += ["", f"Estratos muestreados (seed {SEED}): "
          + ", ".join(f"`{k}` n={len(v)}" for k, v in strata.items()), "",
          "Fidelidad NO se mide aqui (" + out["not_measured"] + ")."]
    (OUT_DIR / f"survey_config_latency_{date.today()}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
