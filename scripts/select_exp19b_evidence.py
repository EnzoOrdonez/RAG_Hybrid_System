"""exp19b — the selector: re-rank the frozen k=50 pool by (claim, chunk) and take five.

Seccion de Claude Code — 2026-08-21 14:40 (hora local).

This is the module that decides what evidence the scored arm gets to see, so it is the one
`tests/test_selector_hygiene.py` guards at source level. It may reach exactly one model: the
PRODUCTION cross-encoder ms-marco-MiniLM-L-12-v2. It may not reach any faithfulness verifier,
because a component that both picks the evidence and judges the grounding measures its own
preferences rather than the system's. It also may not reach bge-reranker-large, the
independent retrieval oracle (Flag 17): putting it inside the method would burn every later
claim of independence made with it.

That is why the draft's claims arrive from a file. Extraction lives in
extract_exp19b_claims.py precisely so this script cannot import the module the extractor
lives in.

The greedy is IMPORTED from compute_exp19a_selector_probe.py, never copied: the offline gate
and the generative arm have to select identically or exp19a's PASS says nothing about exp19b.
That greedy serves each claim by its own best chunk, which is what "grounding-guided" has to
mean; ranking by mean score would concentrate on chunks mildly relevant to everything, the
exact failure mode of query-level ranking that this experiment exists to escape.

FALLBACK, DECLARED BEFORE RUNNING. A draft with no genuine claim gives the selector nothing
to condition on. That query keeps the baseline top-5, so its arm-vs-baseline difference is
exactly zero and the query is counted in `n_fallback`. The alternative -- an empty selection
-- would generate with no context at all and score as a spectacular win on zero claims.

SANITY CHECK, mandatory before any selection is used. Ranking the same pool by (query, chunk)
with this same model must reproduce exp18's baseline top-5 (exp19a got 5.0/5 on 188/188). If
it does not, this harness is not scoring what exp18 scored and the selection is meaningless.

GPU ISOLATION. This selector runs between the draft and regeneration. Loading its cross-encoder
on CUDA rearranges Granite's Ollama GPU state and correctly trips the session-fingerprint gate,
so the reranker is pinned to CPU even when CUDA is available. Nothing may touch the GPU between
the two generative arms.

Usage: python scripts/select_exp19b_evidence.py [--smoke] [--exp-dir DIR] [--max-queries N]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes <exp dir>/selection_ids.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXP_ID = "exp19b_anchored_selector"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
EXP18_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
RERANKER = PROJECT_ROOT / "data/models/ms-marco-MiniLM-L-12-v2"
FINAL_K = 5
BATCH = 64
SANITY_MIN_OVERLAP = 4.0
IO_RETRIES = 3
IO_RETRY_DELAY_SECONDS = 2

_spec = importlib.util.spec_from_file_location(
    "exp19a_probe", PROJECT_ROOT / "scripts/compute_exp19a_selector_probe.py")
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
select_by_claims = _probe.select_by_claims


def atomic_write_text(path, text, encoding="utf-8", retries=IO_RETRIES,
                      delay_seconds=IO_RETRY_DELAY_SECONDS, sleep_fn=None):
    """Write beside the destination, then atomically replace it with Windows retries."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    sleeper = sleep_fn or time.sleep
    for attempt in range(retries):
        try:
            temporary.write_text(text, encoding=encoding)
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            sleeper(delay_seconds)


def unlink_with_retry(path, missing_ok=False, retries=IO_RETRIES,
                      delay_seconds=IO_RETRY_DELAY_SECONDS, sleep_fn=None):
    """Unlink with the same transient-Windows-failure policy as checkpoint replacement."""
    path = Path(path)
    sleeper = sleep_fn or time.sleep
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=missing_ok)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            sleeper(delay_seconds)


def load_reranker():
    """The production cross-encoder on CPU, so it cannot perturb Ollama's GPU state."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(str(RERANKER), max_length=512, device="cpu")


def selection_for_query(pool_ids, R, baseline_ids, k=FINAL_K):
    """(selected chunk ids, fallback reason or None) for one query.

    R is the claim x chunk score matrix, or None when the draft asserted nothing genuine.
    """
    if R is None or getattr(R, "size", 0) == 0 or R.shape[0] == 0:
        return list(baseline_ids), "no_genuine_claims"
    idx = select_by_claims(R, k)
    return [pool_ids[j] for j in idx], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else (
        EXP_DIR / "_smoke" if args.smoke else EXP_DIR)
    claims_path = exp_dir / "draft_claims.json"
    if not claims_path.exists():
        sys.exit(f"{claims_path} missing — run extract_exp19b_claims.py first")

    drafts = json.loads(claims_path.read_text(encoding="utf-8"))["per_query"]
    ids_doc = json.loads((EXP18_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    qids = [q for q in ids_doc if q in drafts]
    if args.max_queries:
        qids = qids[: args.max_queries]

    ckpt = exp_dir / "selection_ids.partial.json"
    done = json.loads(ckpt.read_text(encoding="utf-8")) if (
        ckpt.exists() and not args.no_resume) else {}
    todo = [q for q in qids if q not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)

    model = load_reranker() if todo else None
    for n, qid in enumerate(todo, 1):
        pool_ids = [c for c in ids_doc[qid]["pool_ids"] if c in chunk_map]
        texts = [chunk_map[c]["text"] for c in pool_ids]
        baseline_ids = ids_doc[qid]["baseline_repro_ids"]
        genuine = drafts[qid]["genuine"]

        # (query, chunk): the production ranking. Kept as the harness sanity check, exactly as
        # in exp19a -- it is not a second comparator.
        q_scores = np.array(model.predict([(ids_doc[qid]["question"], t) for t in texts],
                                          batch_size=BATCH))
        q_top = [pool_ids[int(j)] for j in np.argsort(-q_scores)[:FINAL_K]]
        overlap = len(set(q_top) & set(baseline_ids))

        R = None
        if genuine:
            pairs = [(c, t) for c in genuine for t in texts]
            flat = []
            for i in range(0, len(pairs), BATCH):
                flat.extend(float(x) for x in model.predict(pairs[i:i + BATCH],
                                                            batch_size=BATCH))
            R = np.array(flat).reshape(len(genuine), len(pool_ids))

        sel_ids, fallback = selection_for_query(pool_ids, R, baseline_ids)
        done[qid] = {
            "qid": qid, "n_pool": len(pool_ids), "n_genuine_claims": len(genuine),
            "baseline_ids": baseline_ids, "claim_rank_ids": sel_ids,
            "query_rank_ids": q_top, "fallback": fallback,
            "top5_overlap_with_baseline": overlap,
            "n_changed_vs_baseline": len(set(sel_ids) - set(baseline_ids)),
            # A selection can differ from the baseline by ORDER alone. Tier A tested context
            # order and reversal and found them null, so those queries are near-no-ops for the
            # arm even though a set-difference count would read 0 and hide them either way.
            # Recording both makes the dilution of the effect visible instead of implied.
            "same_set_as_baseline": set(sel_ids) == set(baseline_ids),
            "same_order_as_baseline": list(sel_ids) == list(baseline_ids),
        }
        atomic_write_text(ckpt, json.dumps(done))
        if n % 10 == 0:
            print(f"  [{n}/{len(todo)}] {qid} changed={done[qid]['n_changed_vs_baseline']}/5",
                  flush=True)

    rows = [done[q] for q in qids if q in done]
    if len(rows) != len(qids):
        sys.exit(f"INCOMPLETE: {len(qids) - len(rows)} of {len(qids)} queries unselected. "
                 f"Re-run to resume from {ckpt.name}; generating from a partial selection "
                 f"would pair the arms on different query sets.")

    mean_ov = round(float(np.mean([r["top5_overlap_with_baseline"] for r in rows])), 3)
    sanity_ok = mean_ov >= SANITY_MIN_OVERLAP
    n_fallback = sum(1 for r in rows if r["fallback"])
    doc = {
        "experiment_id": EXP_ID, "final_k": FINAL_K, "n_queries": len(rows),
        "selector": "ms-marco-MiniLM-L-12-v2 over (claim, chunk); greedy max over claims",
        "greedy_source": "imported from scripts/compute_exp19a_selector_probe.py",
        "hygiene": ("no faithfulness verifier is reachable from this module, so the three "
                    "evaluators stay clean for the scored arm; the independent retrieval "
                    "oracle is untouched"),
        "sanity_check": {
            "what": "(query, chunk) ranking with this model must reproduce exp18's baseline top-5",
            "mean_overlap_at_5": mean_ov, "threshold": SANITY_MIN_OVERLAP,
            "passed": sanity_ok,
            "why": ("below ~4/5 the harness is not scoring what exp18 scored and the whole "
                    "selection is meaningless"),
        },
        "n_fallback": n_fallback,
        "fallback_rule": ("no genuine claim in the draft -> keep the baseline top-5 -> exact "
                          "zero difference for that query; declared before running"),
        "mean_chunks_changed_vs_baseline": round(
            float(np.mean([r["n_changed_vs_baseline"] for r in rows])), 3),
        "n_same_set_as_baseline": sum(1 for r in rows if r["same_set_as_baseline"]),
        "n_identical_selection": sum(1 for r in rows if r["same_order_as_baseline"]),
        "dilution_note": ("queries with the same SET differ from the baseline by order only, "
                          "and Tier A found context order null; queries with an identical "
                          "ORDERED selection are exact no-ops. Both counts bound how much of "
                          "the arm could have moved at all"),
        "generated_by": "scripts/select_exp19b_evidence.py",
        "per_query": {r["qid"]: r for r in rows},
    }
    atomic_write_text(exp_dir / "selection_ids.json",
                      json.dumps(doc, indent=1, ensure_ascii=False))
    unlink_with_retry(ckpt, missing_ok=True)

    print(f"wrote {exp_dir / 'selection_ids.json'}: n={len(rows)}, sanity overlap {mean_ov}/5 "
          f"({'OK' if sanity_ok else 'FAILED — do not regenerate'}), fallback {n_fallback}, "
          f"mean chunks changed {doc['mean_chunks_changed_vs_baseline']}/5")
    if not sanity_ok:
        sys.exit("SANITY CHECK FAILED — refusing to hand this selection to the generator")


if __name__ == "__main__":
    main()
