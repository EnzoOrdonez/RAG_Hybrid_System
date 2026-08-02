"""exp15 ablation harness (summer phase) — Pass G (generation) + Pass N (NLI).

Arms come from experiments/ablation_arms.json. Tier A arms transform the SIGNED
exp11 top-5 id lists (identity / slice:K / reverse / perm:...) — no retrieval
re-run. Generation uses the CANONICAL prompt path (build_prompt imported from
scripts/run_generation_matrix.py + src/generation/prompt_templates), with the
chunk dicts read straight from the chunk_map JSON — byte-identical prompts to
the exp12 path (HybridIndex.get_chunk returns entries of the same JSON) without
loading FAISS/embedder, and Pass G loads NO NLI model: H5 mitigation (nothing
GPU-resident besides Ollama during generation).

Pass G (per arm): 3x determinism probe (cache off) on the first subset query —
STOP the arm if outputs differ — then greedy generation (temp 0, seed 42),
config_name "<arm> | <model-label>" (unique per arm -> segregated LLM cache),
checkpoint every 10 queries with resume, folded into results.json.
hallucination_metrics are DEFERRED to Pass N (method "pending_nli").

Pass N (per arm, per verifier): claim extraction (extractor only) + one pooled
fp16 CrossEncoder predict per arm, persisting raw per-(arm, query, claim,
chunk) probabilities (gzip) and v3-format rows — same machinery as Tier 0
(scripts/rescore_nli_exp15.py) so threshold/variant sweeps stay CPU-pure.

Usage:
  python scripts/run_exp15_ablation.py --exp-id exp15_ablation_tierA \
      --pass G --arms baseline_repro[,...] [--max-queries 2] [--no-resume]
  python scripts/run_exp15_ablation.py --exp-id exp15_ablation_tierA \
      --pass N --arms baseline_repro[,...] --verifier small|base
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import gzip
import importlib.util
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp15_ablation")

SEED = 42
CHECKPOINT_EVERY = 10
ARMS_PATH = PROJECT_ROOT / "experiments" / "ablation_arms.json"
SUBSET_PATH = PROJECT_ROOT / "data" / "evaluation" / "summer_subset.json"
EXP11_PATH = PROJECT_ROOT / "experiments/results/exp11_retrieval194_fullrerank/results.json"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"

# canonical prompt builder, imported from the frozen exp12 runner (no side effects)
_spec = importlib.util.spec_from_file_location(
    "rgm", PROJECT_ROOT / "scripts" / "run_generation_matrix.py")
rgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgm)


class ChunkMapIndex:
    """Minimal stand-in for HybridIndex in build_prompt: get_chunk only.

    HybridIndex.load() reads chunk_map from the same JSON, so entries are
    byte-identical to what the exp12 path saw.
    """

    def __init__(self, chunk_map):
        self._m = chunk_map

    def get_chunk(self, cid):
        return self._m.get(cid)


def apply_transform(ids, transform):
    if transform == "identity":
        return list(ids)
    if transform == "reverse":
        return list(reversed(ids))
    if transform.startswith("slice:"):
        return list(ids)[: int(transform.split(":", 1)[1])]
    if transform.startswith("perm:"):
        perm = [int(x) for x in transform.split(":", 1)[1].split(",")]
        if sorted(perm) != list(range(len(ids))):
            raise SystemExit(f"perm {perm} is not a permutation of 0..{len(ids)-1} "
                             f"(arm list has {len(ids)} ids)")
        return [ids[i] for i in perm]
    raise SystemExit(f"unknown transform: {transform}")


def load_inputs(registry, arm_names):
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    qids = subset["query_ids"]
    exp11 = json.loads(EXP11_PATH.read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    contexts, questions = {}, {}
    for arm in arm_names:
        spec = registry["arms"][arm]
        cfg = exp11[spec["exp11_config"]]
        per_q = {}
        for r in cfg["results"]:
            if r["query_id"] in set(qids):
                per_q[r["query_id"]] = apply_transform(r["retrieved_ids"], spec["transform"])
                questions[r["query_id"]] = r["question"]
        missing = [q for q in qids if q not in per_q]
        if missing:
            raise SystemExit(f"arm {arm}: {len(missing)} subset queries missing in exp11 "
                             f"config '{spec['exp11_config']}' (e.g. {missing[:3]})")
        contexts[arm] = per_q
    return qids, contexts, questions, chunk_map


def ckpt_path(exp_dir, label, arm):
    return exp_dir / f"checkpoint__{label}__{arm}.json"


# ---------------------------------------------------------------------------
# Pass G — generation (Ollama only; no NLI, no FAISS/embedder)
# ---------------------------------------------------------------------------
def pass_g(args, registry, exp_dir):
    from src.generation.llm_manager import LLMManager
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT",
          "build_context", "get_template")}

    arm_names = args.arms
    qids, contexts, questions, chunk_map = load_inputs(registry, arm_names)
    if args.max_queries:
        qids = qids[: args.max_queries]
    index = ChunkMapIndex(chunk_map)
    tag = registry["model"]
    label = rgm.model_label(tag)

    qp = QueryProcessor()
    qtype = {qid: qp.process(questions[qid]).query_type for qid in qids}

    llm = LLMManager(provider="ollama", model=tag,
                     cache_enabled=not getattr(args, "no_cache", False), seed=SEED)
    probe_report = {}

    # Warmup (H5): the observed nondeterminism was "1st generation after load
    # differs, 2nd/3rd identical". One throwaway generation stabilizes the
    # runner before any measured query so per-arm generations are drawn from the
    # warm state. Cache off so it never persists.
    if qids:
        qid0 = qids[0]
        pr0, sp0, _ = rgm.build_prompt("hibrido", questions[qid0], contexts[arm_names[0]][qid0],
                                       index, qtype[qid0], P)
        LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED).generate(
            prompt=pr0, system_prompt=sp0, temperature=0.0, config_name="warmup")
        logger.info("warmup generation done")

    for arm in arm_names:
        config_name = f"{arm} | {label}"
        variant = registry["arms"][arm].get("prompt_variant", "baseline")
        # --- determinism probe: 3x first query, cache OFF. Per Enzo (relaxed
        # gate): WARN + record as metadata, do NOT skip. Bit-determinism is not a
        # validity requirement for the paired arm-vs-baseline contrast (same
        # queries, same environment/session; H5 cell means stable Δ≤0.017);
        # single-sample non-bit-reproducibility is a documented limitation.
        qid0 = qids[0]
        pr, sp, _ = rgm.build_prompt("hibrido", questions[qid0], contexts[arm][qid0],
                                     index, qtype[qid0], P, variant=variant)
        llm_nc = LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED)
        outs = []
        for _ in range(3):
            r = llm_nc.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                                config_name=f"detprobe|{arm}|{label}")
            outs.append(r.text)
        det_ok = all(o == outs[0] for o in outs)
        vram = rgm.gpu_mem_used_mb()
        tok_s = (r.tokens_output / (r.latency_ms / 1000)) if r.latency_ms else 0.0
        probe_report[arm] = {"determinism_3x_identical": det_ok,
                             "tok_per_s": round(tok_s, 2), "vram_used_mb": vram,
                             "answer_lens_3x": [len(o) for o in outs]}
        logger.info("[%s] probe: determinism=%s tok/s=%.1f vram=%sMB",
                    arm, det_ok, tok_s, vram)
        if not det_ok:
            logger.warning("[%s] determinism probe NOT bit-identical (lens %s) - running anyway "
                           "(relaxed gate); results flagged single-sample.",
                           arm, [len(o) for o in outs])

        cpath = ckpt_path(exp_dir, label, arm)
        results, done = [], set()
        if not args.no_resume and cpath.exists():
            ck = json.loads(cpath.read_text(encoding="utf-8"))
            results = ck["results"]; done = set(ck["completed_ids"])
            logger.info("[%s] resume: %d done", config_name, len(done))
        todo = [q for q in qids if q not in done]
        for i, qid in enumerate(todo):
            prompt, sysp, _ = rgm.build_prompt("hibrido", questions[qid], contexts[arm][qid],
                                               index, qtype[qid], P, variant=variant)
            t = time.perf_counter()
            resp = llm.generate(prompt=prompt, system_prompt=sysp,
                                temperature=0.0, config_name=config_name)
            gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
            results.append({
                "query_id": qid, "config_name": config_name,
                "scenario": arm, "model": tag,
                "question": questions[qid], "answer": resp.text,
                "retrieved_ids": contexts[arm][qid],
                "hallucination_metrics": {"method": "pending_nli"},
                "tokens": {"input": resp.tokens_input, "output": resp.tokens_output},
                "latency": {"generation_ms": round(gen_ms, 1)},
                "from_cache": resp.from_cache,
                "error": resp.error,
                "timestamp": datetime.now().isoformat(),
            })
            done.add(qid)
            if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
                cpath.write_text(json.dumps(
                    {"config_name": config_name, "completed_ids": sorted(done),
                     "results": results}, ensure_ascii=False), encoding="utf-8")
                logger.info("[%s] %d/%d", config_name, len(done), len(qids))

    # fold checkpoints into results.json (exp8 schema; arm sits in the scenario slot)
    configs = {}
    for arm in registry["arms"]:
        cpath = ckpt_path(exp_dir, label, arm)
        if not cpath.exists():
            continue
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        rs = ck["results"]
        configs[ck["config_name"]] = {
            "total_queries": len(rs), "errors": sum(1 for r in rs if r.get("error")),
            "scenario": arm, "model": tag, "results": rs}
    # merge probe_report with any prior run so re-running a SUBSET of arms (e.g. only
    # baseline with --no-cache) preserves the other arms' probe/determinism metadata.
    prior_probe = {}
    rj = exp_dir / "results.json"
    if rj.exists():
        try:
            prior_probe = json.loads(rj.read_text(encoding="utf-8")).get("probe_report", {})
        except Exception:
            prior_probe = {}
    merged_probe = {**prior_probe, **probe_report}
    payload = {
        "experiment_id": args.exp_id,
        "name": registry.get("name", "exp15 ablation Tier A (generation from signed exp11 id lists)"),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED, "temperature": 0.0,
        "context_source": registry.get(
            "context_source", "exp11_retrieval194_fullrerank (transformed id lists; no re-retrieval)"),
        "queries": str(SUBSET_PATH.relative_to(PROJECT_ROOT)),
        "num_queries": len(qids),
        "model": tag, "arms": {a: registry["arms"][a] for a in registry["arms"]},
        "probe_report": merged_probe,
        "all_arms_bit_deterministic": all(p.get("determinism_3x_identical")
                                          for p in merged_probe.values()) if merged_probe else None,
        "reproducibility_note": ("Generations are single-sample. Bit-determinism per arm is recorded "
                                 "in probe_report; where False, arm-vs-baseline paired contrasts remain "
                                 "valid (same queries/session) but absolute answers are not "
                                 "bit-reproducible (H5: VRAM-pressure CPU/GPU split; cell means stable)."),
        "configs": configs,
    }
    (exp_dir / "results.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", exp_dir / "results.json")
    for cname, c in configs.items():
        rs = c["results"]
        lat = [r["latency"]["generation_ms"] for r in rs if not r.get("from_cache")]
        p50 = sorted(lat)[len(lat) // 2] / 1000 if lat else 0
        print(f"  {cname:<38} n={len(rs):>3} errors={c['errors']} p50_gen={p50:.1f}s "
              f"cached={sum(1 for r in rs if r.get('from_cache'))}")


# ---------------------------------------------------------------------------
# Pass N — NLI scoring with raw-prob persistence (mirrors rescore_nli_exp15)
# ---------------------------------------------------------------------------
def pass_n(args, registry, exp_dir):
    from src.generation.hallucination_detector import (
        HallucinationDetector, classify_artifact, decide_nli_status)
    from sentence_transformers import CrossEncoder
    import torch

    local = PROJECT_ROOT / "data" / "models" / f"nli-deberta-v3-{args.verifier}"
    name = str(local) if local.exists() else f"cross-encoder/nli-deberta-v3-{args.verifier}"
    model = CrossEncoder(name, max_length=512)
    if torch.cuda.is_available():
        model.model.half()
    det = HallucinationDetector(use_nli=False)
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    results = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))["configs"]

    probs_out = {"verifier": name, "verifier_tag": args.verifier,
                 "classes": ["contradiction", "entailment", "neutral"],
                 "pooling": "per-arm predict, batch 64, fp16 (rescore_nli_v3 mirror)",
                 "strip_inline_cites": bool(getattr(args, "strip_inline_cites", False)),
                 "generated_by": "scripts/run_exp15_ablation.py::pass_n", "configs": {}}
    claims_out = {"generated_by": "scripts/run_exp15_ablation.py::pass_n", "configs": {}}
    rows_v3 = {"verifier": name, "variant": "vb_agree", "margin": 0.0,
               "generated_by": "scripts/run_exp15_ablation.py::pass_n", "configs": {}}

    # Per-arm checkpoint. pass_g has had one since Tier A; pass_n never did, and it is the
    # longer pass on exp18 (~51k pairs per verifier, the top-10 arm alone is 29k because it
    # carries 10 chunks). The environment has killed long jobs repeatedly, and without this
    # a kill throws away the whole verifier pass. Scoring is deterministic re-aggregation of
    # a fixed model over fixed text -- unlike generation, resuming it is exact and carries no
    # H5 exposure.
    part_path = exp_dir / f"pass_n__{args.verifier}.partial.json.gz"
    if not getattr(args, "no_resume", False) and part_path.exists():
        with gzip.open(part_path, "rt", encoding="utf-8") as f:
            prev = json.load(f)
        probs_out["configs"] = prev["probs"]
        claims_out["configs"] = prev["claims"]
        rows_v3["configs"] = prev["rows"]
        logger.info("pass N resume: %d arms already scored", len(prev["probs"]))

    def _save_partial():
        with gzip.open(part_path, "wt", encoding="utf-8") as f:
            json.dump({"probs": probs_out["configs"], "claims": claims_out["configs"],
                       "rows": rows_v3["configs"]}, f)

    t0 = time.time()

    for cname, cdata in results.items():
        arm = cdata["scenario"]
        if cname in probs_out["configs"]:
            logger.info("[%s] already scored, skipping", cname)
            continue
        if args.arms and arm not in args.arms:
            continue
        cfg_probs, cfg_claims, cfg_rows = {}, {}, {}
        pairs, spans = [], []
        for r in cdata["results"]:
            answer = r.get("answer") or ""
            if getattr(args, "strip_inline_cites", False):
                answer = re.sub(r"\[\d+\]", "", answer)
            if not answer.strip():
                cfg_rows[r["query_id"]] = {"total_claims": 0, "not_a_claim": 0, "genuine": 0,
                                           "supported": 0, "contradicted": 0, "unsupported": 0,
                                           "faithfulness": None, "method": "none"}
                continue
            claims = det._extract_claims(answer)
            artifact = [bool(classify_artifact(c)) for c in claims]
            genuine = [c for c, a in zip(claims, artifact) if not a]
            n_art = len(claims) - len(genuine)
            cids = [cid for cid in r["retrieved_ids"] if cid in chunk_map]
            texts = [chunk_map[cid]["text"] for cid in cids]
            cfg_claims[r["query_id"]] = {"claims": claims, "artifact": artifact,
                                         "chunk_ids": cids}
            if not claims or not texts:
                cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                           "genuine": 0, "supported": 0, "contradicted": 0,
                                           "unsupported": 0, "faithfulness": None,
                                           "method": "none"}
                continue
            if not genuine:
                cfg_probs[r["query_id"]] = []
                cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                           "genuine": 0, "supported": 0, "contradicted": 0,
                                           "unsupported": 0, "faithfulness": 1.0,
                                           "method": "vacuous"}
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
                st, _, _ = decide_nli_status([float(p[0]) for p in rows_p],
                                             [float(p[1]) for p in rows_p],
                                             0.7, 0.7, variant="vb_agree", margin=0.0)
                agg[st] += 1
            g = len(genuine)
            cfg_probs[qid] = q_probs
            cfg_rows[qid] = {"total_claims": g + n_art, "not_a_claim": n_art, "genuine": g,
                             **agg, "faithfulness": round(agg["supported"] / g, 4),
                             "method": "nli"}
        probs_out["configs"][cname] = cfg_probs
        claims_out["configs"][cname] = cfg_claims
        rows_v3["configs"][cname] = cfg_rows
        logger.info("[%s] %d responses, %d pairs (%.0fs)",
                    cname, len(cfg_rows), len(pairs), time.time() - t0)
        _save_partial()

    with gzip.open(exp_dir / f"nli_probs__{args.verifier}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(probs_out, f)
    (exp_dir / "claims_extraction.json").write_text(
        json.dumps(claims_out, indent=1), encoding="utf-8")
    (exp_dir / f"faithfulness_rows__{args.verifier}__vb_agree.json").write_text(
        json.dumps(rows_v3, indent=1), encoding="utf-8")
    part_path.unlink(missing_ok=True)
    logger.info("Pass N done (%s) in %.0fs", args.verifier, time.time() - t0)


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser(description="exp15 ablation harness")
    ap.add_argument("--exp-id", default="exp15_ablation_tierA")
    ap.add_argument("--pass", dest="pass_", required=True, choices=["G", "N"])
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names (default: all in registry)")
    ap.add_argument("--arms-file", default=str(ARMS_PATH),
                    help="arm registry JSON (default: Tier A ablation_arms.json)")
    ap.add_argument("--verifier", default="small", choices=["small", "base"],
                    help="Pass N only")
    ap.add_argument("--strip-inline-cites", action="store_true",
                    help="Pass N only: strip standalone [N] citation markers from answers "
                         "before claim extraction (exp16 anchored arms; default off keeps the "
                         "signed claim-extraction path untouched)")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="Pass G: disable the LLM cache so answers are generated fresh in "
                         "THIS session (avoids serving a same-config answer cached from an "
                         "earlier session/boot — H5 cross-session drift). Checkpoints still resume.")
    args = ap.parse_args()

    set_all_seeds(SEED)
    registry = json.loads(Path(args.arms_file).read_text(encoding="utf-8"))
    args.arms = ([a.strip() for a in args.arms.split(",")] if args.arms
                 else list(registry["arms"].keys()))
    unknown = [a for a in args.arms if a not in registry["arms"]]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.pass_ == "G":
        pass_g(args, registry, exp_dir)
    else:
        pass_n(args, registry, exp_dir)


if __name__ == "__main__":
    main()
