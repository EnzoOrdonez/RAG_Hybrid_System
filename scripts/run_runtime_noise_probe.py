"""Runtime noise floor — how much does faithfulness move when the GENERATOR STATE changes?

Seccion de Claude Code — 2026-08-21 14:50 (hora local).

THE QUESTION, AND WHY IT BLOCKS SPENDING. exp21 asks whether a hosted granite is equivalent to
the local one within +/- 0.081 on faithfulness. But hosted runs 41/41 layers on the GPU and
local runs 30/41: that IS a change of generator state. And on 2026-08-21 a plain Ollama restart
-- same binary, same model, same seed, minutes apart -- changed the answer to a byte-identical
prompt (q001 vs the checkpoint: jaccard-5gram 0.0705). If merely changing state moves MEAN
faithfulness by something comparable to 0.081, then exp21's band cannot separate "hosting" from
"the generator moved", and no number of extra queries fixes that.

WHAT IS AND IS NOT AT RISK. Per-query noise is large but that is not the estimand. exp14's 140
replicas (same runtime, cache off, temp 0, seed 42) give |delta faithfulness| between replicas
of the same query: mean 0.0983, median 0.0000, p90 0.2895, and 28.1 % of replica pairs exceed
0.081. Yet the primary is the MEAN paired difference over ~190 queries, whose SE at SD~0.2 is
~0.015. So the band is fine for the mean PROVIDED the noise is zero-mean. What would break
exp21 is a SYSTEMATIC shift of the mean between two runtime states. exp14 cannot answer that --
it ran in one runtime. That is the gap this probe fills, and the only reason it generates.

THE VERDICT RULE, DECLARED HERE BEFORE ANY DATA EXIST. Let d = mean(state_B - state_A)
faithfulness, with a 95 % bootstrap CI:

  (b) BAND SURVIVES     CI entirely inside +/- 0.0203 (= 0.081/4). A state change does not
                        systematically move mean faithfulness at a quarter of the band, so
                        exp21 may keep the pre-registered band.
  (a) SYSTEMATIC BIAS   CI excludes 0 AND |d| > 0.0203. Hosting cannot be separated from the
                        state change: exp21 needs probe normalisation or a different gate.
  (c) UNDERPOWERED      neither -- the CI half-width is too wide to decide. Report as
                        inconclusive and run the larger tier. NOT a pass.

A quarter of the band is the threshold because the band has to absorb the effect being tested
AND this nuisance; anything above a quarter makes the nuisance a first-order term.

NOT EVIDENCE. Everything is written under experiments/probes/runtime_noise/, outside
experiments/results/. This enters no BH family and no TOST; it is a measurement of the
instrument, used to decide whether another experiment's band is meaningful.

Modes:
  --mode state-b     regenerate the queries of state_A in the CURRENT runtime state (GPU)
  --mode analyze     score both states with HHEM and apply the verdict rule (GPU, no LLM)
  --mode exp14-hhem  re-score exp14's stored replicas with HHEM (GPU, no LLM); converts the
                     within-runtime floor to the instrument exp21 actually uses
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import importlib.util
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROBE_DIR = PROJECT_ROOT / "experiments/probes/runtime_noise"
EXP18_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
STATE_A = PROBE_DIR / "state_A_2026-08-21.json"

TAU = 0.5
PREMISE_CHARS = 1500          # == rescore_grounding_exp15, so numbers stay comparable
BATCH = 16
SEED = 42
BOOT = 10000

TOST_BAND = 0.081
NUISANCE_LIMIT = round(TOST_BAND / 4, 4)     # 0.0203
DECISION_HALF_WIDTH = NUISANCE_LIMIT * 2     # wider than this CI => cannot decide

_spec = importlib.util.spec_from_file_location(
    "exp19b_gen", PROJECT_ROOT / "scripts/run_exp19b_generation.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def verdict(mean_diff, ci_lo, ci_hi):
    """(code, text). Declared before the data exist; see the module docstring."""
    half = (ci_hi - ci_lo) / 2
    inside = ci_lo >= -NUISANCE_LIMIT and ci_hi <= NUISANCE_LIMIT
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    if inside:
        return "b_band_survives", (
            f"the whole CI sits inside +/-{NUISANCE_LIMIT}: a runtime state change does not "
            f"systematically move mean faithfulness at a quarter of the {TOST_BAND} band, so "
            f"exp21 may keep its pre-registered band")
    if excludes_zero and abs(mean_diff) > NUISANCE_LIMIT:
        return "a_systematic_bias", (
            f"the CI excludes 0 and |d|={abs(mean_diff):.4f} exceeds {NUISANCE_LIMIT}: hosting "
            f"could not be separated from the state change, so exp21 needs probe normalisation "
            f"or a different gate")
    return "c_underpowered", (
        f"CI half-width {half:.4f} is too wide to decide (limit {DECISION_HALF_WIDTH}). "
        f"INCONCLUSIVE, which is not a pass: run the larger tier before reading anything")


def score_hhem(model, rows, chunk_map):
    """Faithfulness per row with the phase's exact rule: supported iff max chunk p > tau.

    `rows` are dicts with `answer` and `chunk_ids`. Mirrors rescore_grounding_exp15 so these
    numbers are directly comparable to every other HHEM figure in the project.
    """
    from src.generation.hallucination_detector import HallucinationDetector, classify_artifact
    import torch
    det = HallucinationDetector(use_nli=False)

    pairs, spans = [], []
    for i, r in enumerate(rows):
        answer = r.get("answer") or ""
        claims = det._extract_claims(answer) if answer.strip() else []
        genuine = [c for c in claims if not classify_artifact(c)]
        texts = [chunk_map[c]["text"][:PREMISE_CHARS] for c in r["chunk_ids"] if c in chunk_map]
        if not genuine or not texts:
            spans.append((i, 0, 0, None))
            continue
        spans.append((i, len(genuine), len(texts), len(pairs)))
        pairs.extend((t, cl) for cl in genuine for t in texts)

    scores = []
    for s in range(0, len(pairs), BATCH):
        with torch.no_grad():
            scores.extend(float(x) for x in model.predict(pairs[s:s + BATCH]))

    out = []
    for i, n_gen, k, start in spans:
        if not n_gen:
            out.append({"idx": i, "genuine": 0, "faithfulness": None})
            continue
        sup = sum(1 for c in range(n_gen)
                  if max(scores[start + c * k: start + (c + 1) * k]) > TAU)
        out.append({"idx": i, "genuine": n_gen, "supported": sup,
                    "faithfulness": round(sup / n_gen, 4)})
    return out


def paired_shift(fa, fb):
    """Mean paired shift B-A with a bootstrap CI, dropping pairs the metric cannot define."""
    pairs = [(a, b) for a, b in zip(fa, fb) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    d = np.array([b - a for a, b in pairs], float)
    rng = np.random.default_rng(SEED)
    boot = np.array([float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(BOOT)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    code, text = verdict(float(d.mean()), lo, hi)
    return {"n_paired": len(pairs), "n_dropped": len(fa) - len(pairs),
            "mean_shift_B_minus_A": round(float(d.mean()), 4),
            "mean_abs_shift": round(float(np.abs(d).mean()), 4),
            "sd_paired": round(float(d.std(ddof=1)), 4),
            "boot95": [round(lo, 4), round(hi, 4)],
            "band": TOST_BAND, "nuisance_limit": NUISANCE_LIMIT,
            "verdict": code, "reading": text,
            "not_evidence": ("instrument measurement; enters no BH family and no TOST")}


def mode_state_b(args):
    """Regenerate state_A's queries in whatever runtime state is live now."""
    doc = json.loads(STATE_A.read_text(encoding="utf-8"))
    prior = {r["query_id"]: r for r in doc["results"]}
    qids = list(prior)[: args.n]

    ids_doc = json.loads((EXP18_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    from src.generation.llm_manager import LLMManager
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT", "build_context",
          "get_template")}
    index, qp = _gen.ChunkMapIndex(chunk_map), QueryProcessor()
    llm = LLMManager(provider="ollama", model=_gen.MODEL_TAG, cache_enabled=False, seed=SEED)

    def prompt_for(q):
        return _gen.rgm.build_prompt("hibrido", ids_doc[q]["question"],
                                     ids_doc[q]["baseline_repro_ids"], index,
                                     qp.process(ids_doc[q]["question"]).query_type, P)

    pr0, sp0, _ = prompt_for(qids[0])
    warm = llm.generate(prompt=pr0, system_prompt=sp0, temperature=0.0, config_name="warmup")
    fp = _gen.session_fingerprint(warm.text)
    print(f"state B fingerprint {fp}; generating {len(qids)} queries", flush=True)

    out = PROBE_DIR / f"state_B_{date.today().isoformat()}.json"
    results = []
    for i, q in enumerate(qids, 1):
        pr, sp, _ = prompt_for(q)
        r = llm.generate(prompt=pr, system_prompt=sp, temperature=0.0, config_name="state_B")
        results.append({"query_id": q, "answer": r.text,
                        "retrieved_ids": ids_doc[q]["baseline_repro_ids"],
                        "tokens": {"input": r.tokens_input, "output": r.tokens_output}})
        out.write_text(json.dumps({"session_fingerprint": fp, "generated": datetime.now()
                                   .isoformat(), "results": results}, ensure_ascii=False),
                       encoding="utf-8")
        if i % 5 == 0:
            print(f"  {i}/{len(qids)}", flush=True)
    print(f"wrote {out}")


def mode_analyze(args):
    a_doc = json.loads(STATE_A.read_text(encoding="utf-8"))
    b_path = args.state_b or max(PROBE_DIR.glob("state_B_*.json"), default=None)
    if not b_path:
        sys.exit("no state_B_*.json found — run --mode state-b first")
    b_doc = json.loads(Path(b_path).read_text(encoding="utf-8"))

    A = {r["query_id"]: r for r in a_doc["results"]}
    B = {r["query_id"]: r for r in b_doc["results"]}
    qids = [q for q in A if q in B]
    if not qids:
        sys.exit("state A and state B share no queries")

    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    from scripts.rescore_grounding_exp15 import load_hhem
    model = load_hhem()

    def rows_for(src):
        return [{"answer": src[q].get("answer"), "chunk_ids": src[q]["retrieved_ids"]}
                for q in qids]

    fa = [r["faithfulness"] for r in score_hhem(model, rows_for(A), chunk_map)]
    fb = [r["faithfulness"] for r in score_hhem(model, rows_for(B), chunk_map)]
    res = paired_shift(fa, fb)
    if res is None:
        sys.exit("too few scorable pairs")
    res.update({"n_queries": len(qids), "verifier": "hhem-2.1", "tau": TAU,
                "state_b_fingerprint": b_doc.get("session_fingerprint"),
                "generated_by": "scripts/run_runtime_noise_probe.py"})
    out = PROBE_DIR / f"runtime_shift_{date.today().isoformat()}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nVERDICT: {res['verdict']}\n{res['reading']}")


def mode_exp14_hhem(args):
    """Within-runtime replica noise, re-read with the instrument exp21 uses."""
    src = PROJECT_ROOT / "experiments/results/exp14_h5_replicas/replicas_checkpoint.json"
    rows_raw = json.loads(src.read_text(encoding="utf-8"))["results"]
    ctx, _ = _gen.rgm.load_exp11_contexts()
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))

    usable = [r for r in rows_raw if r["query_id"] in ctx.get(r["scenario"], {})]
    rows = [{"answer": r.get("answer"), "chunk_ids": ctx[r["scenario"]][r["query_id"]]}
            for r in usable]
    from scripts.rescore_grounding_exp15 import load_hhem
    scored = score_hhem(load_hhem(), rows, chunk_map)

    cells = {}
    for r, s in zip(usable, scored):
        cells.setdefault((r["pair_id"], r["model"], r["query_id"]), {})[r["replica"]] = \
            s["faithfulness"]
    import itertools
    deltas = [abs(v[a] - v[b]) for v in cells.values()
              for a, b in itertools.combinations(sorted(v), 2)
              if v[a] is not None and v[b] is not None]
    d = np.array(deltas, float)
    out = {"source": "exp14_h5_replicas (read-only)", "verifier": "hhem-2.1", "tau": TAU,
           "n_rows_scored": len(scored), "n_replica_pairs": len(d),
           "mean_abs_delta": round(float(d.mean()), 4),
           "median_abs_delta": round(float(np.median(d)), 4),
           "p90_abs_delta": round(float(np.percentile(d, 90)), 4),
           "frac_exceeding_band": round(float((d > TOST_BAND).mean()), 4),
           "what_this_is": ("within-runtime run-to-run noise per query. NOT the runtime-state "
                            "shift exp21 needs: exp14 ran in a single runtime"),
           "generated_by": "scripts/run_runtime_noise_probe.py"}
    p = PROBE_DIR / "exp14_replica_noise_hhem.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["state-b", "analyze", "exp14-hhem"])
    ap.add_argument("--n", type=int, default=20, help="queries for --mode state-b (tier 1: 20)")
    ap.add_argument("--state-b", default=None)
    args = ap.parse_args()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    {"state-b": mode_state_b, "analyze": mode_analyze, "exp14-hhem": mode_exp14_hhem}[args.mode](args)


if __name__ == "__main__":
    main()
