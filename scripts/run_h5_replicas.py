"""
H5 (N9) — réplicas dirigidas para cuantificar la varianza run-a-run de los pares
entre-modelos supervivientes bajo la métrica v4.

A temp=0 el seed de Ollama es inoperante (decodificación greedy); lo que amenaza
las cifras es el no-determinismo de kernels GPU. Este runner regenera, con caché
DESACTIVADO, las celdas implicadas en los 2 pares significativos post-v4:

  par A (verificador small): denso  granite4.1 vs mistral   (d_z=+0,42 p_bh=0,014)
  par B (verificador base):  lexico gemma4     vs granite4.1 (d_z=-0,53 p_bh=0,038)

Diseño: 25 queries por par (muestreo determinista seed=42 sobre la intersección
elegible: genuine>0 en el rescore v3 y no-pure_decline en AMBOS brazos) × 2
modelos × 3 réplicas = 300 generaciones (~3,3 h con latencias p50 de exp12).

Paridad con exp12: prompts vía scripts.run_generation_matrix.build_prompt, los
MISMOS retrieved_ids de exp11 (SCENARIO_SOURCE), mismo detector runtime
(post-N8: exclusión de artefactos + vb_agree). La comparación oficial se hace
offline sobre la salida con la convención v4 (vacuas excluidas).

Salida (archivos NUEVOS; nada firmado se toca):
  experiments/results/exp14_h5_replicas/replicas_checkpoint.json  (resumible)
  experiments/results/exp14_h5_replicas/meta.json

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42; Ollama arriba.
Usage: python scripts/run_h5_replicas.py [--n-queries 25] [--n-replicas 3]
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds  # noqa: E402
from scripts.run_generation_matrix import (  # noqa: E402
    SEED, load_exp11_contexts, build_prompt, model_label,
)
from scripts.compute_faithfulness_metrics import classify_response  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("h5_replicas")

EXP12_DIR = PROJECT_ROOT / "experiments" / "results" / "exp12_matrix"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "exp14_h5_replicas"
RESCORE = {
    "small": EXP12_DIR / "faithfulness_rescore_v3__small__vb_agree.json",
    "base": EXP12_DIR / "faithfulness_rescore_v3__base__vb_agree.json",
}
# (pair_id, scenario, tag_a, tag_b, verificador que lo marcó)
PAIRS = [
    ("A_denso_granite_vs_mistral", "denso", "granite4.1:8b", "mistral:7b-instruct", "small"),
    ("B_lexico_gemma_vs_granite", "lexico", "gemma4:e4b", "granite4.1:8b", "base"),
]
CHECKPOINT_EVERY = 10


def eligible_qids(pair, contexts):
    """Intersección elegible del par: genuine>0 (rescore v3 del verificador que
    lo marcó) y no-pure_decline en AMBOS brazos, con contexto exp11 presente."""
    _pid, scenario, tag_a, tag_b, verifier = pair
    rescore = json.loads(RESCORE[verifier].read_text(encoding="utf-8"))["configs"]
    out = None
    for tag in (tag_a, tag_b):
        label = model_label(tag)
        cfg = f"{scenario} | {label}"
        rows = rescore[cfg]
        ck = json.loads((EXP12_DIR / f"checkpoint__{label}__{scenario}.json")
                        .read_text(encoding="utf-8"))
        answers = {r["query_id"]: r["answer"] for r in ck["results"]}
        ok = {qid for qid, r in rows.items()
              if r.get("genuine", 0) > 0
              and classify_response(answers.get(qid)) != "pure_decline"}
        out = ok if out is None else (out & ok)
    out &= set(contexts[scenario].keys())
    return sorted(out)


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser(description="H5 replicas (N9)")
    ap.add_argument("--n-queries", type=int, default=25)
    ap.add_argument("--n-replicas", type=int, default=3)
    args = ap.parse_args()
    set_all_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "replicas_checkpoint.json"

    from src.pipeline.rag_pipeline import load_hybrid_index
    from src.generation.llm_manager import LLMManager
    from src.generation.hallucination_detector import HallucinationDetector
    from src.generation import prompt_templates as PT
    from src.retrieval.query_processor import QueryProcessor
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT",
          "build_context", "get_template")}

    logger.info("Cargando índice + contextos exp11 ...")
    index = load_hybrid_index(embedding_model="bge-large",
                              chunking_strategy="adaptive", chunk_size=500)
    contexts, questions = load_exp11_contexts()
    qp = QueryProcessor()

    # Plan determinista: 25 qids por par (rng seed=42 sobre la lista ordenada)
    plan = []  # (pair_id, scenario, tag, qid, rep)
    meta_pairs = {}
    for pair in PAIRS:
        pid, scenario, tag_a, tag_b, verifier = pair
        elig = eligible_qids(pair, contexts)
        rng = random.Random(SEED)
        chosen = sorted(rng.sample(elig, min(args.n_queries, len(elig))))
        meta_pairs[pid] = {"scenario": scenario, "models": [tag_a, tag_b],
                           "verifier": verifier, "eligible": len(elig),
                           "sampled": chosen}
        logger.info("%s: elegibles=%d muestreadas=%d", pid, len(elig), len(chosen))
        for tag in (tag_a, tag_b):
            for qid in chosen:
                for rep in range(1, args.n_replicas + 1):
                    plan.append((pid, scenario, tag, qid, rep))

    # Agrupar por modelo para minimizar swaps en la GPU
    plan.sort(key=lambda t: (t[2], t[0], t[3], t[4]))
    logger.info("Plan total: %d generaciones", len(plan))

    (OUT_DIR / "meta.json").write_text(json.dumps({
        "experiment_id": "exp14_h5_replicas",
        "purpose": "H5 (N9): varianza run-a-run de los pares entre-modelos post-v4",
        "seed": SEED, "temperature": 0.0, "cache_enabled": False,
        "n_replicas": args.n_replicas, "n_queries_per_pair": args.n_queries,
        "pairs": meta_pairs, "created": datetime.now().isoformat(),
        "instrument": "HallucinationDetector runtime post-N8 (artefactos excluidos, vb_agree); análisis oficial offline bajo convención v4",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    results, done = [], set()
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        results = ck["results"]
        done = {(r["pair_id"], r["model"], r["query_id"], r["replica"]) for r in results}
        logger.info("Resume: %d hechas", len(done))

    detector = HallucinationDetector(use_nli=True)
    _ = detector.nli_model

    def save():
        ckpt_path.write_text(json.dumps(
            {"results": results, "updated": datetime.now().isoformat()},
            indent=1, ensure_ascii=False), encoding="utf-8")

    current_tag, llm = None, None
    todo = [t for t in plan if (t[0], t[2], t[3], t[4]) not in done]
    t0 = time.perf_counter()
    for i, (pid, scenario, tag, qid, rep) in enumerate(todo, 1):
        if tag != current_tag:
            logger.info("=== MODELO %s (caché OFF) ===", tag)
            llm = LLMManager(provider="ollama", model=tag,
                             cache_enabled=False, seed=SEED)
            current_tag = tag
        question = questions[qid]
        rids = contexts[scenario][qid]
        qtype = qp.process(question).query_type
        prompt, sysp, chunk_dicts = build_prompt(scenario, question, rids,
                                                 index, qtype, P)
        t = time.perf_counter()
        resp = llm.generate(prompt=prompt, system_prompt=sysp, temperature=0.0,
                            config_name=f"h5|{model_label(tag)}|{scenario}|rep{rep}")
        report = detector.check(resp.text, chunk_dicts)
        results.append({
            "pair_id": pid, "scenario": scenario, "model": tag,
            "model_label": model_label(tag), "query_id": qid, "replica": rep,
            "answer": resp.text,
            "faithfulness": report.faithfulness_score,
            "hallucination_rate": report.hallucination_rate,
            "total_claims": report.total_claims,
            "supported_claims": report.supported_claims,
            "method": getattr(report, "method", None),
            "class_v2": classify_response(resp.text),
            "latency_ms": resp.latency_ms or (time.perf_counter() - t) * 1000,
            "tokens_output": resp.tokens_output,
            "timestamp": datetime.now().isoformat(),
        })
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            save()
            el = time.perf_counter() - t0
            eta_h = el / i * (len(todo) - i) / 3600
            logger.info("[%d/%d] eta=%.1fh (%.1fs/gen)", i, len(todo), eta_h, el / i)
    save()
    logger.info("COMPLETO: %d generaciones en %s", len(results), ckpt_path)


if __name__ == "__main__":
    main()
