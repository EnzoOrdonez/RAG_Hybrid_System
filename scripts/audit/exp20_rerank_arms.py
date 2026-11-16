"""exp20: brazos de control con el mismo reranker (recuperación sola, sin LLM ni NLI).

Pregunta: la evidencia firmada compara «híbrido + reranker» con «denso sin reranker» y con
«léxico sin reranker». ¿Qué parte de la ventaja es de la fusión y qué parte del reranker?

Brazos, sobre las 194 consultas y con los mismos objetos que `scripts/run_retrieval_only.py`:
  - los cuatro sistemas de exp11 (para comprobar que hoy se reproducen los ids firmados);
  - denso top-50 + reranker;  léxico top-50 + reranker;
  - cota de referencia: reranker sobre la unión de los tres pools.

Modelos que SÍ carga (autorizados por Enzo el 2026-11-15, locales, sin descargas):
  el encoder denso `bge-large-en-v1.5` y el cross-encoder `ms-marco-MiniLM-L-12-v2`.
No carga Ollama, NLI ni HHEM. La corrida de recuperación no usa el oráculo independiente.

Tres modos:
  (sin opciones)     corre los brazos y guarda results.json;
  --grade            puntúa el top-5 de los siete brazos con el oráculo independiente
                     `BAAI/bge-reranker-large`, leído de una copia local (`--oracle-path`), sin red;
  --analyze          no carga modelos: lee lo anterior y escribe analysis.md y analysis.json.

Salida: output/audit/exp20_rerank_arms_2026-11-15/ (nunca experiments/results). NO es evidencia firmada.

Uso:  python scripts/audit/exp20_rerank_arms.py [--max-queries N | --grade --oracle-path DIR | --analyze]
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "output" / "audit" / "exp20_rerank_arms_2026-11-15"
SEED = 42
ORACLE_MODEL = "BAAI/bge-reranker-large"
ORACLE_FILE = "oracle_scores__bge-reranker-indep.json"
GRADE_BLOCK = 64


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_retrieval_only", ROOT / "scripts" / "run_retrieval_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scored(reranker, question, candidates):
    """Reranquea una COPIA del pool (rerank muta los objetos) y devuelve [(id, puntaje)] ordenado."""
    pool = [copy.copy(c) for c in candidates]
    if not pool:
        return []
    ranked = reranker.rerank(question, pool, top_k=len(pool))
    return [(r.chunk_id, float(r.score)) for r in ranked]


def _paired(a, b):
    """Diferencia pareada a - b: media, d_z, p de Wilcoxon y conteo de victorias, empates y derrotas."""
    import numpy as np
    from scipy import stats
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    spread = diff.std(ddof=1) if len(diff) > 1 else 0.0
    p_value = float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.0
    return {"mean": float(diff.mean()), "dz": float(diff.mean() / spread) if spread else 0.0, "p": p_value,
            "wins": int((diff > 1e-9).sum()), "ties": int((np.abs(diff) <= 1e-9).sum()),
            "losses": int((diff < -1e-9).sum()), "n": int(len(diff))}


def grade(oracle_path: Path) -> int:
    """Puntúa con el oráculo independiente el top-5 de los siete brazos y guarda los puntajes.

    Misma llamada que `scripts/compute_retrieval_metrics.py`: `CrossEncoder(modelo).predict((pregunta, texto))`
    con la activación por defecto. Solo cambia de dónde se lee el modelo (ruta local, sin red) y que los
    pares se ordenan por longitud para acelerar. Guarda avances parciales: se puede reanudar.
    """
    import hashlib

    from sentence_transformers import CrossEncoder

    payload = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    chunk_map = json.loads((ROOT / "data" / "indices" / "chunk_map_bge-large_adaptive_500.json")
                           .read_text(encoding="utf-8"))
    out_path = OUT_DIR / ORACLE_FILE
    scores = json.loads(out_path.read_text(encoding="utf-8"))["scores"] if out_path.exists() else {}
    pending = [(r["query_id"], r["question"], cid)
               for r in payload["results"]
               for cid in sorted({c for ids in r["arms"].values() for c in ids})
               if cid not in scores.get(r["query_id"], {})]
    pending.sort(key=lambda item: len(chunk_map[item[2]]["text"]))
    digest = hashlib.sha256()
    with open(oracle_path / "model.safetensors", "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)

    def save(done: bool) -> None:
        body = {"oracle_model": ORACLE_MODEL, "model_sha256": digest.hexdigest(), "complete": done,
                "status": "NO FIRMADO. Corrida de auditoría.", "scores": scores}
        out_path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"[exp20] oráculo {ORACLE_MODEL}: {len(pending)} pares pendientes", flush=True)
    if pending:
        model = CrossEncoder(str(oracle_path), device="cpu")
        started = time.perf_counter()
        for start in range(0, len(pending), GRADE_BLOCK):
            block = pending[start:start + GRADE_BLOCK]
            values = model.predict([(question, chunk_map[cid]["text"]) for _, question, cid in block],
                                   batch_size=8, show_progress_bar=False)
            for (qid, _, cid), value in zip(block, values):
                scores.setdefault(qid, {})[cid] = float(value)
            save(done=False)
            print(f"[exp20] {start + len(block)}/{len(pending)} pares, {time.perf_counter() - started:.0f} s", flush=True)
    save(done=True)
    print(f"[exp20] puntajes guardados en {out_path}")
    return 0


def _independent_section(rows, oracle, arms, mapping, log, table, data, failures) -> None:
    """NDCG@5 con el oráculo independiente: reproduce exp11 y califica los brazos de control."""
    import numpy as np

    spec = importlib.util.spec_from_file_location("fable_concept_probes", ROOT / "scripts" / "audit" / "fable_concept_probes.py")
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    signed = json.loads((ROOT / "experiments" / "results" / "exp11_retrieval194_fullrerank" /
                         "retrieval_metrics__bge-reranker-indep.json").read_text(encoding="utf-8"))["systems"]
    n = len(rows)
    four, seven, probability = {a: [] for a in mapping}, {a: [] for a in arms}, {a: [] for a in arms}
    for r in rows:
        scores = oracle["scores"][r["query_id"]]
        pool4 = {cid: scores[cid] for arm in mapping for cid in r["arms"][arm]}
        pool7 = {cid: scores[cid] for arm in arms for cid in r["arms"][arm]}
        for arm in arms:
            seven[arm].append(probes.ndcg_graded(r["arms"][arm], pool7))
            probability[arm].append(float(np.mean([scores[cid] for cid in r["arms"][arm]])))
            if arm in mapping:
                four[arm].append(probes.ndcg_graded(r["arms"][arm], pool4))
    log("## 6. NDCG@5 con el oráculo independiente (BAAI/bge-reranker-large, copia local verificada por SHA-256)")
    log()
    worst = max(abs(float(np.mean(four[arm])) - signed[name]["ndcg@5_mean"]) for arm, name in mapping.items())
    ok = worst < 5e-4
    log(f"- [{'OK' if ok else 'FALLO'}] reproduce el NDCG@5 independiente firmado de exp11 con el pool de cuatro "
        f"sistemas: desviación máxima {worst:.1e}.")
    if not ok:
        failures.append("reproducción del NDCG independiente firmado")
    log()
    table(["Brazo", "NDCG@5, pool de 4 (firmado)", "NDCG@5, pool de 7", "Probabilidad media de relevancia del top-5"],
          [[arm, f"{np.mean(four[arm]):.4f}" if arm in four else "no existe", f"{np.mean(seven[arm]):.4f}",
            f"{np.mean(probability[arm]):.4f}"] for arm in arms])
    templates = [probes.template_of(r["question"]) for r in rows]
    body = []
    for a, b in (("hibrido_rerank", "denso_rerank"), ("hibrido_rerank", "lexico_rerank"), ("denso_rerank", "denso"),
                 ("lexico_rerank", "lexico"), ("hibrido_rerank", "hibrido_pre"), ("hibrido_pre", "denso"),
                 ("union_rerank", "hibrido_rerank")):
        res = _paired(seven[a], seven[b])
        res["p_template"] = probes.cluster_mean_test([x - y for x, y in zip(seven[a], seven[b])], templates)["p_cluster"]
        data[f"indep_ndcg7_{a}_minus_{b}"] = res
        body.append([f"{a} − {b}", f"{res['mean']:+.4f}", f"{res['dz']:+.2f}", f"{res['p']:.2g}",
                     f"{res['p_template']:.2g}", f"{res['wins']}/{res['ties']}/{res['losses']}"])
    table(["A − B (NDCG@5, pool de 7)", "Diferencia media", "d_z", "p (Wilcoxon)", "p agrupando por plantilla",
           "A gana / empate / A pierde"], body)
    body = []
    for label, wanted in (("con filtro de proveedor", True), ("sin filtro", False)):
        index = [i for i, r in enumerate(rows) if bool(r["provider_filter"]) == wanted]
        res = _paired([seven["hibrido_rerank"][i] for i in index], [seven["denso_rerank"][i] for i in index])
        data[f"indep_filter_{wanted}"] = res
        body.append([label, len(index), f"{res['mean']:+.4f}", f"{res['p']:.2g}", f"{res['wins']}/{res['ties']}/{res['losses']}"])
    table(["Consultas", "n", "híbrido+reranker − denso+reranker", "p (Wilcoxon)", "gana / empate / pierde"], body)
    # Coste de los primeros resultados densos que la fusión deja fuera del top-5.
    dropped = [r for r in rows if r["arms"]["denso"][0] not in r["arms"]["hibrido_pre"]]
    if dropped:
        top1 = [oracle["scores"][r["query_id"]][r["arms"]["denso"][0]] for r in dropped]
        fused_top1 = [oracle["scores"][r["query_id"]][r["arms"]["hibrido_pre"][0]] for r in dropped]
        fused_mean = [float(np.mean([oracle["scores"][r["query_id"]][c] for c in r["arms"]["hibrido_pre"]])) for r in dropped]
        versus_top1, versus_mean = _paired(top1, fused_top1), _paired(top1, fused_mean)
        in_pool = sum(1 for r in dropped if r["arms"]["denso"][0] in set(r["pools"]["hibrido"]))
        rescued = sum(1 for r in dropped if r["arms"]["denso"][0] in r["arms"]["hibrido_rerank"])
        log(f"- La fusión deja fuera de su top-5 el primer resultado denso en {len(dropped)}/{n} consultas. Según el "
            f"oráculo independiente, ese fragmento tiene una probabilidad media de relevancia de {np.mean(top1):.3f}; "
            f"el primero de la lista fusionada, {np.mean(fused_top1):.3f} (diferencia {versus_top1['mean']:+.3f}, "
            f"p = {versus_top1['p']:.2g}); la media del top-5 fusionado, {np.mean(fused_mean):.3f} "
            f"(diferencia {versus_mean['mean']:+.3f}, p = {versus_mean['p']:.2g}). "
            f"De esos {len(dropped)}, {in_pool} seguían en el pool de 50 y el reranker devolvió {rescued} al top-5 final.")
        log()
        data["dropped_dense_top1"] = {"queries": len(dropped), "mean_prob_dropped": float(np.mean(top1)),
                                      "mean_prob_fused_top1": float(np.mean(fused_top1)),
                                      "mean_prob_fused_top5": float(np.mean(fused_mean)),
                                      "vs_fused_top1": versus_top1, "vs_fused_top5": versus_mean,
                                      "still_in_pool": in_pool, "rescued_by_reranker": rescued}
    # Acuerdo entre los dos oráculos sobre los mismos pares: si es bajo, «relevante» depende del juez.
    from scipy import stats
    agreement = []
    for r in rows:
        scores = oracle["scores"][r["query_id"]]
        ids = sorted(scores)
        rho = stats.spearmanr([scores[c] for c in ids], [r["reranker_scores"][c] for c in ids]).statistic
        if rho == rho:
            agreement.append(float(rho))
    log(f"- Acuerdo entre oráculos (Spearman dentro de cada consulta, ms-marco frente a bge-reranker, sobre el top-5 "
        f"de los siete brazos): media {np.mean(agreement):+.3f}, mediana {np.median(agreement):+.3f}, "
        f"por debajo de 0,5 en {sum(1 for x in agreement if x < 0.5)}/{len(agreement)} consultas.")
    log()
    data["oracle_agreement"] = {"mean": float(np.mean(agreement)), "median": float(np.median(agreement)),
                                "below_half": sum(1 for x in agreement if x < 0.5), "n": len(agreement)}
    # Afinidad de cada oráculo con cada pierna: con el juicio del otro oráculo fijo, ¿puntúa más alto lo que
    # trajo la pierna densa (misma familia BGE que el oráculo «independiente») o lo que trajo BM25?
    fixed_effects = probes._load_why().fixed_effects_ols
    y_bge, y_ms, x_bge, x_ms, groups = [], [], [], [], []
    for r in rows:
        scores = oracle["scores"][r["query_id"]]
        ids = sorted(scores)
        rank_bge = stats.rankdata([scores[c] for c in ids]) / len(ids)
        rank_ms = stats.rankdata([r["reranker_scores"][c] for c in ids]) / len(ids)
        dense_pool, lexical_pool = set(r["pools"]["denso"]), set(r["pools"]["lexico"])
        for i, cid in enumerate(ids):
            membership = [float(cid in dense_pool), float(cid in lexical_pool)]
            y_bge.append(rank_bge[i]); x_bge.append([rank_ms[i]] + membership)
            y_ms.append(rank_ms[i]); x_ms.append([rank_bge[i]] + membership)
            groups.append(r["query_id"])
    body = []
    for label, y, x in (("bge-reranker (el «independiente»)", y_bge, x_bge), ("ms-marco (el del pipeline)", y_ms, x_ms)):
        fit = fixed_effects(y, np.array(x), groups, groups)
        body.append([label, f"{fit['beta'][0]:+.3f}", f"{fit['beta'][1]:+.3f} (p = {fit['p'][1]:.2g})",
                     f"{fit['beta'][2]:+.3f} (p = {fit['p'][2]:.2g})"])
        data.setdefault("oracle_affinity", {})[label] = {"other_oracle": float(fit["beta"][0]),
                                                         "in_dense_pool": float(fit["beta"][1]), "p_dense": float(fit["p"][1]),
                                                         "in_lexical_pool": float(fit["beta"][2]), "p_lexical": float(fit["p"][2])}
    log("Afinidad de cada oráculo con cada pierna. Variable explicada: puesto percentil del fragmento según ese "
        "oráculo dentro de su consulta. Controles: el puesto según el otro oráculo y efectos fijos de consulta.")
    log()
    table(["Oráculo", "β puesto según el otro oráculo", "β «estaba en el top-50 denso»", "β «estaba en el top-50 léxico»"], body)
    data["indep_ndcg"] = {"pool4": {a: float(np.mean(v)) for a, v in four.items()},
                          "pool7": {a: float(np.mean(v)) for a, v in seven.items()},
                          "probability": {a: float(np.mean(v)) for a, v in probability.items()},
                          "max_dev_vs_signed": float(worst), "n": n}


def analyze() -> int:
    """Lee results.json y escribe analysis.md. No carga ningún modelo."""
    import hashlib
    import re
    from collections import Counter

    import numpy as np

    payload = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    rows = payload["results"]
    signed = json.loads((ROOT / "experiments" / "results" / "exp11_retrieval194_fullrerank" / "results.json")
                        .read_text(encoding="utf-8"))["configs"]
    signed_ids = {name: {r["query_id"]: r["retrieved_ids"] for r in cfg["results"]} for name, cfg in signed.items()}
    chunk_map = json.loads((ROOT / "data" / "indices" / "chunk_map_bge-large_adaptive_500.json")
                           .read_text(encoding="utf-8"))
    text_hash = {cid: hashlib.sha1(re.sub(r"\s+", " ", c["text"].strip().lower()).encode()).hexdigest()
                 for cid, c in chunk_map.items()}
    out, data, failures = [], {}, []

    def log(text=""):
        out.append(text)

    def table(header, body):
        log("| " + " | ".join(header) + " |")
        log("|" + "---|" * len(header))
        for line in body:
            log("| " + " | ".join(str(cell) for cell in line) + " |")
        log()

    n = len(rows)
    log("# exp20: brazos de control con el mismo reranker")
    log()
    log(f"Corrida del {payload['timestamp']}, {n} consultas, {payload['elapsed_s']} s. {payload['status']}")
    log(f"Modelos cargados: {'; '.join(payload['models_loaded'])}. No cargados: {'; '.join(payload['models_not_loaded'])}.")
    log()
    log("**Cómo leerlo:** las secciones 4 y 5 usan el puntaje del propio reranker, que es circular: sirve para "
        "comparar qué pool le da al reranker mejores candidatos según su propio criterio, no para decir qué "
        "brazo recupera mejor. La calidad se lee en la sección 6 (oráculo independiente), si está presente.")
    log()

    log("## 1. Reproducción de exp11 con modelos reales")
    log()
    mapping = {"lexico": "RAG Lexico (BM25)", "denso": "RAG Semantico (Dense)",
               "hibrido_pre": "RAG Hibrido (pre-rerank RRF)", "hibrido_rerank": "RAG Hibrido Propuesto"}
    body = []
    for arm, name in mapping.items():
        same = sum(1 for r in rows if r["arms"][arm] == signed_ids[name][r["query_id"]])
        same_set = sum(1 for r in rows if set(r["arms"][arm]) == set(signed_ids[name][r["query_id"]]))
        body.append([arm, f"{same}/{n}", f"{same_set}/{n}"])
        data[f"reproduced_{arm}"] = [same, same_set, n]
        if same_set != n:
            failures.append(f"reproducción de {arm}")
    table(["Brazo", "Top-5 idéntico al firmado (ids y orden)", "Mismo conjunto"], body)

    log("## 2. Pools que ve el reranker")
    log()
    body = []
    for pool in ("lexico", "denso", "hibrido"):
        sizes = np.array([len(r["pools"][pool]) for r in rows])
        body.append([pool, f"{sizes.mean():.1f}", int(sizes.min()), int((sizes < 50).sum()), int((sizes < 25).sum())])
        data[f"pool_{pool}"] = {"mean": float(sizes.mean()), "min": int(sizes.min()), "below_50": int((sizes < 50).sum())}
    table(["Pool", "Tamaño medio", "Mínimo", "Consultas con menos de 50", "Con menos de 25"], body)
    filtered = [r for r in rows if r["provider_filter"]]
    hybrid_filtered = np.mean([len(r["pools"]["hibrido"]) for r in filtered]) if filtered else float("nan")
    dense_filtered = np.mean([len(r["pools"]["denso"]) for r in filtered]) if filtered else float("nan")
    log(f"- Consultas con filtro de proveedor: {len(filtered)}/{n}. En ellas el pool híbrido tiene {hybrid_filtered:.1f} "
        f"candidatos de media y el denso {dense_filtered:.1f}. Los brazos base piden 100 candidatos antes de filtrar "
        "y el híbrido fusiona 50 y filtra después, sin reponer.")
    log(f"- Consultas en las que la consulta densa difiere del texto original: "
        f"{sum(1 for r in rows if r['semantic_query_differs'])}/{n}.")
    log(f"- Desviación máxima al puntuar dos veces el mismo par: {max(r['rescoring_spread'] for r in rows):.2e}.")
    log()
    data["filtered"] = {"n": len(filtered), "hybrid_pool_mean": float(hybrid_filtered), "dense_pool_mean": float(dense_filtered)}

    log("## 3. Híbrido con reranker frente a denso con el mismo reranker")
    log()
    overlap = Counter(len(set(r["arms"]["hibrido_rerank"]) & set(r["arms"]["denso_rerank"])) for r in rows)
    table(["Fragmentos en común en el top-5"] + [str(i) for i in range(6)],
          [["Consultas"] + [overlap.get(i, 0) for i in range(6)]])
    mean_overlap = sum(k * v for k, v in overlap.items()) / n
    in_dense_pool = np.mean([len(set(r["arms"]["hibrido_rerank"]) & set(r["pools"]["denso"])) / max(len(r["arms"]["hibrido_rerank"]), 1) for r in rows])
    in_hybrid_pool = np.mean([len(set(r["arms"]["denso_rerank"]) & set(r["pools"]["hibrido"])) / max(len(r["arms"]["denso_rerank"]), 1) for r in rows])
    log(f"- Solapamiento medio: {mean_overlap:.2f} de 5. Listas idénticas: {overlap.get(5, 0)}/{n}.")
    log(f"- Del top-5 del híbrido, el {in_dense_pool:.1%} ya estaba en el pool denso de 50. "
        f"Del top-5 del denso con reranker, el {in_hybrid_pool:.1%} estaba en el pool híbrido.")
    log()
    data["hybrid_vs_dense_rerank"] = {"overlap": dict(sorted(overlap.items())), "mean_overlap": mean_overlap,
                                      "hybrid_top5_in_dense_pool": float(in_dense_pool),
                                      "dense_rerank_top5_in_hybrid_pool": float(in_hybrid_pool)}

    log("## 4. Qué pool le sirve mejor al reranker (medida circular, ver límite)")
    log()
    arms = ("lexico", "denso", "hibrido_pre", "lexico_rerank", "denso_rerank", "hibrido_rerank", "union_rerank")
    utility = {arm: [float(np.mean([r["reranker_scores"][cid] for cid in r["arms"][arm]])) if r["arms"][arm] else float("nan")
                     for r in rows] for arm in arms}
    lengths = {arm: float(np.mean([np.mean([len(chunk_map[cid]["text"]) for cid in r["arms"][arm]])
                                   for r in rows if r["arms"][arm]])) for arm in arms}
    duplicated = {arm: sum(1 for r in rows if len({text_hash[cid] for cid in r["arms"][arm]}) < len(r["arms"][arm]))
                  for arm in arms}
    table(["Brazo", "Puntaje medio del reranker en el top-5 (logit)", "Longitud media del fragmento (caracteres)",
           "Consultas con texto repetido en el top-5"],
          [[arm, f"{np.nanmean(utility[arm]):+.3f}", f"{lengths[arm]:.0f}", f"{duplicated[arm]}/{n}"] for arm in arms])
    body = []
    for a, b in (("hibrido_rerank", "denso_rerank"), ("hibrido_rerank", "lexico_rerank"),
                 ("union_rerank", "hibrido_rerank"), ("union_rerank", "denso_rerank")):
        keep = [i for i in range(n) if not (np.isnan(utility[a][i]) or np.isnan(utility[b][i]))]
        res = _paired([utility[a][i] for i in keep], [utility[b][i] for i in keep])
        data[f"utility_{a}_minus_{b}"] = res
        body.append([f"{a} − {b}", f"{res['mean']:+.3f}", f"{res['dz']:+.2f}", f"{res['p']:.2g}",
                     f"{res['wins']}/{res['ties']}/{res['losses']}"])
    table(["A − B", "Diferencia media", "d_z", "p (Wilcoxon)", "A gana / empate / A pierde"], body)
    data["utility_mean"] = {arm: float(np.nanmean(utility[arm])) for arm in arms}
    data["length_mean"] = lengths
    data["duplicated_top5"] = duplicated

    log("## 5. Lo que la fusión le quita al reranker")
    log()
    lost_queries = lost_docs = lost_dense_only = lost_lexical_only = lost_both = 0
    for r in rows:
        missing = [cid for cid in r["arms"]["union_rerank"] if cid not in set(r["pools"]["hibrido"])]
        if missing:
            lost_queries += 1
        for cid in missing:
            lost_docs += 1
            in_dense, in_lexical = cid in r["pools"]["denso"], cid in r["pools"]["lexico"]
            lost_both += in_dense and in_lexical
            lost_dense_only += in_dense and not in_lexical
            lost_lexical_only += in_lexical and not in_dense
    log(f"- En {lost_queries}/{n} consultas, al menos un fragmento que el reranker pondría en su top-5 sobre la "
        f"unión de pools no llegó al pool híbrido. Son {lost_docs} fragmentos de {5 * n}: {lost_dense_only} estaban "
        f"solo en el pool denso, {lost_lexical_only} solo en el léxico y {lost_both} en ambos.")
    composition = Counter()
    for r in rows:
        for cid in r["pools"]["hibrido"]:
            composition[(cid in r["pools"]["denso"], cid in r["pools"]["lexico"])] += 1
    total = sum(composition.values())
    log(f"- Composición del pool híbrido: {composition[(True, True)] / total:.1%} en ambos pools base, "
        f"{composition[(True, False)] / total:.1%} solo en el denso, {composition[(False, True)] / total:.1%} solo "
        f"en el léxico, {composition[(False, False)] / total:.1%} en ninguno.")
    log()
    data["fusion_loss"] = {"queries": lost_queries, "docs": lost_docs, "dense_only": lost_dense_only,
                           "lexical_only": lost_lexical_only, "both": lost_both}
    oracle_path = OUT_DIR / ORACLE_FILE
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.exists() else None
    if oracle and oracle["complete"]:
        _independent_section(rows, oracle, arms, mapping, log, table, data, failures)

    log("## Comprobaciones")
    log()
    log(f"- Fallos: {len(failures)}" + (f" ({'; '.join(failures)})" if failures else ""))
    text = "\n".join(out) + "\n"
    (OUT_DIR / "analysis.md").write_text(text, encoding="utf-8", newline="\n")
    (OUT_DIR / "analysis.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    return 1 if failures else 0


def main() -> int:
    from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds
    ensure_hashseed_at_startup(SEED)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--analyze", action="store_true",
                        help="no ejecuta nada: lee results.json y escribe analysis.md")
    parser.add_argument("--grade", action="store_true",
                        help="puntúa el top-5 de los brazos con el oráculo independiente (requiere --oracle-path)")
    parser.add_argument("--oracle-path", type=Path, default=None,
                        help="carpeta local de BAAI/bge-reranker-large (no se descarga nada)")
    args = parser.parse_args()
    if args.analyze:
        return analyze()
    if args.grade:
        if args.oracle_path is None or not (args.oracle_path / "model.safetensors").exists():
            parser.error("--grade necesita --oracle-path con una copia local del oráculo")
        set_all_seeds(SEED)
        return grade(args.oracle_path)
    set_all_seeds(SEED)

    from src.pipeline.pipeline_config import BASELINE_LEXICAL, BASELINE_SEMANTIC, PROPOSED_HYBRID
    from src.pipeline.rag_pipeline import load_hybrid_index
    runner = _load_runner()
    queries = runner.load_queries(args.max_queries)
    started = time.perf_counter()
    index = load_hybrid_index(embedding_model="bge-large", chunking_strategy="adaptive", chunk_size=500)
    processor, bm25, dense, hybrid, reranker = runner.build_systems(index)
    final_k = PROPOSED_HYBRID.final_top_k

    rows = []
    for number, query in enumerate(queries, 1):
        question = query["question"]
        processed = processor.process(question)
        pool_bm25 = bm25.search(question, top_k=BASELINE_LEXICAL.retrieval_top_k,
                                use_expansion=BASELINE_LEXICAL.query_expansion)
        pool_dense = dense.search(question, top_k=BASELINE_SEMANTIC.retrieval_top_k)
        pool_hybrid = hybrid.search(question, top_k=PROPOSED_HYBRID.retrieval_top_k,
                                    top_k_candidates=PROPOSED_HYBRID.retrieval_top_k, use_reranker=False)
        ranked_bm25 = _scored(reranker, question, pool_bm25)
        ranked_dense = _scored(reranker, question, pool_dense)
        ranked_hybrid = _scored(reranker, question, pool_hybrid)
        # Unión de los tres pools con el puntaje del reranker. Un fragmento presente en dos pools
        # se puntúa dos veces (en lotes distintos); se guarda la desviación máxima observada.
        merged, spread = {}, 0.0
        for ranked in (ranked_hybrid, ranked_dense, ranked_bm25):
            for cid, score in ranked:
                if cid in merged:
                    spread = max(spread, abs(merged[cid] - score))
                else:
                    merged[cid] = score
        union_rank = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        rows.append({
            "query_id": query["query_id"], "question": question,
            "provider_filter": sorted(processed.provider_filter) if processed.provider_filter else [],
            "semantic_query_differs": processed.semantic_query != question,
            "pools": {"lexico": [r.chunk_id for r in pool_bm25], "denso": [r.chunk_id for r in pool_dense],
                      "hibrido": [r.chunk_id for r in pool_hybrid]},
            "arms": {
                "lexico": [r.chunk_id for r in pool_bm25[:final_k]],
                "denso": [r.chunk_id for r in pool_dense[:final_k]],
                "hibrido_pre": [r.chunk_id for r in pool_hybrid[:final_k]],
                "hibrido_rerank": [cid for cid, _ in ranked_hybrid[:final_k]],
                "denso_rerank": [cid for cid, _ in ranked_dense[:final_k]],
                "lexico_rerank": [cid for cid, _ in ranked_bm25[:final_k]],
                "union_rerank": [cid for cid, _ in union_rank[:final_k]],
            },
            "reranker_scores": merged,
            "rescoring_spread": spread,
        })
        if number % 25 == 0:
            print(f"[exp20] {number}/{len(queries)} consultas", flush=True)

    payload = {
        "experiment_id": "exp20_rerank_arms",
        "status": "NO FIRMADO. Corrida de auditoría en output/audit; no es evidencia de experiments/results.",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "models_loaded": ["BAAI/bge-large-en-v1.5 (encoder denso)",
                          "cross-encoder/ms-marco-MiniLM-L-12-v2 (reranker)"],
        "models_not_loaded": ["Ollama", "NLI", "HHEM", "BAAI/bge-reranker-large (oráculo independiente; no está en disco)"],
        "configs": {"lexico": BASELINE_LEXICAL.name, "denso": BASELINE_SEMANTIC.name, "hibrido": PROPOSED_HYBRID.name},
        "retrieval_top_k": PROPOSED_HYBRID.retrieval_top_k, "final_top_k": final_k,
        "num_queries": len(rows), "elapsed_s": round(time.perf_counter() - started, 1),
        "results": rows,
    }
    if args.max_queries is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
        print(f"[exp20] guardado en {OUT_DIR / 'results.json'}")
    else:
        print(f"[exp20] prueba de humo con {len(rows)} consultas: no se guarda nada")
        for row in rows:
            print(row["query_id"], {arm: ids[:2] for arm, ids in row["arms"].items()})
    print(f"[exp20] {len(rows)} consultas en {payload['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
