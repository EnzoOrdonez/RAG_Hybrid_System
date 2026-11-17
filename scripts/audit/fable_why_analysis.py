"""Auditoría Fable: por qué el híbrido no siempre gana y por qué unos modelos puntúan más.

Todo sale de la evidencia firmada, en solo lectura, y de los índices en disco. No carga ningún
modelo: ni Ollama, ni el encoder denso, ni el cross-encoder, ni NLI, ni HHEM. BM25 se recalcula
porque es aritmética sobre un índice ya construido.

Qué hace:
  1. Integridad de los índices (FAISS, BM25 y mapa de fragmentos) sin embeber ninguna consulta.
  2. Recuperación (exp11): reproduce el brazo léxico con el índice real, demuestra la propiedad
     de dominancia por consenso de RRF con k=60 y profundidad 50, y mide qué hace la fusión con
     las listas firmadas.
  3. Generación (exp12): comprueba que el contexto es el mismo para los cuatro modelos y
     descompone de forma exacta cada diferencia de fidelidad en selección de consultas, número
     de afirmaciones apoyadas y número de afirmaciones emitidas.

Uso:
    python scripts/audit/fable_why_analysis.py            # imprime el reporte
    python scripts/audit/fable_why_analysis.py --write    # además lo guarda en output/audit/

Las funciones puras de la primera sección tienen tests en tests/test_fable_why_analysis.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
EXP11 = RESULTS / "exp11_retrieval194_fullrerank"
EXP12 = RESULTS / "exp12_matrix"
INDICES = ROOT / "data" / "indices"
REPORT_STEM = "fable_why_analysis_2026-11-15"

RRF_K = 60
POOL_DEPTH = 50
FINAL_K = 5
LEXICAL, DENSE = "RAG Lexico (BM25)", "RAG Semantico (Dense)"
HYBRID_POST, HYBRID_PRE = "RAG Hibrido Propuesto", "RAG Hibrido (pre-rerank RRF)"
RAG_SCENARIOS = ("lexico", "denso", "hibrido")
VERIFIERS = {"small": "faithfulness_rescore_v3__small__vb_agree.json",
             "base": "faithfulness_rescore_v3__base__vb_agree.json"}


# =============================================================================
# 1. Funciones puras (con tests)
# =============================================================================
def rrf_score(ranks: Iterable[Optional[int]], k: int = RRF_K) -> float:
    """Puntaje RRF de un documento. `ranks` son posiciones 1-based; None = ausente de esa lista."""
    return sum(1.0 / (k + r) for r in ranks if r is not None)


def consensus_dominates(k: int, depth: int) -> bool:
    """True si TODO documento presente en las dos listas supera a TODO documento de una sola.

    El peor documento de consenso está último en ambas listas: 2/(k+depth). El mejor de una
    sola lista está primero: 1/(k+1). La dominancia es estricta si 2/(k+depth) > 1/(k+1), es
    decir, si depth < k + 2.
    """
    return 2.0 / (k + depth) > 1.0 / (k + 1)


def rrf_fuse(leg_a: Sequence[str], leg_b: Sequence[str], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Reimplementación independiente de la fusión RRF (misma regla de empates que el código).

    Empates: orden de primera aparición (lista A y después lista B), porque el código ordena
    con un sort estable sobre un dict poblado en ese orden.
    """
    order: Dict[str, int] = {}
    for cid in list(leg_a) + list(leg_b):
        order.setdefault(cid, len(order))
    rank_a = {cid: i + 1 for i, cid in enumerate(leg_a)}
    rank_b = {cid: i + 1 for i, cid in enumerate(leg_b)}
    scored = [(cid, rrf_score((rank_a.get(cid), rank_b.get(cid)), k)) for cid in order]
    return sorted(scored, key=lambda item: (-item[1], order[item[0]]))


def selection_within(fa: Dict[str, float], fb: Dict[str, float]) -> Dict[str, float]:
    """Descompone media(A) - media(B) cuando A y B puntúan conjuntos distintos de consultas.

    Identidad exacta, con C = consultas comunes:
        media_A - media_B = [media_A(C) - media_B(C)]            (dentro: mismas consultas)
                          + [media_A - media_A(C)]               (selección de A)
                          - [media_B - media_B(C)]               (selección de B)
    """
    common = sorted(set(fa) & set(fb))
    mean_a, mean_b = _mean(fa.values()), _mean(fb.values())
    mean_a_c, mean_b_c = _mean(fa[q] for q in common), _mean(fb[q] for q in common)
    return {
        "n_a": len(fa), "n_b": len(fb), "n_common": len(common),
        "mean_a": mean_a, "mean_b": mean_b, "total": mean_a - mean_b,
        "within": mean_a_c - mean_b_c,
        "selection_a": mean_a - mean_a_c, "selection_b": mean_b - mean_b_c,
        "selection": (mean_a - mean_a_c) - (mean_b - mean_b_c),
    }


def ratio_shapley(s_a: float, g_a: float, s_b: float, g_b: float) -> Tuple[float, float]:
    """Reparte s_a/g_a - s_b/g_b entre el numerador (apoyadas) y el denominador (emitidas).

    Promedio de los dos órdenes de sustitución (valor de Shapley con dos factores):
        numerador   = (s_a - s_b) * (1/g_a + 1/g_b) / 2
        denominador = (s_a + s_b) * (1/g_a - 1/g_b) / 2
    La suma es exactamente s_a/g_a - s_b/g_b.
    """
    numerator = (s_a - s_b) * (1.0 / g_a + 1.0 / g_b) / 2.0
    denominator = (s_a + s_b) * (1.0 / g_a - 1.0 / g_b) / 2.0
    return numerator, denominator


def mde_dz(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Mínimo |d_z| detectable en un contraste pareado bilateral (aproximación normal)."""
    if n <= 0:
        return float("nan")
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / math.sqrt(n))


def required_n(dz: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Pares necesarios para detectar un d_z dado (aproximación normal)."""
    if not dz:
        return float("inf")
    return float(math.ceil(((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / abs(dz)) ** 2))


def fixed_effects_ols(y: Sequence[float], x: np.ndarray, groups: Sequence, clusters: Sequence
                      ) -> Dict[str, np.ndarray]:
    """MCO con efectos fijos de grupo (se resta la media de cada grupo) y errores por clúster.

    Devuelve coeficientes, errores estándar robustos por clúster (sándwich con corrección
    G/(G-1)) y p bilateral por aproximación normal.
    """
    y_arr = np.asarray(y, dtype=float).copy()
    x_arr = np.asarray(x, dtype=float).copy()
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    index_by_group: Dict = defaultdict(list)
    for i, group in enumerate(groups):
        index_by_group[group].append(i)
    for idx in index_by_group.values():
        y_arr[idx] -= y_arr[idx].mean()
        x_arr[idx] -= x_arr[idx].mean(axis=0)
    bread = np.linalg.pinv(x_arr.T @ x_arr)
    beta = bread @ x_arr.T @ y_arr
    residual = y_arr - x_arr @ beta
    index_by_cluster: Dict = defaultdict(list)
    for i, cluster in enumerate(clusters):
        index_by_cluster[cluster].append(i)
    meat = np.zeros((x_arr.shape[1], x_arr.shape[1]))
    for idx in index_by_cluster.values():
        score = x_arr[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n_clusters = len(index_by_cluster)
    correction = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
    cov = correction * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, np.nan)
    return {"beta": beta, "se": se, "p": 2 * stats.norm.sf(np.abs(z)),
            "n": np.array(len(y_arr)), "n_groups": np.array(len(index_by_group)),
            "n_clusters": np.array(n_clusters)}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _dz(diffs: Sequence[float]) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return float("nan")
    return float(diffs.mean() / diffs.std(ddof=1))


def _wilcoxon_p(diffs: Sequence[float]) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 3 or not np.any(diffs):
        return 1.0
    return float(stats.wilcoxon(diffs).pvalue)


def _bh(p_values: Sequence[float]) -> List[float]:
    from statsmodels.stats.multitest import multipletests
    return [float(p) for p in multipletests(list(p_values), method="fdr_bh")[1]]


# =============================================================================
# 2. Reporte
# =============================================================================
class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.data: Dict = {}
        self.failures: List[str] = []

    def log(self, text: str = "") -> None:
        self.lines.append(text)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.log(f"- [{'OK ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def table(self, header: Sequence[str], rows: Iterable[Sequence]) -> None:
        self.log("| " + " | ".join(header) + " |")
        self.log("|" + "---|" * len(header))
        for row in rows:
            self.log("| " + " | ".join(str(cell) for cell in row) + " |")
        self.log()


def _n_needed(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "n/d"
    return "> 10 000" if value > 10_000 else str(int(value))


def _f(value: float, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


# =============================================================================
# 3. Integridad de los índices
# =============================================================================
def section_indices(rep: Report) -> Dict:
    import faiss
    from src.embedding.index.bm25_index import BM25Index

    rep.log("## 1. Integridad de los índices (sin embeber ninguna consulta)")
    rep.log()
    chunk_map = json.loads((INDICES / "chunk_map_bge-large_adaptive_500.json").read_text(encoding="utf-8"))
    mapping = json.loads((INDICES / "faiss_bge-large_adaptive_500.mapping.json").read_text(encoding="utf-8"))
    dense_ids = mapping["chunk_ids"] if isinstance(mapping, dict) else mapping
    index = faiss.read_index(str(INDICES / "faiss_bge-large_adaptive_500.index"))
    bm25 = BM25Index()
    bm25.load(str(INDICES / "bm25_adaptive_500.pkl"))

    rep.check("FAISS, BM25 y mapa tienen los mismos identificadores",
              set(dense_ids) == set(bm25.chunk_ids) == set(chunk_map),
              f"{len(dense_ids)} / {len(bm25.chunk_ids)} / {len(chunk_map)}")
    rep.check("sin identificadores repetidos",
              len(set(dense_ids)) == len(dense_ids) and len(set(bm25.chunk_ids)) == len(bm25.chunk_ids))
    rep.check("FAISS: un vector por identificador", index.ntotal == len(dense_ids),
              f"ntotal={index.ntotal}, dimensión={index.d}")
    rep.check("FAISS usa producto interno (coseno sobre vectores normalizados)",
              index.metric_type == faiss.METRIC_INNER_PRODUCT)

    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(index.ntotal, size=min(512, index.ntotal), replace=False))
    vectors = np.vstack([index.reconstruct(int(i)) for i in sample]).astype("float32")
    norms = np.linalg.norm(vectors, axis=1)
    rep.check("vectores normalizados (muestra de 512, semilla 42)",
              bool(np.all(np.abs(norms - 1.0) < 1e-3)), f"norma mín {norms.min():.5f}, máx {norms.max():.5f}")
    scores, neighbours = index.search(vectors, 1)
    self_hit = int(np.sum((neighbours[:, 0] == sample) | (np.abs(scores[:, 0] - norms ** 2) < 1e-4)))
    rep.check("cada vector de la muestra se recupera a sí mismo en la posición 1",
              self_hit == len(sample), f"{self_hit}/{len(sample)}")
    providers = Counter(chunk.get("cloud_provider", "") for chunk in chunk_map.values())
    rep.check("solo aws, azure y gcp en el corpus", set(providers) == {"aws", "azure", "gcp"},
              ", ".join(f"{k}={v}" for k, v in sorted(providers.items())))
    empty_docs = sum(1 for tokens in bm25.corpus_tokens if not tokens)
    rep.check("BM25: ningún documento sin tokens", empty_docs == 0, f"{empty_docs} vacíos")
    rep.log()
    rep.data["indices"] = {"chunks": len(dense_ids), "dimension": int(index.d),
                           "self_retrieval": [self_hit, int(len(sample))],
                           "providers": dict(providers)}
    return {"bm25": bm25, "chunk_map": chunk_map}


# =============================================================================
# 4. Recuperación (exp11)
# =============================================================================
class _LexicalOnlyIndex:
    """Lo mínimo de HybridIndex que necesita BM25Retriever. No tiene pierna densa."""

    def __init__(self, bm25, chunk_map: Dict[str, dict]) -> None:
        self.bm25_index, self.chunk_map = bm25, chunk_map

    def search_bm25(self, query: str, top_k: int = POOL_DEPTH) -> List[Tuple[str, float]]:
        ids, scores = self.bm25_index.search(query, top_k)
        return list(zip(ids, scores))

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self.chunk_map.get(chunk_id)


def section_retrieval(rep: Report, bm25, chunk_map: Dict[str, dict]) -> None:
    from src.pipeline.pipeline_config import BASELINE_LEXICAL
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.query_processor import QueryProcessor

    rep.log("## 2. Recuperación: qué hace la fusión RRF (exp11, 194 consultas)")
    rep.log()
    signed = json.loads((EXP11 / "results.json").read_text(encoding="utf-8"))["configs"]
    by_system = {name: {row["query_id"]: row for row in cfg["results"]} for name, cfg in signed.items()}
    query_ids = [row["query_id"] for row in signed[LEXICAL]["results"]]

    # --- 2.1 Teorema ---------------------------------------------------------
    rep.log("### 2.1 Propiedad demostrada: con k=60 y 50 candidatos por pierna, RRF es un filtro de consenso")
    rep.log()
    worst_consensus, best_single = rrf_score((POOL_DEPTH, POOL_DEPTH)), rrf_score((1, None))
    rep.log(f"- Peor documento presente en las dos listas (puesto 50 en ambas): 2/(60+50) = {worst_consensus:.6f}.")
    rep.log(f"- Mejor documento presente en una sola lista (puesto 1): 1/(60+1) = {best_single:.6f}.")
    rep.check("2/(k+N) > 1/(k+1) con k=60, N=50", consensus_dominates(RRF_K, POOL_DEPTH),
              "todo documento de consenso queda por encima de todo documento de una sola pierna")
    limit = next(depth for depth in range(1, 1000) if not consensus_dominates(RRF_K, depth))
    rep.log(f"- La propiedad vale mientras N < k + 2 = {RRF_K + 2}; deja de valer en N = {limit}.")
    rep.log("- Consecuencia: el top-5 fusionado se llena primero con la intersección de las dos listas "
            "de 50. El primer resultado denso entra solo si BM25 también lo tiene entre sus 50, o si "
            "la intersección (tras el filtro de proveedor) tiene menos de 5 documentos.")
    rep.log()

    # --- 2.2 Reproducción del brazo léxico -----------------------------------
    processor = QueryProcessor()
    lexical_index = _LexicalOnlyIndex(bm25, chunk_map)
    retriever = BM25Retriever(lexical_index, query_processor=processor)
    leg_rank: Dict[str, Dict[str, int]] = {}
    filtered: Dict[str, bool] = {}
    reproduced = 0
    for qid in query_ids:
        question = by_system[LEXICAL][qid]["question"]
        found = retriever.search(question, top_k=BASELINE_LEXICAL.retrieval_top_k,
                                 use_expansion=BASELINE_LEXICAL.query_expansion)
        ids = [r.chunk_id for r in found[:BASELINE_LEXICAL.final_top_k]]
        reproduced += ids == by_system[LEXICAL][qid]["retrieved_ids"]
        leg = lexical_index.search_bm25(question, POOL_DEPTH)
        leg_rank[qid] = {cid: i + 1 for i, (cid, _) in enumerate(leg)}
        filtered[qid] = bool(processor.process(question).provider_filter)
    rep.log("### 2.2 Prueba de extremo a extremo de la pierna léxica con el índice real")
    rep.log()
    rep.check("el top-5 léxico recalculado hoy coincide con el firmado en exp11",
              reproduced == len(query_ids), f"{reproduced}/{len(query_ids)} consultas idénticas (ids y orden)")
    n_filtered = sum(filtered.values())
    rep.log(f"- Consultas con filtro de proveedor activo: {n_filtered}/{len(query_ids)}. En ellas el top-5 "
            "denso firmado es el top-5 después del filtro, así que su puesto en la pierna cruda no se "
            "conoce; las comprobaciones de puntaje exacto usan solo las consultas sin filtro.")
    rep.log()

    # --- 2.3 Invariantes sobre las listas firmadas ---------------------------
    order_violations = score_violations = score_checked = 0
    for qid in query_ids:
        pre = by_system[HYBRID_PRE][qid]["retrieved_ids"]
        dense = by_system[DENSE][qid]["retrieved_ids"]
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense)}
        seen_dense_only = False
        for cid in pre:
            in_lexical = cid in leg_rank[qid]
            if not in_lexical:
                seen_dense_only = True
            elif seen_dense_only and not filtered[qid] and cid in dense_rank:
                order_violations += 1
        if not filtered[qid]:
            known = [(cid, rrf_score((leg_rank[qid].get(cid), dense_rank[cid])))
                     for cid in pre if cid in dense_rank]
            score_checked += len(known)
            score_violations += sum(1 for (_, s1), (_, s2) in zip(known, known[1:]) if s2 > s1 + 1e-12)
    rep.log("### 2.3 Las listas firmadas obedecen la aritmética de RRF")
    rep.log()
    rep.check("ningún documento solo-denso precede a un documento de consenso conocido",
              order_violations == 0, f"{order_violations} violaciones en {len(query_ids)} consultas")
    rep.check("los documentos con puntaje RRF exacto conocido aparecen en orden no creciente",
              score_violations == 0,
              f"{score_violations} violaciones sobre {score_checked} documentos de "
              f"{len(query_ids) - n_filtered} consultas sin filtro")
    rep.log()

    # --- 2.4 Mecanismo medido -------------------------------------------------
    overlap = Counter()
    pre_dense_only = Counter()
    top1_kept = top1_dropped_no_lexical = top1_dropped_with_lexical = 0
    retained_with_lexical = retained_without_lexical = total_with_lexical = total_without_lexical = 0
    post_from_pre = post_dense_only = post_in_dense5 = post_in_lexical5 = post_total = 0
    pre_in_dense5 = pre_in_lexical5 = pre_total = 0
    for qid in query_ids:
        lexical = by_system[LEXICAL][qid]["retrieved_ids"]
        dense = by_system[DENSE][qid]["retrieved_ids"]
        pre = by_system[HYBRID_PRE][qid]["retrieved_ids"]
        post = by_system[HYBRID_POST][qid]["retrieved_ids"]
        overlap[len(set(lexical) & set(dense))] += 1
        pre_dense_only[sum(1 for cid in pre if cid not in leg_rank[qid])] += 1
        pre_total += len(pre)
        pre_in_dense5 += len(set(pre) & set(dense))
        pre_in_lexical5 += len(set(pre) & set(lexical))
        if dense:
            if dense[0] in pre:
                top1_kept += 1
            elif dense[0] in leg_rank[qid]:
                top1_dropped_with_lexical += 1
            else:
                top1_dropped_no_lexical += 1
        for cid in dense:
            if cid in leg_rank[qid]:
                total_with_lexical += 1
                retained_with_lexical += cid in pre
            else:
                total_without_lexical += 1
                retained_without_lexical += cid in pre
        post_total += len(post)
        post_from_pre += len(set(post) & set(pre))
        post_dense_only += sum(1 for cid in post if cid not in leg_rank[qid])
        post_in_dense5 += len(set(post) & set(dense))
        post_in_lexical5 += len(set(post) & set(lexical))
    n = len(query_ids)
    rep.log("### 2.4 Qué hace la fusión con las listas firmadas")
    rep.log()
    rep.log("Solapamiento entre el top-5 léxico y el top-5 denso (cuántos fragmentos comparten):")
    rep.log()
    rep.table(["Fragmentos en común"] + [str(i) for i in range(FINAL_K + 1)],
              [["Consultas"] + [overlap.get(i, 0) for i in range(FINAL_K + 1)]])
    mean_overlap = sum(k * v for k, v in overlap.items()) / n
    rep.log(f"- Media: {mean_overlap:.2f} de 5. Las dos piernas casi nunca proponen los mismos fragmentos "
            "arriba, así que la intersección que RRF premia vive sobre todo en puestos profundos.")
    rep.log(f"- Top-5 fusionado (pre-rerank): {pre_in_dense5 / pre_total:.1%} de sus fragmentos estaban en el "
            f"top-5 denso y {pre_in_lexical5 / pre_total:.1%} en el top-5 léxico.")
    rep.log(f"- Retención del top-5 denso dentro del top-5 fusionado: "
            f"{retained_with_lexical}/{total_with_lexical} ({retained_with_lexical / max(total_with_lexical, 1):.1%}) "
            f"si BM25 también lo tenía entre sus 50; "
            f"{retained_without_lexical}/{total_without_lexical} "
            f"({retained_without_lexical / max(total_without_lexical, 1):.1%}) si no.")
    dropped = top1_dropped_no_lexical + top1_dropped_with_lexical
    rep.log(f"- Primer resultado denso: se conserva en el top-5 fusionado en {top1_kept}/{n} consultas y se "
            f"pierde en {dropped}/{n}. De las pérdidas, {top1_dropped_no_lexical} son fragmentos que BM25 no "
            f"tenía entre sus 50 (los expulsa el filtro de consenso) y {top1_dropped_with_lexical} sí los tenía "
            "pero fueron superados por otros de consenso.")
    rep.log()
    rep.log("Fragmentos solo-densos (fuera del top-50 de BM25) dentro del top-5 fusionado:")
    rep.log()
    rep.table(["Solo-densos en el top-5"] + [str(i) for i in range(FINAL_K + 1)],
              [["Consultas"] + [pre_dense_only.get(i, 0) for i in range(FINAL_K + 1)]])
    rep.log(f"- Tras el reranker: {post_from_pre / post_total:.1%} del top-5 final ya estaba en el top-5 "
            f"fusionado; el {1 - post_from_pre / post_total:.1%} restante sube desde los puestos 6 a 50 del "
            "pool. Esa es la parte de la ventaja del híbrido que depende de la profundidad del pool y del "
            "reranker, no de la fusión.")
    rep.log(f"- Top-5 final: {post_in_dense5 / post_total:.1%} en el top-5 denso, "
            f"{post_in_lexical5 / post_total:.1%} en el top-5 léxico, "
            f"{post_dense_only / post_total:.1%} fuera del top-50 de BM25.")
    rep.log()
    rep.data["retrieval"] = {
        "lexical_reproduced": [reproduced, n], "provider_filtered": n_filtered,
        "overlap_lexical_dense_top5": dict(sorted(overlap.items())), "mean_overlap": mean_overlap,
        "pre_share_in_dense5": pre_in_dense5 / pre_total, "pre_share_in_lexical5": pre_in_lexical5 / pre_total,
        "dense5_retained_with_lexical": [retained_with_lexical, total_with_lexical],
        "dense5_retained_without_lexical": [retained_without_lexical, total_without_lexical],
        "dense_top1": {"kept": top1_kept, "dropped_not_in_bm25_top50": top1_dropped_no_lexical,
                       "dropped_outscored": top1_dropped_with_lexical},
        "pre_dense_only_histogram": dict(sorted(pre_dense_only.items())),
        "post_share_from_pre": post_from_pre / post_total,
        "post_share_in_dense5": post_in_dense5 / post_total,
        "post_share_in_lexical5": post_in_lexical5 / post_total,
        "post_share_dense_only": post_dense_only / post_total,
        "order_violations": order_violations, "score_violations": [score_violations, score_checked],
    }


# =============================================================================
# 5. Generación (exp12)
# =============================================================================
def _load_cfm():
    spec = importlib.util.spec_from_file_location("cfm", ROOT / "scripts" / "compute_faithfulness_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXCLUDED_METHODS.clear()
    module.EXCLUDED_METHODS.update({"none", "error", "vacuous"})
    return module


def _included(cfm, per_config: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
    """Filas que entran en `faithfulness_answered` v4: mismas reglas que la tabla citable."""
    out: Dict[str, Dict[str, dict]] = {}
    for config, rows in per_config.items():
        kept = {}
        for qid, rec in rows.items():
            if rec["method"] in cfm.EXCLUDED_METHODS or rec["faithfulness"] is None:
                continue
            if not cfm._incl_primary(rec):
                continue
            genuine = rec["supported_claims"] + rec["contradicted_claims"] + rec["unsupported_claims"]
            if genuine <= 0:
                raise ValueError(f"{config} {qid}: fila incluida sin afirmaciones genuinas")
            # f exacta (apoyadas / genuinas). El valor firmado es este mismo cociente redondeado
            # a 4 decimales; `stored` se conserva para comprobarlo.
            kept[qid] = {"f": rec["supported_claims"] / genuine, "stored": float(rec["faithfulness"]),
                         "s": rec["supported_claims"], "g": genuine}
        out[config] = kept
    return out


def section_generation(rep: Report) -> None:
    cfm = _load_cfm()
    raw = json.loads((EXP12 / "results.json").read_text(encoding="utf-8"))["configs"]
    parsed = {config: cfm.parse_config(config) for config in raw}
    models = sorted({model for _, model in parsed.values()})
    config_of = {(scenario, model): config for config, (scenario, model) in parsed.items()}
    rows_raw = {config: {r["query_id"]: r for r in cfg["results"]} for config, cfg in raw.items()}
    included = {tag: _included(cfm, cfm.load_per_config(EXP12 / "results.json",
                                                         json.loads((EXP12 / name).read_text(encoding="utf-8"))["configs"],
                                                         exclude_vacuous=True))
                for tag, name in VERIFIERS.items()}

    rep.log("## 3. Generación: por qué unos modelos puntúan más (exp12, 4 modelos x 3 escenarios RAG)")
    rep.log()

    # --- 3.1 Diseño -----------------------------------------------------------
    rep.log("### 3.1 Hechos de diseño comprobados sobre los datos firmados")
    rep.log()
    same_context = total_cells = 0
    for scenario in RAG_SCENARIOS:
        reference = rows_raw[config_of[(scenario, models[0])]]
        for qid, row in reference.items():
            total_cells += 1
            same_context += all(rows_raw[config_of[(scenario, m)]][qid]["retrieved_ids"] == row["retrieved_ids"]
                                for m in models)
    rep.check("los cuatro modelos recibieron los mismos fragmentos, en el mismo orden, en cada consulta",
              same_context == total_cells, f"{same_context}/{total_cells} celdas (escenario, consulta)")
    rep.log("- Por tanto, ninguna diferencia entre modelos dentro de un escenario puede venir de la "
            "recuperación. Solo quedan tres fuentes: qué consultas decide contestar cada modelo, cuántas "
            "afirmaciones emite y cuántas de ellas apoya el verificador.")
    cells_ok = cells = 0
    worst = 0.0
    for tag, stored_name in (("small", "faithfulness_metrics_v4_small.json"), ("base", "faithfulness_metrics_v4.json")):
        stored = json.loads((EXP12 / stored_name).read_text(encoding="utf-8"))["systems_v2"]
        for config, rows in included[tag].items():
            worst = max([worst] + [abs(rec["f"] - rec["stored"]) for rec in rows.values()])
            if parsed[config][0] not in RAG_SCENARIOS:
                continue
            cells += 1
            signed_cell = stored[config]["primary_answered"]
            cells_ok += (len(rows) == signed_cell["n"]
                         and abs(_mean(r["f"] for r in rows.values()) - signed_cell["mean"]) < 1e-4)
    rep.check("la fidelidad firmada de cada fila es apoyadas / genuinas redondeada a 4 decimales",
              worst <= 5.0e-5 + 1e-12, f"desvío máximo {worst:.1e}")
    rep.check("las 24 celdas recalculadas coinciden con la tabla citable v4 (n exacto, media a 1e-4)",
              cells_ok == cells, f"{cells_ok}/{cells} celdas, dos verificadores")
    identical_context = sum(
        1 for model in models for s1, s2 in itertools.combinations(RAG_SCENARIOS, 2)
        for qid, row in rows_raw[config_of[(s1, model)]].items()
        if row["retrieved_ids"] == rows_raw[config_of[(s2, model)]][qid]["retrieved_ids"])
    rep.log(f"- Pares (modelo, consulta) en los que dos escenarios entregaron exactamente el mismo contexto: "
            f"{identical_context}. Sin pares así, la evidencia firmada no permite separar el ruido propio de "
            "la generación del efecto del contexto dentro de exp12.")
    rep.log()
    rep.data["generation"] = {"same_context_cells": [same_context, total_cells],
                              "identical_context_pairs": identical_context,
                              "cells_matching_signed_v4": [cells_ok, cells]}

    for tag in VERIFIERS:
        _between_models(rep, included[tag], config_of, models, tag)
    _verbosity_regression(rep, included["small"], parsed, models, "small")
    _verbosity_regression(rep, included["base"], parsed, models, "base")
    for tag in VERIFIERS:
        _between_scenarios(rep, included[tag], rows_raw, config_of, models, tag)


def _pair_components(rows_a: Dict[str, dict], rows_b: Dict[str, dict]) -> Dict[str, float]:
    decomposition = selection_within({q: r["f"] for q, r in rows_a.items()},
                                     {q: r["f"] for q, r in rows_b.items()})
    common = sorted(set(rows_a) & set(rows_b))
    numerator = denominator = 0.0
    for qid in common:
        num, den = ratio_shapley(rows_a[qid]["s"], rows_a[qid]["g"], rows_b[qid]["s"], rows_b[qid]["g"])
        numerator += num
        denominator += den
    count = max(len(common), 1)
    diffs = [rows_a[q]["f"] - rows_b[q]["f"] for q in common]
    decomposition.update({
        "supported_effect": numerator / count, "emitted_effect": denominator / count,
        "mean_s_a": _mean(rows_a[q]["s"] for q in common), "mean_s_b": _mean(rows_b[q]["s"] for q in common),
        "mean_g_a": _mean(rows_a[q]["g"] for q in common), "mean_g_b": _mean(rows_b[q]["g"] for q in common),
        "dz": _dz(diffs), "p": _wilcoxon_p(diffs), "mde": mde_dz(len(common)),
    })
    return decomposition


def _between_models(rep: Report, included: Dict[str, Dict[str, dict]], config_of: Dict, models: List[str],
                    tag: str) -> None:
    rep.log(f"### 3.2 ({tag}) Descomposición exacta de cada diferencia entre modelos")
    rep.log()
    rep.log("`total = dentro + selección`, y `dentro = apoyadas + emitidas`. *Selección*: parte de la brecha "
            "que existe porque cada modelo contesta consultas distintas. *Apoyadas*: brecha por número de "
            "afirmaciones que el verificador apoya. *Emitidas*: brecha por número de afirmaciones emitidas "
            "(el denominador). Las dos últimas se miden solo sobre las consultas que ambos contestaron.")
    rep.log()
    rows, records = [], []
    for scenario in RAG_SCENARIOS:
        for m1, m2 in itertools.combinations(models, 2):
            comp = _pair_components(included[config_of[(scenario, m1)]], included[config_of[(scenario, m2)]])
            comp.update({"scenario": scenario, "a": m1, "b": m2})
            records.append(comp)
    for comp, p_bh in zip(records, _bh([c["p"] for c in records])):
        comp["p_bh"] = p_bh
        closes = abs(comp["within"] + comp["selection"] - comp["total"]) < 1e-9
        closes &= abs(comp["supported_effect"] + comp["emitted_effect"] - comp["within"]) < 1e-9
        comp["closes"] = bool(closes)
        rows.append([comp["scenario"], f"{comp['a']} − {comp['b']}", _f(comp["total"], signed=True),
                     _f(comp["selection"], signed=True), _f(comp["within"], signed=True),
                     _f(comp["supported_effect"], signed=True), _f(comp["emitted_effect"], signed=True),
                     f"{comp['n_a']}/{comp['n_b']}/{comp['n_common']}",
                     f"{_f(comp['mean_s_a'], 2)} vs {_f(comp['mean_s_b'], 2)}",
                     f"{_f(comp['mean_g_a'], 2)} vs {_f(comp['mean_g_b'], 2)}",
                     _f(comp["dz"], 2, signed=True), _f(comp["p_bh"]), _f(comp["mde"], 2)])
    rep.table(["Escenario", "A − B", "Total", "Selección", "Dentro", "· apoyadas", "· emitidas",
               "n A/B/común", "Apoyadas A vs B", "Emitidas A vs B", "d_z", "p_BH", "|d_z| mín. detectable"], rows)
    rep.check(f"({tag}) las 18 descomposiciones cierran exactamente", all(c["closes"] for c in records))
    share = [abs(c["selection"]) / (abs(c["selection"]) + abs(c["within"]))
             for c in records if abs(c["selection"]) + abs(c["within"]) > 0]
    underpowered = sum(1 for c in records if not math.isnan(c["dz"]) and abs(c["dz"]) < c["mde"])
    rep.log(f"- Peso medio de la selección en la brecha: {np.mean(share):.1%} (mediana {np.median(share):.1%}).")
    rep.log(f"- Pares significativos tras BH (familia de 18): {sum(1 for c in records if c['p_bh'] < 0.05)}/18. "
            f"Pares cuyo |d_z| observado queda por debajo del mínimo detectable con su n: {underpowered}/18.")
    rep.log()
    rep.data["generation"][f"between_models_{tag}"] = records


def _verbosity_regression(rep: Report, included: Dict[str, Dict[str, dict]], parsed: Dict, models: List[str],
                          tag: str) -> None:
    y, log_g, g_raw, s_raw, groups, clusters, model_of = [], [], [], [], [], [], []
    for config, rows in included.items():
        scenario, model = parsed[config]
        if scenario not in RAG_SCENARIOS:
            continue
        for qid, rec in rows.items():
            y.append(rec["f"]); log_g.append(math.log(rec["g"])); g_raw.append(rec["g"]); s_raw.append(rec["s"])
            groups.append((scenario, qid)); clusters.append(qid); model_of.append(model)
    sizes = Counter(groups)
    keep = [i for i, group in enumerate(groups) if sizes[group] >= 2]
    y, log_g, g_raw, s_raw = (np.array(v, dtype=float)[keep] for v in (y, log_g, g_raw, s_raw))
    groups = [groups[i] for i in keep]; clusters = [clusters[i] for i in keep]; model_of = [model_of[i] for i in keep]
    reference, others = models[0], models[1:]
    dummies = np.array([[1.0 if m == other else 0.0 for other in others] for m in model_of])
    unadjusted = fixed_effects_ols(y, dummies, groups, clusters)
    adjusted = fixed_effects_ols(y, np.column_stack([log_g, dummies]), groups, clusters)
    pooled = []
    for ref in models[:-1]:
        rest = [m for m in models if m != ref]
        design = np.array([[1.0 if m == other else 0.0 for other in rest] for m in model_of])
        plain = fixed_effects_ols(y, design, groups, clusters)
        controlled = fixed_effects_ols(y, np.column_stack([log_g, design]), groups, clusters)
        for i, other in enumerate(rest):
            if models.index(other) > models.index(ref):
                pooled.append({"a": other, "b": ref, "gap": float(plain["beta"][i]), "p": float(plain["p"][i]),
                               "gap_adjusted": float(controlled["beta"][i + 1]),
                               "p_adjusted": float(controlled["p"][i + 1])})
    from statsmodels.stats.multitest import multipletests
    for item, p_holm, p_holm_adj in zip(
            pooled, multipletests([c["p"] for c in pooled], method="holm")[1],
            multipletests([c["p_adjusted"] for c in pooled], method="holm")[1]):
        item["p_holm"], item["p_adjusted_holm"] = float(p_holm), float(p_holm_adj)
    marginal = fixed_effects_ols(s_raw, g_raw, groups, clusters)
    average_rate = float(s_raw.sum() / g_raw.sum())

    rep.log(f"### 3.3 ({tag}) Contraste: con la misma pregunta y el mismo contexto, ¿emitir más afirmaciones baja la fidelidad?")
    rep.log()
    rep.log("Modelo: `f = efecto fijo de (escenario, consulta) + beta·ln(emitidas) + efecto del modelo`. "
            "El efecto fijo absorbe la dificultad de la pregunta y el contexto recuperado, que son idénticos "
            f"para los modelos comparados. Errores agrupados por consulta. n = {int(adjusted['n'])} respuestas "
            f"en {int(adjusted['n_groups'])} grupos de {int(adjusted['n_clusters'])} consultas.")
    rep.log()
    beta, se, p = adjusted["beta"][0], adjusted["se"][0], adjusted["p"][0]
    rep.log(f"- beta = {beta:+.4f} (EE {se:.4f}, p = {p:.2g}). Duplicar el número de afirmaciones cambia la "
            f"fidelidad en {beta * math.log(2):+.3f}. "
            + ("La hipótesis se sostiene con este verificador." if p < 0.05 and beta < 0 else
               "La hipótesis NO se sostiene con este verificador: el efecto no se distingue de cero."))
    rep.log(f"- Tasa media de apoyo: {average_rate:.3f} apoyadas por afirmación. Tasa marginal dentro de la "
            f"misma consulta: {marginal['beta'][0]:.3f} (EE {marginal['se'][0]:.3f}). "
            + f"Diferencia: {average_rate - marginal['beta'][0]:+.3f}, "
            + f"{(average_rate - marginal['beta'][0]) / marginal['se'][0]:.1f} errores estándar.")
    rep.log()
    rep.log("Brecha de fidelidad entre modelos con la pregunta y el contexto fijos, juntando los tres "
            "escenarios (A − B), antes y después de descontar ln(emitidas). Holm sobre los 6 pares. Es un "
            "análisis exploratorio nuevo: no sustituye a la familia BH por escenario de la tabla citable.")
    rep.log()
    rep.table(["A − B", "Brecha", "p", "p Holm", "Brecha ajustada", "p", "p Holm"],
              [[f"{c['a']} − {c['b']}", _f(c["gap"], signed=True), f"{c['p']:.2g}", f"{c['p_holm']:.2g}",
                _f(c["gap_adjusted"], signed=True), f"{c['p_adjusted']:.2g}", f"{c['p_adjusted_holm']:.2g}"]
               for c in pooled])
    rep.data["generation"][f"verbosity_{tag}"] = {
        "beta_log_emitted": float(beta), "se": float(se), "p": float(p), "n": int(adjusted["n"]),
        "average_support_rate": average_rate, "marginal_support_rate": float(marginal["beta"][0]),
        "marginal_se": float(marginal["se"][0]), "reference": reference,
        "gap_unadjusted": {o: float(unadjusted["beta"][i]) for i, o in enumerate(others)},
        "gap_adjusted": {o: float(adjusted["beta"][i + 1]) for i, o in enumerate(others)},
        "pooled_pairs": pooled,
    }


def _between_scenarios(rep: Report, included: Dict[str, Dict[str, dict]], rows_raw: Dict, config_of: Dict,
                       models: List[str], tag: str) -> None:
    rep.log(f"### 3.4 ({tag}) Por qué el híbrido no siempre gana en fidelidad")
    rep.log()
    rows, records = [], []
    pairs = (("hibrido", "lexico"), ("hibrido", "denso"), ("denso", "lexico"))
    for model in models:
        for s1, s2 in pairs:
            a, b = included[config_of[(s1, model)]], included[config_of[(s2, model)]]
            comp = _pair_components(a, b)
            comp.update({"model": model, "a": s1, "b": s2,
                         "n_needed": required_n(comp["dz"]) if not math.isnan(comp["dz"]) else float("nan")})
            raw_a, raw_b = rows_raw[config_of[(s1, model)]], rows_raw[config_of[(s2, model)]]
            changed, absolute = [], []
            for qid in sorted(set(a) & set(b)):
                changed.append(FINAL_K - len(set(raw_a[qid]["retrieved_ids"]) & set(raw_b[qid]["retrieved_ids"])))
                absolute.append(abs(a[qid]["f"] - b[qid]["f"]))
            if len(set(changed)) > 1 and len(set(absolute)) > 1:
                rho, p_rho = stats.spearmanr(changed, absolute)
                comp["dose_rho"], comp["dose_p"] = float(rho), float(p_rho)
            else:
                comp["dose_rho"], comp["dose_p"] = float("nan"), float("nan")
            comp["mean_changed"] = _mean(changed)
            records.append(comp)
    for comp, p_bh in zip(records, _bh([c["p"] for c in records])):
        comp["p_bh"] = p_bh
        rows.append([comp["model"], f"{comp['a']} − {comp['b']}", _f(comp["total"], signed=True),
                     _f(comp["selection"], signed=True), _f(comp["within"], signed=True),
                     _f(comp["supported_effect"], signed=True), _f(comp["emitted_effect"], signed=True),
                     f"{comp['n_a']}/{comp['n_b']}/{comp['n_common']}", _f(comp["dz"], 2, signed=True),
                     _f(comp["p_bh"]), _f(comp["mde"], 2),
                     _n_needed(comp["n_needed"]),
                     _f(comp["mean_changed"], 2), f"{_f(comp['dose_rho'], 2, signed=True)} (p={_f(comp['dose_p'], 2)})"])
    rep.table(["Modelo", "A − B", "Total", "Selección", "Dentro", "· apoyadas", "· emitidas", "n A/B/común",
               "d_z", "p_BH", "|d_z| mín. detectable", "n para 80 % de potencia", "Fragmentos distintos (media)",
               "rho(|Δf|, fragmentos distintos)"], rows)
    significant = sum(1 for c in records if c["p_bh"] < 0.05)
    powered = sum(1 for c in records if not math.isnan(c["dz"]) and abs(c["dz"]) >= c["mde"])
    sign_flip = sum(1 for c in records if c["a"] == "hibrido" and c["total"] * c["within"] < 0)
    rep.log(f"- Contrastes significativos tras BH (familia de 12): {significant}/12. Contrastes cuyo |d_z| "
            f"observado alcanza el mínimo detectable con su n: {powered}/12.")
    rep.log(f"- Contrastes del híbrido en los que la brecha total y la brecha dentro de las mismas consultas "
            f"tienen signo opuesto: {sign_flip}/8. En esos casos el orden de la tabla publicada lo decide qué "
            "consultas se contestan, no cómo se contestan.")
    rep.log()
    rep.data["generation"][f"between_scenarios_{tag}"] = records


# =============================================================================
# 6. Programa
# =============================================================================
def build_report() -> Report:
    rep = Report()
    rep.log("# Por qué el híbrido no siempre gana y por qué unos modelos puntúan más")
    rep.log()
    rep.log("Generado por `scripts/audit/fable_why_analysis.py` desde la evidencia firmada (solo lectura) y "
            "los índices en disco. No se ejecutó ningún modelo. Cada línea `[OK ]` es una comprobación que "
            "el script hace y que falla con exit 1 si deja de cumplirse.")
    rep.log()
    handles = section_indices(rep)
    section_retrieval(rep, handles["bm25"], handles["chunk_map"])
    section_generation(rep)
    rep.log("## 4. Resultado de las comprobaciones")
    rep.log()
    rep.log(f"- Fallos: {len(rep.failures)}" + (f" ({'; '.join(rep.failures)})" if rep.failures else ""))
    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="guarda el reporte y el JSON en output/audit/")
    args = parser.parse_args()
    rep = build_report()
    text = "\n".join(rep.lines) + "\n"
    sys.stdout.write(text)
    if args.write:
        out_dir = ROOT / "output" / "audit"
        (out_dir / f"{REPORT_STEM}.md").write_text(text, encoding="utf-8", newline="\n")
        (out_dir / f"{REPORT_STEM}.json").write_text(
            json.dumps(rep.data, indent=2, ensure_ascii=False, default=float) + "\n",
            encoding="utf-8", newline="\n")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
