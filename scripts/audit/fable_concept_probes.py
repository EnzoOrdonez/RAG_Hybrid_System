"""Sondas de concepto sobre el diseño de evaluación de CloudRAG (sin ejecutar modelos).

Cada sonda convierte una sospecha de concepto en un número que se puede refutar:

  1. Truncado silencioso a 512 tokens en el embedder, el reranker y el verificador NLI
     (Flags 33, 130 y 144 del ledger: conocidos, nunca medidos).
  2. NDCG relativo al pool: el mismo ranking cambia de nota al añadir brazos.
  3. Asimetría de sobre-recuperación entre los brazos base y el híbrido cuando hay filtro de proveedor.
  4. Preferencia del reranker por fragmentos largos.
  5. Consultas agrupadas por plantilla: correlación intraclase y efecto de diseño.
  6. Texto duplicado que ocupa huecos del top-5.
  7. Qué significa en términos absolutos el umbral p50 y el puntaje del oráculo.
  8. Tipado de consultas por reglas frente a los proveedores anotados en el conjunto de prueba.

No carga pesos de ningún modelo. Usa solo los archivos `tokenizer.json` de `data/models/` (vocabularios)
para contar tokens, la corrida de auditoría exp20 (`output/audit/exp20_rerank_arms_2026-11-15/`) y la
evidencia firmada en modo lectura.

Uso:  python scripts/audit/fable_concept_probes.py [--write]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
EXP11 = RESULTS / "exp11_retrieval194_fullrerank"
EXP20 = ROOT / "output" / "audit" / "exp20_rerank_arms_2026-11-15" / "results.json"
CHUNK_MAP = ROOT / "data" / "indices" / "chunk_map_bge-large_adaptive_500.json"
MODELS = ROOT / "data" / "models"
QUERIES = ROOT / "data" / "evaluation" / "test_queries.json"
REPORT_STEM = "fable_concept_probes_2026-11-15"

MAX_LENGTH = 512               # límite de los tres modelos (embedder, reranker, NLI)
SINGLE_SPECIALS = 2            # [CLS] texto [SEP]
PAIR_SPECIALS = 3              # [CLS] a [SEP] b [SEP]
CLAIM_TOKENS = (20, 40, 60)    # las afirmaciones no están guardadas: se barre su longitud
PROVIDER_WORDS = ("Amazon", "AWS", "Azure", "Microsoft", "Google Cloud", "GCP", "Google")
SIGNED_ARMS = {"lexico": "RAG Lexico (BM25)", "denso": "RAG Semantico (Dense)",
               "hibrido_rerank": "RAG Hibrido Propuesto", "hibrido_pre": "RAG Hibrido (pre-rerank RRF)"}
ALL_ARMS = ("lexico", "denso", "hibrido_pre", "lexico_rerank", "denso_rerank", "hibrido_rerank", "union_rerank")


# --------------------------------------------------------------------- funciones puras
def lost_tokens(n_tokens: int, budget: int) -> int:
    """Tokens de un texto que el modelo no llega a leer si solo caben `budget`."""
    return max(0, n_tokens - max(budget, 0))


def ndcg_graded(retrieved_ids: Sequence[str], pool_scores: Mapping[str, float], k: int = 5) -> float:
    """NDCG@k con la misma definición que `scripts/compute_retrieval_metrics.py`.

    Ganancia = max(0, puntaje). El DCG ideal sale de los k mejores puntajes de TODO el pool de la
    consulta, así que el valor depende de qué sistemas aportaron fragmentos al pool.
    """
    dcg = sum(max(0.0, pool_scores.get(cid, 0.0)) / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:k]))
    best = sorted(pool_scores.values(), reverse=True)[:k]
    idcg = sum(max(0.0, score) / math.log2(i + 2) for i, score in enumerate(best))
    return dcg / idcg if idcg > 0 else 0.0


def icc_oneway(values: Sequence[float], groups: Sequence) -> float:
    """Correlación intraclase de un factor (ANOVA, grupos de tamaño desigual). NaN si no se puede."""
    by_group: Dict[object, List[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    n, g = len(values), len(by_group)
    if g < 2 or n <= g:
        return float("nan")
    grand = float(np.mean(values))
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in by_group.values())
    ss_within = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in by_group.values())
    ms_between, ms_within = ss_between / (g - 1), ss_within / (n - g)
    n0 = (n - sum(len(v) ** 2 for v in by_group.values()) / n) / (g - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator > 0 else float("nan")


def design_effect(icc: float, sizes: Iterable[int]) -> float:
    """Efecto de diseño de Kish para grupos desiguales: 1 + (m* - 1)·ICC, con m* = Σn²/Σn."""
    sizes = [int(s) for s in sizes]
    if not sizes or math.isnan(icc):
        return float("nan")
    weighted = sum(s * s for s in sizes) / sum(sizes)
    return 1.0 + (weighted - 1.0) * icc


def cluster_mean_test(diff: Sequence[float], clusters: Sequence) -> Dict[str, float]:
    """Contraste de media = 0 con error estándar clásico y con error robusto por grupos."""
    values = np.asarray(diff, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se_iid = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    sums: Dict[object, float] = defaultdict(float)
    for value, cluster in zip(values, clusters):
        sums[cluster] += value - mean
    g = len(sums)
    se_cluster = math.sqrt(g / (g - 1) * sum(s * s for s in sums.values())) / n if g > 1 else float("nan")

    def p_value(se: float, dof: int) -> float:
        if not se or math.isnan(se) or dof < 1:
            return float("nan")
        return float(2 * stats.t.sf(abs(mean / se), dof))

    return {"mean": mean, "se_iid": se_iid, "p_iid": p_value(se_iid, n - 1),
            "se_cluster": se_cluster, "p_cluster": p_value(se_cluster, g - 1), "n": n, "clusters": g}


def template_of(question: str, words: int = 4) -> str:
    """Plantilla heurística: proveedor enmascarado y primeras `words` palabras."""
    text = question
    for word in PROVIDER_WORDS:
        text = text.replace(word, "<P>")
    return " ".join(text.split()[:words])


# ------------------------------------------------------------------------ utilidades
class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.data: Dict[str, object] = {}
        self.failures: List[str] = []

    def log(self, text: str = "") -> None:
        self.lines.append(text)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.log(f"- [{'OK' if ok else 'FALLO'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def table(self, header: Sequence[str], rows: Iterable[Sequence]) -> None:
        self.log("| " + " | ".join(header) + " |")
        self.log("|" + "---|" * len(header))
        for row in rows:
            self.log("| " + " | ".join(str(cell) for cell in row) + " |")
        self.log()


def _token_counter(model_dir: str):
    """Cuenta tokens con el vocabulario local. No carga pesos."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODELS / model_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(texts: Sequence[str]) -> List[int]:
        return [len(e.ids) for e in tokenizer.encode_batch(list(texts), add_special_tokens=False)]
    return count


def _load_why():
    spec = importlib.util.spec_from_file_location("fable_why_analysis", ROOT / "scripts" / "audit" / "fable_why_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pct(value: float) -> str:
    return f"{100 * value:.1f} %"


# --------------------------------------------------------------------------- sondas
def probe_truncation(rep: Report, chunk_map: Dict[str, dict], rows: List[dict]) -> Dict[str, Dict[str, int]]:
    rep.log("## 1. Truncado silencioso a 512 tokens (Flags 33, 130 y 144: conocidos, nunca medidos)")
    rep.log()
    ids = list(chunk_map)
    texts = [chunk_map[cid]["text"] for cid in ids]
    tokens = {name: dict(zip(ids, _token_counter(directory)(texts)))
              for name, directory in (("embedder", "bge-large-en-v1.5"), ("reranker", "ms-marco-MiniLM-L-12-v2"),
                                      ("nli", "nli-deberta-v3-small"))}
    query_tokens = dict(zip([r["query_id"] for r in rows],
                            _token_counter("ms-marco-MiniLM-L-12-v2")([r["question"] for r in rows])))

    # 1a. Embedder: el índice denso solo representa los primeros 510 tokens de cada fragmento.
    budget = MAX_LENGTH - SINGLE_SPECIALS
    counts = np.array(list(tokens["embedder"].values()))
    lost = np.array([lost_tokens(int(c), budget) for c in counts])
    rep.log(f"**Embedder (bge-large, {budget} tokens útiles).** Fragmentos del índice: {len(counts)}. "
            f"Longitud en tokens p50/p95/máx: {int(np.percentile(counts, 50))}/{int(np.percentile(counts, 95))}/{int(counts.max())}. "
            f"Fragmentos que no caben: {int((lost > 0).sum())} ({_pct((lost > 0).mean())}). "
            f"Tokens del corpus que el vector no representa: {_pct(lost.sum() / counts.sum())}. "
            "BM25 sí indexa el texto completo, así que las dos piernas no ven el mismo documento.")
    rep.log()
    rep.data["embedder"] = {"chunks": len(counts), "over_budget": int((lost > 0).sum()),
                            "share_over": float((lost > 0).mean()), "token_share_lost": float(lost.sum() / counts.sum())}

    # 1b. Reranker: par (consulta, fragmento).
    body = []
    for arm in ("pool_hibrido",) + ALL_ARMS:
        cut = total = lost_sum = token_sum = 0
        for r in rows:
            chunk_ids = r["pools"]["hibrido"] if arm == "pool_hibrido" else r["arms"][arm]
            room = MAX_LENGTH - PAIR_SPECIALS - query_tokens[r["query_id"]]
            for cid in chunk_ids:
                n_tokens = tokens["reranker"][cid]
                missing = lost_tokens(n_tokens, room)
                total += 1
                cut += missing > 0
                lost_sum += missing
                token_sum += n_tokens
        body.append([arm, total, f"{cut} ({_pct(cut / total)})", _pct(lost_sum / token_sum)])
        rep.data.setdefault("reranker", {})[arm] = {"pairs": total, "truncated": cut, "token_share_lost": lost_sum / token_sum}
    rep.log("**Reranker (ms-marco, par consulta + fragmento).**")
    rep.log()
    rep.table(["Fragmentos de", "Pares", "Pares truncados", "Tokens de fragmento no leídos"], body)

    # 1c. NLI: par (fragmento, afirmación). La afirmación no está guardada: se barre su longitud.
    body = []
    for arm in ("lexico", "denso", "hibrido_rerank"):
        line = [arm]
        for claim in CLAIM_TOKENS:
            room = MAX_LENGTH - PAIR_SPECIALS - claim
            per_chunk = [lost_tokens(tokens["nli"][cid], room) for r in rows for cid in r["arms"][arm]]
            all_tokens = sum(tokens["nli"][cid] for r in rows for cid in r["arms"][arm])
            line.append(f"{_pct(np.mean([x > 0 for x in per_chunk]))} / {_pct(sum(per_chunk) / all_tokens)}")
            rep.data.setdefault("nli", {}).setdefault(arm, {})[str(claim)] = {
                "share_chunks_truncated": float(np.mean([x > 0 for x in per_chunk])),
                "token_share_lost": sum(per_chunk) / all_tokens}
        any_cut = sum(1 for r in rows if any(lost_tokens(tokens["nli"][cid], MAX_LENGTH - PAIR_SPECIALS - 40) > 0
                                             for cid in r["arms"][arm]))
        line.append(f"{any_cut}/{len(rows)}")
        body.append(line)
    rep.log("**Verificador NLI (DeBERTa-v3, par fragmento + afirmación).** Cada celda: fragmentos del top-5 "
            "truncados / tokens de contexto que el verificador no lee. El LLM sí recibe el fragmento completo, "
            "así que una afirmación apoyada en la cola de un fragmento largo cuenta como no apoyada.")
    rep.log()
    rep.table(["Contexto del escenario"] + [f"afirmación de {c} tokens" for c in CLAIM_TOKENS]
              + ["Consultas con algún fragmento truncado (40 tokens)"], body)
    return tokens


def probe_truncation_effect(rep: Report, nli_tokens: Dict[str, int]) -> None:
    """Regresión con efectos fijos: ¿baja la fidelidad cuando el verificador pierde más contexto?"""
    why = _load_why()
    cfm = why._load_cfm()
    raw = json.loads((why.EXP12 / "results.json").read_text(encoding="utf-8"))["configs"]
    parsed = {config: cfm.parse_config(config) for config in raw}
    rows_raw = {config: {r["query_id"]: r for r in cfg["results"]} for config, cfg in raw.items()}
    room = MAX_LENGTH - PAIR_SPECIALS - 40
    rep.log("**¿Se nota en la fidelidad?** Regresión dentro de cada par (modelo, consulta): la única variación es "
            "el escenario de recuperación. `x1` = fracción del contexto que el NLI no lee; `x2` = tokens de "
            "contexto en miles. Errores agrupados por consulta.")
    rep.log()
    body = []
    for tag, name in why.VERIFIERS.items():
        included = why._included(cfm, cfm.load_per_config(
            why.EXP12 / "results.json", json.loads((why.EXP12 / name).read_text(encoding="utf-8"))["configs"],
            exclude_vacuous=True))
        y, x, groups, clusters = [], [], [], []
        for config, per_query in included.items():
            scenario, model = parsed[config]
            if scenario not in why.RAG_SCENARIOS:
                continue
            for qid, rec in per_query.items():
                chunk_ids = rows_raw[config][qid]["retrieved_ids"]
                total = sum(nli_tokens[cid] for cid in chunk_ids)
                if not total:
                    continue
                missing = sum(lost_tokens(nli_tokens[cid], room) for cid in chunk_ids)
                y.append(rec["f"])
                x.append([missing / total, total / 1000.0])
                groups.append((model, qid))
                clusters.append(qid)
        fit = why.fixed_effects_ols(y, np.array(x), groups, clusters)
        body.append([tag, int(fit["n"]), int(fit["n_groups"]),
                     f"{fit['beta'][0]:+.3f} (p = {fit['p'][0]:.3f})", f"{fit['beta'][1]:+.3f} (p = {fit['p'][1]:.3f})"])
        rep.data.setdefault("truncation_effect", {})[tag] = {
            "n": int(fit["n"]), "beta_lost_share": float(fit["beta"][0]), "p_lost_share": float(fit["p"][0]),
            "beta_ktokens": float(fit["beta"][1]), "p_ktokens": float(fit["p"][1])}
    rep.table(["Verificador", "Filas", "Pares (modelo, consulta)", "β fracción no leída", "β miles de tokens"], body)


def probe_pool_relative_ndcg(rep: Report, rows: List[dict]) -> Dict[str, List[float]]:
    rep.log("## 2. El NDCG es relativo al pool: el mismo ranking cambia de nota al añadir brazos")
    rep.log()
    signed = json.loads((EXP11 / "retrieval_metrics__ms-marco-circular.json").read_text(encoding="utf-8"))["systems"]
    four, seven = defaultdict(list), defaultdict(list)
    for r in rows:
        scores = r["reranker_scores"]
        pool4 = {cid: scores[cid] for arm in SIGNED_ARMS for cid in r["arms"][arm]}
        pool7 = {cid: scores[cid] for arm in ALL_ARMS for cid in r["arms"][arm]}
        for arm in ALL_ARMS:
            seven[arm].append(ndcg_graded(r["arms"][arm], pool7))
            if arm in SIGNED_ARMS:
                four[arm].append(ndcg_graded(r["arms"][arm], pool4))
    worst = max(abs(np.mean(four[arm]) - signed[name]["ndcg@5_mean"]) for arm, name in SIGNED_ARMS.items())
    rep.check("exp20 reproduce el NDCG@5 circular firmado de exp11 con el pool original de cuatro sistemas",
              worst < 5e-4, f"desviación máxima {worst:.1e}")
    rep.log()
    rep.table(["Brazo", "NDCG@5 circular, pool de 4 (firmado)", "NDCG@5 circular, pool de 7", "Cambio"],
              [[arm, f"{np.mean(four[arm]):.4f}" if arm in four else "no existe",
                f"{np.mean(seven[arm]):.4f}",
                f"{np.mean(seven[arm]) - np.mean(four[arm]):+.4f}" if arm in four else ""] for arm in ALL_ARMS])
    rep.log("Ningún ranking cambió entre las dos columnas: solo cambió quién más aportó fragmentos al pool. "
            "Oráculo circular (el propio reranker): la tabla mide la dependencia del pool, no la calidad.")
    rep.log()
    rep.data["ndcg_circular"] = {"pool4": {a: float(np.mean(v)) for a, v in four.items()},
                                 "pool7": {a: float(np.mean(v)) for a, v in seven.items()},
                                 "max_dev_vs_signed": float(worst)}
    return seven


def probe_overfetch(rep: Report, rows: List[dict], ndcg7: Dict[str, List[float]]) -> None:
    rep.log("## 3. Asimetría de sobre-recuperación cuando hay filtro de proveedor")
    rep.log()
    body = []
    for label, wanted in (("con filtro", True), ("sin filtro", False)):
        index = [i for i, r in enumerate(rows) if bool(r["provider_filter"]) == wanted]
        diff = [ndcg7["hibrido_rerank"][i] - ndcg7["denso_rerank"][i] for i in index]
        pool = np.mean([len(rows[i]["pools"]["hibrido"]) for i in index])
        short = sum(1 for i in index if len(rows[i]["pools"]["hibrido"]) < 50)
        lost = sum(1 for i in index if any(cid not in set(rows[i]["pools"]["hibrido"]) for cid in rows[i]["arms"]["union_rerank"]))
        p = float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        body.append([label, len(index), f"{pool:.1f}", short, f"{np.mean(diff):+.4f} (p = {p:.3f})", f"{lost}/{len(index)}"])
        rep.data.setdefault("overfetch", {})[label] = {"n": len(index), "hybrid_pool_mean": float(pool), "short_pools": short,
                                                       "ndcg7_hybrid_minus_dense_rerank": float(np.mean(diff)), "p": p,
                                                       "queries_losing_union_top5": lost}
    rep.table(["Consultas", "n", "Pool híbrido medio", "Pools con menos de 50",
               "NDCG@5 circular: híbrido+reranker − denso+reranker", "Pierden algún fragmento del top-5 de la unión"], body)


def probe_length_preference(rep: Report, rows: List[dict], reranker_tokens: Dict[str, int]) -> None:
    rep.log("## 4. El reranker prefiere fragmentos largos")
    rep.log()
    rhos = []
    for r in rows:
        ids = list(r["reranker_scores"])
        rho = stats.spearmanr([r["reranker_scores"][c] for c in ids], [reranker_tokens[c] for c in ids]).statistic
        if not math.isnan(rho):
            rhos.append(float(rho))
    p = float(stats.wilcoxon(rhos).pvalue)
    means = {arm: float(np.mean([reranker_tokens[c] for r in rows for c in r["arms"][arm]])) for arm in ALL_ARMS}
    pool_mean = float(np.mean([reranker_tokens[c] for r in rows for c in r["reranker_scores"]]))
    rep.log(f"- Correlación de Spearman entre puntaje del reranker y longitud en tokens, dentro de cada consulta "
            f"(unión de pools): media {np.mean(rhos):+.3f}, positiva en {sum(1 for x in rhos if x > 0)}/{len(rhos)} "
            f"consultas, p = {p:.1e} (Wilcoxon).")
    rep.log(f"- Tokens medios por fragmento: pool {pool_mean:.0f}; " +
            "; ".join(f"{arm} {means[arm]:.0f}" for arm in ALL_ARMS) + ".")
    rep.log("- Consecuencia: los escenarios con reranker entregan más texto al LLM. La comparación de generación "
            "«híbrido frente a base» mezcla qué fragmentos se eligen con cuánto texto se entrega.")
    rep.log()
    rep.data["length_preference"] = {"mean_rho": float(np.mean(rhos)), "positive": sum(1 for x in rhos if x > 0),
                                     "n": len(rhos), "p": p, "tokens_top5": means, "tokens_pool": pool_mean}


def probe_clustering(rep: Report, rows: List[dict], ndcg7: Dict[str, List[float]]) -> None:
    rep.log("## 5. Las 194 consultas no son independientes: salen de plantillas")
    rep.log()
    templates = [template_of(r["question"]) for r in rows]
    sizes = Counter(templates)
    rep.log(f"- Plantillas (proveedor enmascarado, primeras 4 palabras): {len(sizes)} para {len(rows)} consultas. "
            f"Mayor grupo: {max(sizes.values())}. Grupos de una sola consulta: {sum(1 for v in sizes.values() if v == 1)}.")
    rep.log()
    body = []
    contrasts = {"híbrido+reranker − denso (firmado, sin reranker)": ("hibrido_rerank", "denso"),
                 "híbrido+reranker − léxico (firmado, sin reranker)": ("hibrido_rerank", "lexico"),
                 "híbrido+reranker − denso+reranker (control exp20)": ("hibrido_rerank", "denso_rerank"),
                 "híbrido+reranker − léxico+reranker (control exp20)": ("hibrido_rerank", "lexico_rerank")}
    for label, (a, b) in contrasts.items():
        diff = [x - y for x, y in zip(ndcg7[a], ndcg7[b])]
        icc = icc_oneway(diff, templates)
        deff = design_effect(icc, sizes.values())
        test = cluster_mean_test(diff, templates)
        body.append([label, f"{test['mean']:+.4f}", f"{icc:+.3f}", f"{deff:.2f}", f"{len(rows) / deff:.0f}",
                     f"{test['p_iid']:.2g}", f"{test['p_cluster']:.2g}"])
        rep.data.setdefault("clustering", {})[label] = {"mean": test["mean"], "icc": icc, "deff": deff,
                                                        "p_iid": test["p_iid"], "p_cluster": test["p_cluster"]}
    rep.table(["Contraste (NDCG@5 circular, pool de 7)", "Diferencia media", "ICC por plantilla", "Efecto de diseño",
               "n efectivo", "p con consultas independientes", "p agrupando por plantilla"], body)
    rep.data["templates"] = {"count": len(sizes), "largest": max(sizes.values())}


def probe_model_contrasts(rep: Report, rows: List[dict]) -> None:
    """Brechas de fidelidad entre modelos (exp12) con la consulta o con la plantilla como unidad."""
    from statsmodels.stats.multitest import multipletests
    why = _load_why()
    cfm = why._load_cfm()
    template = {r["query_id"]: template_of(r["question"]) for r in rows}
    rep.log("Lo mismo sobre la generación (exp12): brecha de fidelidad entre modelos con la pregunta y el contexto "
            "fijos, juntando los tres escenarios. Diferencias pareadas por (escenario, consulta). Holm sobre los 6 pares.")
    rep.log()
    for tag, name in why.VERIFIERS.items():
        included = why._included(cfm, cfm.load_per_config(
            why.EXP12 / "results.json", json.loads((why.EXP12 / name).read_text(encoding="utf-8"))["configs"],
            exclude_vacuous=True))
        cells = {cfm.parse_config(config): per_query for config, per_query in included.items()}
        models = sorted({model for scenario, model in cells if scenario in why.RAG_SCENARIOS})
        results = []
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                diff, by_query, by_template = [], [], []
                for scenario in why.RAG_SCENARIOS:
                    for qid in sorted(set(cells[(scenario, a)]) & set(cells[(scenario, b)])):
                        diff.append(cells[(scenario, a)][qid]["f"] - cells[(scenario, b)][qid]["f"])
                        by_query.append(qid)
                        by_template.append(template[qid])
                per_query = cluster_mean_test(diff, by_query)
                per_template = cluster_mean_test(diff, by_template)
                results.append({"a": a, "b": b, "gap": per_query["mean"], "n": per_query["n"],
                                "p_query": per_query["p_cluster"], "p_template": per_template["p_cluster"],
                                "templates": per_template["clusters"], "icc": icc_oneway(diff, by_template)})
        for item, holm_q, holm_t in zip(results, multipletests([c["p_query"] for c in results], method="holm")[1],
                                        multipletests([c["p_template"] for c in results], method="holm")[1]):
            item["holm_query"], item["holm_template"] = float(holm_q), float(holm_t)
        rep.log(f"Verificador `{tag}`:")
        rep.log()
        rep.table(["A − B", "Brecha", "Celdas", "ICC por plantilla", "p Holm (unidad: consulta)",
                   "p Holm (unidad: plantilla)"],
                  [[f"{c['a']} − {c['b']}", f"{c['gap']:+.3f}", c["n"], f"{c['icc']:+.3f}",
                    f"{c['holm_query']:.3g}", f"{c['holm_template']:.3g}"] for c in results])
        rep.data.setdefault("model_contrasts", {})[tag] = results


def probe_duplicates(rep: Report, chunk_map: Dict[str, dict], rows: List[dict]) -> None:
    rep.log("## 6. Texto duplicado que ocupa huecos del top-5")
    rep.log()
    digest = {cid: hashlib.sha1(re.sub(r"\s+", " ", c["text"].strip().lower()).encode()).hexdigest()
              for cid, c in chunk_map.items()}
    groups = Counter(digest.values())
    repeated = sum(v for v in groups.values() if v > 1)
    rep.log(f"- Fragmentos del índice con texto idéntico a otro: {repeated} de {len(digest)} "
            f"({_pct(repeated / len(digest))}), en {sum(1 for v in groups.values() if v > 1)} grupos (el mayor, {max(groups.values())}).")
    body = []
    for arm in ALL_ARMS:
        slots = sum(len(r["arms"][arm]) for r in rows)
        wasted = sum(len(r["arms"][arm]) - len({digest[c] for c in r["arms"][arm]}) for r in rows)
        queries = sum(1 for r in rows if len({digest[c] for c in r["arms"][arm]}) < len(r["arms"][arm]))
        body.append([arm, f"{queries}/{len(rows)}", f"{wasted}/{slots} ({_pct(wasted / slots)})"])
        rep.data.setdefault("duplicates", {})[arm] = {"queries": queries, "wasted_slots": wasted, "slots": slots}
    rep.log()
    rep.table(["Brazo", "Consultas con texto repetido en el top-5", "Huecos del top-5 gastados en repeticiones"], body)
    rep.data["duplicates_corpus"] = {"repeated": repeated, "chunks": len(digest)}


def probe_query_typing(rep: Report, chunk_map: Dict[str, dict], rows: List[dict]) -> None:
    """Tipo de consulta que asigna el procesador (reglas) frente a los proveedores anotados en el conjunto."""
    from src.retrieval.query_processor import QueryProcessor
    rep.log("## 8. Tipado de consultas: «cross_cloud» no significa «varios proveedores»")
    rep.log()
    gold = {q["query_id"]: {p.lower() for p in q["cloud_providers"]}
            for q in json.loads(QUERIES.read_text(encoding="utf-8"))}
    processor = QueryProcessor()
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        processed = processor.process(r["question"])
        if len(gold[r["query_id"]]) != 1:
            label = "varios proveedores (anotado)"
        elif processed.provider_filter:
            label = "un proveedor, con filtro"
        elif processed.query_type == "cross_cloud":
            label = "un proveedor, tipada cross_cloud, sin filtro"
        else:
            label = "un proveedor, otro tipo, sin filtro"
        groups[label].append(r)
    typed_cross = sum(1 for r in rows if processor.process(r["question"]).query_type == "cross_cloud")
    single_typed_cross = len(groups["un proveedor, tipada cross_cloud, sin filtro"])
    rep.log(f"- El procesador tipa {typed_cross} consultas como `cross_cloud`. De ellas, {single_typed_cross} comparan "
            "dos servicios del MISMO proveedor (por ejemplo, dos servicios de AWS): reciben la plantilla de "
            "comparación entre nubes y se recuperan sin filtro de proveedor.")
    rep.log()
    arms = ("lexico", "denso", "hibrido_pre", "hibrido_rerank", "denso_rerank")
    body = []
    for label in sorted(groups):
        line = [label, len(groups[label])]
        for arm in arms:
            slots = sum(len(r["arms"][arm]) for r in groups[label])
            off = sum(1 for r in groups[label] for cid in r["arms"][arm]
                      if (chunk_map[cid].get("cloud_provider") or "").lower() not in gold[r["query_id"]])
            line.append(f"{off}/{slots} ({_pct(off / slots)})")
            rep.data.setdefault("typing", {}).setdefault(label, {})[arm] = {"off_provider": off, "slots": slots}
        body.append(line)
    rep.log("Fragmentos del top-5 que son de un proveedor que la pregunta no nombra:")
    rep.log()
    rep.table(["Grupo", "n"] + list(arms), body)
    rep.data["typing_counts"] = {"typed_cross_cloud": typed_cross, "single_provider_typed_cross_cloud": single_typed_cross,
                                 "groups": {label: len(items) for label, items in groups.items()}}


def probe_absolute_meaning(rep: Report) -> None:
    rep.log("## 7. Qué dicen las métricas en términos absolutos")
    rep.log()
    indep = json.loads((EXP11 / "retrieval_metrics__bge-reranker-indep.json").read_text(encoding="utf-8"))
    p50 = indep["thresholds"]["p50"]
    rep.log(f"- El oráculo independiente da probabilidades en (0, 1). La mediana del pool es {p50:.3f}: "
            "«relevante» en P@k, R@k y MRR significa «por encima de la mediana del pool», no «relevante». "
            "Por construcción la mitad del pool es relevante, sea cual sea su calidad real.")
    body = [[name, f"{s['ndcg@5_mean']:.4f}", f"{s['avg_score@5_mean']:.4f}", f"{s['precision@5__t0_mean']:.2f}"]
            for name, s in indep["systems"].items()]
    rep.log()
    rep.table(["Sistema", "NDCG@5 (relativo al pool)", "Probabilidad media de relevancia del top-5 según el oráculo",
               "P@5 con umbral 0"], body)
    best = max(indep["systems"].values(), key=lambda s: s["ndcg@5_mean"])
    rep.log(f"- Un NDCG@5 de {best['ndcg@5_mean']:.2f} no significa que esa fracción del top-5 sea relevante: el propio "
            f"oráculo asigna al top-5 del mejor sistema una probabilidad media de relevancia de "
            f"{best['avg_score@5_mean']:.2f}. La columna de umbral 0 vale 1,00 "
            "para todos con este oráculo (una probabilidad nunca es negativa): esa variante no discrimina.")
    rep.log()
    rep.data["absolute"] = {"p50_indep": p50, "avg_score": {n: s["avg_score@5_mean"] for n, s in indep["systems"].items()}}


def build_report() -> Report:
    rep = Report()
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    payload = json.loads(EXP20.read_text(encoding="utf-8"))
    rows = payload["results"]
    rep.log("# Sondas de concepto sobre el diseño de evaluación (2026-11-15)")
    rep.log()
    rep.log("Sin pesos de modelos: solo vocabularios (`tokenizer.json`), la corrida de auditoría exp20 (no firmada) "
            "y la evidencia firmada en modo lectura.")
    rep.log()
    tokens = probe_truncation(rep, chunk_map, rows)
    probe_truncation_effect(rep, tokens["nli"])
    ndcg7 = probe_pool_relative_ndcg(rep, rows)
    probe_overfetch(rep, rows, ndcg7)
    probe_length_preference(rep, rows, tokens["reranker"])
    probe_clustering(rep, rows, ndcg7)
    probe_model_contrasts(rep, rows)
    probe_duplicates(rep, chunk_map, rows)
    probe_absolute_meaning(rep)
    probe_query_typing(rep, chunk_map, rows)
    rep.log("## 9. Comprobaciones")
    rep.log()
    rep.log(f"- Fallos: {len(rep.failures)}" + (f" ({'; '.join(rep.failures)})" if rep.failures else ""))
    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="guarda el informe en output/audit/")
    args = parser.parse_args()
    rep = build_report()
    text = "\n".join(rep.lines) + "\n"
    sys.stdout.write(text)
    if args.write:
        out = ROOT / "output" / "audit"
        (out / f"{REPORT_STEM}.md").write_text(text, encoding="utf-8", newline="\n")
        (out / f"{REPORT_STEM}.json").write_text(json.dumps(rep.data, indent=1, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
