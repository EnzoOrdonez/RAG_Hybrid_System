"""
Paper tables for Nota 3 — faithfulness matrix (Tabla 6) + primary-model detail
(Tabla 5), from exp12 faithfulness_metrics.json.

Reads experiments/results/<exp>/faithfulness_metrics.json and emits, into
output/tables/nota3/ (.csv + .md, comma decimals "0,923"):
  - tabla6_fidelidad__<exp>.{md,csv}     faithfulness mean, scenario x model
  - tabla6_declinacion__<exp>.{md,csv}   honest-decline rate, scenario x model
  - tabla5_modelo_principal__<exp>.{md,csv}  granite per-scenario detail

v2 (ledger N5): if faithfulness_metrics_v2.json exists, also emits:
  - tabla6_fidelidad_v2__<exp>            PRIMARY faithfulness_answered (+n)
  - tabla6_sensibilidad_denominador__<exp> 4 denominators side by side
  - tabla6c_clasificacion_v2__<exp>        pure/hedged/answered census
  - tabla5_modelo_principal_v2__<exp>      primary-model v2 detail
  - tabla_claims_desglose__<exp>           %supp/%contr/%unsup, claims/resp

Robust to which models are present, so re-running after qwen3.5 completes simply
adds its column. "Sin RAG" faithfulness is 0-by-construction (N3) and is
annotated; read it together with the decline table, not as a comparable score.

Usage: python scripts/_export_paper_tables_nota3.py [exp12_matrix] [primary_model_label]
"""
import json
import os
import sys

exp = sys.argv[1] if len(sys.argv) > 1 else "exp12_matrix"
primary = sys.argv[2] if len(sys.argv) > 2 else "granite4.1-8b"
base = os.path.join("experiments", "results", exp)
out_dir = os.path.join("output", "tables", "nota3")
os.makedirs(out_dir, exist_ok=True)

fm = json.loads(open(os.path.join(base, "faithfulness_metrics.json"), encoding="utf-8").read())
systems = fm["systems"]

SCEN_ORDER = ["sin_rag", "lexico", "denso", "hibrido"]
SCEN_DISP = {"sin_rag": "Sin RAG", "lexico": "RAG léxico (BM25)",
             "denso": "RAG denso (BGE)", "hibrido": "RAG híbrido"}
MODEL_DISP = {"granite4.1-8b": "Granite 4.1 8B", "gemma4-e4b": "Gemma 4 E4B",
              "mistral-7b-instruct": "Mistral 7B", "qwen3.5-9b": "Qwen 3.5 9B"}


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


# Discover models present, ordered.
models = []
for c in systems:
    m = systems[c]["model"]
    if m not in models:
        models.append(m)
order = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
models = [m for m in order if m in models] + [m for m in models if m not in order]

# cell lookup: (scenario, model) -> system dict
cell = {(s["scenario"], s["model"]): s for s in systems.values()}


def matrix_table(value_fn, title, note):
    cols = ["Escenario"] + [MODEL_DISP.get(m, m) for m in models]
    rows = []
    for sc in SCEN_ORDER:
        if not any((sc, m) in cell for m in models):
            continue
        row = [SCEN_DISP.get(sc, sc)]
        for m in models:
            s = cell.get((sc, m))
            row.append(value_fn(s) if s else "—")
        rows.append(row)
    md = [f"# {title} — {exp}\n", note, "",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    csv = [";".join(cols)] + [";".join(r) for r in rows]
    return "\n".join(md), "\n".join(csv)


def w(name, md, csv):
    open(os.path.join(out_dir, f"{name}__{exp}.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(out_dir, f"{name}__{exp}.csv"), "w", encoding="utf-8").write(csv)


# Tabla 6a — faithfulness (sin_rag annotated 0-by-construction)
def faith_cell(s):
    v = fmt(s["faithfulness_mean"])
    if s["scenario"] == "sin_rag":
        v += "*"
    return v
md, csv = matrix_table(
    faith_cell, "Tabla 6 — Fidelidad (faithfulness NLI) por escenario y modelo",
    "Media sobre respuestas con claims verificables (excluye method none/error). "
    "`*` Sin RAG = 0 por construcción (sin contexto no hay claim verificable, N3): "
    "léase junto a la tabla de declinación, no como fidelidad comparable.")
w("tabla6_fidelidad", md, csv)

# Tabla 6b — honest-decline rate
def decl_cell(s):
    dr = s.get("honest_decline_rate")
    return fmt(100 * dr, 1) + "%" if dr is not None else "—"
md, csv = matrix_table(
    decl_cell, "Tabla 6b — Tasa de declinación honesta por escenario y modelo",
    "Fracción de respuestas marcadas como declinación honesta "
    '("no hay información suficiente"). Confunde la comparación de fidelidad entre '
    "modelos: granite declina mucho más que mistral.")
w("tabla6_declinacion", md, csv)

# Tabla 5 — primary model per-scenario detail
pcols = ["Escenario", "Fidelidad", "Alucinación (1−fid)", "Declinación %", "No-evidencia %", "n_efectivo"]
prows = []
for sc in SCEN_ORDER:
    s = cell.get((sc, primary))
    if not s:
        continue
    dr = s.get("honest_decline_rate")
    prows.append([
        SCEN_DISP.get(sc, sc),
        fmt(s["faithfulness_mean"]) + ("*" if sc == "sin_rag" else ""),
        fmt(1 - s["faithfulness_mean"]),
        (fmt(100 * dr, 1) + "%") if dr is not None else "—",
        fmt(100 * s["no_evidence_rate"], 1) + "%",
        str(s["n_effective"]),
    ])
pmd = [f"# Tabla 5 — Modelo principal ({MODEL_DISP.get(primary, primary)}) por escenario — {exp}\n",
       "`*` Sin RAG fidelidad = 0 por construcción (N3).", "",
       "| " + " | ".join(pcols) + " |", "|" + "|".join(["---"] * len(pcols)) + "|"]
for r in prows:
    pmd.append("| " + " | ".join(r) + " |")
pcsv = [";".join(pcols)] + [";".join(r) for r in prows]
w("tabla5_modelo_principal", "\n".join(pmd), "\n".join(pcsv))

print(f"Models in matrix: {models}")
print(f"Wrote tabla6_fidelidad, tabla6_declinacion, tabla5_modelo_principal (__{exp}) to {out_dir}")
print(f"  (primary model = {primary})")

# ===========================================================================
# v2 tables (ledger N5) — only when faithfulness_metrics_v2.json exists
# ===========================================================================
v2_path = os.path.join(base, "faithfulness_metrics_v2.json")
if not os.path.exists(v2_path):
    print("No faithfulness_metrics_v2.json; skipping v2 tables.")
    sys.exit(0)

fm2 = json.loads(open(v2_path, encoding="utf-8").read())
sys2 = fm2["systems_v2"]
cell2 = {(s["scenario"], s["model"]): s for s in sys2.values()}
cfg_by_cell = {(s["scenario"], s["model"]): name for name, s in sys2.items()}

NOTE_V2 = ("Métrica primaria v2 (N5): media de fidelidad NLI sobre respuestas NO "
           "declinadas (excluye pure_decline = marcador de rechazo en los primeros "
           "300 caracteres; incluye respuestas con hedge tardío). n_answered entre "
           "paréntesis. `*` Sin RAG = 0 por construcción (N3).")


def faith_v2_cell(s):
    p = s["primary_answered"]
    v = fmt(p["mean"]) + ("*" if s["scenario"] == "sin_rag" else "")
    return f"{v} ({p['n']})"
md, csv = matrix_table(
    lambda s: faith_v2_cell(cell2[(s["scenario"], s["model"])]),
    "Tabla 6 v2 — Fidelidad en respuestas contestadas (faithfulness_answered)",
    NOTE_V2)
w("tabla6_fidelidad_v2", md, csv)

# Sensibilidad: 4 denominadores lado a lado (tabla larga por config)
DEN_COLS = [("primary_answered", "Primaria (sin pure_decline)"),
            ("sens_a_v1flag", "Sens. A (flag v1)"),
            ("sens_b_strict", "Sens. B (estricta)"),
            ("sens_c_published", "Sens. C (publicada v1)")]
hdr = ["Config"] + [f"{d[1]}" for d in DEN_COLS] + ["n primaria", "n publicada"]
rows_s = []
for sc in SCEN_ORDER:
    for m in models:
        s = cell2.get((sc, m))
        if not s:
            continue
        rows_s.append([f"{SCEN_DISP.get(sc, sc)} — {MODEL_DISP.get(m, m)}"]
                      + [fmt(s[d[0]]["mean"]) for d in DEN_COLS]
                      + [str(s["primary_answered"]["n"]), str(s["sens_c_published"]["n"])])
md_s = ["# Tabla — Sensibilidad del denominador (4 definiciones) — " + exp, "",
        "Las conclusiones deben leerse bajo las 4 definiciones (espejo del multi-oráculo).", "",
        "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
for r in rows_s:
    md_s.append("| " + " | ".join(r) + " |")
csv_s = [";".join(hdr)] + [";".join(r) for r in rows_s]
w("tabla6_sensibilidad_denominador", "\n".join(md_s), "\n".join(csv_s))

# Census v2: % pure / hedged / answered por config
hdr_c = ["Config", "% pure_decline", "% hedged_partial", "% answered", "% vacías", "n"]
rows_c = []
for sc in SCEN_ORDER:
    for m in models:
        s = cell2.get((sc, m))
        if not s:
            continue
        cen = s["decline_census_v2"]
        n_cls = max(1, s["n_total"] - cen["(empty)"])
        rows_c.append([f"{SCEN_DISP.get(sc, sc)} — {MODEL_DISP.get(m, m)}",
                       fmt(100 * cen["pure_decline"] / n_cls, 1),
                       fmt(100 * cen["hedged_partial"] / n_cls, 1),
                       fmt(100 * cen["answered"] / n_cls, 1),
                       fmt(100 * cen["(empty)"] / s["n_total"], 1),
                       str(s["n_total"])])
md_c = ["# Tabla 6c — Clasificación v2 de respuestas (census) — " + exp, "",
        "pure_decline: rechazo en apertura (300c). hedged_partial: marcador de rechazo "
        "tardío con contenido sustantivo. answered: sin marcador. % vacías sobre n total "
        "(qwen sin_rag: 91,2 % vacías).", "",
        "| " + " | ".join(hdr_c) + " |", "|" + "|".join(["---"] * len(hdr_c)) + "|"]
for r in rows_c:
    md_c.append("| " + " | ".join(r) + " |")
csv_c = [";".join(hdr_c)] + [";".join(r) for r in rows_c]
w("tabla6c_clasificacion_v2", "\n".join(md_c), "\n".join(csv_c))

# Tabla 5 v2 — primary model detail
pcols2 = ["Escenario", "Fidelidad (primaria)", "n_answered", "Sens. A", "Sens. B",
          "Publicada v1", "% pure_decline", "% hedged"]
prows2 = []
for sc in SCEN_ORDER:
    s = cell2.get((sc, primary))
    if not s:
        continue
    cen = s["decline_census_v2"]
    n_cls = max(1, s["n_total"] - cen["(empty)"])
    prows2.append([
        SCEN_DISP.get(sc, sc),
        fmt(s["primary_answered"]["mean"]) + ("*" if sc == "sin_rag" else ""),
        str(s["primary_answered"]["n"]),
        fmt(s["sens_a_v1flag"]["mean"]),
        fmt(s["sens_b_strict"]["mean"]),
        fmt(s["sens_c_published"]["mean"]),
        fmt(100 * cen["pure_decline"] / n_cls, 1),
        fmt(100 * cen["hedged_partial"] / n_cls, 1),
    ])
pmd2 = [f"# Tabla 5 v2 — Modelo principal ({MODEL_DISP.get(primary, primary)}) — {exp}\n",
        NOTE_V2, "",
        "| " + " | ".join(pcols2) + " |", "|" + "|".join(["---"] * len(pcols2)) + "|"]
for r in prows2:
    pmd2.append("| " + " | ".join(r) + " |")
pcsv2 = [";".join(pcols2)] + [";".join(r) for r in prows2]
w("tabla5_modelo_principal_v2", "\n".join(pmd2), "\n".join(pcsv2))

# Desglose de claims por config (f3)
cb = fm2.get("claims_breakdown", {})
hdr_k = ["Config", "% supported", "% contradicted", "% unsupported",
         "claims/resp", "claims tot", "n resp"]
rows_k = []
for sc in SCEN_ORDER:
    for m in models:
        name = cfg_by_cell.get((sc, m))
        k = cb.get(name) if name else None
        if not k:
            continue
        rows_k.append([f"{SCEN_DISP.get(sc, sc)} — {MODEL_DISP.get(m, m)}",
                       fmt(k["supported_pct"], 1), fmt(k["contradicted_pct"], 1),
                       fmt(k["unsupported_pct"], 1), fmt(k["claims_per_response"], 1),
                       str(k["total_claims"]), str(k["n_responses"])])
md_k = ["# Tabla — Desglose de claims NLI por config — " + exp, "",
        "Vista a nivel instrumento (todas las respuestas con claims, método nli). "
        "La banda de % contradicted ~27-39 % casi insensible al escenario es evidencia "
        "del artefacto del verificador (N5/A3).", "",
        "| " + " | ".join(hdr_k) + " |", "|" + "|".join(["---"] * len(hdr_k)) + "|"]
for r in rows_k:
    md_k.append("| " + " | ".join(r) + " |")
csv_k = [";".join(hdr_k)] + [";".join(r) for r in rows_k]
w("tabla_claims_desglose", "\n".join(md_k), "\n".join(csv_k))

print(f"Wrote v2 tables: tabla6_fidelidad_v2, tabla6_sensibilidad_denominador, "
      f"tabla6c_clasificacion_v2, tabla5_modelo_principal_v2, tabla_claims_desglose (__{exp})")

# ===========================================================================
# Tabla 4 — retrieval (exp11, multi-oráculo) — I5/N7
# ===========================================================================
EXP11 = os.path.join("experiments", "results", "exp11_retrieval194_fullrerank")
ORACLES = [("bge-reranker-indep", "bge-reranker-large (independiente)"),
           ("ms-marco-circular", "ms-marco (circular — referencia)")]
if all(os.path.exists(os.path.join(EXP11, f"retrieval_metrics__{o}.json")) for o, _ in ORACLES):
    SYS_ORDER = [("RAG Lexico (BM25)", "Léxico (BM25)"),
                 ("RAG Semantico (Dense)", "Denso (BGE)"),
                 ("RAG Hibrido (pre-rerank RRF)", "Híbrido pre-rerank (RRF)"),
                 ("RAG Hibrido Propuesto", "Híbrido post-rerank")]
    METRICS4 = [("precision@1__p50_mean", "P@1"), ("precision@3__p50_mean", "P@3"),
                ("precision@5__p50_mean", "P@5"), ("recall@5__p50_mean", "R@5"),
                ("mrr__p50_mean", "MRR"), ("ndcg@5_mean", "NDCG@5")]
    md4 = ["# Tabla 4 — Métricas de retrieval por sistema y oráculo — exp11 (194 q)\n",
           "Oráculo titular: **bge-reranker-large (independiente)**; ms-marco se reporta "
           "solo como referencia porque es el reranker del propio pipeline (circularidad "
           "cuantificada en N2: NDCG@5 0,995 por construcción). P/R/MRR con umbral de "
           "relevancia p50 del oráculo; NDCG@5 graded sin umbral. Coma decimal.", ""]
    csv4 = []
    for okey, olabel in ORACLES:
        d4 = json.loads(open(os.path.join(EXP11, f"retrieval_metrics__{okey}.json"),
                             encoding="utf-8").read())
        s4 = d4["systems"]
        hdr4 = ["Sistema"] + [m[1] for m in METRICS4]
        md4 += [f"## Oráculo: {olabel}", "",
                "| " + " | ".join(hdr4) + " |",
                "|" + "|".join(["---"] * len(hdr4)) + "|"]
        csv4.append(f"# {olabel}")
        csv4.append(";".join(hdr4))
        for skey, slabel in SYS_ORDER:
            row = [slabel] + [fmt(s4[skey][mk]) for mk, _ in METRICS4]
            md4.append("| " + " | ".join(row) + " |")
            csv4.append(";".join(row))
        md4.append("")
    w_exp = "exp11_retrieval194_fullrerank"
    open(os.path.join(out_dir, f"tabla4_retrieval__{w_exp}.md"), "w", encoding="utf-8").write("\n".join(md4))
    open(os.path.join(out_dir, f"tabla4_retrieval__{w_exp}.csv"), "w", encoding="utf-8").write("\n".join(csv4))
    print(f"Wrote tabla4_retrieval__{w_exp} (2 oráculos)")
