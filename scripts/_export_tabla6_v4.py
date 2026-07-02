"""
Tabla 6 v4 (N9) — faithfulness_answered sin filas vacuas, desde faithfulness_metrics_v4*.json.

v4 = v3 (artefactos excluidos del numerador/denominador de claims, N8) + exclusión
de las respuestas cuyo claims extraídos son TODOS artefactos (genuine==0), que en
v3 entraban al denominador primario con faithfulness=1.0 vacuo (59/1798, ledger N9).

Emite output/tables/nota3/tabla6_fidelidad_v4__<exp>.{md,csv} en el mismo formato
que tabla6_fidelidad_v3 (Escenario x modelo, coma decimal, n entre paréntesis).
NO toca las tablas v1/v2/v3. Fuente por defecto = el verificador small (comparable
con la Tabla 6 publicada).

Usage:
  python scripts/_export_tabla6_v4.py [exp12_matrix] [faithfulness_metrics_v4_small.json]
"""
import json
import os
import sys

exp = sys.argv[1] if len(sys.argv) > 1 else "exp12_matrix"
src_name = sys.argv[2] if len(sys.argv) > 2 else "faithfulness_metrics_v4_small.json"
base = os.path.join("experiments", "results", exp)
out_dir = os.path.join("output", "tables", "nota3")
os.makedirs(out_dir, exist_ok=True)

fm = json.loads(open(os.path.join(base, src_name), encoding="utf-8").read())
if not fm.get("vacuous_exclusion"):
    raise SystemExit(f"{src_name} no tiene vacuous_exclusion=true — no es un artefacto v4")
sv = fm["systems_v2"]
cell = {(s["scenario"], s["model"]): s for s in sv.values()}

SCEN_ORDER = ["sin_rag", "lexico", "denso", "hibrido"]
SCEN_DISP = {"sin_rag": "Sin RAG", "lexico": "RAG léxico (BM25)",
             "denso": "RAG denso (BGE)", "hibrido": "RAG híbrido"}
MODEL_DISP = {"granite4.1-8b": "Granite 4.1 8B", "gemma4-e4b": "Gemma 4 E4B",
              "mistral-7b-instruct": "Mistral 7B", "qwen3.5-9b": "Qwen 3.5 9B"}
order = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
models = [m for m in order if any((sc, m) in cell for sc in SCEN_ORDER)]


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def cellval(sc, m):
    s = cell.get((sc, m))
    if not s:
        return "—"
    p = s["primary_answered"]
    v = fmt(p["mean"]) + ("*" if sc == "sin_rag" else "")
    return f"{v} ({p['n']})"


cols = ["Escenario"] + [MODEL_DISP.get(m, m) for m in models]
rows = []
for sc in SCEN_ORDER:
    if not any((sc, m) in cell for m in models):
        continue
    rows.append([SCEN_DISP.get(sc, sc)] + [cellval(sc, m) for m in models])

note = (f"Métrica v4 (N9, verificador {fm.get('faithfulness_source','?')}): "
        "faithfulness_answered = v3 (artefactos de formato excluidos, H1; guarda "
        "vb_agree, H2) + exclusión de respuestas 100%-artefactos (genuine==0, "
        "faithfulness=1,0 vacuo en v3; 59/1798, mismo trato que method='none' "
        "por Flag 137). `*` Sin RAG = 0 por construcción (N3). v1/v2/v3 se "
        "conservan como superseded. **Veredictos:** retrieval n.s. en fidelidad "
        "(robusto, 0/12 bajo AMBOS verificadores); entre-modelos 1/18 sig bajo "
        "small (denso granite-vs-mistral, d_z=+0,42, p_bh=0,014) y 1/18 bajo "
        "base (léxico gemma-vs-granite, d_z=−0,53, p_bh=0,038) — corrige el "
        "'2/18' de N8, que dependía de filas vacuas.")

md = [f"# Tabla 6 v4 — Fidelidad sin vacuas (N9) — {exp}\n", note, "",
      "| " + " | ".join(cols) + " |",
      "|" + "|".join(["---"] * len(cols)) + "|"]
for r in rows:
    md.append("| " + " | ".join(r) + " |")
csv = [";".join(cols)] + [";".join(r) for r in rows]

open(os.path.join(out_dir, f"tabla6_fidelidad_v4__{exp}.md"), "w", encoding="utf-8").write("\n".join(md))
open(os.path.join(out_dir, f"tabla6_fidelidad_v4__{exp}.csv"), "w", encoding="utf-8").write("\n".join(csv))
print("\n".join(md))
print(f"\nWrote tabla6_fidelidad_v4__{exp}.{{md,csv}} to {out_dir}")
