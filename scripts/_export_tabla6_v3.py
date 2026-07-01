"""
Tabla 6 v3 (N8) — faithfulness_answered corregida, desde faithfulness_metrics_v3*.json.

Emite output/tables/nota3/tabla6_fidelidad_v3__<exp>.{md,csv} en el mismo formato
que tabla6_fidelidad_v2 (Escenario x modelo, coma decimal, n entre paréntesis).
NO toca las tablas v1/v2. Fuente por defecto = el verificador small (comparable
con la Tabla 6 publicada).

Usage:
  python scripts/_export_tabla6_v3.py [exp12_matrix] [faithfulness_metrics_v3_small.json]
"""
import json
import os
import sys

exp = sys.argv[1] if len(sys.argv) > 1 else "exp12_matrix"
src_name = sys.argv[2] if len(sys.argv) > 2 else "faithfulness_metrics_v3_small.json"
base = os.path.join("experiments", "results", exp)
out_dir = os.path.join("output", "tables", "nota3")
os.makedirs(out_dir, exist_ok=True)

fm = json.loads(open(os.path.join(base, src_name), encoding="utf-8").read())
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

note = (f"Métrica v3 (N8, verificador {fm.get('faithfulness_source','?')}): "
        "faithfulness_answered corregida = artefactos de formato excluidos del "
        "denominador (H1) + guarda de contradicción vb_agree (H2). `*` Sin RAG = 0 "
        "por construcción (N3). v1/v2 se conservan como superseded. "
        "**Veredictos:** retrieval n.s. en fidelidad (robusto, 0/12); entre-modelos "
        "2/18 sig bajo small (6/18 base) — corrige el 'todo n.s.' de N5.")

md = [f"# Tabla 6 v3 — Fidelidad corregida (N8) — {exp}\n", note, "",
      "| " + " | ".join(cols) + " |",
      "|" + "|".join(["---"] * len(cols)) + "|"]
for r in rows:
    md.append("| " + " | ".join(r) + " |")
csv = [";".join(cols)] + [";".join(r) for r in rows]

open(os.path.join(out_dir, f"tabla6_fidelidad_v3__{exp}.md"), "w", encoding="utf-8").write("\n".join(md))
open(os.path.join(out_dir, f"tabla6_fidelidad_v3__{exp}.csv"), "w", encoding="utf-8").write("\n".join(csv))
print("\n".join(md))
print(f"\nWrote tabla6_fidelidad_v3__{exp}.{{md,csv}} to {out_dir}")
