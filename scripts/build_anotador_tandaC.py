"""Genera output/audit/anotador_tandaC.html — re-anotacion ciega de control (guia seccion 4.4).

20 idx aleatorios de la etapa A con seed 42, SIN mostrar el juicio original.
El export incluye el mapeo C-xx -> A-xxx para que el analisis de auto-acuerdo
lo haga un script; Enzo nunca ve sus juicios previos aqui.

Uso: python scripts/build_anotador_tandaC.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_anotador_gold import HTML_TEMPLATE, load_questions, load_stage_a  # noqa: E402

OUT = ROOT / "output" / "audit" / "anotador_tandaC.html"


def main() -> None:
    rng = random.Random(42)
    elegidos = sorted(rng.sample(range(1, 151), 20))
    items_a = {it["key"]: it for it in load_stage_a(load_questions())}
    items = []
    for i, idx in enumerate(elegidos, start=1):
        it = dict(items_a[f"A-{idx:03d}"])
        it["key"] = f"C-{i:02d}"
        it["ref"] = f"A-{idx:03d}"  # va al export, no se muestra en la UI
        it.pop("calibracion", None)
        items.append(it)

    data = {"C": items}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    # una sola seccion: reemplazar la tabla de secciones y el titulo
    html = html.replace(
        'const SECCIONES = [\n  {id:"T", titulo:"Taxonomía exp18", items:DATA.T,\n   nota:"40 claims que ningún chunk del pool superó el umbral. Aquí NO hay chunk: juzga si el claim es factualmente correcto según tu conocimiento (correcto = verdadero aunque el corpus no lo respalde; incorrecto = falso/alucinado; dudoso = no puedes saberlo)."},\n  {id:"A", titulo:"Etapa A (1 chunk)", items:DATA.A,\n   nota:"150 claims, cada uno con el mejor chunk. Juzga claim↔chunk, no claim↔mundo. A-001 a A-010 son tu calibración (tanda 0)."},\n  {id:"B", titulo:"Etapa B (5 chunks)", items:DATA.B,\n   nota:"50 claims con los 5 chunks que vio el modelo. Correcto = respaldado por ALGUNO. No mires lo que pusiste en A."}\n];',
        'const SECCIONES = [\n  {id:"C", titulo:"Tanda C · control", items:DATA.C,\n   nota:"20 claims de la etapa A elegidos al azar (seed 42). Re-anótalos a ciegas: NO consultes lo que pusiste antes; el archivo tampoco te lo muestra."}\n];')
    html = html.replace('let seccion = "T";', 'let seccion = "C";')
    html = html.replace('const LS_KEY = "gold_v4_annotations_v1";',
                        'const LS_KEY = "gold_v4_tandaC_annotations_v1";')
    html = html.replace("<title>Anotador gold humano v4 — Enzo</title>",
                        "<title>Tanda C — control de auto-acuerdo — Enzo</title>")
    html = html.replace("<h1>Anotador gold humano v4 <span", "<h1>Tanda C · re-anotación ciega <span")
    # export incluye el mapeo C->A
    html = html.replace(
        'const payload = {formato:"gold_v4_anotaciones", version:1,\n                   exportado:new Date().toISOString(), juicios:estado};',
        'const payload = {formato:"gold_v4_tandaC", version:1,\n                   exportado:new Date().toISOString(),\n                   mapeo:Object.fromEntries(DATA.C.map(it=>[it.key,it.ref])),\n                   juicios:estado};')
    html = html.replace('a.download = "gold_v4_juicios_enzo_"', 'a.download = "gold_v4_tandaC_enzo_"')
    OUT.write_text(html, encoding="utf-8")
    print(f"OK {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print("idx elegidos (seed 42):", elegidos)


if __name__ == "__main__":
    main()
