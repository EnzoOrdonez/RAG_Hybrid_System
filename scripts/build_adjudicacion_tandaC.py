"""Genera output/audit/adjudicacion_tandaC.html — adjudicacion de los discordantes test-retest.

Los 9 items donde la tanda C difirio del gold original. Muestra pregunta, claim, chunk y
AMBOS juicios previos de Enzo (1a pasada y re-test): la adjudicacion post-hoc es la unica
fase donde verlos es correcto. Enzo elige el veredicto final y escribe la razon
(obligatoria). El export lo fusiona un script con marca 'adjudicado' en el comentario.

Uso: python scripts/build_adjudicacion_tandaC.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_anotador_gold import HTML_TEMPLATE, load_questions, load_stage_a  # noqa: E402

AUDIT = ROOT / "output" / "audit"
OUT = AUDIT / "adjudicacion_tandaC.html"


def main() -> None:
    res = json.loads((AUDIT / "gold_v4_tandaC_resultado.json").read_text(encoding="utf-8"))
    items_a = {it["key"]: it for it in load_stage_a(load_questions())}
    items = []
    for i, disc in enumerate(res["discordantes"], start=1):
        it = dict(items_a[disc["ref"]])
        it["key"] = f"J-{i:02d}"
        it["ref"] = disc["ref"]
        it.pop("calibracion", None)
        it["chunk_label"] = (
            f"1.a pasada (gold): {disc['gold']}  ·  re-test (tanda C): {disc['retest']}"
        )
        items.append(it)

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps({"J": items}, ensure_ascii=False))
    html = html.replace(
        'const SECCIONES = [\n  {id:"T", titulo:"Taxonomía exp18", items:DATA.T,\n   nota:"40 claims que ningún chunk del pool superó el umbral. Aquí NO hay chunk: juzga si el claim es factualmente correcto según tu conocimiento (correcto = verdadero aunque el corpus no lo respalde; incorrecto = falso/alucinado; dudoso = no puedes saberlo)."},\n  {id:"A", titulo:"Etapa A (1 chunk)", items:DATA.A,\n   nota:"150 claims, cada uno con el mejor chunk. Juzga claim↔chunk, no claim↔mundo. A-001 a A-010 son tu calibración (tanda 0)."},\n  {id:"B", titulo:"Etapa B (5 chunks)", items:DATA.B,\n   nota:"50 claims con los 5 chunks que vio el modelo. Correcto = respaldado por ALGUNO. No mires lo que pusiste en A."}\n];',
        'const SECCIONES = [\n  {id:"J", titulo:"Adjudicación · 9 discordantes", items:DATA.J,\n   nota:"En cada item ves tus DOS juicios previos. Lee el chunk de nuevo, elige el veredicto FINAL y escribe la razón (obligatoria). Hazlo descansado, no ahora si llevas 5 h anotando."}\n];')
    html = html.replace('let seccion = "T";', 'let seccion = "J";')
    html = html.replace('const LS_KEY = "gold_v4_annotations_v1";',
                        'const LS_KEY = "gold_v4_adjudicacion_v1";')
    html = html.replace("<title>Anotador gold humano v4 — Enzo</title>",
                        "<title>Adjudicación tanda C — Enzo</title>")
    html = html.replace("<h1>Anotador gold humano v4 <span", "<h1>Adjudicación · discordantes test-retest <span")
    # seccion J usa el mismo render que A (chunk unico); obligar comentario en TODOS
    html = html.replace('if (s.id === "A") {', 'if (s.id === "A" || s.id === "J") {')
    html = html.replace(
        'placeholder="Comentario (obligatorio si dudoso): qué faltó, qué se contradijo…"',
        'placeholder="Razón de la adjudicación (OBLIGATORIA): por qué este veredicto y no el otro…"')
    html = html.replace(
        'if (b.dataset.v === "dudoso") document.getElementById("coment").classList.add("falta");',
        'document.getElementById("coment").classList.add("falta");')
    html = html.replace(
        'const payload = {formato:"gold_v4_anotaciones", version:1,\n                   exportado:new Date().toISOString(), juicios:estado};',
        'const payload = {formato:"gold_v4_adjudicacion", version:1,\n                   exportado:new Date().toISOString(),\n                   mapeo:Object.fromEntries(DATA.J.map(it=>[it.key,it.ref])),\n                   juicios:estado};')
    html = html.replace('a.download = "gold_v4_juicios_enzo_"', 'a.download = "gold_v4_adjudicacion_enzo_"')
    assert '"J"' in html and "gold_v4_adjudicacion" in html
    OUT.write_text(html, encoding="utf-8")
    print(f"OK {OUT} ({OUT.stat().st_size/1024:.0f} KB), {len(items)} items")


if __name__ == "__main__":
    main()
