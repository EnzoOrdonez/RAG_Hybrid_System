"""Genera output/audit/anotador_gold.html — anotador offline del gold humano.

Un solo HTML autocontenido (sin red, sin servidor) para anotar desde el
telefono las tres tandas del gold:
  T = taxonomia exp18 (40 claims, output/audit/unsupported_claims_sample_v2.csv)
  A = etapa A v4 (150 claims, output/audit/claim_audit_sample_v4.csv)
  B = etapa B v4 (50 claims, output/audit/claim_audit_sample_v4_stageB.csv)

Cegamiento: NO incluye scores automaticos, estratos, configs ni ningun juicio
de LLM (los archivos *_blind nunca se leen aqui). Solo pregunta, claim y
chunk(s), que es exactamente lo que la guia permite ver.

Persistencia: localStorage del navegador + exportar/importar JSON.
El JSON exportado lo vuelca Kimi/Codex a los CSV con un script aparte;
Enzo NO edita los CSV a mano.

Uso: python scripts/build_anotador_gold.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "output" / "audit"
QUERIES = ROOT / "data" / "evaluation" / "test_queries.json"
OUT = AUDIT / "anotador_gold.html"


def load_questions() -> dict[str, str]:
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("queries", [])
    out = {}
    for q in items:
        qid = q.get("query_id") or q.get("id")
        text = q.get("question") or q.get("query") or q.get("text") or ""
        if qid:
            out[qid] = text
    return out


def clean(text: str) -> str:
    # ⏎ marca saltos de linea reales en los chunks exportados
    return (text or "").replace("⏎", "\n").strip()


def load_taxonomy(questions: dict[str, str]) -> list[dict]:
    rows = []
    with (AUDIT / "unsupported_claims_sample_v2.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter=","), start=1):
            rows.append(
                {
                    "key": f"T-{i:02d}",
                    "query_id": row["query_id"],
                    "question": questions.get(row["query_id"], ""),
                    "claim": clean(row["claim"]),
                }
            )
    return rows


def load_stage_a(questions: dict[str, str]) -> list[dict]:
    rows = []
    with (AUDIT / "claim_audit_sample_v4.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            idx = int(row["idx"])
            rows.append(
                {
                    "key": f"A-{idx:03d}",
                    "query_id": row["query_id"],
                    "question": row["question"] or questions.get(row["query_id"], ""),
                    "claim": clean(row["claim"]),
                    "chunk": clean(row["best_chunk_text"]),
                    "chunk_label": row.get("best_chunk_source", ""),
                    "calibracion": idx <= 10,
                }
            )
    return rows


def split_evidence(text: str) -> list[dict]:
    """Divide evidence_all_chunks en segmentos E1..E5."""
    text = clean(text)
    parts = re.split(r"\[E(\d)\]\s*", text)
    # parts = ['', '1', 'texto1', '2', 'texto2', ...]
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        label = ""
        m = re.match(r"([^\n]*?::[^\n]*?)\n", body)
        if m:
            label = m.group(1).strip()
        chunks.append({"n": int(parts[i]), "label": label, "text": body})
    return chunks


def load_stage_b(questions: dict[str, str]) -> list[dict]:
    rows = []
    with (AUDIT / "claim_audit_sample_v4_stageB.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            idx = int(row["idx"])
            rows.append(
                {
                    "key": f"B-{idx:02d}",
                    "query_id": row["query_id"],
                    "question": row["question"] or questions.get(row["query_id"], ""),
                    "claim": clean(row["claim"]),
                    "chunks": split_evidence(row["evidence_all_chunks"]),
                }
            )
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anotador gold humano v4 — Enzo</title>
<style>
  :root { --bg:#faf9f7; --ink:#1c1b1a; --muted:#6b6763; --line:#e3e0db;
          --ok:#1a7f4b; --bad:#b3261e; --dud:#9a6a00; --accent:#2b4c7e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
         font-size:16px; line-height:1.5; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
           padding:10px 14px; z-index:10; }
  header h1 { font-size:17px; margin:0 0 6px; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; }
  .tabs button { border:1px solid var(--line); background:#fff; border-radius:20px;
                 padding:6px 12px; font-size:14px; cursor:pointer; }
  .tabs button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .tabs .prog { color:var(--muted); font-size:13px; align-self:center; }
  main { max-width:820px; margin:0 auto; padding:14px 14px 120px; }
  details.guia { background:#fff; border:1px solid var(--line); border-radius:10px;
                 padding:10px 14px; margin-bottom:14px; font-size:14px; }
  details.guia summary { cursor:pointer; font-weight:600; }
  details.guia table { border-collapse:collapse; width:100%; margin-top:8px; }
  details.guia td, details.guia th { border:1px solid var(--line); padding:6px 8px;
                                     vertical-align:top; text-align:left; }
  .card { background:#fff; border:1px solid var(--line); border-radius:12px;
          padding:14px; margin-bottom:12px; }
  .meta { font-size:13px; color:var(--muted); margin-bottom:6px; display:flex;
          justify-content:space-between; gap:8px; flex-wrap:wrap; }
  .badge { background:#f0e8d8; color:#7a5b12; border-radius:10px; padding:1px 8px;
           font-size:12px; }
  .pregunta { font-size:14px; color:var(--muted); margin-bottom:10px; }
  .pregunta b { color:var(--ink); }
  .claim { font-size:17px; font-weight:600; background:#f4f6fb; border-left:4px solid var(--accent);
           padding:10px 12px; border-radius:0 8px 8px 0; margin-bottom:12px; }
  .chunk { background:#f7f6f4; border:1px solid var(--line); border-radius:8px;
           padding:10px 12px; font-size:13.5px; white-space:pre-wrap; word-break:break-word;
           max-height:340px; overflow-y:auto; margin-bottom:10px; }
  .chunk .clabel { font-weight:600; color:var(--accent); font-size:12.5px; }
  .opciones { display:flex; gap:8px; margin:12px 0 10px; }
  .opciones button { flex:1; padding:14px 6px; font-size:16px; font-weight:600;
                     border-radius:10px; border:2px solid var(--line); background:#fff;
                     cursor:pointer; }
  .opciones button small { display:block; font-weight:400; font-size:11px; color:var(--muted);
                           margin-top:2px; }
  .opciones button.sel-correcto { border-color:var(--ok); background:#e8f5ee; color:var(--ok); }
  .opciones button.sel-incorrecto { border-color:var(--bad); background:#fbecea; color:var(--bad); }
  .opciones button.sel-dudoso { border-color:var(--dud); background:#fdf3dc; color:var(--dud); }
  textarea { width:100%; min-height:64px; border:1px solid var(--line); border-radius:8px;
             padding:8px 10px; font-size:14px; font-family:inherit; resize:vertical; }
  textarea.falta { border-color:var(--dud); background:#fdf8ec; }
  .nav { position:fixed; left:0; right:0; bottom:0; background:#fff;
         border-top:1px solid var(--line); padding:10px 14px; display:flex; gap:8px;
         align-items:center; z-index:10; }
  .nav button { padding:10px 16px; border-radius:8px; border:1px solid var(--line);
                background:#fff; font-size:15px; cursor:pointer; }
  .nav .spacer { flex:1; }
  .nav .io button { font-size:13px; padding:8px 10px; }
  .indice { display:grid; grid-template-columns:repeat(auto-fill,minmax(64px,1fr)); gap:6px; }
  .indice button { padding:8px 2px; font-size:13px; border-radius:6px;
                   border:1px solid var(--line); background:#fff; cursor:pointer; }
  .indice button.ok { background:#e8f5ee; border-color:var(--ok); }
  .indice button.actual { outline:2px solid var(--accent); }
  .aviso { background:#fdf3dc; border:1px solid #e8d9a8; border-radius:8px;
           padding:10px 12px; font-size:14px; margin-bottom:12px; }
  .guardado { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Anotador gold humano v4 <span class="guardado" id="guardado"></span></h1>
  <div class="tabs" id="tabs"></div>
</header>
<main>
  <details class="guia">
    <summary>📖 Reglas rápidas (guía §1) — tócame para recordar</summary>
    <p><b>No respondes la pregunta: juzgas si el claim queda respaldado por el chunk que
    tienes delante.</b> Tu conocimiento de AWS/Azure/GCP sirve para entender el texto,
    nunca como fuente de verdad.</p>
    <p><b>Regla de bolsillo:</b> «¿Un lector cuidadoso, usando SOLO este texto, podría
    escribir este claim?» Sí → correcto. No → incorrecto o dudoso.</p>
    <table>
      <tr><th>Etiqueta</th><th>Definición</th></tr>
      <tr><td><b>correcto</b></td><td>Todo lo material del claim está dicho o se sigue
        directamente del chunk. Vale paráfrasis. Números, entidades y alcance deben
        coincidir.</td></tr>
      <tr><td><b>incorrecto</b></td><td>El chunk contradice el claim, trata de otra cosa,
        o el claim añade hechos que el chunk no menciona. Chunk on-topic que silencia el
        dato concreto del claim → incorrecto.</td></tr>
      <tr><td><b>dudoso</b></td><td>Chunk del tema correcto pero genuinamente a medias:
        truncado justo donde estaría la respuesta, ambiguo, o respalda solo una parte.
        <b>Comentario obligatorio.</b></td></tr>
    </table>
    <p>⚠️ Trampas: provider swap, números cambiados, negaciones (<i>not / only / except</i>),
    generalizaciones, claims multi-parte (todas las partes deben estar respaldadas),
    markdown corrupto, y la «confianza por familiaridad» (si piensas "esto es obviamente
    cierto", vuelve al chunk y busca el respaldo literal).</p>
    <p>En <b>etapa B</b>: correcto = respaldado por ALGUNO de los 5 chunks; para incorrecto
    o dudoso lee los 5. No compares con lo que pusiste en etapa A.</p>
  </details>

  <div id="avisoB" class="aviso" style="display:none">
    ⚠️ La guía pide terminar y entregar la etapa A completa antes de anotar B
    (B mide cuántos juicios cambian al ver los 5 chunks; solo funciona si es independiente).
  </div>

  <div id="vista"></div>
</main>

<nav class="nav">
  <button id="prev">← Anterior</button>
  <button id="next">Siguiente →</button>
  <span class="spacer"></span>
  <span class="io">
    <button id="exportar">💾 Exportar</button>
    <button id="importar">📥 Importar</button>
    <input type="file" id="archivo" accept="application/json" style="display:none">
  </span>
</nav>

<script>
const DATA = __DATA__;
const LS_KEY = "gold_v4_annotations_v1";
const SECCIONES = [
  {id:"T", titulo:"Taxonomía exp18", items:DATA.T,
   nota:"40 claims que ningún chunk del pool superó el umbral. Aquí NO hay chunk: juzga si el claim es factualmente correcto según tu conocimiento (correcto = verdadero aunque el corpus no lo respalde; incorrecto = falso/alucinado; dudoso = no puedes saberlo)."},
  {id:"A", titulo:"Etapa A (1 chunk)", items:DATA.A,
   nota:"150 claims, cada uno con el mejor chunk. Juzga claim↔chunk, no claim↔mundo. A-001 a A-010 son tu calibración (tanda 0)."},
  {id:"B", titulo:"Etapa B (5 chunks)", items:DATA.B,
   nota:"50 claims con los 5 chunks que vio el modelo. Correcto = respaldado por ALGUNO. No mires lo que pusiste en A."}
];

let estado = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
let seccion = "T";
let pos = 0;
let soloPend = false;

function guardar() {
  localStorage.setItem(LS_KEY, JSON.stringify(estado));
  document.getElementById("guardado").textContent =
    "✓ guardado " + new Date().toLocaleTimeString();
}
function anot(key) { return estado[key] || null; }
function sec() { return SECCIONES.find(s => s.id === seccion); }
function pendientes(s) { return s.items.filter(it => !anot(it.key)); }

function renderTabs() {
  const el = document.getElementById("tabs");
  el.innerHTML = "";
  for (const s of SECCIONES) {
    const hechas = s.items.length - pendientes(s).length;
    const b = document.createElement("button");
    b.textContent = `${s.titulo} ${hechas}/${s.items.length}`;
    if (s.id === seccion) b.classList.add("active");
    b.onclick = () => { seccion = s.id; pos = 0; render(); };
    el.appendChild(b);
  }
  const p = document.createElement("span");
  p.className = "prog";
  const total = SECCIONES.reduce((a,s)=>a+s.items.length,0);
  const done = total - SECCIONES.reduce((a,s)=>a+pendientes(s).length,0);
  p.textContent = `Total ${done}/${total}`;
  el.appendChild(p);
  const f = document.createElement("button");
  f.textContent = soloPend ? "Ver todos" : "Solo pendientes";
  f.onclick = () => { soloPend = !soloPend; pos = 0; render(); };
  el.appendChild(f);
}

function listaVisible() {
  const s = sec();
  return soloPend ? pendientes(s) : s.items;
}

function render() {
  renderTabs();
  document.getElementById("avisoB").style.display =
    (seccion === "B" && pendientes(SECCIONES[1]).length > 0) ? "block" : "none";
  const s = sec();
  const lista = listaVisible();
  const vista = document.getElementById("vista");
  if (lista.length === 0) {
    vista.innerHTML = `<div class="card">🎉 ${s.titulo}: todo anotado en esta vista.</div>` + indice(s);
    return;
  }
  if (pos >= lista.length) pos = lista.length - 1;
  const it = lista[pos];
  const a = anot(it.key) || {};
  let html = `<div class="card">
    <div class="meta"><span><b>${it.key}</b> · ${pos+1}/${lista.length} · ${it.query_id}</span>
    ${it.calibracion ? '<span class="badge">tanda 0 · calibración</span>' : ''}</div>
    <div class="pregunta"><b>Pregunta (contexto):</b> ${esc(it.question)}</div>
    <div class="claim">${esc(it.claim)}</div>`;
  if (s.id === "A") {
    html += `<div class="chunk">${it.chunk_label ? `<div class="clabel">${esc(it.chunk_label)}</div>` : ""}${esc(it.chunk)}</div>`;
  } else if (s.id === "B") {
    for (const c of it.chunks) {
      html += `<div class="chunk"><div class="clabel">E${c.n}${c.label ? " · " + esc(c.label) : ""}</div>${esc(c.text)}</div>`;
    }
  } else {
    html += `<div class="aviso">Sin chunk: juzga el claim con tu conocimiento del tema.</div>`;
  }
  html += `<div class="opciones">`;
  for (const v of ["correcto","incorrecto","dudoso"]) {
    const defs = {correcto:"todo respaldado", incorrecto:"contradicho / sin respaldo", dudoso:"a medias, comenta"};
    html += `<button data-v="${v}" class="${a.juicio===v ? "sel-"+v : ""}">${v}<small>${defs[v]}</small></button>`;
  }
  html += `</div>
    <textarea id="coment" placeholder="Comentario (obligatorio si dudoso): qué faltó, qué se contradijo…">${esc(a.comentario||"")}</textarea>
  </div>`;
  html += indice(s);
  vista.innerHTML = html;

  vista.querySelectorAll(".opciones button").forEach(b => {
    b.onclick = () => {
      estado[it.key] = {juicio: b.dataset.v,
                        comentario: document.getElementById("coment").value.trim(),
                        ts: new Date().toISOString()};
      guardar();
      if (b.dataset.v === "dudoso") document.getElementById("coment").classList.add("falta");
      // avanza solo si no es dudoso (dudoso suele necesitar comentario)
      if (b.dataset.v !== "dudoso" && pos < lista.length - 1) { pos++; }
      render();
    };
  });
  document.getElementById("coment").oninput = (e) => {
    if (estado[it.key]) { estado[it.key].comentario = e.target.value; guardar(); }
    e.target.classList.toggle("falta",
      estado[it.key] && estado[it.key].juicio === "dudoso" && !e.target.value.trim());
  };
}

function indice(s) {
  const lista = listaVisible();
  let html = `<div class="card"><div class="meta"><b>Índice ${s.titulo}</b>
    <span>${s.items.length - pendientes(s).length}/${s.items.length}</span></div>
    <div class="indice">`;
  s.items.forEach((it) => {
    const done = !!anot(it.key);
    const esActual = lista[pos] && lista[pos].key === it.key;
    html += `<button class="${done ? "ok" : ""} ${esActual ? "actual" : ""}"
      data-key="${it.key}">${it.key.split("-")[1]}</button>`;
  });
  html += `</div></div>`;
  return html;
}

document.getElementById("vista").addEventListener("click", (e) => {
  const b = e.target.closest(".indice button");
  if (!b) return;
  const lista = listaVisible();
  const i = lista.findIndex(it => it.key === b.dataset.key);
  if (i >= 0) { pos = i; render(); window.scrollTo(0,0); }
});
document.getElementById("prev").onclick = () => { if (pos > 0) { pos--; render(); window.scrollTo(0,0);} };
document.getElementById("next").onclick = () => {
  const lista = listaVisible();
  if (pos < lista.length - 1) { pos++; render(); window.scrollTo(0,0); }
};

document.getElementById("exportar").onclick = () => {
  const payload = {formato:"gold_v4_anotaciones", version:1,
                   exportado:new Date().toISOString(), juicios:estado};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gold_v4_juicios_enzo_" + new Date().toISOString().slice(0,10) + ".json";
  a.click();
};
document.getElementById("importar").onclick = () => document.getElementById("archivo").click();
document.getElementById("archivo").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    try {
      const d = JSON.parse(r.result);
      const j = d.juicios || d;
      if (!confirm(`Importar ${Object.keys(j).length} juicios y SOBRESCRIBIR lo actual?`)) return;
      estado = j; guardar(); render();
    } catch (err) { alert("Archivo no válido: " + err.message); }
  };
  r.readAsText(f);
};

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

render();
</script>
</body>
</html>
"""


def main() -> None:
    questions = load_questions()
    data = {
        "T": load_taxonomy(questions),
        "A": load_stage_a(questions),
        "B": load_stage_b(questions),
    }
    counts = {k: len(v) for k, v in data.items()}
    assert counts == {"T": 40, "A": 150, "B": 50}, f"conteos inesperados: {counts}"
    # Cegamiento: ningun campo prohibido debe colarse
    prohibidos = {"best_over_pool", "decline_class", "stratum", "config",
                  "juicio_llm", "juicio_humano", "human_verdict"}
    for seccion, items in data.items():
        for it in items:
            filtrados = prohibidos & set(it)
            assert not filtrados, f"{seccion}: campos prohibidos {filtrados}"

    payload = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"OK {OUT} ({OUT.stat().st_size/1024:.0f} KB) conteos={counts}")


if __name__ == "__main__":
    main()
