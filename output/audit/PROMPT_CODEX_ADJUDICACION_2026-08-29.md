# PROMPT MAESTRO PARA CODEX — maquinaria de adjudicación + sección del paper (2026-08-29)

```
CONTEXTO. Repo C:\Users\enziz\projects\hybrid-rag-system, rama summer/taxonomia-759.
El gold humano (240/240) ya está fusionado en los 3 CSV y analizado (entradas 27, 28 y 28b
de paper/summer_ablation_log.md — LÉELAS antes de escribir nada). La tanda C dio
auto-acuerdo 11/20 (55%, κ=0,268); los 9 discordantes se adjudicarán HOY por Enzo con
output/audit/adjudicacion_tandaC.html, que exportará un JSON con formato:
  {"formato":"gold_v4_adjudicacion","mapeo":{"J-01":"A-007",...},
   "juicios":{"J-01":{"juicio":"correcto|incorrecto|dudoso","comentario":"...","ts":"..."}}}

TAREA A — scripts/merge_adjudicacion_tandaC.py (nuevo):
  - Lee el JSON de adjudicación (ruta por argv, default: el más reciente
    output/audit/gold_v4_adjudicacion_enzo_*.json).
  - Exige: 9/9 claves J, mapeo a A-xxx válido, vocabulario exacto, comentario NO vacío
    en todos (la razón es obligatoria).
  - Actualiza SOLO las 9 filas correspondientes de output/audit/claim_audit_sample_v4.csv
    (delimitador ';', UTF-8 BOM): juicio_humano = veredicto final,
    comentario = "adjudicado: <razón>" (conserva el comentario previo entre corchetes al
    final si existía). Backup del CSV en output/audit/backups_adjudicacion_<fecha>/ antes
    de escribir. Si el JSON no existe todavía, el script aborta limpio con mensaje.
  - Tests nuevos en tests/test_merge_adjudicacion.py con JSON sintético en tmp_path:
    merge correcto, aborta si falta clave, aborta si comentario vacío, no toca otras filas.

TAREA B — scripts/run_gold_sensitivity.py (nuevo):
  - Ejecuta el análisis del gold en 3 variantes y escribe
    output/audit/gold_v4_sensitivity.{json,md}:
    (1) etiquetas actuales (post-adjudicación), (2) excluyendo los 9 idx adjudicados,
    (3) etiquetas pre-adjudicación (leyendo el backup en output/audit/backups_pre_merge_2026-08-29/
    y aplicando tanda C? NO: pre-adjudicación = estado actual del CSV menos los cambios del
    JSON de adjudicación; reconstruye usando el JSON, nunca re-anotes nada).
  - Reutiliza la lógica de scripts/analyze_gold_v4.py (impórtala; NO la copies ni la edites).
    Si analyze_gold_v4 no es importable sin efectos laterales, declara el bloqueo en vez de
    refactorizarla.
  - Reporta por variante: κ ponderada y κ anchor por candidato, y Δ vs la variante (1).
  - Si el JSON de adjudicación aún no existe, corre solo la variante (2) usando
    output/audit/gold_v4_tandaC_resultado.json (campo discordantes[].ref) y lo declara.

TAREA C — docs/SECCION_VALIDACION_HUMANA.md (nuevo, en español, borrador para el informe):
  Sección "Validación con gold humano" lista para adaptar al paper: diseño del gold
  (estratificado, 150+50+40, cegamiento, anotador HTML offline), resultados de la entrada
  28 (tabla de κ por verificador, nivel de fidelidad ~0,55 vs ~0,30, flips A→B 32%),
  taxonomía (53%/45%), triple juez ciego, y subsección "Confiabilidad del anotador" con el
  test-retest crudo (55%, κ=0,268), la adjudicación y la sensibilidad (rellena con los
  números de TAREA B cuando existan; si no, deja marcadores [PENDIENTE-ADJUDICACION]).
  Tono: honesto, sin inflar; cita solo archivos y entradas reales del repo.

PROHIBIDO: hacer push; cambiar de rama; borrar nada; editar analyze_gold_v4.py,
compute_exp15_ensemble_sweep.py ni ningún CSV de output/audit salvo via TAREA A cuando el
JSON exista; leer o usar los juicios de gold_v4_juicios_enzo_*.json como criterio de nada;
ejecutar GPU/Ollama/NLI/HHEM (todo offline CPU); tocar experiments/.

VERIFICACIÓN: corre la suite segura:
  & "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pytest -q -m "not slow and not gpu"
Commits atómicos en la rama actual (test/feat/docs por separado), SIN push.

REPORTE FINAL con exactamente estas secciones: Hecho / Auditoría (git status --short
--untracked-files=all + git log --oneline -5) / Bloqueos / Falta.
```
