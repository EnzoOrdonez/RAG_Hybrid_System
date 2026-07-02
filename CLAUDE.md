# CLAUDE.md — hybrid-rag-system

Contexto permanente para Claude Code en este repo (tesis CloudRAG, Nota 3, LACCI 2026).
Se carga automáticamente cada sesión; estas reglas no hace falta repetirlas.

## Reglas de trabajo (permanentes)

1. **Sé crítico con tu propio trabajo.** No des nada por bueno solo porque "corrió sin error".
   Audita y testea **antes Y después** de cada cambio de código.
2. **Sé proactivo.** Si encuentras una falla que nadie pidió revisar, dila — no te limites a la
   lista de tareas del momento.
3. **Ante ambigüedad, pregunta.** Cuando algo sea ambiguo, se contradiga, o no tengas certeza de
   la interpretación correcta — sobre todo si afecta una cifra ya entregada en el **A.3** o el
   **paper de LACCI** — pregúntale a Enzo directamente en vez de asumir. Prefiere una pregunta a
   una decisión silenciosa.
4. **Da contexto suficiente en tus reportes** para que una decisión se pueda tomar sin ambigüedad.

## Restricciones inviolables (evidencia de la tesis)

- **NO modificar/borrar** `experiments/results/exp3..exp13` (+`exp8b`; no existen exp1/exp2)
  — evidencia firmada, tag `nota3-evidencia-2026-06-11`. Todo recálculo = archivos
  **`_vN` nuevos** (v3=N8, v4=N9, ...).
- **NO generación LLM nueva**; solo re-análisis offline de respuestas guardadas.
- Entorno: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42`, seed=42. Intérprete con
  stack ML = `py 3.14` (`...\pythoncore-3.14-64\python.exe`), no el `python` del PATH (3.11).
- **GATE antes de `git push`**: reportar qué se publicaría y esperar OK explícito. Parar si
  aparece `.env`/secreto.
- `paper/audit_findings.md` y `exp8_stats_corrected.csv` son **inmutables**.
- `main.tex`/A.3 = prosa del paper; corregir solo con OK explícito frase-por-frase.

## Punteros
- Ledger de auditoría: `paper/audit_findings_cc_addenda.md` (N1–N8), `paper/correction_log.md`.
- Trazabilidad: `docs/TRACEABILITY_nota3.md`. Resultados: `RESULTADOS_RESUMEN.md`.
