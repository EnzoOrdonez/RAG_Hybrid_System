# Guardrails: reglas inviolables y zonas sensibles

Texto literal de las tres secciones normativas del `CLAUDE.md` anterior (migración del 2026-10-25).
`CLAUDE.md` resume lo esencial; ante cualquier duda manda este archivo. Cambiar una regla de aquí
requiere OK explícito de Enzo y un ADR nuevo en [../decisions/adr-log.md](../decisions/adr-log.md).

Rutas actuales de los documentos citados: `REPRODUCE.md` es `docs/guides/reproduce.md`; el resto
del mapa está en [../README.md](../README.md).

---

## Sensible / no tocar sin permiso

- `experiments/results/exp3..exp14` y `exp8b`: evidencia firmada. Solo se admite reanálisis offline en
  archivos `_vN` nuevos.
- `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv`: inmutables.
- `paper/overleaf_ready/main.tex` y la prosa A.3: cualquier corrección requiere aprobación explícita
  frase por frase.
- `experiments/results/exp15..exp19a`: evidencia de verano ya committeada. No regenerar, sobrescribir
  ni reinterpretar sin fijar antes el alcance y revisar el ledger. Desde el 2026-08-21 esto lo hace
  cumplir el código: `src/utils/signed_evidence.py` protege por defecto y solo `LIVE_EXPERIMENTS`
  (actualmente vacío) permite sobrescritura; exp19b también está congelado.
- `.env`: existe localmente, está ignorado y puede contener secretos. No leer, imprimir ni versionar.

## Reglas de trabajo (permanentes)

1. **Sé crítico con tu propio trabajo.** No des nada por bueno solo porque corrió sin error.
   Audita y prueba antes y después de cada cambio de código.
2. **Sé proactivo.** Si encuentras una falla que nadie pidió revisar, repórtala.
3. **Ante ambigüedad, pregunta.** Cuando algo sea ambiguo, se contradiga, o no tengas certeza de la
   interpretación correcta, sobre todo si afecta una cifra ya entregada en A.3 o el paper de LACCI,
   pregunta a Enzo en vez de asumir.
4. **Da contexto suficiente en tus reportes** para que una decisión se pueda tomar sin ambigüedad.

## Restricciones inviolables (evidencia de la tesis)

- **NO modificar ni borrar** `experiments/results/exp3..exp14` ni `exp8b`. No existen exp1/exp2.
  Evidencia firmada con tags `nota3-evidencia-2026-06-11` y `nota3-N9-cierre-2026-07-02`.
  Todo recálculo crea archivos **`_vN` nuevos**: v3=N8, v4=N9 y así sucesivamente.
- **Generación LLM nueva solo bajo IDs `exp15+`**, fase de verano autorizada el 2026-07-22.
  Sobre `exp3..exp14`, solo reanálisis offline. Ledger: `paper/summer_ablation_log.md`.
  Perillas: `docs/KNOB_MAP_summer.md`. Para la paridad de despliegue se autorizó `summer/mejoras`.
- Entorno: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42`, seed=42. Intérprete con
  stack ML: `C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe`. No usar el Python
  3.11 del PATH para reproducir experimentos.
- **Compuerta antes de `git push`**: reportar qué se publicaría y esperar un OK explícito. Parar si
  aparece `.env` o un secreto.
- `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv` son inmutables.
- `paper/overleaf_ready/main.tex` y A.3 son prosa del paper. Corregir solo con OK explícito frase por
  frase.
