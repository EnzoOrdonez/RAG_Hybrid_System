# PROMPT MAESTRO PARA CODEX — pulido final pre-entrega (2026-08-29/30)

```
CONTEXTO. Repo C:\Users\enziz\projects\hybrid-rag-system, rama summer/taxonomia-759.
Entrega LACCI: 30-31 agosto. El gold humano esta fusionado; faltan la adjudicacion de los
9 discordantes de tanda C (Enzo la hace hoy) y el cierre documental. Ya existen:
scripts/merge_adjudicacion_tandaC.py, scripts/run_gold_sensitivity.py,
docs/SECCION_VALIDACION_HUMANA.md (borrador con marcadores [PENDIENTE-ADJUDICACION]),
paper/summer_ablation_log.md entradas 27/28/28b.

FASE 1 (hazla ya, no depende de la adjudicacion):

T1. Probe descriptivo de cobertura por proveedor (SOLO LECTURA de resultados).
  scripts/probe_provider_coverage.py: para las queries multi-nube del set de evaluacion
  (data/evaluation/test_queries.json; detecta las que nombran >=2 proveedores o dicen
  "across"), mide sobre los resultados YA PERSISTIDOS de retrieval/experimentos (lee
  experiments/results/, no re-ejecutes nada): fraccion con >=1 chunk por cada proveedor
  mencionado, y tasa de chunks cuya carpeta (aws/azure/gcp) no corresponde al proveedor
  de la query. Salida: output/audit/provider_coverage_probe.{json,md}. Si los resultados
  persistidos no contienen suficiente informacion, DECLARA el bloqueo; no re-ejecutes
  retrieval. Es descriptivo, fuera de toda familia BH, para la seccion de limitaciones.
  Tests con fixtures sinteticos en tmp_path.

T2. docs/LIMITACIONES_Y_TRABAJO_FUTURO.md (espanol, borrador para el informe):
  - Limitaciones medidas: sesgo de evidencia etapa A->B (32% flips), confiabilidad del
    anotador (55% crudo -> adjudicacion), varianza de κ por diseno estratificado (Kish),
    corpus en markdown con restos de conversion, y lo que salga del probe T1.
  - Trabajo futuro concreto (1 parrafo c/u): routing por metadatos de proveedor,
    descomposicion de consultas multi-nube, grafo de tripletas (servicio-relacion-atributo)
    con BD de grafos embebida (p. ej. LadybugDB) como tercer canal de retrieval y
    verificacion estructural. Cita solo archivos/entradas reales del repo.

T3. Pulido de repo (sin tocar logica):
  - CITATION.cff minimo (titulo, autor Enzo, licencia si el repo la declara).
  - Verifica que README enlace: guia de anotacion, SECCION_VALIDACION_HUMANA.md,
    REPRODUCE.md y el ledger. Solo lineas de enlace.
  - Re-corre la suite segura al final:
    & "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pytest -q -m "not slow and not gpu"

FASE 2 (SOLO si ya existe output/audit/gold_v4_adjudicacion_enzo_*.json; si no, salta):
  T4. python scripts/merge_adjudicacion_tandaC.py  (merge real de los 9)
  T5. python scripts/run_gold_sensitivity.py       (3 variantes con deltas)
  T6. Rellena los marcadores [PENDIENTE-ADJUDICACION] de docs/SECCION_VALIDACION_HUMANA.md
      con las cifras de T5 y el auto-acuerdo crudo (11/20, kappa 0,268, ya registrado en
      la entrada 28b).

PROHIBIDO: push; cambiar de rama; borrar nada; editar analyze_gold_v4.py ni
compute_exp15_ensemble_sweep.py; modificar CSV de output/audit salvo via T4; re-ejecutar
experimentos; GPU/Ollama/NLI/HHEM; leer gold_v4_juicios_enzo_*.json como insumo de analisis
(solo T4 lo usa para escribir los 9 veredictos finales).

COMMITS atomicos por tarea, SIN push. REPORTE FINAL: Hecho / Auditoria (git status +
git log --oneline -8) / Bloqueos / Falta.
```
