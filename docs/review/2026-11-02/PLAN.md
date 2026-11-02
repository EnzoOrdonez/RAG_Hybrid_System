# Plan de auditoría: 2026-11-02

Rama de trabajo: `claude/blissful-euler-0a3xkp` (ver DECISIONES.md D-01). Base: `main` @ `8fdaa65`.
Presupuesto: 3 h de reloj, 6 pasadas (valores por defecto; [ALCANCE] vacío: todo el repo por riesgo).

## Mapa de módulos y riesgo estimado

| Módulo | Líneas | Riesgo | Motivo |
|---|---|---|---|
| `scripts/study_operator/**` (zona congelada) | ~14.7k | Muy alto | Crea/borra recursos GCP, presupuesto, TTL, recibos, datos de participantes, SSH/IAP |
| `scripts/cloud_entrypoint.py` (congelado) | 197 | Alto | Punto de entrada en la imagen desplegada |
| `src/utils/signed_evidence.py` + scripts `compute_*`/`recompute_*`/`rescore_*` | — | Alto | Única guarda de evidencia firmada; cifras del paper |
| `src/evaluation/**` (métricas, estadística) | — | Alto | Cifras de A.3 / LACCI |
| `src/retrieval`, `src/reranking`, `src/pipeline` | — | Medio | Lógica RAG congelada; determinismo |
| `scripts/scan_secrets.py`, `check_docs.py`, `init.sh`, CI | — | Medio | Las tres capas de verificación |
| `src/ingestion`, `src/chunking`, `src/preprocessing` | — | Bajo | No se re-ejecutan |
| `src/ui/**` (congelado) | — | Medio | Superficie expuesta a participantes |

## Pasadas (orden por riesgo) y subagentes

| # | Pasada (§1 Fase 3) | Ejecutor |
|---|---|---|
| A | 5 Dinero/límites + 3 Concurrencia/recuperación en `study_operator` | subagente A |
| B | 4 Seguridad (subprocess, rutas, deserialización, secretos, privacidad) en `study_operator`, `cloud_entrypoint`, `src/ui` | subagente B |
| C | 1 Corrección lógica + 7 Reproducibilidad en `src/` y scripts de métricas/estadística, guarda de evidencia | subagente C |
| D | 2 Invariantes duplicadas + 8 Docs↔código (ADRs, guardrails, runbooks, init.sh, CI) | subagente D |
| E | 6 Calidad de pruebas + 9 Estático/dependencias (ruff, bandit, pip-audit, flakes F-016) | subagente E |
| V | §0.7 Verificación adversarial de cada S0/S1 | subagentes independientes |

## Crítica previa (§0.2)

- **Supuestos ocultos del plan:** (1) que el código de `study_operator` se pueda razonar sin GCP real: todo lo que
  dependa del API real queda NO EJECUTADO; (2) que los tests existentes reflejen el contrato: pueden codificar el
  defecto (mocks); (3) que Linux sea representativo: 30+ tests son solo-Windows y la ruta de producción del operador es
  PowerShell/Windows; (4) que el venv reconstruido (3.14.6, `requirements-app.txt` con hashes) iguale al `.venv-app`
  del dueño (3.14.x en Windows).
- **Lo que queda sin cubrir:** los `.ps1` solo se leen (no hay PowerShell nativo); Docker, GCP, GPU, Ollama y
  modelos HF no se ejecutan; los datos ignorados por git (`data/`, `.env`) no están en este clon; mutation testing
  completo sobre 15k líneas no cabe en el presupuesto (solo muestreo dirigido).
- **Sesgo a vigilar:** 15k líneas de operador con >1800 tests invitan a hallar "algo" para justificar la pasada.
  La verificación adversarial y la regla "sin reproductor = sospecha" son el contrapeso.
