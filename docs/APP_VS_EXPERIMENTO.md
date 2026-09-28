# Interfaz Streamlit frente a la receta experimental

**Addendum 2026-09-27:** esta comparación describe la receta histórica. La nueva
entrada participante usa `study_pipeline.py`: `SURVEY_DEPLOY` y una configuración
separada `STUDY_NO_RAG`, ambas con Granite, iguales parámetros de generación y
presentación. La segunda no consulta índice ni reordenador. `LLM_ONLY_NO_RAG` y
las configuraciones experimentales no se modifican. Ver [runbook](STUDY_DEPLOYMENT_RUNBOOK.md).
La equivalencia de parámetros se prueba con dobles; rendimiento real aún NO verificado.

Auditoría de configuración realizada el 2026-08-23 contra `REPRODUCE.md` y
`scripts/run_exp19b_generation.py`. La comparación distingue valores configurables de
diferencias de mecanismo: igualar un número es seguro; convertir la interfaz en una
reproducción de exp19b requeriría una ruta explícita y no se hizo aquí.

| Parámetro | Interfaz Streamlit | Receta exp19b | Estado |
|---|---|---|---|
| Proveedor | Ollama local | Ollama local | Alineado |
| Modelo por defecto | `granite4.1:8b`, derivado de `SURVEY_DEPLOY`; el chat permite override visible | `MODEL_TAG = "granite4.1:8b"` | Alineado por defecto |
| Temperatura | `0.0` en las configuraciones | `0.0` en cada generación | Alineado |
| Semilla | `42`, explícita al construir `LLMManager` | `SEED = 42` | Alineado |
| Caché LLM | Desactivada para toda ruta Streamlit | `--no-cache` en draft y regen | Alineado |
| Máximo de salida | 1024 tokens por defecto; el chat permite cambiarlo | 1024, default de `LLMManager.generate` | Alineado por defecto |
| Pool inicial | Retrieval híbrido vivo, top 50 | Pool top 50 congelado en exp18 | Diferencia de lógica |
| Evidencia final | 5 chunks por defecto | 5 IDs por query | Mismo k, distinto mecanismo |
| Reranking | MiniLM-L-12-v2 sobre `(query, chunk)` | MiniLM-L-12-v2 sobre `(claim, chunk)`, ejecutado en CPU | Diferencia de lógica |
| Selección cross-cloud | `SURVEY_DEPLOY` rebalancea proveedores en queries cross-cloud | `claim_selected` usa el selector condicionado por claims | Diferencia de lógica |
| Tipo de query y prompt | `QueryProcessor` + `build_context` + `get_template` + `SYSTEM_PROMPT` | `build_prompt("hibrido", ...)` reutiliza esas mismas funciones | Alineado para iguales IDs y tipo |
| `keep_alive` | `30m` en Chat; Evaluation no lo envía | No se envía | Operación distinta entre páginas |
| Contexto | `num_ctx=4096`, explícito en el cliente de la app | Consultar receta y servidor histórico | No se infiere paridad por el tag |
| Identidad de artefactos | Manifiesto de índices, consultas y snapshots; digest del modelo comprobado por generación en modo participante | Evidencia congelada | Trazabilidad distinta, explícita |

Desde la corrección de preparación para entrevistas, `src/ui/app.py` abre únicamente
Evaluation por defecto. Chat y las herramientas requieren `CLOUDRAG_MODE=development`.
Evaluation persiste intentos y respuestas antes de admitir ratings, muestra las fuentes
en un desplegable y rechaza fallos técnicos o verificación NLI degradada. Instalación y
compuerta de validación real: [INTERVIEW_READINESS.md](INTERVIEW_READINESS.md).

## Diferencias que no son configuración pura

La interfaz recupera evidencia en vivo. Exp19b, en cambio, lee los
`baseline_repro_ids` congelados de exp18 o los `claim_rank_ids` producidos desde los claims
del draft. Sustituir una ruta por la otra cambiaría la lógica, el origen de evidencia y el
significado de la interfaz; no se implementó.

El control “Hybrid Alpha” se muestra en el chat, pero `SURVEY_DEPLOY` usa fusión RRF y el
retriever ya está construido cuando se crean los overrides por consulta. Por tanto, mover el
slider no reproduce ninguna perilla de exp19b ni altera el retriever activo. La recomendación
es ocultarlo para RRF o reconstruir explícitamente un retriever lineal en un cambio de lógica
separado y testeado.

Si se necesita una demo bit-comparable con exp19b, la recomendación es añadir un modo de
replay explícito que cargue los IDs archivados y llame a la misma maquinaria de generación,
sin reemplazar el flujo general de Streamlit ni reinterpretar la evidencia congelada.
