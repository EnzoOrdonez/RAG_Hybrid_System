# Registro de decisiones de arquitectura (ADR)

Formato [MADR](https://adr.github.io/madr/) reducido: Contexto, Decisión, Consecuencias y Fuentes.
Cada ADR enlaza la evidencia de origen; el relato completo de cada sesión está en
[session-log.md](session-log.md). Un ADR aceptado no se edita: si cambia la decisión, se escribe uno
nuevo con estado «Sustituye a ADR-XXXX» y el anterior pasa a «Sustituido por ADR-YYYY».

| ID | Fecha | Decisión | Estado |
|---|---|---|---|
| [ADR-0001](#adr-0001) | 2026-06-11 | Evidencia firmada inmutable; todo recálculo en archivos `_vN` nuevos | Aceptado |
| [ADR-0002](#adr-0002) | 2026-06 | Granite 4.1 8B como modelo principal | Aceptado |
| [ADR-0003](#adr-0003) | ≤ 2026-06-11 | Expansión cross-cloud apagada por defecto | Aceptado |
| [ADR-0004](#adr-0004) | 2026-07-22 | Generación LLM nueva solo bajo IDs `exp15+` | Aceptado |
| [ADR-0005](#adr-0005) | 2026-08-03 | Paridad de despliegue con `SURVEY_DEPLOY`, sin tocar `PROPOSED_HYBRID` | Aceptado |
| [ADR-0006](#adr-0006) | 2026-08-04 | Separar soporte del paquete (3.11+) del entorno reproducible (3.14) | Aceptado |
| [ADR-0007](#adr-0007) | 2026-08-21 | `PYTHONUTF8=1` obligatoria en la receta | Aceptado |
| [ADR-0008](#adr-0008) | 2026-08-21 | Texto canónico del ruteo de prompts | Aceptado |
| [ADR-0009](#adr-0009) | 2026-08-21 | Guarda de evidencia con registro invertido | Aceptado |
| [ADR-0010](#adr-0010) | 2026-08-21 | Huella de sesión del runtime: no reanudar entre estados del generador | Aceptado |
| [ADR-0011](#adr-0011) | 2026-08-21 | Material no-evidencia se mueve fuera de `experiments/results/`, no se borra | Aceptado |
| [ADR-0012](#adr-0012) | 2026-08-21 | Regla de veredicto de tres ramas para el suelo de ruido | Aceptado |
| [ADR-0013](#adr-0013) | sin fecha de origen | Compuerta humana antes de `git push` | Aceptado |
| [ADR-0014](#adr-0014) | 2026-10-25 | Estandarización documental: Diátaxis, `.harness/` y `CLAUDE.md` índice | Aceptado |

---

<a id="adr-0001"></a>
## ADR-0001: Evidencia firmada inmutable y recálculos `_vN`

- **Estado:** Aceptado. **Fecha:** 2026-06-11 (tag `nota3-evidencia-2026-06-11`; reforzado con `nota3-N9-cierre-2026-07-02`).
- **Contexto:** las cifras de la Nota 3 y del paper LACCI salen de `experiments/results/exp3..exp14` y `exp8b`. Varias auditorías (N5, N8, N9) cambiaron métricas e instrumentos después de entregar cifras.
- **Decisión:** no se modifica ni se borra ningún artefacto de esos directorios. Todo recálculo se escribe en un archivo nuevo con sufijo `_vN` (v3 = N8, v4 = N9). `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv` son inmutables.
- **Consecuencias:** cada cifra citada es rederivable desde un commit firmado y las versiones conviven. A cambio, el árbol acumula variantes y cada documento debe decir qué versión cita.
- **Fuentes:** [guardrails.md](../guides/guardrails.md), `paper/audit_findings_cc_addenda.md` (N1 a N9).

<a id="adr-0002"></a>
## ADR-0002: Granite 4.1 8B como modelo principal

- **Estado:** Aceptado. **Fecha:** 2026-06 (smoke de determinismo previo a exp12).
- **Contexto:** la matriz exp12 compara cuatro modelos abiertos en una GPU de 6 GB. Con temperatura 0, Granite 4.1 y Qwen 3.5 resultaron bit-idénticos en 3 repeticiones; Gemma 4 E4B y Mistral 7B no.
- **Decisión:** Granite 4.1 8B es el modelo principal porque fue determinista, cabe en 6 GB e IBM le declara capacidad para RAG. La fuente original dice «RAG-tuned»; la ficha del proveedor solo lo lista como capacidad, y así se registra aquí.
- **Consecuencias:** las cifras principales son reproducibles desde caché. La falta de determinismo de Gemma y Mistral queda documentada como limitación en [models.md](../architecture/models.md).
- **Fuentes:** [nota3-next-steps-2026-06.md](../archive/nota3-next-steps-2026-06.md) (sección «Determinismo por modelo»).

<a id="adr-0003"></a>
## ADR-0003: Expansión cross-cloud apagada por defecto

- **Estado:** Aceptado. **Fecha:** anterior a N5 (2026-06-11), ledger N4.
- **Contexto:** exp7 atribuía +16,8 % a la expansión cross-cloud, pero sus dos brazos corrieron una recuperación idéntica (N1). Con el mecanismo D11 bien conectado, exp13 no encontró mejora en recuperación ni en fidelidad.
- **Decisión:** `query_expansion=False` en `PROPOSED_HYBRID`. El mecanismo D11 sigue implementado y testeable mediante el flag y el runner de exp13. La afirmación del +16,8 % queda retirada.
- **Consecuencias:** exp11 y exp12 corrieron sin expansión, en coherencia con el sistema medido. La opción de reactivarla existe, pero exige un experimento nuevo.
- **Fuentes:** `src/pipeline/pipeline_config.py` (comentario de `PROPOSED_HYBRID`), `paper/audit_findings_cc_addenda.md` N1 y N4.

<a id="adr-0004"></a>
## ADR-0004: Generación LLM nueva solo bajo IDs `exp15+`

- **Estado:** Aceptado. **Fecha:** 2026-07-22 (autorización de la fase de verano).
- **Contexto:** la fase de verano necesitaba generar respuestas nuevas sin contaminar la evidencia de la Nota 3.
- **Decisión:** toda generación nueva usa IDs `exp15` o superiores. Sobre `exp3..exp14` solo se admite reanálisis offline. Ledger: `paper/summer_ablation_log.md`; perillas: `docs/KNOB_MAP_summer.md`.
- **Consecuencias:** separación limpia entre evidencia firmada y exploración. Las comparaciones entre fases requieren cuidado, porque el runtime cambió entre ellas (ver ADR-0010).
- **Fuentes:** [guardrails.md](../guides/guardrails.md).

<a id="adr-0005"></a>
## ADR-0005: Paridad de despliegue con `SURVEY_DEPLOY`

- **Estado:** Aceptado. **Fecha:** 2026-08-03 (autorización de Enzo); corrección del modelo el 2026-08-04 (defecto #12, ledger entrada 24).
- **Contexto:** Chat y Evaluation debían servir el sistema que se encuesta, con routing de prompts y balanceo cross-cloud. `PROPOSED_HYBRID` es el registro del sistema enviado a LACCI (Llama 3.1 8B Q4) y no debe alterarse. La fase de verano midió todo con `granite4.1:8b`.
- **Decisión:** `SURVEY_DEPLOY` es una copia de `PROPOSED_HYBRID` con `prompt_routing=True`, `balance_cross_cloud_providers=True` y `llm_model="granite4.1:8b"`. Se usa solo mediante un mapeo exclusivo de Streamlit y queda fuera de `PIPELINE_CONFIGS`, de modo que `get_config("hybrid")` sigue devolviendo el sistema medido. `query_stream()` aplica el mismo routing y balanceo que `query()`.
- **Consecuencias:** el despliegue mide un sistema con cifras de fidelidad detrás y el registro experimental no se falsea. A cambio, hay dos configuraciones parecidas que conviene no confundir.
- **Fuentes:** `src/pipeline/pipeline_config.py`, [session-log.md](session-log.md) (entrada del 2026-08-03).

<a id="adr-0006"></a>
## ADR-0006: Soporte del paquete 3.11+ frente a entorno reproducible 3.14

- **Estado:** Aceptado. **Fecha:** decidido el 2026-08-04; ejecutado el 2026-08-21 (P-DOC).
- **Contexto:** README, `setup.py` y `REPRODUCE.md` declaraban versiones distintas de Python (3.10+, 3.11+ y 3.14).
- **Decisión:** README y `setup.py` declaran soporte del paquete para 3.11+; [reproduce.md](../guides/reproduce.md) §0 declara el entorno reproducible en 3.14. No se eligió una versión nueva.
- **Consecuencias:** las dos afirmaciones dejan de contradecirse. Reproducir experimentos sigue exigiendo el intérprete 3.14 con el stack ML, no el 3.11 del PATH.
- **Fuentes:** [session-log.md](session-log.md) (entradas del 2026-08-04 y del 2026-08-21 09:08).

<a id="adr-0007"></a>
## ADR-0007: `PYTHONUTF8=1` obligatoria

- **Estado:** Aceptado. **Fecha:** 2026-08-21.
- **Contexto:** `verify_v4_offline.py` falló con `UnicodeEncodeError` en una consola `cp1252` al imprimir `≈`. En PowerShell 7.6.5 con `chcp 65001` da exit 0 sin la variable: el fallo depende de la consola.
- **Decisión:** la receta fija `PYTHONUTF8=1` siempre, para que sea determinista en cualquier consola.
- **Consecuencias:** se elimina un fallo dependiente del entorno. La variable aparece en la receta, en el lanzador de exp19b y en el Dockerfile.
- **Fuentes:** [reproduce.md](../guides/reproduce.md) §0, [session-log.md](session-log.md).

<a id="adr-0008"></a>
## ADR-0008: Texto canónico del ruteo de prompts

- **Estado:** Aceptado. **Fecha:** decisión registrada el 2026-08-04; texto fijado el 2026-08-21.
- **Contexto:** `TRACEABILITY_nota3.md` y `KNOB_MAP_summer.md` describían el ruteo de prompts de forma distinta. La re-auditoría mostró que la construcción del prompt es idéntica (`rag_pipeline.py` frente a `run_generation_matrix.py`) y que solo cambia el origen del `query_type`: `prompt_routing` es opcional y por defecto `False`.
- **Decisión:** un mismo texto canónico, idéntico en los dos documentos. No se tocó código.
- **Consecuencias:** la contradicción era de redacción y queda cerrada sin alterar resultados.
- **Fuentes:** `docs/TRACEABILITY_nota3.md`, `docs/KNOB_MAP_summer.md`, [session-log.md](session-log.md).

<a id="adr-0009"></a>
## ADR-0009: Guarda de evidencia con registro invertido

- **Estado:** Aceptado. **Fecha:** 2026-08-21 (aprobado por Enzo); `LIVE_EXPERIMENTS` vaciado después, en la corrección de entrevistas.
- **Contexto:** `src/utils/signed_evidence.py` protegía una lista explícita (`exp3..exp14`, `exp8b`) y dejaba `exp15..exp19a` al cuidado humano. Las listas de lo que se protege ya habían causado dos defectos.
- **Decisión:** todo directorio `expN*` bajo `experiments/results/` queda protegido contra sobrescritura salvo los declarados en `LIVE_EXPERIMENTS`. `SIGNED_EXPERIMENTS` se conserva para que el error cite el motivo real. `guard_write` solo falla si el archivo ya existe, así que se pueden crear artefactos nuevos. Hoy `LIVE_EXPERIMENTS = frozenset()`.
- **Consecuencias:** una entrada obsoleta produce un rechazo ruidoso e inofensivo en lugar de evidencia pisada en silencio. Limitación declarada: los runners de verano no llaman a `guard_write`.
- **Fuentes:** `src/utils/signed_evidence.py`, [session-log.md](session-log.md) (entrada del 2026-08-21 02:30).

<a id="adr-0010"></a>
## ADR-0010: Huella de sesión del runtime y prohibición de reanudar entre estados

- **Estado:** Aceptado. **Fecha:** 2026-08-21.
- **Contexto:** con prompts byte-idénticos, Ollama devuelve la misma respuesta dentro de una sesión ya calentada, pero otra distinta tras reiniciarse (jaccard-5grama 0,0705 en q001). La corrida de exp19b se abortó en la query 79.
- **Decisión:** el runner persiste una huella de sesión (hash de la respuesta de warmup) y rechaza checkpoints de otro estado o sin huella. El pipeline vuelve a tomar la huella antes de `regen` y aborta con `RUNTIME_STATE_CHANGED` sin puntuar nada.
- **Consecuencias:** ningún brazo mezcla dos estados del generador. A cambio, una corrida completa (~6,6 h) tiene que caber en una sola sesión de servidor, y exp21 queda con un suelo de ruido del orden de su banda TOST.
- **Fuentes:** `paper/summer_ablation_log.md` entradas 25 y 26, [session-log.md](session-log.md).

<a id="adr-0011"></a>
## ADR-0011: Mover el material no-evidencia fuera de `experiments/results/`, sin borrarlo

- **Estado:** Aceptado. **Fecha:** 2026-08-21 (decisión de Enzo).
- **Contexto:** el checkpoint huérfano de 70 queries de exp19b era a la vez una trampa de reanudación y una muestra ya pagada.
- **Decisión:** se mueve a `experiments/probes/runtime_noise/state_A_2026-08-21.json`, donde no lo descubre `verify_summer_offline.py` ni ninguna familia BH, y el README del directorio lo declara no-evidencia.
- **Consecuencias:** se conserva el cómputo (~70 min de GPU) sin contaminar la evidencia. Es el patrón general: mover y declarar, nunca borrar.
- **Fuentes:** [session-log.md](session-log.md) (entrada del 2026-08-21 14:05).

<a id="adr-0012"></a>
## ADR-0012: Regla de veredicto de tres ramas para el suelo de ruido

- **Estado:** Aceptado. **Fecha:** 2026-08-21, declarada en código antes de tener datos.
- **Contexto:** el riesgo era leer como aprobado un resultado inconcluyente sobre si un cambio de estado del runtime desplaza la media pareada (banda TOST ±0,081).
- **Decisión:** hay tres desenlaces posibles:
  - (b) la banda sobrevive si el IC95 cabe en ±0,0203;
  - (a) hay sesgo sistemático si el IC excluye 0 y supera ese límite;
  - (c) el resultado está infrapotenciado, y eso no cuenta como aprobado.
- **Consecuencias:** el resultado inconcluyente queda nombrado y testeado; los tests atacan sobre todo la rama (c).
- **Fuentes:** [session-log.md](session-log.md) (entrada del 2026-08-21 14:05).

<a id="adr-0013"></a>
## ADR-0013: Compuerta humana antes de `git push`

- **Estado:** Aceptado. **Fecha:** sin fecha de origen documentada; vigente en todas las versiones conservadas de `CLAUDE.md`.
- **Contexto:** el repositorio es público y contiene evidencia de tesis; un push con `.env`, un secreto o evidencia alterada no se puede deshacer.
- **Decisión:** antes de cualquier `git push`, el agente informa qué se publicaría y espera un OK explícito de Enzo. Se detiene si aparece `.env` o un secreto. No hay push a `main` ni force push.
- **Consecuencias:** se ralentiza la publicación, pero ningún agente publica por su cuenta.
- **Fuentes:** [guardrails.md](../guides/guardrails.md).

<a id="adr-0014"></a>
## ADR-0014: Estandarización documental (Diátaxis, `.harness/` y `CLAUDE.md` índice)

- **Estado:** Aceptado. **Fecha:** 2026-10-25 (directiva de Enzo; rama `chore/docs-harness`).
- **Contexto:** la raíz acumulaba siete documentos y un `CLAUDE.md` de 543 líneas que mezclaba contexto, estado, reglas y un log append-only. Un agente en arranque en frío tenía que leerlo entero para saber qué hacer. La documentación estaba además bifurcada entre `fix/interview-readiness` y `summer/taxonomia-759`.
- **Decisión:**
  1. Se integran primero las dos ramas, con conflictos resueltos sin perder texto, más el trabajo sin commitear.
  2. Copia inmutable del `CLAUDE.md` anterior en `docs/archive/`.
  3. Los documentos se mueven con `git mv` a `docs/` según Diátaxis (mapa en [../README.md](../README.md)).
  4. El contenido del `CLAUDE.md` se reparte de forma literal: reglas en `guides/guardrails.md`, estado y log en `decisions/session-log.md`, decisiones formalizadas en este registro.
  5. El estado operativo pasa a `.harness/` (`feature_list.json`, `progress.md`) y la verificación a `init.sh`, con tres capas que deben dar exit 0.
  6. `CLAUDE.md` queda como índice de menos de 100 líneas.
- **Alternativas descartadas:**
  - Raíz estrictamente limpia: mover `LICENSE`, `CITATION.cff`, `setup.py`, `pytest.ini`, `requirements*.txt`, `secrets_baseline.json`, `Dockerfile` y `run.py` rompe la CI, pip, la detección de GitHub y la imagen. Solo se movió documentación.
  - Construir `feature_list.json` desde `NOTA3_NEXT_STEPS.md`: ese archivo declara historial sus TODO desde el 2026-08-30.
  - Reescribir el log como ADRs: se perderían matices. Se conserva literal y aquí solo se formalizan las decisiones.
- **Consecuencias:** el arranque en frío lee unas 100 líneas en lugar de 543. `scripts/check_docs.py` verifica que ninguna línea del `CLAUDE.md` anterior se pierda y que los enlaces resuelvan. Los documentos fechados y los ledgers conservan rutas antiguas a propósito.
- **Fuentes:** `docs/archive/README.md`, commits de `chore/docs-harness`.
