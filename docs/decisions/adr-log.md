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
| [ADR-0015](#adr-0015) | 2026-10-27 | Topes de gasto del estudio: Fase B USD 20 + 3 de cierre, total USD 100 | Aceptado |
| [ADR-0016](#adr-0016) | 2026-10-27 | Fecha mínima de pago de la Fase B I7 adelantada al 2026-10-28 UTC, con respaldo al 2026-11-01 | Aceptado |
| [ADR-0017](#adr-0017) | 2026-10-27 | Decisiones de arranque de F-020: ruta humana en nube, cohorte NLI, config sellada vigente | Aceptado |
| [ADR-0018](#adr-0018) | 2026-10-27 | Fase humana sobre la imagen I7: GO verificado, ledger propio de USD 70 y firewall restringido | Aceptado |
| [ADR-0019](ADR-0019-i7-ledger-corrections.md) | 2026-10-27 | Correcciones del ledger y la admisión humana I7; preparación offline | Aceptado |

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
- **Contexto:** la raíz acumulaba siete documentos y un `CLAUDE.md` de 543 líneas que mezclaba contexto, estado, reglas y un log append-only. Un agente en arranque en frío tenía que leerlo entero para saber qué hacer. La documentación también estaba bifurcada entre `fix/interview-readiness` y `summer/taxonomia-759`.
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

<a id="adr-0015"></a>
## ADR-0015: Topes de gasto del estudio con usuarios

- **Estado:** Aceptado. **Fecha:** 2026-10-27 (directiva de Enzo en la tarea F-020).
- **Contexto:** el tope de la Fase B I7 existía solo en el código y en el runbook; el tope del estudio completo no estaba escrito. [DECLARADO] `scripts/study_operator/i7_config.py:26` fija `budget_usd=20` y `closure_reserve_usd=3`; `i7.py:57-59` rechaza otra configuración; `i7_costs.py:102-103` devuelve `ADMIT` solo si el total previsto con cierre es de USD 20 o menos; `i7_costs.py:129-130` corta a las 24 h de retención.
- **Decisión:**
  1. Fase B I7: USD 20 de exposición prevista a precio de lista, incluidos USD 3 de cierre. Los precios de lista no son factura.
  2. Estudio completo (Fase B, ensayo, sesiones y retención): USD 100 a precio de lista, Fase B incluida.
  3. El gasto posterior a la Fase B se pide a Enzo por bloques, cada uno con su costo máximo, dentro del saldo de USD 100.
  4. No se amplía ningún tope para obtener un resultado.
- **Consecuencias:** una tarifa nueva o una retención mayor de 24 h pueden bloquear la Fase B aunque la factura real sea menor. Ese bloqueo es aceptado.
- **Fuentes:** [STUDY_I7_RUNBOOK_FASE_B.md](../STUDY_I7_RUNBOOK_FASE_B.md) (sección «Costos y regla de corte»); `.harness/feature_list.json` (F-020).

<a id="adr-0016"></a>
## ADR-0016: Fecha mínima de pago de la Fase B I7

- **Estado:** Aceptado. **Fecha:** 2026-10-27. Enzo delegó la decisión en Claude, con preferencia por adelantar.
- **Contexto:** el código impide todo efecto pagado antes de 2026-11-01T00:00Z: `i7.py:76-77` (`NO_PAID_EFFECT_BEFORE_NOVEMBER`), `i7.py:374-375`, `i7_phase_b.py:258` y `:261-263`, y el bloqueo con dueño Enzo de `i7_readiness.py:41-42`. [VERIFICADO por lectura] La fecha no figura en ningún preregistro ni en `stimulus_calendar.py`; la introdujo el commit `f26395c` (2026-10-15) como guarda de pago, y ningún documento da su motivo. Clase (b): decisión del dueño, no restricción técnica ni calendario preregistrado.
- **Decisión:**
  1. El piso pasa a 2026-10-28T00:00Z, en una constante única, con cambio por TDD en `fix/interview-readiness`. Los códigos de error no cambian.
  2. El cambio exige regenerar el recibo de preparación local: suites de Windows, Linux e I7 y `verify_local` (19 módulos y 3 documentos).
  3. Respaldo: si una suite falla por una causa distinta de F-016, se revierte con un commit nuevo y rige el 2026-11-01T00:00Z (31/10, 19:00 en Lima).
  4. La autorización de gasto sigue siendo explícita: el piso de fecha no la reemplaza.
- **Consecuencias:** la Fase B pagada puede empezar unos tres días antes y terminar antes de la ausencia de Enzo del 3 al 6/11. Cuesta un cambio de fuente y evidencia local nueva antes del build.
- **Fuentes:** [STUDY_I7_RUNBOOK_FASE_B.md](../STUDY_I7_RUNBOOK_FASE_B.md) (líneas 22-23); `tests/test_study_i7_operator.py:72-78`.

<a id="adr-0017"></a>
## ADR-0017: Decisiones de arranque de F-020

- **Estado:** Aceptado. **Fecha:** 2026-10-27 (respuestas de Enzo en la sesión de planificación de F-020).
- **Contexto:** el checklist local del 1/10 ([STUDY_HUMAN_CHECKLIST.md](../STUDY_HUMAN_CHECKLIST.md)) y el runbook I7 describen flujos distintos, y el código no desempata tres puntos.
- **Decisión:**
  1. Ruta de las sesiones humanas: nube con el navegador propio del participante por HTTPS (enmienda del 2026-10-02). La forma de operarla sobre la imagen I7 se investiga en la fase B de F-020, porque `i7.py:136-138` solo admite propósitos técnicos.
  2. La cohorte NLI no es precondición: la enmienda del 2026-10-02 la exige solo para el remedio R2 y la app está congelada en `per_claim` (`scripts/study_operator/rag_freeze.py:71`).
  3. Config sellada vigente: `study-config-reviewed-v2` (protocolo v2 con UEQ-S, misma asignación). [VERIFICADO] El sello del 1/10 falla con el código vigente; no hace falta un sorteo ni un sello nuevos.
  4. Tablero: una entrada F-020 con las fases A a G como pasos (los ids deben cumplir `F-NNN`, `scripts/check_docs.py:114`); F-017 es una entrada propia.
- **Consecuencias:** los pasos del checklist que dependen de Zoom y de un servidor local quedan superados para el estudio; P998 se ensaya en la nube.
- **Fuentes:** [STUDY_GATE_PREREGISTRATION_AMENDMENT_2026-10-02.md](../STUDY_GATE_PREREGISTRATION_AMENDMENT_2026-10-02.md); `.harness/feature_list.json` (F-020).

<a id="adr-0018"></a>
## ADR-0018: Fase humana sobre la imagen I7

- **Estado:** Aceptado. **Fecha:** 2026-10-27 (respuestas de Enzo en la fase B de F-020).
- **Contexto:** [VERIFICADO por lectura] el operador I7 solo admitía los propósitos `technical` y `smoke` (`scripts/study_operator/i7.py`), y la CLI de la iteración 5 no puede operar la imagen I7 sin varios cambios. Los topes de la Fase B (USD 20, 17 h de reserva y 24 h de retención) hacen fallar por diseño cualquier boot posterior a la compuerta. Como `verify_tool_source` amarra el commit del operador al de la imagen, un cambio de código posterior a `cpu-build` obligaría a repetir la Fase B.
- **Decisión:**
  1. El código de la fase humana entra en `fix/interview-readiness` antes de `cpu-build`, de modo que la imagen y la compuerta lo cubran.
  2. `rehearsal`, `pilot` y `study` solo arrancan con la fase humana abierta. Para abrirla hacen falta recibos de compuerta descargados y verificados, todos con veredicto GO, y la reserva de la Fase B cerrada y verificada.
  3. El propósito `study` exige además los archivos de aprobación ética (`policy.purpose_allowed`). Los boots humanos usan el perfil `standard`.
  4. Ledger propio de USD 70 a precio de lista, dentro del total de USD 100 (ADR-0015):
     - cada boot lleva una reserva de 4 h como máximo;
     - desde la apertura se cuentan las horas retenidas de disco (540 GiB), snapshots (552 GiB) e IP, con un mínimo de 24 h, más un día proyectado;
     - se reservan USD 3 de cierre, igual que en la Fase B;
     - los precios de cómputo y de almacenamiento deben tener 7 días o menos; `refresh-prices` rechaza archivos más antiguos que los vigentes;
     - el GO debe estar ligado a su prueba privada, a la imagen y al commit, y no puede cambiar después de abrir la fase.
  5. Las reglas `default-allow-ssh` y `default-allow-rdp` quedan restringidas al rango de IAP; no se borran. [VERIFICADO] con una lectura posterior al cambio.
- **Consecuencias:** [ESTIMADO, lista del 27/10] la retención cuesta unos USD 2,87 por día y cada boot humano unos USD 3,08. Con 2 sesiones por boot (10 boots), el tope alcanza para unos 12 días de fase humana; con 1 sesión por boot, no alcanza para 20 sesiones. Al llegar al tope solo se bloquean nuevas admisiones: la retención de recursos la decide Enzo. El calendario de sesiones debe ajustarse a esto, o Enzo decide otro tope con un ADR nuevo.
- **Fuentes:** `tests/test_study_i7_human.py`; [ADR-0015](#adr-0015); `.harness/feature_list.json` (F-020).

## ADR-0020: Ajustes finales y pendientes de I7

Ver [ADR-0020](ADR-0020-i7-final-freeze.md): H3, H9, positivo integrado de study,
fe de erratas de los conteos de mutación de ADR-0019 y backlog sin corregir.
