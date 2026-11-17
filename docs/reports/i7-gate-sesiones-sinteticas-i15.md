# I7, intento 15: sesiones técnicas sintéticas del estímulo y del gate

[DECLARADO] Informe descriptivo del 2026-11-15, tarea F-020, decisión en
[ADR-0035](../decisions/ADR-0035-i7-intento-15-gate-go.md). Lo pidió Enzo tras el cierre.
Se calculó solo leyendo la evidencia ya recolectada: sin gasto nuevo y sin correr modelos.

**Qué son estas sesiones.** El operador las genera con códigos técnicos (P999 y P9000xx) y
preguntas fijas, sin personas. Miden si el despliegue responde a tiempo y sin fallos. No son datos
del estudio con usuarios y no prueban H1, H2 ni H3. Este informe solo publica agregados: ningún
texto de respuesta, contexto ni identificador.

## Fuente y método

[VERIFICADO]
- **Gate:** prueba privada recolectada con recibo `DOWNLOAD_VERIFIED`. Tiene 140 filas: 20 de
  piloto (fuera del cálculo) y 120 del gate, en dos ventanas de 60.
- **Estímulo:** 12 pruebas privadas de boots en frío, con 144 filas en total.
- **Percentiles:** `scripts/study_operator/i7_gate.py` (`percentile`, `summarize`), con
  interpolación lineal. El recálculo del gate coincide con el recibo.
- **Clases de respuesta:** `src/evaluation/decline_classifier.py`, versión
  `faithfulness_v2_28_patterns_300_chars`. Son patrones de texto deterministas, sin modelo.
- **Entorno:** modelo `granite4.1:8b`, temperatura 0, seed 42, `num_ctx` 4096 y `num_predict` 1024.
  Máquina g2-standard-4 con una L4 en us-central1-a, según el anexo `output/audit/I7_ENVIRONMENT_ANNEX_I15.md`.

## Gate: latencia y fallos (criterio preregistrado)

[VERIFICADO]

| Condición | n válidas | Fallos | Inválidas | Mín. | Media | p50 | p95 | Máx. |
|---|---|---|---|---|---|---|---|---|
| hybrid | 60 | 0 | 0 | 14,26 s | 25,42 s | 26,03 s | **34,88 s** | 35,01 s |
| no_rag | 60 | 0 | 0 | 17,51 s | 22,19 s | 22,70 s | **27,64 s** | 27,69 s |

Decisión: **GO**. Se exigía p95 ≤ 60 s por condición, 60 válidas por condición y cero fallos e
inválidas.

La latencia es estable entre tramos (p95, en segundos):

| Tramo | hybrid | no_rag |
|---|---|---|
| Piloto (n = 10 por condición) | 33,41 | 27,50 |
| Ventana 1 (n = 30) | 34,87 | 27,62 |
| Ventana 2 (n = 30) | 34,91 | 27,65 |

Del inicio de la admisión al final de las 140 llamadas pasaron 63,7 min, con 907 muestras de
telemetría. El aislamiento y el bloqueo de metadatos del invitado quedaron verificados.

### En qué se va el tiempo (media por llamada del gate)

| Etapa | hybrid | no_rag |
|---|---|---|
| Generación del LLM | 23,65 s | 22,18 s |
| Verificación de alucinaciones | 1,42 s | < 0,01 s |
| Reranking | 0,25 s | — |
| Recuperación | 0,09 s | — |

La generación es el 93 % del tiempo de hybrid. La recuperación híbrida y el reranking suman
menos de 0,35 s por consulta.

## Arranque en frío (estímulo)

[VERIFICADO]

| Llamadas del estímulo | hybrid p50 / p95 / máx. | no_rag p50 / p95 / máx. |
|---|---|---|
| Objetivo (n = 60 por condición) | 30,57 / 35,51 / 47,44 s | 22,79 / 27,65 / 27,94 s |
| Primera tras el boot (n = 6) | 39,64 / 46,90 / 47,44 s | 23,00 / 26,92 / 27,94 s |

La primera consulta hybrid tras un boot en frío tuvo p50 39,64 s, frente a 26,03 s en el gate en
caliente. Sigue bajo 60 s, pero justifica el calentamiento antes de cada participante que prevé
`docs/INTERVIEW_READINESS.md`.

## Clases de respuesta: nube frente a local

[VERIFICADO en las pruebas y en `experiments/results/exp12_matrix`; la comparación es DESCRIPTIVA]

Las 6 tareas son el subconjunto sellado del estímulo y del gate. Cada celda de nube resume 10
ejecuciones de la misma tarea, que dieron siempre la misma clase. La columna local es una sola
ejecución de exp12 con el mismo modelo (`granite4.1-8b`).

| Tarea | hybrid gate | hybrid estímulo | hybrid local | no_rag gate | no_rag estímulo | no_rag local |
|---|---|---|---|---|---|---|
| q001 | hedged_partial | hedged_partial | answered | answered | answered | answered |
| q016 | hedged_partial | hedged_partial | answered | answered | answered | answered |
| q068 | answered | answered | answered | answered | answered | answered |
| q070 | answered | answered | answered | answered | answered | answered |
| q172 | answered | answered | answered | answered | answered | answered |
| q180 | answered | answered | pure_decline | answered | answered | answered |

| Resumen | hybrid | no_rag |
|---|---|---|
| Gate: `pure_decline` | 0/60 | 0/60 |
| Gate: `hedged_partial` | 20/60 (33 %) | 0/60 |
| Local, mismas 6 tareas: `pure_decline` | 1/6 | 0/6 |
| Local, 194 consultas de exp12: `pure_decline` (igual que `decline_census_v2` en `faithfulness_metrics_v4.json`) | 107/194 | 5/194 |

Lectura:
- En la nube la clase es estable: el gate y los 12 boots en frío del estímulo coinciden tarea por
  tarea.
- La nube y el local no coinciden en 3 de las 6 tareas hybrid. El modelo es el mismo, pero cambian
  el hardware, el arnés de servicio y quizá parámetros de decodificación que exp12 no registra en su
  checkpoint. Con una sola ejecución local por tarea, no se atribuye la diferencia a ninguna causa.
- Las 6 tareas no representan las 194: la tasa de rechazo del gate no estima la del corpus.

## Latencia local de referencia

[VERIFICADO en `experiments/results/exp12_matrix`; DESCRIPTIVO]
En exp12 (194 consultas, hardware local), `granite4.1-8b` tardó:
- hybrid: p50 67,6 s y p95 161,3 s;
- sin RAG: p50 65,7 s y p95 105,0 s.

Con la L4 del despliegue, el p95 del gate es unas 4,6 veces menor en hybrid. No es una comparación
controlada: cambian el hardware y el arnés.

## Gasto

[VERIFICADO en el ledger del operador; no es la factura]
- El intento 15 completo, del cpu-build al close, registra USD 9,46 de exposición conservadora,
  dentro del tope de USD 20 + 3.
- La primera raíz del intento 15, abandonada al fallar cpu-build por cuota SSD llena, conserva
  USD 3,95 de exposición abierta conservadora. No entra en los USD 9,46.
- Los intentos 11 a 15 suman USD 40,02 según sus ADR (6,91 + 7,95 + 6,31 + 9,39 + 9,46), más
  esos USD 3,95 abiertos.
- La factura real se contrasta en la consola de facturación de Google Cloud.
- Fuera de estas cifras siguen cobrando los recursos retenidos (discos, snapshots e IP reservada).

## Límites

[DECLARADO]
- Las sesiones son sintéticas y repiten 6 preguntas. Miden el servicio, no la experiencia de una
  persona ni la calidad de las respuestas.
- El gate mide en caliente y en serie, con una sola sesión a la vez. No mide concurrencia.
- Las sesiones sintéticas de los intentos 14 y 15 siguen sin purgar en el disco GPU y en el bucket
  del período (ADR-0035). No contienen datos de personas.
