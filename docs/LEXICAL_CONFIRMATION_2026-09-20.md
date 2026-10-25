# Diagnóstico emparejado completo — 2026-09-20

**VERIFICADO: hipótesis matizada; NO-GO vigente por p95 léxico e híbrido.**
Esta entrega analiza una cohorte ejecutada por el usuario. No implementa una
optimización, no cambia el criterio de compuerta y no autoriza entrevistas.
El [pre-registro candidato](NLI_SHARED_PREREGISTRATION_2026-09-20.md) requiere
aprobación formal antes de modificar código. La fecha del título es la fecha
local del estudio; la auditoría documental comenzó el 21-sep a las 00:26 UTC,
todavía 20-sep en Lima.

## Fuentes, integridad y reproducción

- Build medido y baseline documental: `a06bb98c1d400458b3bd070a8e2eb748de3f1a9f`.
  Rama `fix/interview-readiness`; árbol limpio antes de la edición documental.
- Paquete original, sólo lectura: `C:/CloudRAG/diag-run-20260920T223927918Z/`.
  `summary.json` publicado a las `2026-09-20T23:38:41.305931+00:00`.
- SHA-256 de `manifest.json`:
  `329b492b8c68b2afaa4c5d42c36764c58b034d1f44a6fd85bc66b1cc1f5a32bf`.
- Análisis nuevo y auditoría, fuera del paquete original:
  `C:/CloudRAG/lexical-analysis-20260921T002614931Z/`.
- `analysis.json`, SHA-256:
  `e6b6df63c94f087229cb43d19d472aa6e9e967b2d8d5e205bbc8eed55884ac91`.
- `analyze.py`, SHA-256:
  `9bea28fa5365795644672f73aab1e19846b2901b749820d55cad92f1a3018ecd`.

VERIFICADO: `verify_package` comprobó inventario y hashes de **962 archivos**.
Los 40 registros tienen posiciones únicas y orden cronológico igual al calendario
AB/BA registrado. Los conteos de llamadas/pares NLI coinciden con las trazas de
ejecución. El resumen recalculado desde los registros coincide con el exportado.
Hay 20 éxitos válidos por sistema, cero errores, abortos, invalidaciones y
posiciones pendientes; `complete=true`, `confirmation_ready=true`,
`energy_aborted=false`, `cleanup_pending=[]`. Estos indicadores no equivalen a GO.

`analysis.json` contiene las 40 filas con consulta, posición, ID, timestamp,
ruta y hash del `result.json` original, etapas, carga, diferencias pareadas y
escenarios. No se reemplazaron ni reescribieron registros originales. Para
reproducirlo, crear un directorio de salida nuevo y existente fuera del checkout
y del paquete, y ejecutar desde este worktree:

```powershell
.\.venv-app\Scripts\python.exe -B C:/CloudRAG/lexical-analysis-20260921T002614931Z/analyze.py RUTA_NUEVA_DE_SALIDA
```

El script rechaza sobrescribir `analysis.json` y comprueba, entre otros, unicidad
de posiciones, orden, hashes, igualdad con el resumen y reconstrucción del p95.
No importa ni ejecuta modelos. El baseline de tests se conserva en
`baseline-pytest.txt`: **576 pasan, 5 excluidos, 9 subpruebas, 3 avisos SWIG**, 47,80 s.
Los controles posteriores y el hash del commit documental quedan en la auditoría
final externa de esta misma carpeta; no se atribuyen al build medido.

## Entorno y controles

VERIFICADO en `cohort/source-manifest.json`: i5-12450H, RAM 16.891.633.664 bytes,
RTX 3060 Laptop GPU 6144 MiB, driver 616.64, Ollama 0.22.1, seed 42, caché LLM
desactivada, timeout de lectura 180 s y Granite `granite4.1:8b`, digest
`444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`.

VERIFICADO en 324 muestras de telemetría: AC presente en todas, cero muestras
con errores del colector y RAM disponible mínima **3,29 GiB**. El plan base fue
Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`) con overlay efectivo Best
Performance (`ded574b5-45a0-4f42-8737-46345c09c238`), exactamente lo admitido por
el observador. No confundir este overlay de energía con NVIDIA Overlay.
Estos datos no prueban ausencia absoluta de carga entre muestras ni causalidad
del driver; no se realizó un experimento comparativo de drivers.

Ventana: `windows/ba5126004364403e9927735e9630b02f/`. Las pruebas independientes
del supervisor por vencimiento y muerte del controlador pasaron a las 22:40:17
y 22:40:24 UTC, antes de la ventana real. El contraste de observador registró
10 pares y límite superior unilateral 95 % de **0,28555 %**, menor que 5 %;
no se descontó tiempo a las consultas.

Restauración VERIFICADA por eventos y `restored.json`: NvContainer a las
23:38:34 UTC, SelfUpdate habilitada a las 23:38:35, restauración completa a las
23:38:37 y watchdog retirado a las 23:38:40. AnyDesk y NVDisplay conservaron
su estado inicial. El marcador de restauración se emite después de comprobar
servicios y presencia del overlay; no se reclama una inspección actual del
sistema fuera de la ventana histórica.

## Resultados: 20 posiciones válidas por sistema

Latencias totales en segundos; cuantiles lineales con posición `(n-1)*q`, como
`measure_interview_gate.percentile`. Fallos/abortos/invalidaciones nunca entran
en percentiles. Superar 60 s es incumplimiento de latencia, no error técnico.

| Sistema | p50 | p90 | p95 | Mínimo | Máximo | Media | Respuestas >60 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Léxico | 24,66 | 63,06 | **70,61** | 6,97 | 86,26 | 30,81 | 4/20 |
| Híbrido | 29,54 | 59,63 | **63,18** | 15,65 | 77,62 | 35,11 | 2/20 |

Cada celda siguiente contiene **p50 / p95**; las etapas están en segundos.

| Variable | Léxico | Híbrido |
|---|---:|---:|
| Retrieval | 0,0325 / 0,0430 | 0,1324 / 0,1437 |
| Re-ranking | ≈0 / ≈0 | 4,3680 / 5,0385 |
| Generación | 17,8641 / 50,7990 | 19,1288 / 46,6705 |
| NLI | 2,6784 / 17,6293 | 3,3915 / 13,0238 |
| Otros fuera de las etapas instrumentadas | 2,0485 / 2,0680 | 2,0445 / 2,0633 |
| Tokens de salida | 188,5 / 643,45 | 222 / 559,75 |
| Claims extraídos | 3 / 13,10 | 3,5 / 9,45 |
| Llamadas NLI reales | 3 / 13,05 | 2,5 / 9,45 |
| Pares NLI reales | 15 / 65,25 | 12,5 / 47,25 |
| Caracteres de prompt | 6.916,5 / 10.216,4 | 8.193 / 12.542,2 |
| Chunks al generador | 5 / 5 | 5 / 5 |

La frontera incluye selección/verificación de preparación dentro de la consulta,
pipeline completo y serialización/exportación instrumentada. El montaje del
instrumento y la publicación durable posterior quedan fuera; preparación inicial,
admisión y contraste se registran separados. No se vuelve a definir el cronómetro.
Las regresiones están en `tests/test_diagnostic_clock_boundaries.py` y
`tests/test_paired_diagnostic.py`. **No sumar percentiles marginales de etapas.**

### Comparación histórica separada

| Cohorte | n válido por sistema | p95 léxico | p95 híbrido | Uso |
|---|---:|---:|---:|---|
| Segunda ventana parcial, build 658b748 | 14 | 68,99 s | 64,71 s | Sólo descriptivo |
| Diagnóstico completo 20-sep, build a06bb98 | 20 | 70,61 s | 63,18 s | Contraste prospectivo de carga |

Fuente parcial: `C:/CloudRAG/lexical-clean-20260911T2020Z/partial-analysis.json`,
SHA-256 `f58c09fa46ddc5d50a833e8bd2f6b30ee4daa347b28708c2e41cde294d3a1f8c`.
La invalidez por Brave y los once
no ejecutados siguen conservados. DECLARADO por el usuario: abrió Brave él mismo.
No se mezclan estas muestras ni las cohortes 610.62/616.64 contaminada/pilotos
previos. Tampoco se utiliza la medición antigua semántica para certificar un
verificador compartido que aún no existe.

## Historia causal: veredicto matizado

VERIFICADO: generación y NLI dominan la cola y la cantidad de trabajo realmente
ejecutada depende de la respuesta. Que el híbrido añada RRF y re-ranking no
implica más trabajo total en una respuesta diferente. No hay una violación de
esa estructura; hay carga variable de generación/verificación que debe medirse.

El léxico es más lento en **6/20 pares**, no en todos. Diferencia léxico menos
híbrido: media **−4,30 s**, mediana **−6,14 s**. Su p95 es peor aunque su media y
mediana sean mejores. Un solo resumen no describe toda la distribución.

| Caso | Sistema | Total s | Generación s | NLI s | Tokens | Claims | Pares |
|---|---|---:|---:|---:|---:|---:|---:|
| q011 | Léxico | 86,26 | 67,78 | 16,40 | 861 | 11 | 55 |
| q011 | Híbrido | 42,84 | 23,49 | 12,43 | 245 | 9 | 45 |
| q016 | Léxico | 69,78 | 49,91 | 17,81 | 632 | 15 | 70 |
| q016 | Híbrido | 77,62 | 46,54 | 24,28 | 574 | 18 | 90 |
| q027 | Léxico | 29,87 | 23,69 | 4,12 | 273 | 3 | 15 |
| q027 | Híbrido | 62,42 | 49,13 | 6,06 | 500 | 4 | 20 |

q016 contradice la versión fuerte de la hipótesis histórica: aquí el híbrido
extrae y verifica más claims que el léxico. q011 sí muestra una salida léxica
mucho mayor y 44,29 s adicionales de generación. q027 conserva una cola híbrida
problemática, aunque ya no repita exactamente los 92,73 s de la muestra parcial.
No se oculta ninguna de las dos respuestas híbridas >60 s.

### Cuantificación correcta del exceso sobre 60 s

Con n=20, el p95 interpola 95 % del segundo mayor total y 5 % del máximo.
Aplicando esos mismos pesos a sus etapas, y no a los percentiles marginales:

| Componentes de las observaciones que forman el p95 | Léxico | Híbrido |
|---|---:|---:|
| Generación ponderada | 50,7990 s | 49,0032 s |
| NLI ponderado | 17,7367 s | 6,9712 s |
| Re-ranking ponderado | ≈0 s | 5,0137 s |
| Retrieval ponderado | 0,0392 s | 0,1372 s |
| Resto, incluido procesamiento de consulta | 2,0307 s | 2,0507 s |
| Total p95 | 70,6056 s | 63,1760 s |
| Exceso sobre 60 s | **10,6056 s** | **3,1760 s** |

Ésta es una descomposición contable de latencia, no una estimación causal de
segundos recuperables. Un NLI instantáneo es un contrafactual imposible, útil
como cota: recalculando toda la distribución, p95 sería 52,87 s léxico y
53,50 s híbrido. No se atribuye todo ese ahorro a una optimización futura.

### Crítica y alternativas no descartadas

- Orden AB/BA equilibrado y preparación residente acotan orden/carga, pero no
  eliminan efectos de estado KV, secuencias distintas ni variación de generación.
- Los prompts híbridos son mayores en agregado; el contenido y el truncamiento
  efectivo pueden importar. No se deduce que el prompt tenga costo cero.
- El conteo de pares explica cantidad de verificaciones, no duración universal
  por par: tokenización, longitud, padding y ejecución CPU también intervienen.
- `hallucination_detector.py:503–528` procesa un claim por llamada, normalmente
  cinco pares, con `batch_size=32`. Es una oportunidad comprobable de agrupación,
  **no evidencia de que la agrupación ahorre un porcentaje determinado**.
- La muestra sigue siendo pequeña y específica. No hay un experimento que
  varíe sólo longitud de salida, ni una prueba causal del driver o del hardware.

## Estado de compuerta y siguiente paso

| Condición | Estado en esta entrega |
|---|---|
| Diagnóstico íntegro y limpio | ✅ VERIFICADO, 40/40 |
| Léxico caliente p95 ≤60 s | ❌ VERIFICADO, 70,61 s |
| Híbrido caliente p95 ≤60 s | ❌ VERIFICADO, 63,18 s |
| Semántico con una futura optimización compartida | Pendiente: no existe ese cambio ni una cohorte suya |
| P900 y resiliencia | Cierre histórico documentado, no reejecutado en esta entrega |
| Entrevistas / Fase B | **NO-GO / bloqueada** |

El diagnóstico habilita proponer una intervención falsable; no prueba su
eficacia. Decisiones explícitas del usuario durante la planificación: NLI
compartido y validación nueva de 120 intentos emparejados. Esas decisiones no
sustituyen el visto bueno formal del pre-registro. No se tocaron producción,
modelos, procesos, evidencia congelada, P2s ni nube.
