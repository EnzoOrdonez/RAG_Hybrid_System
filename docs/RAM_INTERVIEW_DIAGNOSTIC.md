# RAM y latencia: diagnóstico local del 2026-09-09

**VERIFICADO:** el piloto de 40 consultas híbridas terminó sin errores de respuesta.
**Conclusión:** estos datos no justifican comprar 32 GB como solución al p95.
La RAM adicional podría dar margen a otras aplicaciones (**SUPUESTO**), pero no
hay una medición con 32 GB que permita estimar una mejora porcentual ni garantizar
60 s. Aumentar RAM del sistema no aumenta los 6 GB de VRAM de esta GPU.

## Evidencia y método

Raíz externa `C:/CloudRAG/managed-pilot-20260908T012621Z/` (en adelante R).
Medición: `R/window-real-05/cohort/`, build
`a761fa28e7af691f2e2bdaee880e20f987ae1a59`, finalizada
2026-09-09T01:24:20Z. Análisis: `R/memory-analysis-01.json`, generado
2026-09-09T10:45:35Z con `f4a60f2a420c63b8d4e04c143a36590bdba49326`.
Incluye hashes de manifiesto, resultados, journals, muestras, HTTP, ETL y JSON
decodificado. No modifica la cohorte. Equipo: i5-12450H, 16 GB nominales de RAM,
RTX 3060 Laptop 6 GB, driver 616.64, Ollama 0.22.1, Granite de digest `444af1c4…ba2852`.
El digest completo y versiones están en `cohort/source-manifest.json`.

Crítica del método: compromiso virtual no equivale a RAM residente; los fallos
duros globales incluyen carga de archivos y otros procesos. Mezclar carga inicial
con generación podría producir una recomendación de compra falsa. Por ello
`scripts/analyze_gate_memory.py` delimita `chat` con sus timestamps HTTP reales,
mantiene aparte la ventana del observador y rechaza hashes o capturas incompletas.
Sus cinco regresiones prueban límites del intervalo, datos ausentes y alteración
de journals. No se suman percentiles de etapas ni se incluyen calentamientos.

## Observaciones durante chat (20 consultas por condición)

| Medida VERIFICADA | Frío | Caliente |
|---|---:|---:|
| RAM disponible: mínimo de todas las muestras | 2,02 GiB | 4,21 GiB |
| Mediana del mínimo disponible por consulta | 4,19 GiB | 4,30 GiB |
| Compromiso / límite: rango observado | 44,16–62,38 % | 63,57–64,47 % |
| Compromiso máximo | 28,53 GiB | 29,48 GiB |
| Fallos duros globales/s: mediana por consulta | 18,06 | 0,79 |
| Fallos duros globales/s: máximo por consulta | 156,09 | 737,40 |
| Temperatura GPU: rango muestreado | 52–61 °C | 53–63 °C |
| Tokens de salida/s: mediana por consulta | 16,87 | 17,40 |

No se observó agotamiento sostenido del compromiso ni de RAM disponible durante
chat. Esto no descarta paginación o presión transitoria entre muestras de 5 s.
El compromiso incluye memoria virtual respaldada por RAM/pagefile: no significa
que 29 GB estuvieran simultáneamente residentes en los 16 GB físicos.
Referencia: [Microsoft, diagnóstico de memoria](https://learn.microsoft.com/en-us/troubleshoot/windows-server/performance/troubleshoot-performance-problems-in-windows).

El pico caliente (índice 6, chat de 11,2504 s) tuvo 8.296 fallos duros:
8.280 de `svchost.exe`, 12 de protección del endpoint, 2 de Python y 2 de System.
La consulta completa tardó 22,55 s y mantuvo >=4,24 GiB disponibles. Numerosas
rutas corresponden a servicing de Windows; 6.595 archivos no pudieron atribuirse.
No convertir ese pico global en «Ollama está paginando por falta de RAM».
El primer chat frío tuvo 4.270 fallos: 907 de Ollama y 2.513 de System, entre otros;
incluye carga de Granite, no solamente evaluación de tokens.

## Qué sí limita y qué no está aislado

**VERIFICADO:** `ollama ps` mostró reparto 43 % CPU / 57 % GPU en el primer frío,
y 19 % CPU / 81 % GPU en los posteriores y calientes, con contexto 4096.
Ese porcentaje expresa colocación del modelo, no utilización instantánea del
procesador. La GPU conservaba alrededor de 481 MiB libres en las muestras calientes.
[Ollama explica cómo interpretar el reparto](https://docs.ollama.com/faq).
El offload parcial existe, pero su costo causal no se midió contra carga 100 % GPU.

**VERIFICADO:** generación domina la mediana de las etapas (23,71 s frío / 15,72 s
caliente); retrieval pasa de 7,08 a 0,13 s y el trabajo fuera de etapas cronometradas
de 8,76 s a 0,002 s. Incluye inicialización y otros costos no desglosados; no se
atribuye todo ese residuo a carga de modelos. La peor respuesta caliente tardó
61,47 s: generación 37,71 s y NLI 19,05 s. No hay un único costo constante.

**VERIFICADO:** AC y plan efectivo Mejor rendimiento en las muestras válidas;
no se registraron flags térmicos GPU activos durante chat. Hubo una muestra con
`sw_power_cap=Active` (caliente índice 15, consulta completa 21,34 s).
**SIN AISLAR:** throttling CPU (sin temperatura CPU), costo de offload y efecto
causal de RAM/energía. La observación no permite certificar causa exclusivamente
de hardware ni descartar toda interferencia del sistema operativo.

**DECLARADO:** la cohorte histórica 616.64 incluyó batería y Brave. El piloto
actual mejora sus p95, pero los historiales no permiten separar driver de carga.
No se demuestra que 610.62 sea más rápido, ni que 616.64 haya corregido una regresión.

## Decisión de compra y siguiente contraste

No recomendar 32 GB como corrección acreditada de CloudRAG. **SUPUESTO:** podría
mitigar competencia de RAM al mantener aplicaciones abiertas; no elimina el
offload causado por capacidad de VRAM. **ESTIMADO:** no hay un porcentaje defendible
de aceleración con estos datos. Para cuantificarlo haría falta una comparación
pareada con 16/32 GB y mismo resto del entorno, sin comprar ni cambiar hardware
por iniciativa del agente. Sigue pendiente el cierre real de la app; véase
[la compuerta](OPERATIONAL_GATE_2026-09-05.md#cierre-del-piloto-y-reanudación-de-p900-2026-09-09).
