# Compuerta operativa del estudio — APROBADA

Aprobada por Enzo el 2026-09-28. Este documento fija criterios antes de medir y no modifica los pre-registros NLI ni WARM; el [borrador histórico](STUDY_GATE_PREREGISTRATION_DRAFT.md) se conserva.

La compuerta tiene dos condiciones, `hybrid` y `no_rag`, seis tareas (`q001`, `q064`, `q171`, `q010`, `q070`, `q172`) y diez repeticiones por tarea/condición. La ventana 1 contiene repeticiones 1–5 (60 intentos) y la ventana 2 las repeticiones 6–10 (60 intentos). El orden alternado se determina antes de medir y cada ventana está balanceada por separado.

El cronómetro monotónico comienza antes de la solicitud durable y termina cuando existe el payload listo para mostrar; incluye residencia/preparación, recuperación, generación, NLI y proyección de citas. Excluye espera del lock, calentamiento, transporte/pintado del navegador y lectura humana. No es latencia de navegador.

Cada ventana tiene un límite independiente de 120 minutos. Una respuesta fallida, una posición inválida, la expiración o una interrupción no planificada termina la cohorte: no hay imputación, reemplazo ni continuación excepcional. La ventana 2 solo puede comenzar cuando la 1 terminó íntegra y se revalidaron identidad, configuración sellada, hashes y entorno.

Solo el agregado de ambas ventanas puede decidir: GO requiere 60 observaciones válidas por condición, cero fallos, cero inválidas y p95 caliente ≤60 s por condición. Una ventana aislada jamás produce GO. El modo seco se etiqueta `SYNTHETIC_ONLY` y nunca es evidencia de rendimiento.

Antes de cada ventana el operador declara Zoom activo y que comparte la ventana de la app; la mera presencia del proceso no es evidencia suficiente. Se conserva telemetría, progreso, ETA, checkpoints, causas de invalidez, resumen y manifiesto con hashes. El runner no termina Ollama, Zoom ni ningún servicio.
