# BORRADOR — compuerta del estudio de dos condiciones

2026-09-27. **NO APROBADO. No ejecutar mediciones reales.** Esta propuesta no
modifica `NLI_SHARED_PREREGISTRATION_2026-09-20.md` ni `WARM_GATE_PREREGISTRATION.md`.
Recomendación D1: conservar íntegra la cohorte NLI de tres sistemas, ejecutándola
desde la copia aislada del build `521f525`; no sustituir sus resultados por esta compuerta.

## Crítica y alcance

La nueva condición sin consulta documental no tiene mediciones operativas. La
latencia histórica léxica ya no describe una condición del estudio, pero eso NO
autoriza declarar GO: cambia el tratamiento y hace necesaria una compuerta propia.
Dos sistemas pueden producir salidas diferentes con iguales parámetros; no es un
experimento de equivalencia de respuestas. El cegamiento solo afecta la etiqueta.

Propuesta: pipeline de estudio congelado, mapeo y CSV congelados, mismo Granite,
digest, parámetros y hardware. Caché de respuestas desactivada. Calentamiento real
de ambas condiciones antes de cada ventana/sesión y comprobación de residencia
antes de cada consulta. La familiarización NO es calentamiento.

## Diseño propuesto, pendiente de aprobación

- Seis tareas configuradas T1/T2, diez repeticiones por tarea y condición:
  **120 intentos, 60 por condición**. Alternar orden de condiciones por posición
  y repetición. No reutilizar observaciones históricas. Reportar por tarea además
  del agregado; n=10 por tarea no permite una estimación precisa de su cola.
- p50/p95 con interpolación lineal; umbral propuesto **p95 caliente ≤60 s en cada
  condición**, cero fallos y cero posiciones inválidas para aceptación. Mostrar
  máximos, tasa de fallos y todos los intentos; no suprimir respuestas lentas.
- Son repeticiones de solo seis tareas, no 60 consultas independientes. El p95
  describe este protocolo operativo; no estima generalización a todo el corpus.
- Frío se reportaría aparte si se autoriza medirlo, sin mezclarlo con caliente.
- Aclaración autorizada del prompt: no se recoge latencia/contenido/errores de la
  familiarización. Por ello se excluye de la medición y del paquete de la compuerta.
  Su único evento persistido por sesión es `familiarization_done` y timestamp.
- Configuración/entorno/versiones/digest/manifiesto/commit y orden fijados antes de
  medir. Una modalidad D2 elegida debe estar presente durante la futura validación:
  navegador participante autorizado y, si aplica, videollamada/túnel. Cerrar otras
  cargas. No trasladar cifras de una medición sin ese canal al despliegue con él.
- Error/aborto entra en tasa de fallos, jamás en percentiles. Contaminación conserva
  respuesta pero marca inválida. No reemplazar posiciones. Par interrumpido: cohorte
  terminal insuficiente, sin completar brazo en otra ventana. Pausa entre pares
  completos requiere nueva ventana explícita; límite de ventana 120 minutos.
- Si híbrido o condición sin consulta documental exceden umbral: **NO-GO para el
  estudio**, documentar y pedir autorización para una intervención separada. No
  cambiar criterio después de ver resultados. El GO requiere además sesión completa,
  recuperación/reinicio/exportación íntegros y preparación válida.

## Frontera y regresión

`study_service.answer`: cronómetro después de adquirir exclusión de inferencia y
antes de guardar solicitud; incluye ese guardado, comprobación de residencia,
retrieval/reordenamiento cuando corresponde, generación, NLI cuando corresponde y
proyección de fuentes. Fin justo antes del guardado final. Excluye espera del lock,
calentamiento previo, lectura humana, transporte y pintado del navegador. Se publica
como latencia de servidor, nunca como tiempo visual completo. Medir por separado
latencia cliente/servidor en la modalidad D2 seleccionada requiere instrumentación
adicional autorizada. Regresión `test_clock_includes_factory_and_query` del servicio
y frontera 2+3=5 del ejecutor sintético.

## Ejecutor diseñado y comprobable sin modelos

`scripts/study_gate_draft.py --dry-run --output <directorio externo nuevo>` recorre
120 posiciones simuladas, checkpoint por intento, hashes, resumen, rechazo de
contaminación simulada, expiración y reanudación explícita entre pares completos.
**No incluye adaptador real ni intervención NVIDIA/AnyDesk**. Sus resultados llevan
`SYNTHETIC_ONLY` y `NOT_A_GO_DECISION`; no son evidencia de rendimiento.

Antes de habilitar un runner real: aprobación D1/D2, adaptación del preflight a la
modalidad escogida, verificación del bundle y warmup, telemetría y monitor de energía,
supervisor independiente probado si se autoriza una intervención de procesos, y
pruebas de esas fronteras. El humano lanzará toda medición larga; el agente no la espera.
