# R1 prospectivo: ejecución de auxiliares en la L4 existente

Autorización: prompt maestro de Enzo, iteración 3, A1/A3/A4 y §4 Fase 7.
Este documento se commitea antes de implementar o medir el candidato.
No modifica tareas, pesos, índices, dependencias, prompts, temperatura, NLI
`per_claim`, frontera, timeout ni umbral. R2 permanece inelegible por el fallo
terminal de calidad de su cohorte; no se activa `cross_claim`.

## Evidencia, diagnóstico y crítica

El agregado real íntegro está en el paquete externo de la iteración 3:
`gate02-independent-analysis23/analysis.json`, respaldado por el manifiesto
generacional de `cloud-i3-gate02-corrected-20261004t142921z` y su verificador.
Son 120 intentos válidos, cero fallos/inválidos; p95 híbrido 113,37 s y sin RAG
22,44 s. Los valores exactos y hashes se leen de esos artefactos, no de esta nota.
`gate-diagnosis24.json` contiene el desglose por intento, condición y tarea.
En q070 NLI promedia 73,45 s, reranking 8,25 s y generación 28,66 s; q068
muestra también predominio NLI. Son observaciones retrospectivas que generan
una hipótesis; no prueban todavía que GPU reduzca la duración ni conserve texto.
No hay paradoja léxico/híbrido en esta compuerta, que no contiene condición
léxica; el hallazgo NLI local separado sigue sin cierre causal.

Opciones consideradas: más CPU exigiría otro entorno/recurso; modificar lotes
es R2 y está prohibido sin su equivalencia; reducir generación es R3 y cambia
el sistema. La ejecución GPU usa la L4 y pesos ya existentes, cambia una sola
variable y es reversible. Riesgos: resultados numéricos distintos en auxiliares,
orden de recuperación y texto distintos aun a temperatura cero, contención
con Ollama y memoria GPU. Se rechaza el candidato ante esas diferencias según
el contrato siguiente; no se seleccionan respuestas históricas favorables.

## Hipótesis y mecanismo

Hipótesis falsable: la ejecución de forward passes auxiliares en CPU explica
el componente dominante del exceso; permitir CUDA en esos auxiliares reduce
al menos 70 % la mediana conjunta de reranking+NLI de las seis tareas híbridas
sin cambiar texto ni insumos. Se espera reducir el p95 híbrido a 60 s o menos:
el componente CPU observado ronda 80 s en q070; dejar 24 s más unos 29 s de
generación ofrece margen aproximado de 7 s. Es una proyección, no un resultado.
Se informa la discrepancia aunque mejore sin llegar a esa expectativa.

Única variable: `CUDA_VISIBLE_DEVICES` del contenedor de la app/runner,
control vacío (auxiliares CPU), candidato `0` (misma L4). Ollama conserva la
misma L4, digest, contexto y opciones. Se amplía la identidad para inventariar
y rechazar deriva de esa variable, sin activar GPU por defecto ni cambiar
la receta. Ambas ramas usan la misma imagen con esa instrumentación.

## Calendario y admisión prospectiva

1. Construcción del candidato de la Fase 7, independiente de los tres builds
   iniciales ya consumidos. Máximo dos builds por causa corregida, tres horas
   verificables dentro de la ronda; `pip check`, ambas suites y siete pruebas
   POSIX sobre el disco persistente. No se borra evidencia de builds anteriores.
2. Enmienda/anexo final e identidad nueva antes de medir. Misma VM, zona,
   pesos, índices, pins y controles de contaminación. Supervisor nativo STOP,
   cierre independiente, máximo 600 s por llamada, 900 s preparación y
   120 min por ventana. Ningún navegador/smoke local durante esta medición.
3. Verificación prospectiva de texto: orden fijo q001/q068/q180/q016/q070/q172,
   híbrido y sin RAG para cada tarea, temperatura cero. Se ejecutan las doce
   posiciones nuevas de control y luego las doce del candidato, sin reintento
   ni sustitución; se conservan textos crudos, insumos y hashes. Elegibilidad
   R1 exige igualdad exacta por posición del texto generado en las doce
   comparaciones, mismos IDs y textos de contextos y misma clase v2/citas.
   Se informan todos los scores NLI, sin redondearlos como prueba de igualdad.
   Una diferencia implica R1_INELEGIBLE; no se redefine equivalencia ni se
   inicia su compuerta. Una interrupción da evidencia insuficiente terminal.
4. Si es elegible, piloto NUEVO caliente de 20 posiciones: primer tramo del
   calendario alternado de la compuerta, que incluye las seis tareas y ambas
   condiciones. Cero fallos/inválidos, comparación descriptiva por etapa y
   expectativa anterior; ningún piloto concede GO. Se conservan las veinte
   posiciones aunque la expectativa no se cumpla; no se afinan parámetros.
5. Después del piloto íntegro, compuerta NUEVA de dos ventanas de 60 intentos
   con identidad idéntica, cero fallos/inválidos y p95 lineal agregado <=60 s
   por condición. Nunca se mezclan control, elegibilidad, piloto o historial
   con sus 120 intentos. Una cohorte terminal no se reanuda ni reemplaza.

Máximo una ronda R1, seis horas verificables de comandos/mediciones, además
del techo global y cierre reservado. No se crean recursos GPU nuevos ni se
cambia arquitectura. Presupuesto acumulado y reserva se comprueban antes
de cada encendido. Si R1 es inelegible o NO-GO, se revierte visibilidad CUDA
vacía; se evalúa R2 (inelegible) y solo entonces se preregistra R3 por separado.

## Rollback y UX

Rollback: detener/desarmar la VM, conservar evidencia, arrancar el mismo
contenedor con CUDA vacía y generar identidad CPU nueva. No se toca el driver
local ni NVDisplay, dependencias, corpus, evidencia congelada o documentos de
Enzo. La cuenta ACME y claves privadas quedan en el disco retenido.
No cambia ningún texto visible: mismas etiquetas A/B, mensajes e instrumentos
en los mismos momentos. Solo se espera menos espera por respuesta. Si R1 se
selecciona, un smoke propio del entorno candidato verifica otra vez las ocho
respuestas v2, instrumentos, exportación y respaldo; no rescata smokes previos.
No se reclutan participantes y un GO no sustituye aprobación ética.
