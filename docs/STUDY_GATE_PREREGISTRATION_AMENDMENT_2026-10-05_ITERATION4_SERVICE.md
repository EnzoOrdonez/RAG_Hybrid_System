# Enmienda prospectiva: compuerta sobre la imagen final de iteración 4

Esta enmienda se publica antes del piloto o de la compuerta que evalúan la
capa de servicio final. No concede GO, no sustituye la aceptación del estímulo
y no autoriza participantes. No modifica la lógica RAG, los prompts, las
opciones de generación, los modelos, los pesos ni los índices. El GO de R1 de
iteración 3 no cubre la capa de servicio final.

## Prerrequisitos

Se exige el congelamiento RAG final igual a la base, incluidos los doce
contextos verificados sin generar contenido, un anexo del entorno efectivo
commiteado y publicado y la aceptación íntegra del estímulo de cláusula 58:
120 posiciones objetivo, doce combinaciones con un texto, una clase v2 y un
conjunto de citas, historiales variados y al menos dos arranques por
combinación. Se exige además el smoke HTTPS completo aprobado mediante el
mismo automatizador que pasó el recorrido sintético. Un resultado parcial o
sintético no satisface estos prerrequisitos.

El paquete externo `C:/CloudRAG/autonomous-run-20261004T230147Z/` conserva
`verified-final-context-provenance.json`, `rag_freeze_final.json`, el recibo
del anexo y el inventario `environment_identity.json` generado en la VM.
Las identidades y hashes se obtienen de esos archivos mediante scripts; no
se introducen manualmente como prueba. Antes de medir se anclan también el
código de los observadores, esta enmienda y el calendario en un commit
publicado o en un objeto GCS con generación y hora de servidor verificadas.

## Hipótesis, intervención y rollback

La hipótesis prospectiva de servicio es que un runner de generación nuevo
para cada consulta elimina el efecto del historial inmediato observado en
el diagnóstico, sin cambiar los bytes del prompt ni las opciones enviadas
a Ollama. Se aplica a toda consulta de ambas condiciones. No se sirven
respuestas almacenadas. La evidencia disponible sostiene aislamiento del
runner como candidato; no prueba por sí sola una causa específica de caché KV.

La aceptación del estímulo comprueba esta hipótesis antes de la compuerta.
Los textos pueden diferir de exp12. No se promete de antemano una latencia
observada ni se ajusta el umbral después de medir. El rollback consiste en
detener la imagen candidata y conservar toda su evidencia; el GO histórico
no se transfiere a otra imagen ni a otro servicio.

## Calendario y fronteras

Las tareas son, en orden: T1 q001/q068/q180 y T2 q016/q070/q172. Se conserva
la reserva de q180; no se buscan reemplazos ni se consultan respuestas para
seleccionar tareas. El runner lee las tareas del protocolo revisado y comprueba
este orden, evitando sus valores de ejemplo antiguos.

Se realiza un piloto nuevo caliente de veinte posiciones: las primeras veinte
del calendario de ventana 1, después de preparar los modelos auxiliares de
ambas condiciones. El runner de generación sigue siendo nuevo en cada
consulta. Se conservan las veinte posiciones, los resultados, la preparación
y los cierres del supervisor. El piloto no concede GO. Si queda incompleto,
falla o contiene una posición inválida, no se inicia automáticamente una
compuerta ni se sustituye el piloto.

La compuerta tiene dos ventanas de sesenta posiciones. Ventana 1 usa las
repeticiones 1–5; ventana 2, 6–10. El orden de condiciones alterna según
repetición y posición, como en el calendario vigente. Se ejecutan en un solo
arranque y con el mismo inventario efectivo. No se mezclan identidades,
ventanas de otras cohortes ni datos del piloto con la compuerta.

El cronómetro conserva la frontera de la app: la consulta pasa por
`study_service.answer` y termina cuando están disponibles la respuesta y su
evaluación. Incluye recuperación, generación, NLI y toda espera del servicio
para descargar, reiniciar y cargar el runner. Preparación, observación en
reposo, escritura posterior de evidencia y transporte desde Lima quedan
fuera de ese cronómetro. La latencia HTTPS desde Lima se describe por separado.

Se mantienen 600 segundos por llamada, 900 de preparación y 7.200 por ventana,
sin modificar una frontera durante la medición. El supervisor de trabajo
completo tiene un límite adicional de 9.000 segundos, compatible con el STOP
nativo de tres horas y el supervisor del invitado de 165 minutos. Se exige
margen suficiente antes de lanzarlo. Si un límite deja datos incompletos, se
conservan como terminales e incompletos: no se convierten en NO-GO medido.

## Observación y seguridad

Se usa el runner y el supervisor contenidos en la imagen final. El observador
sintético es no root, de solo lectura, sin capacidades, sin privilegios nuevos,
sin red y sin logs de Docker. Solo este observador técnico ve los PID del host,
porque el muestreador Linux existente compara los PID publicados por NVIDIA
con `/proc`. La app para participantes conserva su espacio privado de PID y
no alcanza metadatos. El supervisor rechaza propósito `study`. No se crean
participantes, aprobaciones éticas ni invitaciones reales.

La app se cierra para mantenimiento durante la medición. Solo se ejecuta una
medición en la VM. La telemetría comprueba identidad, residencia y transiciones
del servicio, carga y procesos ajenos. El contenedor conserva `--rm=false`.
Los supervisores paran el contenedor exacto y no dependen de la conversación.
Se descargan los archivos por generación, tamaño y SHA-256. Los intentos,
fallos e inválidos se conservan con sus fronteras originales.

## Regla de decisión

La decisión se calcula exclusivamente sobre las dos ventanas íntegras: 120
intentos, sesenta por condición, cero fallos, cero inválidos y p95 lineal
menor o igual a 60 segundos en cada condición. Se publican p50 y p95, conteos
y manifiestos. Una ventana aislada, el piloto o un ensayo sintético no
conceden GO. Un NO-GO completo queda `BLOQUEADO-HUMANO`, con diagnóstico por
etapa; no habilita remedios automáticos que alteren la lógica RAG congelada.
Una ventana parcial permanece incompleta y terminal, sin sustitución automática.
