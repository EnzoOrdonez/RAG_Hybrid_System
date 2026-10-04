# Corrección prospectiva del reloj de residencia

Autorización: Enzo, iteración 3, A1 y cláusulas 15, 21, 23 y 38.
Documento escrito antes de corregir el guard y antes de iniciar otra medición.
Se aplica junto con la enmienda del 2026-10-02 y el anexo del 2026-10-03.

## Defecto y evidencia conservada

La cohorte `i3-gate01-20261004t035751z` terminó en la primera ventana:
31 intentos válidos y el intento 32 abortado. No empezó la segunda ventana.
Su estado es **INSUFFICIENT_TERMINAL**, sin GO ni NO-GO agregado medido.
El campo interno `NO_GO` del resumen terminal no constituye ese veredicto.
No se reanuda, completa, sustituye una posición ni rescata ningún dato suyo.

La evidencia externa está en el paquete de la iteración 3, en
`cloud-i3-gate01-20261004t035751z/`, con descarga verificada por generación y
SHA-256. `gate01-causal-guard-diagnosis.json` reproduce las 308 muestras:
el historial falla únicamente por `residency_lease`, mientras la última
muestra está sana. El reloj de reproducción se declara estimado, anclado al
UTC registrado del preflight y a diferencias monotónicas, sin inventar un
timestamp nativo del aborto. La regresión determinista también reproduce el
fallo al envejecer la primera concesión, aunque las posteriores estén renovadas.

El guard comparaba la caducidad de **cada muestra histórica** con la hora
actual. Una concesión que era válida al observarse podía volverse inválida
retrospectivamente. Esto es un defecto de observación, no una pérdida
demostrada de residencia, ni evidencia de una mejora de rendimiento.

## Corrección y nueva colección

Cada muestra guarda su UTC de observación. Su concesión se evalúa en ese
instante; la última muestra se coteja además con la hora actual. Se mantienen
el margen de 180 segundos, contexto 4096, digest único, comprobaciones de
CPU/GPU, hueco máximo de 15 segundos y rechazo de contaminación. Una hora
ausente, inválida o sin zona horaria invalida la telemetría.

Antes de una nueva colección deben aprobarse las regresiones del historial
renovado, concesión histórica corta, concesión actual caducada y reloj inválido;
ambas suites Windows y Linux, controles y nueva identidad de la imagen.
La app, generación, pesos, índices, recetas y automatizador no cambian. Se
documentará la igualdad de sus archivos y se verificará nuevamente la identidad
viva; el smoke HTTPS previo conserva su alcance funcional, no se afirma que
su commit o inventario sean el de la nueva medición.

Por la cláusula 21, se admite **una colección nueva** con dos ventanas nuevas
de 60 intentos, evidencia en directorios nuevos e identidad nueva e idéntica
entre esas dos ventanas. Es una corrección explícita del instrumento antes de
volver a medir, no una reanudación o sustitución automática de la cohorte
terminal. Todos sus intentos se conservan. Rigen 600 segundos por llamada,
900 de preparación, 120 minutos por ventana, 48 horas globales y corte de USD45.
Solo su agregado íntegro de 120 intentos puede decidir GO/NO-GO: cero fallos,
cero inválidos y p95 lineal <=60 segundos por condición. No se combina con la
cohorte abortada ni cambia el umbral.

Esta corrección no es una optimización R1 y no habilita R2. Los remedios R1–R3
siguen requiriendo un NO-GO completo medido y sus propios requisitos previos.
Rollback: conservar la imagen anterior y revertir la corrección de observación;
la imagen anterior no es admisible para repetir una compuerta larga con ese
defecto conocido. No se modifica evidencia, corpus, artículo o documentos de Enzo.

## UX

No cambia ningún texto, pantalla, respuesta o instrumento del participante.
La información del guard permanece en la evidencia operativa externa.
