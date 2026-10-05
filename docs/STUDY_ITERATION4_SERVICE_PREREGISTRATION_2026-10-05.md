# Iteración 4: runner nuevo antes de cada generación

## Diagnóstico prospectivo y alcance causal

El diagnóstico registrado el 4 de octubre terminó con 48 objetivos nuevos.
Su paquete externo contiene `i4-diag02/diagnosis.json`, el inventario por
generación y `diagnostic-prospective-analysis.json`. La generación del manifiesto
de GCS y su hora de servidor se leen del recibo `collect-diag02-receipt.json`;
este documento no sustituye esos recibos con hashes escritos manualmente.

Los cuerpos de generación, sus opciones y los contextos son idénticos entre
brazos para los cuatro objetivos. Con runner retenido se observaron 3 textos
en q016/no_rag y 2 en cada uno de q068, q070 y q172/hybrid. En q016 variaron
las clases v2 y en q172 las citas. Con descarga comprobada del runner antes de
cada llamada, los cuatro objetivos tuvieron un texto crudo y presentado, una
clase y un conjunto de citas; seis PID distintos por objetivo.

Esto apoya causalmente la intervención conjunta sobre el estado del runner.
No identifica KV como causa específica: el contador de caché no está disponible
en la versión instalada. Tampoco demuestra aún los doce estímulos, las celdas
del estudio, consultas libres, independencia entre arranques o la imagen final.

## Hipótesis, mecanismo y efecto esperado

H: el estado conservado entre generaciones produce las variantes; eliminarlo
antes de toda generación evitará que su texto dependa del antecedente.

Un gateway de servicio serializa las llamadas. Antes de cualquier generación
con contenido, envía una solicitud administrativa vacía `keep_alive=0`, espera
que la API real `ps` no tenga modelos y que el runner anterior haya salido,
y transmite los bytes originales de la solicitud. Comprueba que el runner nuevo
tenga otro PID. Se aplica igual a toda consulta y condición, sin mirar ID de tarea,
sin cambiar prompts/opciones, sin respuestas almacenadas y sin cambios de NLI,
recuperación, índices o pesos. Las solicitudes administrativas y las lecturas de
estado no son respuestas a consultas.

Predicción: doce combinaciones con un único texto, clase y conjunto de citas en
los 120 objetivos nuevos previamente definidos. Se espera añadir aproximadamente
2–8 segundos por consulta frente al servicio retenido, tomando como referencia
las diferencias prospectivas de carga observadas; es una proyección, no un
resultado. La discrepancia se informará en ambas direcciones. La aceptación
final y la compuerta decidirán sobre datos nuevos, nunca sobre el diagnóstico.

Rollback: desplegar la imagen y el servicio R1 anteriores, deshabilitando el
gateway nuevo. Eso pierde la corrección del estímulo y no habilita participantes.
No se combinarán cohortes de servicios o identidades diferentes.

## Fronteras, estado y regresiones

Todo tiempo de cola, descarga, verificación de salida, carga y generación entra
en el cronómetro del servicio de respuesta de la app y del runner. Se conservan
600 s por llamada, 900 s de preparación y 7200 s por ventana; el supervisor
independiente conserva la política terminal original.

La transición intencional de servicio se registra fuera del contenedor de la app
con ID aleatorio de solicitud, fase, reloj monotónico, boot ID y plazo fijo.
RESETTING dura como máximo 10 s; LOADING, como máximo 30 s. Las observaciones
reales de `ps` no se alteran. Solo durante esas fases verificadas se reconoce
como previsto el estado vacío: no hay excepción para otro digest, marcador
ausente/alterado/vencido, otro arranque, nueva fase desconocida o una admisión
en reposo. Al cargar, vuelven a exigirse digest, contexto y lease originales.
No se elimina ninguna muestra ni se excluye su tiempo. El contenedor solo puede
leer el marcador; el gateway lo escribe y valida los PID reales.

Esta enmienda documenta la nueva transición necesaria de la capa de servicio;
no cambia los umbrales de rendimiento, fallos, invalidez, carga ajena, preparación
o decisión agregada. Se prueban transición legítima, marcador vencido, digest
distinto, ausencia, admisión y tiempos de espera incluidos antes de medir.

## Imagen final y aceptación

El anexo de entorno y la identidad efectiva de la imagen final se publicarán
antes de los ensayos. Se conserva el calendario prospectivo ya anclado:
36 objetivos del calendario de compuerta, 48 de dos recorridos de las cuatro
celdas latinas, 12 primeras consultas de 12 arranques y 24 posteriores a
antecedentes libres sintéticos. Diez repeticiones por combinación, con presencia
en al menos dos arranques. No se hace generación de calentamiento antes de las
primeras consultas; la preparación técnica sin contenido se distingue de ellas.

Se comparan bytes de texto crudo y presentado, clase v2 y citas. Los textos
pueden diferir de exp12; no se afirma equivalencia de salidas con el artículo.
Si alguna combinación tiene más de un resultado, estímulo queda
BLOQUEADO-HUMANO y no se declara una variación permitida.

Sobre la imagen final: smoke completo, piloto caliente nuevo de 20 posiciones
fuera del agregado, y dos ventanas nuevas de 60 con identidad idéntica. GO solo
con el agregado: p95 lineal ≤60 s por condición, cero fallos e inválidos.
Un NO-GO no habilita remedios automáticos. Nunca participan personas en los
ensayos de esta iteración.
