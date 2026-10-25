# UEQ-S y análisis ejecutable de la iteración 5

Este anexo implementa las decisiones de Enzo del 6 de octubre de 2026. No
modifica el SUS, los Likert propios, las tareas ni la lógica del RAG. Los datos
de prueba son sintéticos; ninguna prueba concede aptitud para participantes.

## Fuente y captura

Los ocho pares, su orden y su polaridad se transcriben literalmente de la página
en español de [UEQS_Items.pdf](https://www.ueq-online.org/Material/UEQS_Items.pdf).
La puntuación sigue el [manual oficial](https://www.ueq-online.org/Material/Handbook.pdf).
Se conservan `facil` y los dos inicios `convencional` tal como aparecen allí.
La procedencia y los SHA-256 calculados de ambos PDF están en
`config/UEQS_ES_official.json`; `scripts.study_operator.ueq_source` verifica las
descargas contra el recibo externo antes de generar ese archivo. Los documentos
no introducen hashes tecleados como prueba.

Cada bloque presenta diez ítems SUS, ocho UEQ-S y los diez Likert, en ese orden,
antes del siguiente sistema. El UEQ-S usa siete posiciones sin valor inicial.
La posición 1 corresponde al extremo negativo izquierdo y 7 al positivo
derecho. Cada puntuación es posición menos 4; pragmática = promedio 1–4,
hedónica = promedio 5–8 y global = promedio 1–8. El formulario exige los ocho
ítems completos; no imputa valores.

**Nota UX:** inmediatamente después del SUS aparece «UEQ-S» y esta instrucción
exacta de la app: «Para cada par de palabras, marca una de las siete posiciones
según tu experiencia con este sistema. No hay respuestas correctas o incorrectas.
Esta parte toma aproximadamente un minuto.» Es una instrucción de la app, no
una instrucción estandarizada atribuida al manual. El minuto es una estimación
que deberá comprobarse en el piloto; no es una duración medida. Ambos sistemas
reciben los mismos pares e instrucción sin revelar su condición. La leyenda
exacta del UEQ-S es «1 = palabra de la izquierda · 7 = palabra de la derecha».
Al terminar sus ocho ítems se repite, antes de los Likert, su leyenda existente:
«1 = Totalmente en desacuerdo · 5 = Totalmente de acuerdo». Así la escala de
siete posiciones del UEQ-S no queda como última instrucción para los Likert.
No cambian sus textos, opciones, orden ni puntuación; ambas condiciones muestran
la misma separación. La regresión verifica las cuatro leyendas en su orden y
las opciones de cinco posiciones de los diez Likert en ambos bloques.

El protocolo nuevo es versión 2 y la sesión/exportación versión 4. Sus ocho
posiciones y las tres puntuaciones se guardan en `instruments[].ueq_s` y
`instruments[].ueq_s_scores`, dentro del registro asociado al código. La
exportación por código incluye ambos campos; el inventario de retiro y purga
cubre los archivos completos que los contienen. Los registros antiguos versión
3 se analizan únicamente como archivo histórico identificado, nunca se mezclan
con versión 4 ni se les añade un UEQ-S no observado. La entrada del despliegue
rechaza protocolos anteriores al nuevo instrumento.

La actualización de un sorteo revisado debe conservar exactamente la asignación,
las etiquetas, los seis IDs de tareas y su evidencia. Añadir el instrumento exige
un nuevo sello; no se vuelve a sortear ni se reutiliza un directorio de sesiones
antiguo. La igualdad estática del RAG y los doce contextos vivos se verifican por
separado antes de las mediciones finales.

## Regla estadística fijada antes de sesiones

`src.evaluation.study_statistics` y `src.evaluation.study_analysis` implementan
el contraste híbrido menos sin RAG, con el participante emparejado como unidad.

- SUS de Brooke: ítems impares menos 1, 5 menos los pares; suma por 2,5,
  intervalo 0–100. Se recalcula desde las respuestas y se rechaza un score alterado.
- Shapiro-Wilk sobre las diferencias, alfa 0,05; si p ≥ 0,05, t pareada bilateral
  primaria; si p < 0,05, Wilcoxon bilateral primaria. La otra es sensibilidad.
  La misma selección rige las dos escalas UEQ-S, según la decisión D03 de Enzo.
- d_z = media de diferencias dividida entre su desviación estándar muestral.
  Intervalos percentiles 95 % por bootstrap de pares, 10 000 remuestreos,
  semilla 42, para diferencia media y d_z.
- Benjamini-Hochberg exclusivamente sobre los dos p primarios de las escalas
  pragmática y hedónica. SUS, sensibilidades, global UEQ-S, Likert propios y
  perfiles no ingresan en esa familia. Los últimos tres se reportan de forma
  descriptiva; no se añade inferencia para F/U ni comparación de perfiles.

Se excluyen pilotos, sesiones no completas y pares incompletos con motivo;
duplicados, mezclas de sellos, escalas fuera de rango y scores inconsistentes
fallan. Los errores técnicos de reintento de una sesión completa se conservan.

Con menos de tres pares no se inventa Shapiro ni decisión primaria. Si las
diferencias son constantes, se informa normalidad y d_z no definidos y no se
produce p primario; el Wilcoxon calculable queda como diagnóstico. Los bootstrap
con varianza cero se cuentan y no generan infinito. Si alguno no permite d_z,
su intervalo queda no definido: eliminar esos remuestreos produciría un
intervalo condicionado, no un intervalo del 95 % sobre todos los remuestreos.
El intervalo de la diferencia media sigue disponible. Si un p UEQ-S no está definido, no se reduce la familia BH a
una sola escala: ambos ajustes quedan pendientes. Estas decisiones conservadoras
se declaran antes de observar datos de participantes.

## Verificación

Las regresiones prueban extremos, polaridad, captura completa, orden del
formulario, reconexión, exportación, retiro sintético y respaldo automático al
cierre. Prueban diferencias normales y no normales, selección de prueba,
sensibilidad, semilla fija, familia BH de dos, datos degenerados y rechazo de
mezclas y alteraciones. No sustituyen el smoke HTTPS, el piloto ni la compuerta
de la imagen final, ni la auditoría independiente y aprobación ética.
