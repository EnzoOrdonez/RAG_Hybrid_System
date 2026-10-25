# Iteración 4: diagnóstico prospectivo del estado del generador

## Ámbito y ancla

Prompt maestro íntegro: `output/audit/ITERATION4_MASTER_PROMPT_2026-10-04.md`.
Todas las mediciones corren en la VM original de us-central1-a, g2-standard-4,
L4, sobre la imagen R1 heredada identificada por su inventario externo completo.
No participan personas. No se cambia lógica, modelo, opciones, pesos o índices.
El inventario heredado y el recibo de lanzamiento fijan hashes completos;
no se introducen hashes manuales como evidencia de identidad.

## Hipótesis y diseño

H1: solicitudes distintas por historial. Predicción: cambian hashes del cuerpo
efectivo o contextos. Se observa la llamada real al cliente, sin alterar mensajes.
H2: estado residual del runner (caché/RNG/planificación). Predicción: con cuerpos
idénticos, historiales distintos cambian texto y el runner recién cargado elimina
esas variantes. No se distingue KV de otro estado sin observación adicional.
H3: concurrencia o agrupación del servidor. Predicción: configuración efectiva,
PID y actividad revelan solicitudes superpuestas o slots distintos.
H4: variación numérica ajena al historial. Predicción: persisten variantes tras
aislamiento, en repeticiones de cuerpos idénticos.

Antes del diagnóstico se ejecutan las seis tareas en ambas condiciones para
comparar contextos y opciones con la compuerta R1 y completar el freeze baseline.
Se conserva fuente heredada; el helper y las observaciones viven fuera del repo
de la imagen. Opciones efectivas: num_predict1024, temperatura0, seed42, num_ctx4096.

Se evalúan q016/no_rag, q172/hybrid, q068/hybrid y q070/hybrid. Cada objetivo tiene
dos antecedentes fijados: respectivamente q001/no_rag vs q016/hybrid;
q070/hybrid vs q172/no_rag; q001/hybrid vs q068/no_rag;
q016/hybrid vs q070/no_rag. Tres repeticiones por antecedente y brazo: 48 objetivos,
con antecedentes generados realmente. Orden de brazos retenido/aislado en
repeticiones0 y2, invertido en1. No se eligen tareas por resultados nuevos.

Brazo aislado: petición administrativa vacía keep_alive0 antes de toda generación,
espera comprobada hasta API ps sin modelo residente, y llamada original intacta.
Se capturan PID del runner, cuerpo completo hashado, opciones, contextos, métricas
disponibles de la versión instalada, texto crudo, presentación y citas sintéticas.
Campo no disponible queda null; las duraciones no prueban por sí solas uso de KV.

Se espera pasar de variantes según antecedentes a un texto por objetivo y cuerpo,
con un costo positivo de recarga que se medirá. El diagnóstico no concede GO ni
sustituye los 120 objetivos de aceptación de la imagen final. Si no se reproduce
el defecto o no se demuestra una intervención causal defendible, no se optimiza
a ciegas: se conserva el diagnóstico y se bloquea esa rama para revisión humana.
Rollback: servicio heredado sin reset; los datos de brazos nunca se mezclan.

## Fronteras y supervisión

Cronómetro objetivo: antes del reset (si corresponde) y construcción de pipeline,
hasta respuesta final con NLI y presentación; incluye toda espera de reset.
Antecedentes y preparación se registran aparte y no integran ese cronómetro.
Ninguna exclusión de casos lentos o inválidos. Un fallo aborta y conserva la cohorte;
no se sustituyen posiciones. Petición máximo600s, job máximo7200s, STOP nativo3h,
finalización y subida verificadas antes de apagado. Sin UI, TLS ni admisión pública.

## Criterios posteriores fijados

Freeze final idéntico en proyección RAG, servicio declarado aparte. Aceptación final:
120 objetivos, diez por cada una de doce combinaciones; 36 del calendario vigente,
48 de dos recorridos de las cuatro celdas, doce primeras consultas de doce arranques
y24 tras antecedentes libres sintéticos. Cada combinación aparece en al menos dos
arranques. Un texto crudo y presentado, una clase v2 y un conjunto de citas por combinación.
Se declarará que los textos pueden diferir de exp12.

La compuerta final tiene piloto nuevo20 fuera del agregado y dos ventanas60,
identidad única, cero fallos/inválidos, p95 lineal agregado por condición <=60s;
600s por llamada,900s preparación,7200s por ventana. Enmienda e inventario finales
se publican antes de medir. Cualquier NO-GO no habilita remedios automáticos.

## Privacidad y permisos

Enzo precisó durante planificación: cero identificadores de clientes en salidas
nuevas; se inventarían por separado evidencia heredada y direcciones de infraestructura.
Datos de diagnóstico son respuestas a tareas públicas sintéticas, no sesiones study.
No se registran tokens, cookies, claves, encabezados ni consultas de participantes.
No hay administrador Windows, medición local ni cierres de aplicaciones.
