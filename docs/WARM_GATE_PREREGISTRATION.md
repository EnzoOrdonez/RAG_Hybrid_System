# Enmienda prospectiva de operación caliente — 2026-09-09

Decisiones del usuario confirmadas antes de la nueva medición. Baseline `f139cf9`:
476 tests pasan, 5 excluidos, 45,98 s. Evidencia inicial:
`C:/CloudRAG/warm-protocol-20260909T152518Z/baseline-*`.

## Crítica y criterio fijado

El resultado híbrido anterior era conocido: esto es una enmienda prospectiva del
protocolo de despliegue, no un pre-registro retroactivo. El calentamiento solo
justifica evaluar operación caliente si se verifica su persistencia y se conserva
el tiempo de preparación/interrupciones. No equivale a un caché de respuestas.
La RAM no está descartada causalmente; no hubo agotamiento sostenido observado.

Nueva cohorte homogénea: 120 posiciones, 20 frías y 20 calientes por cada sistema,
sin importar ni reemplazar posiciones de las cohortes anteriores. GO exige en
cada condición caliente 20 respuestas válidas, cero fallos/abortos/condiciones
inválidas y p95 <=60 s, además de P900/SUS/exportación completas, resiliencia y
documentación cerradas. Fallos no entran en percentiles; calentamientos van aparte.
El frío se mide y reporta, pero su latencia no bloquea bajo preparación obligatoria.
No cambiar el criterio después de ver resultados. Fase B solo después de GO.

## Receta aprobada y cambios críticos

Lectura HTTP 180 s exclusivamente en participante; conexión 5, escritura/pool 60.
No es un deadline RAG. Modelos, digest, contexto 4096, seed 42, temperatura 0,
1024 tokens y caché desactivada permanecen. Registrar límites efectivos por intento.
El reloj debe vivir en el navegador: un fragmento dependiente del servidor puede
dejar de refrescar mientras la llamada síncrona está bloqueada.

Preparar los tres pipelines en el proceso que atenderá al usuario, después de
autenticar su invitación y antes de consultas/prácticas. Preparación y NLI reales,
no solamente `warm_model()` ni una respuesta que eluda NLI por declinación.
Granite mantiene residencia 30 minutos renovados por consulta. Reinicio, cambio
de identidad o pérdida de residencia invalidan preparación; pausar, registrar la
interrupción y preparar nuevamente. Nunca convertir esa pausa en latencia cero.
La cohorte caliente reproduce los tres pipelines residentes; el frío es diagnóstico
de proceso nuevo sin preparación. Usar ventanas por condición con restauración
probada; no cortar AnyDesk sin necesidad observada y aviso previo.

## Nota UX aprobada

Mismo reloj y textos para los tres sistemas y prácticas:

- Inicial: «Buscando y verificando la respuesta. Tiempo transcurrido: MM:SS.»
- Desde 60 s: «La consulta está tardando más de lo previsto. Sigue en curso; no necesitas enviarla otra vez.»
- Desde 120 s: «La consulta continúa. Puedes esperar o avisar al coordinador. Recargar la página no cancela la consulta.»
- Fallo: «No se pudo completar la consulta. Puedes reintentar o contactar al coordinador.»
- Preparación perdida: «El sistema necesita preparación antes de continuar. Tu progreso está guardado; avisa al coordinador.»

No se muestran nombres técnicos, progreso porcentual inventado ni promesas de
terminación a los 180 s. El reloj recuperado parte del intento persistido.
Estos elementos forman parte de la interfaz evaluada y pueden influir en SUS;
no se afirma ausencia de efecto. Mantenerlos iguales entre participantes y sistemas,
registrar la versión UI y no comparar puntuaciones históricas como si la UI fuera idéntica.

## Pruebas antes de medir

Límites HTTP y otros modos intactos; reloj de navegador a 60/120 s durante bloqueo;
recarga sin duplicación; preparación/residencia y su invalidación; NLI ejercitado;
calentamientos fuera de ratings/percentiles; protocolo durable sin reescritura de
históricos; exportación real reconstruible. Cada cambio de código lleva regresiones,
suite completa, Ruff, diff check y secretos antes de su commit. Esta nota no afirma
que las implementaciones o mediciones nuevas ya estén completadas.
