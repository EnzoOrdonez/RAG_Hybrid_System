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

## Ventanas y trazabilidad de la cohorte nueva

Una inicialización crea las 120 posiciones inmutables, el hash de esta enmienda,
commit, paquetes, fuentes, servidor, driver y manifiesto del bundle. Cada ventana
ejecuta únicamente `--system SISTEMA --phase cold|warm` sobre esa misma cohorte.
No crear una cohorte por ventana ni mezclar resultados de commits diferentes.
El supervisor de 120 minutos conserva un margen antes de comenzar nuevas consultas
(600 s) o preparación completa (900 s). Estos márgenes reducen el riesgo de corte;
no son límites demostrados de ejecución. Un corte conserva el aborto sin duración
final; reanudar posiciones pendientes exige una ventana nueva y preparación nueva.

Crítica operativa: agrupar los 120 intentos en una sola ventana podría exceder el
supervisor y cortar acceso remoto innecesariamente. Las ventanas por condición
conservan identidad y registran interrupciones; el reporte separa condición completa
de cohorte completa. `-KeepAnyDesk` conserva el servicio y procesos de acceso remoto;
solo omitirlo si la admisión/telemetría demuestra necesidad y se aplica el aviso
visible y restauración autorizados. El manifiesto del bundle se verifica en cada
ventana; no se vuelve a crear ni se sobrescribe durante la cohorte.

En condición caliente, cada inicio de proceso registra una preparación que contiene
tres respuestas de calentamiento y tres pruebas NLI explícitas. Se conservan fuera
de las 120 posiciones y fuera de p50/p95. Cada consulta medida guarda la identidad
de preparación y la comprobación de residencia inmediatamente anterior a generar.
No se afirma residencia física permanente de páginas de RAM ni ausencia de toda
actividad del sistema operativo; las condiciones válidas son las del observador.
