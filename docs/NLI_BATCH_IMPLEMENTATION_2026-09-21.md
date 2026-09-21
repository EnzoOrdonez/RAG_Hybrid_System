# Implementación NLI compartida — 2026-09-21

## Crítica previa a la edición

El usuario aprobó explícitamente el pre-registro de 20-sep en la sexta tanda.
El baseline es `976a379`; NO-GO vigente. Sólo se implementa agrupación de pares.
El control secuencial permanece como opción predeterminada de la app: cambiar
el despliegue antes de validar equivalencia y latencia confundiría implementación
con aceptación. El selector del candidato pertenece al ejecutor técnico.

Riesgos: padding/longitudes pueden anular la ganancia o alterar scores; se exigen
las mismas etiquetas, evidencia y puntuaciones publicadas sobre entradas iguales.
No se afirma equivalencia real a partir de mocks. Un lote fallido recuperado por
el camino original debe seguir registrado como fallo, no éxito de compuerta.
Un resultado mal formado no debe desalinear claims y evidencia silenciosamente.
No se añade otro timeout NLI ni se cambian prompts, modelo, dispositivo o lote
interno 32: se prueban excepciones de timeout y se conserva el supervisor temporal.

El calendario de 120 incorpora brazo en la identidad de posición. Reutilizar
sin cambios la clave antigua sistema/fase/índice duplicaría posiciones aparentes;
por ello el protocolo nuevo tiene calendario y agregación explícitos. Los
paquetes históricos conservan sus reglas. Una interrupción dentro de un par no
autoriza completar su otro brazo en otra ventana ni reemplazar lo consumido.

Se reutiliza el supervisor NVIDIA probado. El lanzamiento humano del experimento
exige autorización explícita de esa ventana, también al reanudar. Esta sesión no
interviene procesos ni ejecuta modelos. No se reusa por silencio una autorización
de ventana histórica. Cada ventana real prueba su supervisor antes del corte.

Aceptación de esta entrega: suite y regresiones sin modelos, dry-run rápido de
120 posiciones con alternancia y recuperación, hashes íntegros, documentación
del comando humano. Aceptación científica posterior: la del pre-registro;
no la sustituye un dry-run. Evidencia de desarrollo externa:
`C:/CloudRAG/nli-build-20260921T004410616Z/`.
