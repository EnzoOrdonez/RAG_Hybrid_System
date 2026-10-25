# Retiro de recursos de prueba de la iteración 5

La instantánea final solo se conserva como conjunto mínimo después de una
restauración CPU real. El recibo debe acreditar la imagen final, las 19 fuentes
congeladas, los 79 artefactos, los blobs de Ollama y la prueba pareada del usuario
de runtime. Un test sintético o un build aprobado no sustituyen esa restauración.

`scripts.study_operator.retention_disposable` genera primero un inventario API
completo y un plan inmutable. Solo admite los recursos redundantes heredados
autorizados y los recursos propios declarados desechables, con ID, zona y marcador
de propiedad coincidentes. Rechaza recursos ajenos, VM activas y discos conectados
fuera del conjunto de retiro. Conserva la VM y el disco originales, la instantánea
final restaurada, los buckets y todos los archivos de evidencia.

El modo predeterminado simula sin borrar. `--execute` comprueba nuevamente la
identidad y los recursos protegidos antes de cada efecto, registra la intención
en `DESTRUCTION_LOG.md`, conserva los discos al borrar una VM y verifica ausencia
mediante la API. Un recibo perdido se reconcilia contra la intención y el ID; no
se interpreta una ausencia no registrada como éxito. Una ejecución posterior usa
el plan original, aunque la CPU de la prueba ya haya sido retirada.

La limpieza no concede aceptación GPU, privacidad, continuidad ni GO. Los costes
en reposo se recalculan con un inventario posterior; no se deducen del plan.
