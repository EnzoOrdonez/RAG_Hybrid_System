# Arranque de la imagen final desde la instantánea

El operador nuevo admite `bootstrap --zone <zona revisada>`. No sustituye el disco
original: crea otro disco desde la instantánea final cuya restauración CPU tiene
recibo y SHA-256 verificados. Comprueba imagen, los 19 archivos congelados, los
artefactos y la prueba pareada de `USER` antes de crear recursos. La zona debe
coincidir con la instalación, la subred privada y la IP regional propias.

La reserva, los intentos de disco y VM, los IDs observados y el arranque con STOP
nativo quedan persistidos antes de cada efecto siguiente. Una respuesta de
creación perdida se reconcilia con la misma VM; no se crea otra ni se detiene la
GPU adquirida antes de READY. El costo de retención del disco sigue reservado
después de STOP. La VM y el disco originales permanecen conservados.

Los seis archivos públicos de configuración revisada, manifiesto de artefactos
y preregistro viajan en el arranque con un inventario SHA-256 y destino inmutable.
Así se instala el protocolo 2 y el UEQ-S sin dar a la SA acceso de lectura al bucket
de evidencia técnica. No se incluyen sesiones, invitaciones ni credenciales.
El invitado valida todos los bytes antes de escribir, rechaza rutas simbólicas y
contenido previo distinto. El código de host y la imagen siguen siendo los del
build final verificado; el mecanismo de servicio y el RAG no cambian.

`bootstrap` no concede aceptación, ni GO ni autorización ética. Antes de cualquier
medición siguen siendo obligatorios el anexo y preregistro anclados, identidad,
preflight, congelamiento vivo y las pruebas del protocolo de iteración 5. La
instalación real, la disponibilidad de L4 y el runbook completo están pendientes.
