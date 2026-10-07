# Capacidad regional después de las rondas centrales

El controlador `scripts.study_operator.regional_capacity` exige tres rondas
centrales completas separadas por al menos 45 minutos, con las tres zonas
agotadas en la última ronda y apagado verificado en todas. Una observación
anterior de capacidad que se detuvo no reserva esa GPU ni demuestra capacidad
actual; se conserva como tal y no se reetiqueta como agotamiento. Una ronda
incompleta, un error de herramienta o una GPU sin STOP verificado bloquean el
avance. Los IDs ordinales y el censo de zonas deben coincidir. Conserva todos
los intentos. El catálogo de aceleradores debe coincidir con el hash del
archivo de latencia y todas las regiones deben tener cinco peticiones válidas.
Las regiones se visitan por mediana de latencia y las zonas por el orden fijo del
catálogo. El catálogo describe el tipo L4, no garantiza stock.

Solo crea recursos en las zonas estadounidenses del catálogo verificado.
Cada región usa una subred propia dentro de la red del estudio, sin solapamiento
de rangos, con Private Google Access y sin flow logs. Las reglas públicas siguen
siendo las de esa red: 443. Durante esta prueba no hay IP externa, cuenta de
servicio, app ni tráfico de sesiones. Los discos parten de la instantánea que ya
pasó la restauración CPU. Las VM son g2-standard-4 estándar, con disco retenido,
protección, STOP nativo de tres horas y apagado invitado de 110 minutos.

Antes de cada disco/VM se registra la intención, la tarifa oficial de cómputo y
del disco para esa región, la retención hasta el cierre y una reserva de USD 2
para restaurar hasta 100 GiB desde us-central1. La tarifa de transferencia entre
ubicaciones norteamericanas es USD 0,02/GiB en la
[tabla oficial de instantáneas](https://cloud.google.com/compute/disks-image-pricing#network-charges),
consultada el 7 de octubre de 2026. La tarifa mensual del disco conserva su unidad;
la conversión de 730 horas por mes se etiqueta como estimación. Las reservas no
son gasto facturado. Tras un resultado terminal y STOP comprobado se libera solo
la reserva de cómputo; se mantiene la del disco y la transferencia.

El primer arranque observado con L4 se detiene y se registra como capacidad,
sin afirmar driver, READY, HTTPS, imagen final o GO. Antes de medir todavía hacen
falta IP y certificado regionales, imagen final, identidad viva y anexo publicado.
El bucket de sesiones permanece en us-central1; su transferencia entre regiones
se cotizará para el despliegue y el periodo de sesiones. Si se agota todo el orden,
queda espera documentada; un intento terminal nunca se repite automáticamente.
