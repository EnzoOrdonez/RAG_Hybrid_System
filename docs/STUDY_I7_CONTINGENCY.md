# Contingencia I7 y decisión de cuota

La cuota declarada es una GPU global. El ensayo predeterminado será secuencial,
con interrupción, entre dos zonas de `us-central1`. No demuestra servicio
simultáneo ni ausencia de interrupción. Su ejecución queda para Fase B.

## Traslado y vuelta

1. Comprobar respaldo sintético por generación y SHA-256, cierre de sesiones,
   inventario de copias de recuperación y certificado persistente. No crear
   una instantánea con sesiones ni invitaciones. Conservar todos los intentos.
2. Detener y verificar el origen. Antes de liberar la cuota, conservar su disco
   y la referencia a la imagen calificada. Borrar solo la VM propia de I7 si
   la asignación de una VM detenida sigue consumiendo cuota. No tocar recursos
   originales de Enzo. Registrar ID, intención, respuesta y ausencia por API.
3. Borrar la reserva propia y confirmar identidad y ausencia con la guardia.
   Un resultado desconocido bloquea el destino hasta conciliarlo por lectura.
4. Intentar una reserva específica nueva en otra zona de `us-central1`, con
   autoeliminación verificada. Máximo dos rondas separadas por 30 minutos;
   registrar orden, zona, hora y código de cada intento. No restaurar un disco
   de destino antes de retener la GPU. Sin capacidad, conservar el origen
   detenido y declarar NO_MEDIDO.
5. Restaurar un disco propio en la zona destino desde el entorno limpio y
   calificado. Conservar imagen, artefactos, caché JIT compatible y TLS.
   Reasignar la IP regional existente; no reservar ni borrar otra IP.
6. Exigir identidad de software, preparación, READY dentro de 900 s, misma IP
   y certificado, aislamiento y smoke sintético con respaldo verificable.
   No mezclar esta ejecución con las ventanas de la compuerta.
7. Repetir el ciclo para volver al origen. Cerrar cada reserva y comprobar
   todos los recursos propios. Un fallo deja el servicio cerrado y recibos
   conservados. No reconstruir ni afinar el RAG para rescatar el ensayo.

## Estimación y corte

Referencia provisional archivada en octubre: g2-standard-4 con una L4,
USD 0,706832255/h; disco pd-balanced, USD 0,00013698630137/GiB-h;
IP asociada USD 0,005/h, sin asociación USD 0,01/h. Fuente local:
`C:/CloudRAG/iteration6-cowork-20261015/AUDITORIA_CODEX/evidence/cost-recalculation.json`.
No es una tarifa de noviembre comprobada en vivo.

Supuesto del ensayo: dos horas de asignación total, contando espera y vuelta,
100 GiB de disco de destino durante dos horas y hasta dos horas de IP sin
asociación. Compute: USD 1,4137; disco adicional: USD 0,0274; IP: USD 0,0200.
Subtotal incremental USD 1,4611. Reservar USD 2 para el ensayo, dentro del
tope global de USD 20; no es un presupuesto adicional. La retención de las
instantáneas y discos existentes se suma al ledger general. Dentro de la
región no se presupone transferencia interregional de instantáneas.

Antes de pagar: comprobar precios vigentes por SKU, retención y presupuesto
restante. Si no caben el siguiente paso y USD 3 de cierre, no se inicia.
Una reserva vacía cuesta compute; no se duplica el cargo de su VM consumidora.

## Borrador para Enzo, no enviado

Proyecto: pure-loop-474323-a8. Región: us-central1. Solicitar pasar
GPUS_ALL_REGIONS de 1 a 2 y cuota regional NVIDIA L4 de al menos 2.
Comprobar los nombres efectivos y cuotas de CPU G2 en `regions describe`
antes de presentar la solicitud. El ensayo necesitaría hasta 8 vCPU G2.

Motivo: ensayo temporal de conmutación de un estudio académico, con dos
g2-standard-4, una L4 por VM, reservas específicas y cierre independiente.
Sin participantes durante el ensayo. Dos horas con dos GPU: USD 2,8273
de compute más discos e IP; reservar USD 3,50 dentro del mismo techo global.
El aumento no autoriza dos GPU ni amplía el presupuesto. Enzo debe decidir
si quiere este ensayo y si su costo cabe junto al estímulo y la compuerta.
