# Invitaciones: vigencia y cierre

Cada invitación se conserva solo como SHA-256, caduca a las 24 horas y se
revoca al completar la sesión o por una orden explícita del operador. Una
invitación heredada sin vencimiento se rechaza; no se le inventa una vigencia.
La inscripción remota recibe únicamente el hash generado en el equipo del
operador. Una sesión incompleta puede reanudarse dentro del plazo original.

Nota UX: la persona ve el mismo campo «Invitación» y los mismos instrumentos.
Al terminar ve exactamente «La sesión ha terminado. Gracias por participar.»
en el navegador que completó la sesión; el token se retira de su estado.
Otro navegador no puede reabrir una sesión completa con ese token. El rechazo
conserva el texto existente «No se pudo abrir la sesión. Contacta al coordinador.»
No cambian consultas, respuestas, opciones de generación, cegamiento ni relojes.

Si el respaldo de una sesión cerrada falta o está pendiente, no se permite
emitir ni admitir la siguiente invitación, incluso antes de crear el export.
La recuperación de un export cerrado corresponde al operador, no a reutilizar
el enlace del participante.
