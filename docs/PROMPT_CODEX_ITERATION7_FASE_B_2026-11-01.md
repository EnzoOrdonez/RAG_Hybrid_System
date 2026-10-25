# I7 Fase B: construir, medir y validar

Fecha mínima: 01/11/2026. Rama `fix/interview-readiness`. Este documento no
autoriza gasto: Enzo debe abrir la tarea y aprobar USD 20 con USD 3 de cierre.

Leer el runbook I7 Fase B completo, el informe A2 privado, los preregistros I4 e
I5 y el borrador de declaraciones I6. Comprobar que el código fue publicado
con permiso y que la evidencia local liga esta misma fuente a Windows, Linux,
I7 y el congelamiento 19/19 y 3/3. La prueba Linux incluye los cuatro casos
POSIX omitidos en Windows, constructor y saneamiento offline.

No cambiar RAG, modelo, opciones, calendario, umbrales ni final03. Sin personas
ni registro ético. Solo us-central1 y una L4. No borrar recursos heredados ni
escribir documentos oficiales de Enzo. No publicar IP, hostname, correo,
tokens, consultas, respuestas o inventarios privados en el repositorio público.

La ronda A2 dejó implementados y probados localmente el navegador HTTPS
automático, publicación y hashes del anexo, comparación viva de contextos,
canarios del recorrido completo con dobles, ledger de todos los SKU y ciclo
restauración/purga con UNKNOWN. No son validaciones en nube.

Seguir los comandos del runbook sin crear herramientas nuevas para eludir una
falla. Descargar ambos catálogos de lista en directorios nuevos. UNPRICED,
fuente distinta, exposición con cierre > USD 20 o retención > 24 h bloquean
pagos. Conciliar factura I6 por separado. No confundir lista con factura ni
STOP con fin de cobro del almacenamiento.

Construir la imagen I7 CPU, archivar nueve registros originales privados con
hash, sanear el clon y crear snapshot propio central1. Retener una reserva
específica antes de restaurar GPU. Verificar READY bloqueante en cada boot,
TLS en tres boots consecutivos, doce contextos sin generación y crear el anexo
efectivo. Mostrar a Enzo el diff público del anexo y pedir permiso explícito
antes del commit/push a la rama. No force ni main. No simular publicación.
Anclar el tip remoto real, hashes de observadores, preregistros y calendario.
El check debe dar LOCAL_RUNBOOK_READY antes de medir.

Ejecutar smoke HTTPS con el mismo automatizador validado localmente, verificar
cada checkpoint y SHA del export, restauración desplegada por generaciones,
canarios en el SO real, retiro y purga. Programar la serie independiente de la
sesión del agente: doce boots, 144 llamadas completas y aceptación 12/12 en la
imagen nueva, con descarga y hash verificados antes de ACK. No heredar ACCEPTED
I6 ni mezclar entornos. Preservar terminales y no sustituirlos.

`gate` revalida los requisitos, repite smoke tras el estímulo, restaura/purga,
abre un boot técnico nuevo y ejecuta piloto caliente de 20 y dos ventanas de
60. Solo el agregado de 120 con 60 por condición, cero fallos e inválidos y
p95 lineal <= 60 s concede GO. Límites: llamada 600 s, preparación 900 s,
ventana 7200 s, trabajo 9000 s, invitado 165 minutos y STOP nativo 3 h. El
cierre técnico queda dentro de esos límites. Conservar prueba, telemetría,
hashes por fila y recibos de restauración y las cuatro comprobaciones de purga.

Ante UNKNOWN usar reconciliación por identidad, nunca recrear a ciegas. Un
bloqueo CPU parcial conserva recursos hasta revisión de sus recibos. Ante
falta de capacidad detener esa pieza, registrar intento y aplicar la pausa
preregistrada. No trasladar discos sin GPU retenida ni cambiar región.

Cerrar reserva con DELETE y LIST exitosos, VM detenida y temporales CPU propios
ausentes. Antes de 24 h Enzo decide retención o eliminación de los recursos
propios conservados. Conmutación y posibles cuotas requieren su decisión.
Entregar informe por eje con fuentes, costos de lista/factura separados y
pendientes. No declarar aptitud para personas sin todas las mediciones y ética.
