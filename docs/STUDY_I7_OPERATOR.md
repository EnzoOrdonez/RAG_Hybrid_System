# Operador I7

Estado: interfaces y dobles locales. No hay validación desplegada.

Entrada: `python -m scripts.study_operator.i7 --root <paquete> <operación>`.
`init --config <installation.json>` crea una raíz independiente y no llama a
Google Cloud. Las operaciones en vivo exigen `--live`, SDK explícito y fecha
desde el 1 de noviembre. No se aceptan propósitos de participantes en I7.

La instalación conserva el contrato de despliegue revisado e incorpora
`iteration=7`, `budget_usd=20`, `closure_reserve_usd=3`, `resource_marker`,
`native_stop_s`, `rates`, `local_reviewed_config`, `cuda_cache`,
`cuda_cache_seed` y `cuda_cache_driver`. `rates` contiene precio horario USD,
fecha de comprobación y SHA-256 del recibo de precios. La configuración final
se fija con la imagen nueva y sus recibos CPU; no se copia la identidad final03.

`reserve --hours <h>` exige una reserva específica y autoeliminación nativa
verificada, expuesta por `gcloud beta compute reservations`. Si el SDK o la API
no aceptan ese límite, se bloquea sin alternativa de reserva ilimitada.
Un resultado desconocido conserva la intención y prohíbe crear otra reserva.
El techo de 12 horas por reserva no amplía los 120 minutos de cada ensayo.
La reserva sigue facturando mientras la VM está detenida.

`start` comprueba VM detenida, identidad y consumo específico de la reserva,
plazo suficiente, presupuesto y precio vigente. La preparación se ejecuta en
el host antes de READY. `preflight` conserva las comprobaciones existentes de
IAM, bucket, metadatos, aislamiento, exposición 443 y TLS externo; comprueba
el montaje JIT efectivo y la persistencia NVIDIA. Los seis insumos revisados
deben estar instalados y sellados en el disco CPU: no se reenvían por metadatos.

`schedule` registra una tarea limitada de Windows que ejecuta `series` fuera
de la sesión del agente. La tarea se acota a 24 horas y cada ensayo conserva
su STOP nativo. Se archivan versiones de scripts por commit y SHA-256. Las
intenciones y terminaciones son archivos separados. Un parcial no se sustituye.
La descarga verifica la prueba completa y el archivo local antes del ACK.
La serie siempre intenta STOP y luego la guardia, incluso al fallar.

`guard` exige VM propia detenida, identidad exacta de reserva, éxito del DELETE
y listado exitoso sin el ID ni el nombre en su zona. Un error conserva el estado
abierto y exige conciliación por lectura; no imprime «borrada». La eliminación
nativa de la reserva limita la exposición si la tarea local desaparece.

Cada boot conserva certificado e identidad. La comprobación TLS exige los tres
últimos intentos, no tres recibos READY escogidos. Las versiones históricas de
I6 ausentes siguen sin verificarse: archivar la versión actual no las recupera.

El cálculo de asignación fusiona intervalos de reserva y ejecución de su VM:
incluye la espera y no duplica la VM que consume la reserva. El ledger separa
retención y otros SKU. La conciliación final requiere filas de facturación;
ninguna estimación local se presenta como factura conciliada.
