# Datos del estudio — esquema 3

No migra el esquema 2 ni modifica P900 histórico. El directorio del estudio se
separa de sesiones históricas y de pilotos. `purpose` distingue `study`, `pilot`
y `technical`; pilotos usan P900+ y nunca entran al análisis del estudio.

| Campo | Significado / unidad |
|---|---|
| schema_version | 3, protocolo de dos bloques |
| session_id | Identificador aleatorio; no token de acceso |
| assignment | Código P01–P24, celda 1–4, perfil, rol y plaza primaria reemplazada |
| protocol_fingerprint, protocol_hashes | SHA-256 de configuración, CSV y consultas; congelados |
| build_id | Commit del despliegue; no identifica a la persona |
| stage, block_index, task_index | Máquina de estados; índices base cero |
| events | Avance de práctica o abandono, con timestamp epoch segundos |
| incidents | `session_technical_block` y timestamp; sin detalles de práctica |
| attempts | Consultas puntuadas y libres, incluyendo errores y reintentos |
| analysis_role | `tasks` o `free_query`; nunca familiarización |
| condition, label, task_set | Condición interna, etiqueta ciega A/B y conjunto T1/T2 |
| question, answer, sources | Texto enviado, respuesta original y fuentes realmente presentadas |
| started_at, finished_at, shown_at | Epoch segundos; `shown_at` significa render solicitado en servidor, no prueba de pintura en navegador |
| elapsed_ms | Reloj monotónico de consulta; null si un aborto impide medir el final |
| status, error | Estado del intento y código técnico neutro; sin stacktrace ni credenciales |
| instruments | Dos registros: 10 SUS crudos, SUS 0–100, F/U/R/I crudos y timestamp |
| comparative | C1–C3 y texto abierto C4, una vez al cerrar ambos bloques |
| blinding | Elección A/B/No sabría decir, motivo y timestamp |

## Aclaración del prompt, no cambio del artículo

Artículo V2.11 §4.8: la familiarización **no se registra**. La precisión del
usuario del 27-sep corrige el Paso 2c del prompt: sólo se guarda un evento
`familiarization_done` con bloque y timestamp. No se guarda consulta, respuesta,
fuentes, latencia ni errores de reintentos, tampoco en exportación. Una falla
que impide continuar permite registrar únicamente un incidente de sesión sin
contenido. Los diagnósticos de librerías se suprimen durante esa llamada efímera
en el despliegue de una sola inferencia; el nivel de logging se restaura al salir.

La consulta es fija, igual en ambos bloques y participantes, fuera de T1/T2 y de
las premisas inválidas. La propuesta inicial es `What is cloud computing?`.
La repetición puede enseñar esa respuesta genérica, pero no repite ninguna tarea
puntuada. El pre-flight calienta antes de cada participante, independientemente
de la práctica. La configuración del operador contiene el texto fijo; **no se
copia ese contenido al almacenamiento de sesiones**: se guarda sólo su hash.

La consulta libre se registra completa con `analysis_role=free_query`, previa
advertencia literal de confidencialidad. Se usa cualitativamente y no determina
el desenlace primario SUS. No hay ratings por consulta: SUS/Likert son por bloque.

## Frontera del reloj

Inicio: tras adquirir la exclusión de inferencia, antes de persistir la solicitud.
Fin: después de obtener la respuesta y proyectar fuentes para presentación, antes
del flush final del resultado. Incluye comprobación de preparación/residencia,
retrieval, generación y NLI cuando aplican. Excluye espera del lock, calentamiento,
flush final, red, pintura del navegador y tiempo de lectura. Los timestamps de
presentación se conservan aparte; no se denomina a este reloj latencia del cliente.
Regresión: preparación de 2 s + consulta de 3 s = 5000 ms. Aborto sin final: null,
nunca cero ni imputación. Sólo respuestas exitosas entran a percentiles técnicos.

## Integridad, privacidad y análisis

`full_session.json` tiene manifiesto SHA-256 por sesión; exportación repetida no
altera archivos. La configuración y el CSV se sellan antes de emitir invitaciones.
Los tokens sólo se guardan hasheados en admisión, fuera de las exportaciones.
No existen campos de nombre, correo, empleador, IP o firma. Los textos libres
pueden contener datos que el usuario escriba: el software no garantiza anonimato
del contenido; antes de compartir se requiere revisión humana y copia redactada
separada, conservando la trazabilidad y respetando el consentimiento.

Una reserva ocupa la misma celda/perfil mediante un registro aparte; el CSV no
cambia. La sesión anterior se abandona y su invitación se revoca. No se promedian
dos personas de una plaza ni se imputa una condición faltante. Las sesiones
abandonadas se reportan como exclusiones del contraste pareado. Errores dentro
de sesiones completas se conservan y se reportan; no se borran para mejorar SUS.

## Exportación consolidada y estadística

`manage_study.py export` produce `sessions.json`, `analysis.json`, este diccionario
y `manifest.json` con hashes; no sobrescribe un destino existente. El análisis
solo incluye pares completos `purpose=study`, identifica duplicados por participante
y plaza primaria, y rechaza mezcla de configuraciones congeladas distintas.

Diferencia = híbrido menos condición sin consulta documental, independientemente
del mapeo A/B. SUS se recalcula desde los diez ítems; valores almacenados discrepantes
se rechazan. F = media F1/F2/(6−F3)/F4; U = media U1/U2/U3. Wilcoxon bilateral y
BH conjunto sobre exactamente SUS/F/U, alfa 0,05. d_z = media de diferencias / SD
muestral de diferencias. Si SD=0 con diferencia no nula, d_z es indefinido (null),
no infinito. Bootstrap pareado por participante: 10 000 remuestras, seed42, IC
percentil 95% para diferencia media y d_z; informa remuestras con d_z indefinido.
Todos los contrastes usan la misma secuencia de índices de remuestreo.

R1, R2 crudo, R2 invertido e I1 se presentan separados como descriptivos, sin promedio
de responsividad ni contraste adicional. Perfiles: diferencias descriptivas, sin
tests de subgrupos. C1–C3: conteos; C4, motivo de cegamiento y consultas libres:
insumos cualitativos. Exactitud de cegamiento: aciertos / participantes incluidos
con cierre; «No sabría decir» permanece en el denominador y se informa aparte.
Pilotos, abandonos y pares faltantes aparecen como exclusiones explícitas, sin imputar.
Los scripts no deciden GO ni sustituyen el plan/procedimientos éticos aprobados.
