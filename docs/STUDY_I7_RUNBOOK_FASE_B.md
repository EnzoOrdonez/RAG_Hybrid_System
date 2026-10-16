# I7: runbook de Fase B

Revisión: 16/10/2026, ronda A2. Código probado en local; ninguna validación en
nube. El inicio de la compuerta exige requisitos comprobados, sin un rechazo
incondicional. El check enumera el dueño de cada bloqueo pendiente.

## Límites que se conservan

Solo `fix/interview-readiness`, us-central1 y una L4. Sin personas ni registro
ético. No modificar RAG, calendario, modelo, opciones, umbrales ni final03.
Construir una imagen I7 nueva. Leer completos los dos preregistros y la enmienda
I4 antes de cualquier pago. No extender límites para obtener un resultado.

Piloto caliente de 20 y dos ventanas de 60 en el mismo boot. El piloto no
concede GO. Solo el agregado de 120, 60 por condición, cero fallos e inválidos y
p95 lineal <= 60 s por condición decide. Llamada 600 s, preparación 900 s,
ventana 7200 s, trabajo 9000 s, invitado 165 minutos, STOP nativo 3 h.
El tiempo parte de `study_service.answer`, antes de la escritura durable.

## Preparación privada, antes de pagar

Enzo autoriza el push del código y abre la tarea de noviembre con el presupuesto.
No ejecutar `--live` en octubre. Revisar `DECLARACIONES_I6_BORRADOR.md` por separado.
Confirmar cuota, instantáneas, IP, firewall, IAM, bucket vacío y política de
retención por lecturas API. La factura I6 requiere conciliación propia; los
precios de lista no son factura.

Primero abrir `powershell.exe -NoLogo -NoProfile` y ejecutar el bloque completo
de instalación y suite I7 de `STUDY_I7_BROWSER_VALIDATION.md`. Ese bloque fija
la ruta y comprueba Chromium antes del primer paso de Fase B.

Después, desde la raíz del worktree publicado, con Python 3.14.3:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = 'C:/CloudRAG/i7-browser-runtime'
$py = (Resolve-Path .venv-app/Scripts/python.exe).Path
& $py -m scripts.study_operator.i7_browser_preflight
if ($LASTEXITCODE -ne 0) { throw 'Pinned Chromium preflight failed' }
$sdk = (Get-Command gcloud.cmd).Source
$prep = 'C:/CloudRAG/i7-preparacion-noviembre'
$run = 'C:/CloudRAG/iteration7-faseB-20261101'
$spec = "$prep/fase-b-spec.json"
$evidence = 'C:/CloudRAG/iteration7-browser-stability-20261016/local-readiness.json'
& $py -m scripts.study_operator.i7_phase_b --check --evidence $evidence
```

Debe listar solo bloqueos con dueño Enzo: autorización de noviembre y anexo
efectivo publicado. Un bloqueo con dueño `code` impide empezar. El recibo liga
los bytes de fuente, 19 módulos, 3 documentos y logs de Windows, Linux e I7.
No editar ese recibo para hacer pasar código diferente. Un cambio de fuente
exige repetir las comprobaciones afectadas y emitir evidencia nueva.

El automatizador del dueño se instala fuera de la imagen de servicio. En cada
consola que ejecute Fase B se debe definir la misma ruta explícita:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = 'C:/CloudRAG/i7-browser-runtime'
python -m pip --python $py install --require-hashes -r requirements-i7-browser.txt
if ($LASTEXITCODE -ne 0) { throw 'Browser dependency installation failed' }
& $py -m playwright install --no-shell chromium
if ($LASTEXITCODE -ne 0) { throw 'Chromium installation failed' }
& $py -m scripts.study_operator.i7_browser_preflight
if ($LASTEXITCODE -ne 0) { throw 'Pinned Chromium preflight failed' }
```

Playwright 1.58.0 instala Chromium **1208** por su manifiesto fijado. Ruta
Windows: `$env:PLAYWRIGHT_BROWSERS_PATH/chromium-1208/chrome-win64/chrome.exe`.
SHA-256: `ff0c99fc5bb2301cad77930aa76e5354cf2a5fefb7efdcf3b3bc10ba0550c015`.
En Linux: `chromium-1208/chrome-linux64/chrome`, SHA-256
`481fea1516a1f2b76454664272f12cd9dd1f20117b21e1f1498e08bc7f872c00`.
El pin comprueba revisión, ruta, versión, árbol del paquete y ejecutable.
Un navegador ausente o distinto bloquea el preflight antes de crear el cliente
de nube de cualquier comando Fase B; el check enumera el bloqueo por nombre.
Mantener la variable también en el proceso independiente de la serie. Copiar
el recibo `browser-validation.json` de la validación de estabilidad al nuevo `$run` tras crearlo; el recibo debe
coincidir con fuente y pin. La validación recorrió la app real por HTTPS local,
con dobles del backend, y tres fallas deliberadas. No es el smoke desplegado.

Descargar ambos catálogos en directorios nuevos, como máximo siete días antes
de cada admisión pagada. Archivar stdout privado sin credenciales:

```powershell
& $py -m scripts.study_operator.pricing --sdk $sdk --package $prep --destination "$prep/compute-list"
& $py -m scripts.study_operator.pricing --sdk $sdk --package $prep --destination "$prep/storage-list" --storage
& $py -m scripts.study_operator.i7_costs --compute "$prep/compute-list" --storage "$prep/storage-list" --output "$prep/exposure.json"
```

Exigir `PRICED_LIST` y `ADMIT`. `UNPRICED` bloquea. JSON privado de entradas:

```json
{
  "inherited_package": "C:/CloudRAG/iteration5-run-20261007T020139Z",
  "source_installation": "C:/CloudRAG/iteration6-cowork-20261015/operator/installation.json",
  "price_archive": "C:/CloudRAG/i7-preparacion-noviembre/compute-list",
  "storage_price_archive": "C:/CloudRAG/i7-preparacion-noviembre/storage-list",
  "reviewed_config": "C:/CloudRAG/iteration5-run-20261007T020139Z/study-config-reviewed-v2",
  "vendor": "RUTA_PRIVADA_DEL_VENDOR_VERIFICADO"
}
```

Los junctions C: siguen siendo válidos. Ante una ruta rota, cotejar hashes y
reportar; no trasladar paquetes sellados. El vendor debe corresponder al commit
del lock; no cambiar dependencias para evitar una falla.

```powershell
& $py -m scripts.study_operator.i7_config --root $run --spec $spec
Copy-Item 'C:/CloudRAG/iteration7-browser-stability-20261016/browser-validation.json' "$run/browser-validation.json"
```

La raíz es nueva, con períodos técnicos y smoke distintos. No hereda una VM.
Los siguientes comandos están escritos para noviembre y no constituyen permiso.

## Secuencia de ejecución

Cada comando conserva sus recibos privados. No publicar consultas, respuestas,
tokens, IP, hostname, canarios o inventarios completos.

| Paso | Comando | Criterio de éxito |
|---|---|---|
| 1. CPU, imagen y disco limpio | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --spec $spec --evidence $evidence --live cpu-build` | Fuente publicada; restauración CPU; imagen nueva con corrección de restore; nueve registros originales archivados fuera del servicio con hash y ACK; saneamiento offline; snapshot propio READY en us-central1. |
| 2. Reserva y GPU | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live gpu-restore` | Reserva específica 12 h retenida antes de crear disco; snapshot e identidad exactos; VM propia con STOP 3 h. |
| 3. READY | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live ready` | <= 900 s; persistencia NVIDIA, driver, blob y mincore completo, JIT persistente, Ollama vacío, aislamiento e identidad. Una falla impide READY y admisión. |
| 4. TLS | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live tls-three` | Tres boots distintos y consecutivos READY, mismo certificado y fuente, sin fallo intermedio. |
| 4b. Contextos vivos | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live contexts` | Doce contextos en contenedor aislado, antes de generar y sin consumir historial del servicio. Comparación exacta con la referencia I3 heredada de I5. |

Antes de generar el estímulo o medir: preparar el anexo, revisar que no exponga
identificadores, mostrar a Enzo el diff y pedir permiso para su publicación.

```powershell
& $py -m scripts.study_operator.i7_admission --root $run --draft
```

El archivo `output/audit/I7_ENVIRONMENT_ANNEX.md` contiene solo zona, máquina,
imagen, fuente y hashes. Revisar si corresponde ampliar el inventario público
sin identificadores. Commit y push del anexo solo con permiso a la rama, sin
force. Los observadores, preregistros, calendario y automatizador deben ser
idénticos a la fuente de la imagen. Solo se admiten cambios de documentación.
Tras confirmar la publicación real:

```powershell
& $py -m scripts.study_operator.i7_admission --root $run --anchor-published
& $py -m scripts.study_operator.i7_phase_b --check --root $run --evidence $evidence --live
```

Exigir `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED` y `LOCAL_RUNBOOK_READY`.
La publicación externa no se sustituye con un doble ni con un JSON inventado.
Un cambio del tip remoto exige revisar y anclar de nuevo de manera explícita.

| Paso | Comando | Criterio de éxito |
|---|---|---|
| 5a. Smoke automático HTTPS | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live smoke` | App real, cada dato confirmado en checkpoint, ocho intentos, dos instrumentos, cierre y export completo con SHA igual al respaldo; recibo ligado a boot/sesión/automatizador. |
| 5b. Restauración | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live restore` | Generaciones exactas, control aislado, almacén nuevo sellado, export idéntico, original intacto e invitación revocada. |
| 6. Canarios | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live privacy` | IP propia en memoria, UA generado, navegador y WebSocket confirmados, generación y respaldo; siete dominios completos y cero aciertos nuevos. Error o dominio ausente produce INCOMPLETE. |
| 7. Retiro y purga | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live withdraw-purge` | Dry-run y ejecución sobre datos sintéticos, cuatro controles en disco y versiones GCS, incluidas recuperaciones. |
| 8a. Estímulo | `& $py -m scripts.study_operator.i7 --root $run --sdk $sdk --live schedule` | Proceso del Programador de tareas independiente de la sesión del agente; doce boots, calendario intacto, hash y verificación antes de ACK, ACCEPTED 12/12 actual. Cierra reserva. |
| 8b. Compuerta | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --evidence $evidence --live gate` | Revalida publicación y doce pruebas/contextos actuales; reserva 5 h; repite el mismo smoke después del estímulo, restaura y purga; nuevo boot, piloto y dos ventanas; hash por fila, telemetría y descarga verificada. Restaura un respaldo técnico y purga antes del ACK final. |
| 9. Cierre | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live close` | STOP verificado, DELETE y LIST de reserva exitosos, temporales CPU propios ausentes. Un UNKNOWN o cierre CPU parcial no devuelve CLOSED. |

No correr también `stimulus` si ya está programada la serie. La alternativa en
primer plano es el paso `stimulus` de `i7_phase_b`; no resiste el cierre del
proceso dueño. Guardar el recibo de la tarea y sus resultados. No programar un
segundo intento terminal ni cambiar el calendario. El colector mantiene pruebas
privadas de todos los boots completos. Los terminales se conservan en el host;
los terminales de compuerta se descargan antes del ciclo técnico.

En la Fase B, el canario IP introducido debe ser la **IP pública de origen real
de la conexión del navegador al servicio**, la que observa el extremo servidor.
Comprobarla desde ese mismo equipo y conexión antes del ensayo, considerando
NAT, VPN o proxy. No escribirla en documentos públicos. Una IP RFC 5737 o una
cadena inventada no valida ese control desplegado.

En local, la IP documental viajó como sufijo de User-Agent también por
WebSocket para probar el transporte y el censo con dobles. Ese sufijo no es
la IP de origen TCP ni acredita su control en la Fase B. El recorrido puede
transportar también la IP real como marcador en UA; el criterio desplegado
sigue siendo cero registros nuevos de la IP de origen real y del UA canario.
No declarar PASS si no se confirmó la IP de origen de la conexión.
Caddy elimina la cabecera antes de la app. El control
positivo local provocó una fuga deliberada con dobles; la comprobación del SO,
Caddy, journald, consola serial y bucket reales se hace en el paso 6. No declarar
PASS si hay exclusiones sin fecha/motivo o un dominio incompleto.

## Costos y regla de corte

Referencia de lista archivada el 16/10, no factura y no tarifa garantizada de
noviembre: reserva g2+L4 USD 0,706832255/h. Las reservas 12 h + 5 h suman
USD 12,016148335 de compute; 6 h CPU, 540 GiB de PD, 552 GiB de snapshots,
IP, GCS, operaciones y egreso se registran por SKU. Exposición de referencia
completa por 24 h: **USD 18,69112008**, incluidos **USD 3 de cierre**.
Actualizar ambos catálogos antes de noviembre; otro precio puede bloquear.

| Paso o grupo | Cota estimada con lista de octubre, USD |
|---|---:|
| 1 CPU, máximo 6 h e2 | 0,4021 de compute; PD y snapshots en la cota común de abajo. |
| 2 Reserva inicial, máximo 12 h | 8,4820 de g2+L4, incluida toda espera y VM consumidora. |
| 3 READY; 4 TLS/contextos; 5 smoke/restore; 6 privacidad; 7 retiro; 8a estímulo | Dentro de las mismas 12 h, 0,7068 por hora asignada; no sumar otro cargo GPU por paso. Operaciones/egreso entran en la cota común. |
| 8b Reserva compuerta, máximo 5 h | 3,5342 de g2+L4; incluye smoke posterior y cierre técnico. |
| PD 540 GiB, snapshots 552 GiB e IP sin uso, 24 h | 2,8723 en conjunto, incluidos recursos que quedan después de STOP. |
| GCS, operaciones y ambos egresos acotados | 0,4007 en conjunto. |
| 9 Margen reservado de cierre | 3,0000; es provisión, no un cargo medido. |

Son cotas compartidas: no sumar las filas de pasos como si cada uno tuviese
una reserva distinta. El ledger conserva el desglose exacto por SKU.

Se cobran reservas aun con VM detenida. La unión de intervalos evita contar
VM y reserva consumida dos veces. El ledger automático evalúa todos los SKU,
retención después de STOP y exposición extra, antes de cada efecto pagado.
Corta si el total previsto con cierre supera USD 20, si falta cualquier SKU,
si la tarifa vence o si transcurren 24 h. El límite conservador acumulado de
reservas es 17 h; no se reembolsa automáticamente un TTL cerrado antes. Puede
bloquear una recuperación aunque la factura real sea menor. No ampliar el tope.

Las cantidades previstas son cotas operativas, no una garantía de factura:
17 h reservadas, 6 h CPU, máximo 540 GiB de PD y 552 GiB de snapshots durante
24 h, 1 GiB de egreso VM y 1 GiB GCS, 10 000 operaciones A y 100 000 B.
Los snapshots fuente, intermedios y final deben estar en us-central1; moverlos
fuera del ámbito bloquea la estimación de transferencia cero. Vigilar tamaños,
operaciones y egreso; una exposición extra conocida se añade al ledger privado
antes de admitir otro efecto. Contrastar al cierre con la consola.

STOP no detiene cobro de discos, snapshots ni IP. `close` conserva el disco/VM
GPU detenido y el snapshot limpio. Antes de 24 h Enzo decide su retención o
eliminación, con autorización y verificación de identidad. El corte del
operador no apaga la facturación del proveedor. No borrar recursos heredados.

## UNKNOWN, terminales y contingencia

Un error API no acredita ausencia. No repetir una creación desconocida:

```powershell
& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live reconcile
```

La reconciliación lee CPU, VM, disco y snapshots por nombre, zona, marcador,
ID y procedencia. Devuelve identidad resuelta o bloqueo con comando exacto
`describe`. No crea reemplazos. Si la construcción CPU falló sin un plan de
cierre calificado, revisar recursos propios y recibos antes de cualquier
borrado; `BLOCKED_CPU_CLOSURE_REVIEW` requiere esa revisión, no otro build.

La compuerta conserva todo intento terminal. No volver a ejecutarla para buscar
GO. El host reserva hasta 600 s del límite original para cierre técnico; no
extiende los 9000 s. Si no alcanza, conserva evidencia y datos. Tras autorizar
una recuperación dentro del presupuesto y preparar un boot técnico, el comando
es `i7_phase_b ... --live technical-cycle`; verifica restauración, retiro y
purga, sin repetir consultas. Después, `close`.

Si falta capacidad, detener la pieza. La política de dos rondas y pausa de
30 minutos requiere un recibo humano: el operador no cambia de zona por sí
mismo ni consume otra cuota a ciegas. No restaurar GPU sin capacidad retenida.
El ensayo secuencial con cuota de una GPU sigue en `STUDY_I7_CONTINGENCY.md`;
la solicitud de dos GPU es un borrador que Enzo decide si envía.

El resultado de Fase B debe separar mediciones, fallos, terminales y pendientes.
Ningún resultado local, ni el ACCEPTED de I6, acredita GO I7. La entrada de
usuarios requiere todos los controles desplegados y aprobación ética.
