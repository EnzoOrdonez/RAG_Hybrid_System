# I7: runbook de Fase B, borrador bloqueado

Estado al 16/10/2026: PARCIAL, probado solo en local. `gate` y `gate-start`
rechazan el inicio antes de efectos pagados: falta integrar el automatizador
HTTPS validado y la admisión del anexo de entorno publicado. A1 y A8 quedan
abiertos. No iniciar la Fase B con esta revisión. Una tarea posterior debe
cerrar esos dos prerrequisitos, probarlos y publicar su código con permiso.

El preregistro vigente incluye
`STUDY_GATE_PREREGISTRATION_AMENDMENT_2026-10-05_ITERATION4_SERVICE.md`.
Conserva piloto caliente de 20, ventanas de 60 en el mismo arranque, 600 s por
llamada, 900 s de preparación, 7200 s por ventana, trabajo de 9000 s,
supervisor invitado de 165 minutos y STOP nativo de 3 h. El piloto no concede
GO. Solo el agregado de 120, con 60 por condición, cero fallos e inválidos y
p95 lineal <= 60 s por condición, puede concederlo. El cronómetro es el de
`study_service.answer`, anterior a la petición durable. Los respaldos y
telemetría quedan fuera de ese cronómetro, dentro de los límites del trabajo.

## Preparación local y decisiones previas

Enzo debe autorizar una tarea nueva en noviembre, el push a la rama y el gasto.
El tope elegido es USD 20 total, con USD 3 reservados al cierre. No crear
participantes ni registro ético. Solo us-central1 y una L4. No borrar recursos
heredados ni ejecutar este runbook desde una copia modificada del código.

Antes del primer pago: cerrar F01 local; revisar anexo y observadores, calendario
y congelamiento; comprobar cuota y recursos por API; verificar tarifas por SKU
en USD y conciliarlas con la consola, dejando fecha, moneda, filas y hashes.
La API del catálogo aporta tarifas; no acredita gasto facturado. F06 y las
desviaciones F08 de I6 requieren decisión y declaración propias.

Preparar un JSON privado `fase-b-spec.json` con cinco rutas verificadas:

```json
{
  "inherited_package": "C:/CloudRAG/iteration5-run-20261007T020139Z",
  "source_installation": "C:/CloudRAG/iteration6-cowork-20261015/operator/installation.json",
  "price_archive": "C:/CloudRAG/i7-preparacion/official-compute-skus-nov",
  "reviewed_config": "C:/CloudRAG/iteration5-run-20261007T020139Z/study-config-reviewed-v2",
  "vendor": "RUTA_PRIVADA_VERIFICADA_DEL_VENDOR"
}
```

Si hubo traslado a D:, comprobar hashes y cambiar esas rutas. No mover los
paquetes sellados para preparar esta fase. Verificar la ruta del vendor con
`build_context`, el sello heredado y la presencia de las tres instantáneas.
La caché JIT heredada se espera bajo el driver 580.178.04; cualquier diferencia
requiere revisión antes de pagar. No recompilar ni cambiar el modelo para
salvar una validación fallida.

Comandos de preparación, desde un checkout limpio y publicado de la rama:

```powershell
$py = (Resolve-Path .venv-app/Scripts/python.exe).Path
$sdk = 'C:/Users/enziz/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
$run = 'C:/CloudRAG/iteration7-faseB-20261101'
$spec = 'C:/CloudRAG/i7-preparacion/fase-b-spec.json'
& $py -m scripts.study_operator.i7_phase_b --check
```

El resultado actual es `LOCAL_RUNBOOK_PARTIAL`: detenerse. Tras cerrar los
bloqueos y revisar este documento, descargar el catálogo en un directorio nuevo:

```powershell
& $py -m scripts.study_operator.pricing --sdk $sdk --package C:/CloudRAG/i7-preparacion --destination C:/CloudRAG/i7-preparacion/official-compute-skus-nov
& $py -m scripts.study_operator.i7_config --root $run --spec $spec
```

El creador local asigna períodos nuevos, elimina identidad de VM y disco
heredados y deja un destino sin asignar. No prueba la nube ni autoriza gasto.
La tarifa debe tener como máximo siete días al admitir cada efecto pagado.

## Secuencia prevista, pendiente de cierre de prerrequisitos

Cada fila tiene un punto de entrada. Conservar stdout y stderr en el paquete
privado, salvo la consola de invitación sintética, que contiene un token.
Nunca incluir textos sintéticos ni canarios en el informe público.

| Paso | Comando PowerShell | Criterio de éxito | Costo incremental de referencia |
|---|---|---|---|
| 1. Imagen CPU y disco limpio | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --spec $spec --live cpu-build` | Fuente publicada, suites Linux y POSIX, RAG idéntico, imagen nueva, nueve originales descargados con hash, snapshot propio READY. | 6 h CPU y dos discos de 180 GiB por 72 h como cota conservadora; tarifas de noviembre obligatorias. |
| 2. Restaurar con reserva específica | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live gpu-restore` | Reserva específica con autoeliminación, identidad leída, disco de snapshot calificado, VM propia y STOP 3 h. | Reserva 12 h: USD 8,4820 con tarifa de octubre; disco e IP se suman. |
| 3. READY | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live ready` | <= 900 s desde start, persistencia NVIDIA, blob con hash y mincore completo, JIT persistente, Ollama vacío, aislamiento, identidad y TLS verificados. | Dentro de la misma reserva. |
| 4. TLS tres arranques | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live tls-three` | Tres boot ID distintos y consecutivos, sin fallo intermedio, mismo certificado, URL y commit. | Dentro de la misma reserva; cada espera también se cobra. |
| 5a. Sesión sintética | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live smoke` | Dos sistemas, ocho intentos, instrumentos, cierre, export y generaciones de respaldo. | Dentro de reserva, límite de espera 1 h. Actualmente requiere recorrido manual de navegador. |
| 5b. Restauración | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live restore` | Puente IAP, contenedor de control aislado, nuevo almacén sellado, export con igual hash, invitación revocada y original intacto. | Dentro de reserva. |
| 6. Canarios | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live privacy` | Siete dominios completos, cero aciertos, serial desde offset cero sin pérdida, versiones y políticas GCS comprobadas. | Dentro de reserva, más lecturas GCS. |
| 7. Retiro y purga | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live withdraw-purge` | Dry-run y ejecución sintéticos; cuatro controles positivos en original y recuperaciones. | Dentro de reserva. |
| 8a. Estímulo frío | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live stimulus` | Doce boots completos, calendario intacto, prueba privada verificada por SHA antes de ACK, ACCEPTED 12/12 en imagen I7. Cierra reserva. | Hasta TTL de 12 h, ya contabilizado; no sumar VM dos veces. |
| 8b. Smoke validado, piloto y compuerta | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live gate` | **BLOQUEADO en esta revisión.** Integrar automatizador validado y anexo publicado antes de habilitar. | Cero al rechazar. Propuesta futura: reserva 5 h, USD 3,5342 de compute, dentro del mismo tope. |
| 9. Cierre | `& $py -m scripts.study_operator.i7_phase_b --root $run --sdk $sdk --live close` | VM propia TERMINATED, reserva ausente por listado exitoso y borrado confirmado, temporales CPU propios ausentes; conservar snapshot limpio, disco/VM GPU detenidos y evidencia privada. | Cubierto por USD 3 reservados; retención sigue generando gasto. |

El paso 8b aún necesita integrar la purga de las 140 sesiones técnicas respaldadas,
restaurar un respaldo representativo y guardar el resultado antes del cierre.
También necesita enlazar los contextos vivos con los de I3 y exigir los hashes
publicados del anexo, observadores y calendario antes de la primera llamada.
El núcleo local mide el recorrido correcto y verifica dos preflights en reposo;
esas piezas no acreditan los prerrequisitos todavía ausentes.

## Corte, fallos y cierre

La reserva cobra aun con VM detenida. La contabilidad toma la unión de intervalos
de reserva y ejecución. Mantiene costo conservador CPU separado conceptualmente
de la factura. El siguiente efecto se rechaza si gasto + exposición abierta +
costo siguiente + USD 3 supera USD 20. La cota actual admite solo el compute y
un margen fijo de USD 0,25 por reserva; falta integrar retención exacta de
snapshot, IP y transferencias al ledger automático. **Bloqueo previo a pagos:**
comprobar que esa cota cubre todos los SKU o ampliar la contabilidad conservando
el tope de USD 20. No presentar este control parcial como garantía del total.

Referencia de octubre, procedencia en `STUDY_I7_CONTINGENCY.md`: GPU
USD 0,706832255/h; PD USD 0,00013698630137/GiB-h; snapshot
USD 0,000068493150685/GiB-h; IP USD 0,005/h asociada y USD 0,01/h libre.
Ejemplo: 180 GiB de disco más 60 GiB de snapshot por 24 h cuestan USD 0,6904,
antes de IP y operaciones. Definir retención de 24 h y luego revisar el gasto;
el código no elimina automáticamente el disco GPU ni las instantáneas heredadas.

Una creación con resultado UNKNOWN bloquea otro intento. Reconciliar por lectura
con `& $py -m scripts.study_operator.i7 --root $run --sdk $sdk --live reconcile`.
Ese comando reconcilia reservas; para VM/disco con resultado desconocido se
requiere revisión por identidad y recibos antes de continuar. No interpretar
errores de API como ausencia ni repetir construcciones/cohortes para obtener GO.
Si CPU falló antes de asignar VM GPU, el cierre global requiere revisar los
recursos propios del subpaquete `cpu`; no invocar STOP sobre el placeholder.

Si no hay capacidad: dos rondas separadas por 30 minutos, con recibos y orden
de zonas preregistrado. El operador no automatiza esa política todavía. No
restaurar el disco GPU antes de retener capacidad ni salir de us-central1.
Conmutación: seguir la propuesta de `STUDY_I7_CONTINGENCY.md`, previa decisión
de Enzo. No mezclar un ensayo de conmutación con una cohorte.

Conservar todos los terminales antes de borrar temporales. Un límite vencido
deja INCOMPLETE_TERMINAL. Un agregado completo que incumple el umbral deja
BLOQUEADO-HUMANO. La aprobación ética sigue pendiente y ningún resultado local
autoriza usuarios. El cierre final debe cotejar API y consola de facturación,
retención restante y ausencia de sesiones sintéticas en disco y GCS.
