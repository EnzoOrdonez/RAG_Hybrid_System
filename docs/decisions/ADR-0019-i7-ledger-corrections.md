# ADR-0019: Correcciones del ledger y la admisión humana I7

[DECLARADO] Estado: aceptado para preparación local; no autoriza publicación,
gasto ni ejecución de Fase B. Base: d8954cb más los cherry-picks ab267dd y d0ba95a
(HEAD de integración 5ae784f). Tarea F-020, revisión cruzada del 2026-10-27.

## Entorno y línea base

[VERIFICADO] El worktree i7-human-ledger-ready usa una junction .venv-app al
mismo destino que f020-docs: el entorno existente de interview-readiness.
No se instalaron ni copiaron paquetes. El enlace está ignorado por Git.

[VERIFICADO] Git Bash seleccionó .venv-app/Scripts/python.exe sin PYTHON
sobrescrito. Con HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, PYTHONHASHSEED=42 y
PYTHONUTF8=1, bash init.sh dio secretos=0, CPU=0 y docs=0: 1896 pasan,
0 fallan, 11 omitidas, 5 excluidas y 9 subtests pasan (211,22 s).

## F1: aislar el ledger heredado

[DECLARADO] reserve_cost e ip_reserve rechazan iteration=7 antes de consultar
nube o tocar el ledger de I5. La comprobación vive en lifecycle, por lo que
también protege una instancia de la clase heredada con instalación I7.
I7 debe usar su admisión/reserva propia; no se eleva ningún tope.

[VERIFICADO] El test parametrizado de fase B/humana y clase I7/heredada falló
en sus ocho casos antes del cambio. Comando de reproducción offline:

```text
.venv-app/Scripts/python.exe -m pytest tests/test_study_i7_ledger.py -o addopts= -q -p no:cacheprovider
```

[DECLARADO] Las demás correcciones se registrarán al verificarse, antes del
cierre local. El despliegue, GO real y aprobación ética real siguen pendientes.

[VERIFICADO] F1 verde: 37 tests de ledger y ciclo/IP heredados pasan. Al retirar
por separado la guarda de reserve_cost y la de ip_reserve, fallan cuatro casos
por mutante; los bytes originales se restauraron en ambos casos.

## F3: exposición durable antes de CREATE

[DECLARADO] El intento incluye horas, instante solicitado, tarifa, hash del
recibo de precios e importe máximo. open_exposure_usd se persiste con el intento
antes de invocar CREATE. Un reinicio conserva esa exposición y bloquea otro CREATE.

[VERIFICADO] La regresión falló antes del cambio (importe durable ausente).
Verde: 49 tests de ledger/humano/operador. Retirar la exposición previa o la
tarifa del intento hizo fallar el test durable; se restauraron ambos mutantes.

## F4: recuperación UNKNOWN con la exposición solicitada

[DECLARADO] Un intento encontrado recupera su base de costo durable; si vence
y está ausente, se liquida una sola vez su máximo admitido y se libera la
exposición abierta. No se sustituyen horas por 12 ni precios por la tarifa actual.
Un intento antiguo sin base de costo queda bloqueado para revisión, sin adivinar.

[VERIFICADO] Cuatro regresiones (1/4 h, encontrado/ausente) fallaron antes de
corregir; después, 53 tests pasan. Los tres mutantes (12 h al encontrar, 12 h
al vencer y no liberar exposición) causaron dos fallos cada uno, y se restauraron.

## F5: tarifa fijada en la reserva

[DECLARADO] La reserva exitosa y la recuperada guardan cost_basis con la tarifa
admitida. guard y reconcile liquidan con esa tarifa. refresh_prices puede renovar
el catálogo para admisiones futuras, pero no modifica la reserva ni su exposición.
Reservas antiguas sin base verificable requieren revisión; no se adivina su precio.

[VERIFICADO] Las dos regresiones fallaron sin la base persistida. Los mutantes
sin base, guard con tarifa actual y reconcile con tarifa actual produjeron
respectivamente dos, uno y un fallo. Se restauraron los bytes originales.
La primera mutación encontró un doble que filtraba cost_basis como si fuese un
campo del proveedor; se corrigió el doble antes de repetir las tres mutaciones.
No se contactó el proveedor ni se alteraron reservas privadas reales.

## F6: códigos primarios rechazados en rehearsal/pilot

[DECLARADO] No se cambia el comportamiento de producción: rehearsal admite
P998 y pilot códigos sintéticos P900+, nunca códigos primarios P01-P20.

[VERIFICADO] Las 40 combinaciones de rechazo y dos admisiones sintéticas pasan.
Al retirar temporalmente la restricción de rehearsal fallaron sus 20 casos;
al retirar la de pilot fallaron sus otros 20 casos. study_sessions.py se
restauró byte por byte y no forma parte del commit de esta corrección.

## F9: TTL nativo y no-shows

[VERIFICADO] El límite min(now, deleteAtTime) ya venía de d0ba95a y se conserva.
Las nuevas regresiones de reservas de 1/4 h reconciliadas tres días después,
sin arrancar VM ni sesión, cargan solo hasta el TTL y no duplican tras reinicio.
Retirar el límite produjo dos fallos; se restauró byte por byte. No hubo
cambio de producción en esta corrección, solo cobertura y este registro.

## F2: cota no-compute completa por reserva

[DECLARADO] La cota del runbook es común a la corrida técnica de 24 h, no por
boot ni por sesión. Por decisión del dueño, se reutiliza completa por cada
reserva humana, sin descontarla por no-show o cierre anticipado. Un reintento
UNKNOWN se reconcilia sin incrementar otra vez el contador. Varios boots dentro
de la misma reserva no añaden un paquete; una reserva nueva sí lo añade.

[DECLARADO] Cada paquete usa default_quantities(24): PD 540 GiB, snapshots
552 GiB e IP sin uso durante 24 h; GCS 1 GiB durante 24 h; 10 000 operaciones A,
100 000 B; 1 GiB de egreso VM y 1 GiB de egreso GCS. IP en uso y transferencia de
snapshots conservan cantidad cero bajo el mismo ámbito us-central1 del runbook.
GPU se cuenta por su reserva de compute; CPU no forma parte del paquete humano.

[DECLARADO] Para PD, snapshots e IP se toma el máximo entre los paquetes
acumulados y la retención continua desde la apertura (mínimo 24 h más 24 h
proyectadas), evitando sumar dos veces la misma retención. Los restantes SKU
no-compute se acumulan por reserva. extra_exposure_usd se valida y suma SIEMPRE,
incluso al comprobar un arranque sin crear otra reserva. El margen de USD 3 se
mantiene; los paquetes son exposición conservadora, no cargos de factura.

[DECLARADO] La admisión incluye el paquete futuro antes de CREATE y el contador
se persiste con el intento, solo después de aprobar el presupuesto. Una petición
rechazada no incrementa el contador. Una fase antigua sin contador válido se
bloquea en vez de asumir gasto cero.

[VERIFICADO] Los ocho casos iniciales fallaron antes de la corrección. Se
verificaron siete mutantes: extra omitido (1 fallo), SKU omitidos (8), paquete
prospectivo omitido (1), contador durable omitido (1), retención continua omitida
(1), validación de extra omitida (5) y contador inválido permitido (3). Todos se
restauraron. La suite del ledger tiene 36 casos aprobados.

[VERIFICADO] Se reescribió este ADR en UTF-8 para reparar los caracteres de
interrogación que sustituyeron tildes en el traspaso anterior.

## F7: aprobación ligada a la configuración sellada ejecutada

[DECLARADO] approved_b4_version debe coincidir exactamente con el identificador
del directorio de la configuración sellada que se ejecuta (por ejemplo,
study-config-reviewed-v2). Se obtiene de local_reviewed_config después de
verificar draw_seal.json y exigir que su fingerprint coincida con el de la
instalación. No se compara con schema_version ni con un b4_version libre.
Fuente ausente, sello alterado, fingerprint distinto o etiqueta distinta bloquean
study antes de consultar nube. La prueba usa fixture-reviewed-config; no se
crea ni se modifica el expediente ético real. Su etiqueta la aportará Enzo.

[DECLARADO] La autenticidad y autoridad del PDF son un atestado humano de Enzo.
El código valida campos, fecha, SHA-256 e identidad de la configuración, pero el
hash no autentica al comité ni sustituye la revisión humana del documento.

[VERIFICADO] Las seis regresiones nuevas fallaron antes del cambio. Se retiraron
por separado la comparación de etiqueta (3 fallos), fingerprint (1), verificación
del sello (1), llamada de política desde I7 (1) y hash del PDF (1). Se restauraron
los bytes originales de cada mutante y las pruebas focalizadas volvieron a pasar.

## Congelación y verificación del cierre

[VERIFICADO] Correcciones en commits separados: F1=77a19aa, F3=3660a10,
F4=f30f4cf, F5=32622b5, F6=c06a919, F9=cc01ac9, F2=b9a2d2f y F7=1a7c319.
La integración que resolvió F8 está en los cherry-picks 9dbba79 y 5ae784f sobre
d8954cb. El conjunto focalizado final tiene 171 pruebas aprobadas y los seis
checks documentales pasan. No hay cambios netos en src/, config/ ni en el smoke
congelado; el CLAUDE.md pendiente del worktree original no se ha tocado.

[DECLARADO] El commit documental de cierre se crea antes de ejecutar el comando
siguiente, para que el hash congelado sea exactamente el árbol verificado. La
salida de las tres capas y el hash se entregan juntos en el reporte al dueño:

```powershell
Remove-Item Env:PYTHON -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONHASHSEED='42'
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PLAYWRIGHT_BROWSERS_PATH='C:/CloudRAG/i7-browser-runtime'
$env:PYTEST_ADDOPTS='-v -ra -p no:cacheprovider'
git rev-parse HEAD
& 'C:\Program Files\Git\bin\bash.exe' init.sh
exit $LASTEXITCODE
```

[DECLARADO] La aceptación local exige secretos=0, CPU=0 y docs=0; no equivale a
GO desplegado. La publicación, los recibos ligados a la fuente y el expediente
ético real corresponden al dueño. No se ejecuta Fase B ni se autoriza gasto.
