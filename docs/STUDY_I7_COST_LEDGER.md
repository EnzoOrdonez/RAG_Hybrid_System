# Registro de exposición I7

El registro distingue precios de lista, exposición conservadora y factura.
Los catálogos de Compute Engine y Cloud Storage se archivan completos con fecha,
moneda USD, páginas y SHA-256. La falta de un SKU, moneda, unidad o página
deja UNPRICED y bloquea pagos. Se toma la tarifa máxima de los tramos, sin
suponer créditos ni franjas gratuitas. Las unidades mensuales se convierten
mediante el factor base declarado por el catálogo.

Se incluyen CPU y RAM g2, L4, CPU/RAM e2, pd-balanced, snapshots, IP en uso y
sin uso, egreso a Sudamérica, transferencia de snapshots dentro de Norteamérica,
almacenamiento GCS, operaciones A/B y descarga GCS. Las reservas se valoran
como la unión de su intervalo y el de la VM que las consume. No hay doble cargo
de compute en el modelo. La espera sin VM también cuenta.

La cota inicial cubre 17 h de reserva GPU acumulada, 6 h CPU, 540 GiB de discos,
552 GiB de snapshots propios/heredados durante 24 h, una IP durante 24 h al
precio más alto sin uso, 1 GiB de egreso de VM y 1 GiB GCS, 10 000 operaciones A
y 100 000 B. El modelo incluye retención después de STOP. El sondeo de snapshot
exige ubicación us-central1 antes de restaurar: la transferencia interregional
queda fuera del plan y bloqueada. El SKU se conserva con cantidad cero para
este ámbito; cualquier traslado requiere revaloración antes del efecto.

Cada admisión pagada recalcula todos los SKU y conserva las filas en el estado.
Se rechaza una exposición total con cierre superior a USD 20, precio de más de
siete días, reserva acumulada superior a 17 h o plazo de retención superior a
24 h. Los USD 3 de cierre se conservan dentro del techo. Los controles heredados
de gasto/espera siguen como una comprobación adicional, no reemplazan el censo.

Comandos de lista, sin crear recursos:

```powershell
python -m scripts.study_operator.pricing --sdk $sdk --package $run --destination $compute
python -m scripts.study_operator.pricing --sdk $sdk --package $run --destination $storage --storage
python -m scripts.study_operator.i7_costs --compute $compute --storage $storage --output $plan
```

La entrada storage_price_archive del spec apunta al segundo catálogo. El creador
local guarda el plan calculado en installation.json. En noviembre deben refrescarse
ambos catálogos antes de admitir pagos; lo leído en octubre no es una factura ni
un precio futuro comprobado.

El corte impide efectos nuevos; STOP no elimina discos, snapshots ni una IP
reservada. El cierre debe retirar temporales propios y decidir la retención de
la evidencia. La retención posterior al período requiere otra decisión de
presupuesto. Las facturas reales deben conciliarse por separado; este modelo
no modifica los presupuestos de alertas de Google Cloud.
