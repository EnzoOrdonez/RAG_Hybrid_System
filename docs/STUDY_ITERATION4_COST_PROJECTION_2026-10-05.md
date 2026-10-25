# Proyección del periodo de sesiones — actualización de iteración 4

ESTIMADO; no es factura ni autorización de admisión. Esta actualización reemplaza los supuestos de la versión inicial del 5 de octubre: 90 minutos por sesión, una instantánea futura del tamaño de la bootstrap y retiro supuesto de discos de prueba. Los artefactos anteriores permanecen íntegros en el paquete.

La fuente es `cost-census09.json`, generado por script a partir de 26 intervalos completados de la API, cinco discos de 100 GiB, cuatro instantáneas comprimidas y metadatos de ambos buckets. El bucket nuevo está vacío. El bucket técnico conserva sus objetos y no se borra. Los SHA y rutas se generan en `session-period-cost-projection03.json`; ningún hash tecleado sustituye el inventario.

Base superior ESTIMADA USD 11.822429, incluyendo el extremo superior separado del rango de transferencia de la imagen. El margen inicial de USD 2,72 se conserva separado. Retención actual sin IP: USD 1.977342/día. Se supone liberar la IP al cierre y reservarla tres días antes de la primera sesión.

Calendario supuesto: un piloto más veinte sesiones, una por día durante 21 días, sin fecha conocida. Cada sesión reserva 135 minutos de VM: 60 minutos de encendido anticipado, 60 de sesión y 15 de apagado. Total 47.25 horas y USD 33.397825 de cómputo. La retención incluye los tres días de preparación y los 21 días de sesiones, además de la espera. La IP asociada se cobra durante los 24 días completos, incluso con la VM detenida: USD 2,88; si permanece sin asociar, el extremo superior es USD 5,76.

| Espera | Estimado USD | Margen separado USD | Reserva total USD | Corte 90 | Techo 100 |
|---|---:|---:|---:|---|---|
| 0 días | 95.5565 | 10.97 | 106.5265 | Superado | Superado |
| 30 días | 154.8767 | 10.97 | 165.8467 | Superado | Superado |
| 60 días | 214.1970 | 10.97 | 225.1670 | Superado | Superado |
| 90 días | 273.5173 | 10.97 | 284.4873 | Superado | Superado |

El margen incluye USD 2,72 iniciales, USD 5,25 para operaciones futuras y USD 3,00 para transferencias futuras. No se presenta como gasto facturado. La factura histórica en PEN y el cambio implícito 3,39 PEN/USD son DECLARADOS por el auditor; no son conciliación actual ni tipo de cambio vigente.

Con la retención actual, ningún escenario admite el periodo bajo el corte de USD 90. La eliminación de recursos de prueba propios solo puede reducir la proyección después de recibos reales, verificación de IDs y preservación completa de su evidencia. No se supone esa eliminación, compresión futura ni ahorro de bloques compartidos. El disco original, todas las instantáneas y ambos buckets permanecen protegidos. Tampoco se presupone el costo de una contingencia futura todavía no calificada.

Esta proyección no cubre los ensayos de aceptación restantes ni una nueva factura después del corte histórico. Antes de admitir sesiones hacen falta calendario, conciliación, recursos finales y nueva proyección bajo el corte. Una alerta de presupuesto no sustituye el supervisor ni el apagado.

Fuentes oficiales: [cómputo](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [discos e instantáneas](https://cloud.google.com/compute/disks-image-pricing), [IPv4](https://cloud.google.com/vpc/pricing) y [almacenamiento](https://cloud.google.com/storage/pricing). Los cargos históricos de Network Intelligence Center, PEN 0,102710, permanecen DECLARADOS e incluidos en la base heredada; no se duplican como cargo nuevo.
