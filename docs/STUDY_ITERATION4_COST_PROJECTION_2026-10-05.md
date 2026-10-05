# Proyección del periodo de sesiones — iteración 4

ESTIMADO, sin fecha de primera sesión. No es factura. Se conservan el disco original y dos instantáneas: la bootstrap existente y otra preparada futura, ambas presupuestadas con el tamaño comprimido real de la primera; la segunda todavía no existe. No se supone ahorro por bloques compartidos. Los discos de prueba se retiran al cierre, tras verificar sus IDs y conservar evidencia. La IP se libera durante la espera y se reserva tres días antes.

Supuestos: un piloto y veinte sesiones, 90 minutos de VM por sesión incluidos encendido y apagado, periodo de 21 días. Base USD7,30 estimados y USD3,72 de margen. Se añaden USD3,00 de margen de herramientas/transferencias, separado del gasto. La factura histórica y el cambio implícito PEN/USD3,39 son DECLARADOS por el auditor; no son una nueva conciliación ni un cambio vigente.

| Espera | Estimado USD | Margen USD | Reserva total USD | Techo100 | Corte90 |
|---|---:|---:|---:|---|---|
| 30 días | 57.5130 | 6.72 | 64.2330 | Dentro | Dentro |
| 60 días | 72.2294 | 6.72 | 78.9494 | Dentro | Dentro |
| 90 días | 86.9457 | 6.72 | 93.6657 | Dentro | No admitir bajo estos supuestos |

El escenario de 90 días conserva el techo de USD100 pero supera el corte propio de USD90 al sumar todo el periodo. Se bloquea esa planificación antes de crear recursos; no se elimina margen ni se presupone una ampliación. Enzo debe conciliar la factura y definir el calendario antes de admitir sesiones. La fecha desconocida no autoriza mantener la IP reservada al cierre.

Las tarifas y fórmulas se leen de installation.json y del recibo del tamaño real de la instantánea; el paquete externo conserva session-period-cost-projection01.json. [Cómputo](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [discos e instantáneas](https://cloud.google.com/compute/disks-image-pricing), [IPv4](https://cloud.google.com/vpc/pricing) y [almacenamiento](https://cloud.google.com/storage/pricing).

Los cargos históricos de Network Intelligence Center siguen separados como DECLARADOS (PEN0,102710). Sus APIs no usadas se deshabilitaron con recibos; esto no implica que la factura tardía esté conciliada. Ninguna alerta de presupuesto sustituye el apagado independiente ni el corte del operador.
