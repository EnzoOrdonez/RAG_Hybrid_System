# ADR-0035: Resultado del intento 15: gate GO y limpieza técnica incompleta

[DECLARADO] Registrado el 2026-11-15 sobre la imagen de b5e54f1 y el anexo publicado en 1a4ead8
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Yap te autprizo todo pero termina el despliegue y todo lo qie haga falta para que este
proyecto ya sea palapable y se cumpla el objetivo inicial"), dentro del tope de USD 20 + 3 por
intento. Este ADR no cambia código.

## Resultado del intento 15

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK. |
| 2 gpu-restore | Cinco rechazos de capacidad L4 liberados (ADR-0026); reserva 1 retenida a las 14:47Z. Una ronda abortó antes de crear la reserva por RAM local insuficiente; su carpeta se archivó con nota de traslado. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en 1a4ead8 y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY`. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**: siete dominios completos, 0 aciertos. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | **ACCEPTED**: 12/12 combinaciones, 144 llamadas, 12 boots en frío, de 17:05Z a 20:26Z. Reserva 1 cerrada por guard. |
| 8b gate | Reserva 2 retenida al primer intento (20:28Z). Recolección verificada a las 22:25Z: **GO**. |
| Limpieza técnica | Sesiones técnicas cerradas y respaldadas; restauración verificada; `purge-study` falló en `maintenance` por transporte (abajo). Guard detuvo la VM y borró la reserva 2, ambas verificadas. |
| 9 close | `CLOSED`: VM detenida y verificada, cierre de reserva registrado, temporales CPU ausentes. |

Tras el close, la IP estática se desligó de la VM y quedó RESERVED sin usuarios.
Gasto: USD 9,46 según el ledger del operador (exposición conservadora, no factura).

### Medición del gate

[VERIFICADO: recibo `DOWNLOAD_VERIFIED`, recalculado desde la prueba privada con `i7_gate.summarize`]

| Condición | n válidas | Fallos | Inválidas | p50 | p95 |
|---|---|---|---|---|---|
| hybrid | 60 | 0 | 0 | 26,03 s | 34,88 s |
| no_rag | 60 | 0 | 0 | 22,70 s | 27,64 s |

El criterio preregistrado (120 válidas, 60 por condición, cero fallos e inválidas, p95 lineal ≤ 60 s
por condición) se cumple. El piloto de 20 llamadas no entra en el cálculo. Las sesiones de esta
medición son técnicas y sintéticas (códigos P999 y P9000xx), no personas. El detalle descriptivo
está en [i7-gate-sesiones-sinteticas-i15.md](../reports/i7-gate-sesiones-sinteticas-i15.md).

## Problema: la purga técnica no sobrevive a un corte de transporte en `maintenance`

[VERIFICADO en recibos y en el traceback del gate]
- Tras recolectar el GO, `cleanup` llamó `delete_sessions('purge-study')`, que empieza con
  `maintenance`. El SSH por IAP no pudo verificarse (`TransportUnavailable`).
- `maintenance` tiene efectos. Por diseño (ADR-0034) tiene un solo intento; solo `inventory` y
  `download-disk` reintentan.
- El gate salió con exit 1 por la limpieza, no por la medición: el recibo GO ya estaba escrito y
  verificado.
- Las sesiones sintéticas del gate quedaron sin purgar en el disco GPU y en el bucket del período,
  igual que en el intento 14. No son datos de personas.

## Decisión

[DECLARADO]
1. El GO de este intento es la evidencia de rendimiento de la Fase B de I7. Vale solo para la
   imagen, la fuente b5e54f1, la zona y la máquina del anexo publicado en 1a4ead8. `verified_gate_go`
   ya exige esa identidad.
2. No se cambia código del operador. `verify_tool_source` rechaza cualquier comando `i7` sobre
   esta raíz si el diff contra b5e54f1 sale de `docs/` y `output/audit/`. Un arreglo en `scripts/`
   invalidaría el uso de esta raíz y exigiría otra Fase B.
3. Los comandos `i7` de la fase humana sobre esta raíz se ejecutan desde un checkout del commit que
   introduce este ADR, o de otro posterior cuyo diff contra b5e54f1 solo toque `docs/` y
   `output/audit/`. Los cambios en `.harness/` van en commits aparte, después.
4. La purga de las sesiones sintéticas de los intentos 14 y 15 queda pendiente. La vía es
   `technical-cycle`, que exige un boot pagado. Decide Enzo, con cotización previa.

## Límites conocidos

[DECLARADO]
- Un corte de transporte en una operación con efectos de la limpieza sigue dejando sesiones
  sintéticas sin purgar. No afecta a una medición ya recolectada.
- La primera respuesta tras un boot en frío fue más lenta que en caliente (hybrid hasta 47,4 s en el
  estímulo). El gate mide en caliente. Cada sesión con personas debe empezar con el calentamiento
  ya previsto en `docs/INTERVIEW_READINESS.md`.
- El GO no autoriza entrevistas. Faltan la aprobación ética escrita, `human-phase-open`, el ensayo
  P998 con Enzo y las invitaciones (F-020, pasos F y G).

## Verificación

[VERIFICADO]
- Recibo del gate con `status` `DOWNLOAD_VERIFIED` y `decision` GO. `summarize` sobre las 120 filas
  de la prueba privada da cifras idénticas al recibo.
- Recibos de guard: VM `TERMINATED_VERIFIED` y reserva `RESERVATION_DELETED_VERIFIED`. El close
  devolvió `CLOSED`.
- Lectura de GCP tras el close: 0 reservas, VMs detenidas y la IP estática RESERVED sin usuarios.
