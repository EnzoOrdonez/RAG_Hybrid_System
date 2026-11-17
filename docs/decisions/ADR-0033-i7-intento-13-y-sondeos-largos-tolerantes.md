# ADR-0033: Resultado del intento 13 y sondeos largos tolerantes a cortes de transporte

[DECLARADO] Registrado el 2026-11-14 sobre la imagen de 06ec707 y el anexo publicado en 1349b20
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Yap te autprizo todo pero termina el despliegue y todo lo qie haga falta para que este
proyecto ya sea palapable y se cumpla el objetivo inicial"), dentro del tope de USD 20 + 3 por
intento. El cambio toca `scripts/`, así que exige commit, suites, `local-readiness`, raíz y
cpu-build nuevos (intento 14).

## Resultado del intento 13

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK. |
| 2 gpu-restore | Capacidad L4 en la primera ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en 1349b20 y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY`. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**: siete dominios completos, 0 aciertos. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | Boots 1 y 2 COMPLETE. Fallo terminal en el boot 3 por `TransportUnavailable`. |
| 8b gate | No se lanzó: exige un estímulo ACCEPTED. |
| 9 close | `CLOSED`: VM detenida y verificada, reserva cerrada, temporales CPU ausentes. |

Gasto: USD 6,31 (exposición conservadora, no factura). La latencia del despliegue sigue sin medir:
no hay GO ni NO-GO.

## Problema

[VERIFICADO en recibos]
- El túnel SSH por IAP hacia la VM falló de 20:53:04Z a 20:58:40Z, unos 5,5 minutos.
- Durante ese tiempo las llamadas a la API de Google Cloud (`instances describe`,
  `get-guest-attributes`) siguieron respondiendo.
- `stimulus-status` es un sondeo de solo lectura. Reintentó tras 15 s y 45 s (ADR-0029). Cada intento
  fallido tardó entre 90 y 130 s, así que los tres intentos cubrieron unos 4,5 minutos.
- Al rendirse el sondeo, `series` terminó. El `finally` detuvo la VM y `guard` borró la reserva 1.
- El boot 3 quedó como INTENT sin COMPLETE, que es terminal (`TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW`).
- La RAM libre en el momento del fallo rondaba 2,2 GB. No fue presión de memoria local.

[NO VERIFICADO] La causa del corte (IAP, red o carga del invitado) no se puede leer: la VM está
detenida.

[VERIFICADO en código]
- Mientras el operador sondea, el trabajo del invitado sigue corriendo. Lo acotan su propio límite
  (7200 s el estímulo, 9000 s el gate) y la ventana de acuse de 600 s tras terminar (ADR-0031).
- Un sondeo que espera más no cambia la evidencia: la prueba se descarga, se verifica por hash y se
  acusa igual que antes.

## Decisión

[DECLARADO] En `lifecycle.py`, `stimulus-status` y `gate-status` (`LONG_POLL_OPERATIONS`) reintentan
un `TransportUnavailable` con `LONG_POLL_TRANSPORT_RETRY_DELAYS_S = (15, 45, 60, 60, 60, 60, 60)`.
Son ocho intentos y 6 minutos de espera, unos 23 minutos en total con intentos lentos de ~130 s.

- El calendario empieza con el corto (15, 45). Todo corte que el corto ya sobrevivía se resuelve
  igual, sin regresión.
- Después prueba cada 60 s. Así el dueño ve pronto el fin de un corte y puede recolectar dentro de la
  ventana de acuse de 600 s, que el invitado abre al terminar el trabajo (ADR-0031). Una revisión
  independiente descartó una primera versión con esperas de 120 a 300 s porque retrasaba esa
  detección.
- El resto de las operaciones reintentables conserva (15, 45).
  - `preflight` debe dejar READY dentro de 900 s.
  - La recolección idempotente sigue su calendario de ADR-0030.
- Las operaciones con efectos siguen sin reintento.
- No cambian `series`, el gate, la ventana de acuse ni la contabilidad de reservas.

## Límites conocidos

[DECLARADO]
- Un corte que coincida con el fin de un boot sigue acotado por la ventana de acuse de 600 s. Si el
  invitado se apaga sin acuse, el boot queda terminal como antes.
- Si el invitado muere, el sondeo tarda hasta unos 23 minutos más en rendirse. El costo es tiempo de
  reserva, ya cubierto por el cálculo de exposición.
- Los plazos del dueño (7200 s en `series`, 9000 s en el gate) se comprueban entre sondeos. Una
  llamada que reintenta puede pasarlos por unos 23 minutos antes del `TimeoutError`. La VM sigue
  acotada por su límite nativo de 10 800 s.
- No se ha probado en la nube un corte real con el calendario nuevo.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_long_poll_retry.py`:
  - los dos sondeos largos sobreviven siete fallos seguidos y se rinden al octavo, con las esperas
    exactas;
  - el prefijo es el calendario corto y ninguna espera pasa de 60 s;
  - las demás operaciones reintentables conservan (15, 45).
- RED: error de importación antes del cambio. GREEN: pasan junto con las suites de transporte de los
  intentos 10 y 11 y la del ciclo de vida del operador.
- El test de rendición del calendario corto (`test_short_schedule_op_gives_up_after_bounded_attempts`)
  ahora usa `i7-smoke-observe`, porque `gate-status` sigue el calendario largo.
- Revisión de código independiente:
  - cerró el hallazgo HIGH de la ventana de acuse con esperas de 60 s;
  - el hallazgo MEDIUM de los plazos del dueño queda en límites conocidos;
  - verificó que ninguna operación con efectos reintenta y que READY sigue dentro de 900 s.
