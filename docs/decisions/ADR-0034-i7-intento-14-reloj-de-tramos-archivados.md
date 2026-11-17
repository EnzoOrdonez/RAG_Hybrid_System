# ADR-0034: Resultado del intento 14, reloj de los tramos archivados del gate y lecturas de mantenimiento

[DECLARADO] Registrado el 2026-11-15 sobre la imagen de 4ef8d35 y el anexo publicado en 996db48
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Yap te autprizo todo pero termina el despliegue y todo lo qie haga falta para que este
proyecto ya sea palapable y se cumpla el objetivo inicial"), dentro del tope de USD 20 + 3 por
intento. El cambio toca `scripts/`, así que exige commit, suites, `local-readiness`, raíz y
cpu-build nuevos (intento 15).

## Resultado del intento 14

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK. |
| 2 gpu-restore | Capacidad L4 en la primera ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en 996db48 y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY`. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**: siete dominios completos, 0 aciertos. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | **ACCEPTED**: 12/12 combinaciones, 144 llamadas, 12 boots en frío, 3 h 20 min. Reserva 1 cerrada por guard. |
| 8b gate | Cinco rechazos de capacidad liberados en reserve(5), sin consumir el intento (ADR-0026). La reserva del nombre reutilizado no fue rechazada por nombre (ADR-0032). En la ronda 5 se retuvo la reserva 2 y el gate corrió: 140/140 llamadas y después **`INCOMPLETE_TERMINAL`** (`ValueError` en la verificación del invitado). Recibo `TERMINAL_DOWNLOADED_NOT_GO`. |
| 9 close | `CLOSED`: VM detenida y verificada, reserva 2 borrada y verificada, temporales CPU ausentes. |

Gasto: USD 9,39 (exposición conservadora, no factura). **No hay GO ni NO-GO**: la prueba terminal no
es una medición del gate. Con fines solo descriptivos, las 140 llamadas tardaron p50 23,1 s,
p95 35,1 s y máximo 35,9 s; esos valores no se citan como resultado.

## Problema 1: el gate juzgaba los tramos archivados con el reloj de cierre

[VERIFICADO en código y en la prueba terminal]
- `i7_gate_host.verify` vuelve a evaluar la telemetría al cerrar. Pasaba `reference_time=closing`
  (el reloj de la última muestra de toda la corrida) a tres llamadas: la telemetría continua, el
  tramo de admisión y los tramos de las ventanas 21 y 81.
- `assess` compara el lease de residencia de Ollama de cada fila con su propio instante, salvo la
  **última fila de la lista recibida**, que se compara con ese reloj + 180 s. El lease dura unos
  30 min.
- El gate duró unos 62 min tras la admisión. Al cierre (08:46:51Z), la última muestra de admisión
  (07:44:24Z, lease hasta 08:13:08Z) quedaba 36,7 min vencida; la última de la ventana 21, 27,6 min;
  la de la ventana 81, 1,1 min. Resultado: `residency_lease` y `GATE_TELEMETRY_REJECTED`.
- En vivo esos tramos pasaron: `idle_admission` los evalúa justo después de su última muestra.
- El defecto es determinista: todo gate que dure más de unos 27 min después de su admisión fallaba
  su propia verificación, también al revalidarse en el dueño (`i7.py`, recolección del gate).
- La réplica fuera de línea de las comprobaciones que no dependen del inventario (frontera,
  backups, runners, secuencias, ventanas, censo de 140) pasa sobre la prueba terminal.
- El estímulo no tiene el problema: sus admisiones son en frío, sin modelo residente, y no evalúan
  lease en el tramo admitido.

## Problema 2: la limpieza técnica no reintentaba sus lecturas

[VERIFICADO en recibos]
- Tras recolectar la prueba terminal, `cleanup` cerró, respaldó, restauró y retiró la sesión
  representativa. En `purge-study`, la llamada `download-disk` falló: la autenticación SSH sin carga
  (`--command=true`, límite 90 s) venció. Fue un corte de transporte IAP en una operación sin
  reintento.
- Las sesiones sintéticas del gate (códigos P9000xx) quedaron sin purgar en el disco GPU y en el
  bucket del período. No son datos de personas. `technical-cycle` exige otro boot pagado y no se
  ejecutó; decide Enzo.

## Decisión

[DECLARADO]
1. En `verify`, la telemetría continua se sigue evaluando con el reloj de cierre. Los tramos de
   admisión y de ventanas se evalúan con el reloj de **su propia última muestra** (`observed_at`),
   que es el instante en que se evaluaron en vivo. Un lease que ya era corto cuando se muestreó se
   sigue rechazando. No cambian el criterio, los umbrales ni la regla de residencia.
2. `inventory` y `download-disk` (`MAINTENANCE_READ_OPERATIONS`) reintentan un
   `TransportUnavailable` con el calendario corto (15, 45) s. Solo leen bajo mantenimiento y el
   invitado vuelve a comprobar que el inventario coincide con el plan. Las llamadas con efectos
   (`maintenance`, `i7-technical-close`, `restore`, `clean-disk`) siguen con un intento.
3. No cambian la verificación del estímulo, el gate (`collect`, `GateRun`, p95), la ventana de acuse
   ni la contabilidad de reservas.

## Límites conocidos

[DECLARADO]
- La réplica del tramo usa el instante de su última muestra; la evaluación en vivo usó el reloj
  unos instantes después. La diferencia es de segundos frente al margen de 180 s.
- Un corte de transporte de más de unos 5 min en la limpieza sigue dejando sesiones sintéticas sin
  purgar. La medición ya está recolectada en ese punto.
- El instante de la réplica lo declara la propia prueba del invitado (`at`), igual que antes el
  reloj de cierre. No se comprueba que `at` avance con `monotonic_s`.
- No se ha probado en la nube un gate completo con la corrección.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_gate_slice_clock.py`, con relojes que avanzan y un lease de 300 s:
  - un gate más largo que el lease verifica GO (RED: reproduce `GATE_TELEMETRY_REJECTED`);
  - en la admisión y en las ventanas 21 y 81, `hot_reasons` da `residency_lease` con el reloj de
    cierre y ninguna razón con el de su última muestra;
  - las comprobaciones que solo hace el tramo (CPU ocupada en la ventana inactiva) siguen
    rechazando: `GATE_TELEMETRY_REJECTED` y `WINDOW_PREFLIGHT_DIFFERS`;
  - un lease que ya era de 60 s al muestrearse se rechaza por la telemetría continua;
  - `observed_at` exige un reloj con zona horaria.
- `tests/test_study_i7_maintenance_read_retry.py`: las dos lecturas sobreviven dos fallos con
  esperas (15, 45); las llamadas con efectos mantienen un intento. RED: error de importación.
- El test de conjunto exacto de `test_study_i7_attempt11_fixes.py` incluye ahora las dos lecturas.
- Las suites de I7 relacionadas (gate, admisión, ciclo técnico, almacenes gestionados, evidencia
  del estímulo, intentos 10 a 12 y sondeos largos) pasan.
- Revisión de código independiente: aprobada sin hallazgos CRITICAL ni HIGH. Confirmó que la
  réplica no debilita ninguna comprobación, que el estímulo no tiene el defecto (admisión en frío)
  y que reintentar las dos lecturas no altera la transacción de borrado. Sus dos hallazgos MEDIUM
  (aislar el tramo en los tests y probar `observed_at`) quedaron cubiertos.
