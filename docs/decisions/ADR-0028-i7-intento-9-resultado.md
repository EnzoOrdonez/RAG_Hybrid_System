# ADR-0028: Resultado del intento 9 de la Fase B y defectos nuevos

[DECLARADO] Registrado el 2026-11-13 sobre la imagen de 83382fb y el anexo
publicado en a81038c (`fix/interview-readiness`). Tarea F-020. Claude ejecutó
el intento con el control delegado por Enzo el 13/11 (plan nocturno), dentro
del tope de USD 20 + 3 (ADR-0027). Las correcciones de código esperan un OK
separado de Enzo.

## Resultado

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | Raíz 1 abortada: el create del disco de servicio salió con exit 1 por un corte de red aunque el disco se creó (`reconcile` sin mutaciones). Raíz 2: OK en 59 min. |
| 2 gpu-restore | Seis rechazos de capacidad L4 liberados (ADR-0026), cada 30 min. Capacidad en la séptima ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY`. |
| 5a, 5b | smoke (`BROWSER_EXPORT_VERIFIED`) y restore: exit 0. El primer lanzamiento del smoke falló en local, antes de tocar la nube, por un recibo de navegador desactualizado (defecto 4). |
| 6 privacy | Primer intento: exit 1 al vencer un bridge SSH por IAP (`SSH_TRANSPORT_REJECTED`). Reintento único: exit 0 con estado **INCOMPLETE**. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | Fallo terminal en el boot 1 (defecto 1). La reserva 1 se borró. |
| 8b gate | No corrió: exige `stimulus-acceptance.json` ACCEPTED. |
| 9 close | `CLOSED`: VM GPU detenida y verificada, reserva cerrada, temporales CPU ausentes. |

Detalle de privacy en el reintento:
- 0 aciertos;
- transporte del canario confirmado en los tres canales (contexto del navegador, IP y UA por WebSocket);
- bucket, journald y serial completos;
- caddy, app y runner con `ValueError` antes del primer archivo, y backups con `OSError` tras 67 archivos.

El criterio del paso 6 exige los siete dominios completos, así que no hay PASS.

Gasto: raíz 2 USD 6,07 (exposición conservadora, no factura) y raíz 1 abortada
como máximo USD 3,95. Total del intento ≤ USD 10. La latencia del despliegue sigue sin medir.

## Defectos

[VERIFICADO en código]

1. **stimulus-09.** El paso 8a (`i7 --live schedule`, que ejecuta `i7_workflows.series`)
   llama `operator.start()` sin detener antes la VM. `Operator.start` exige
   TERMINATED y, si no, lanza `STOPPED_VM_AND_SPECIFIC_RESERVATION_REQUIRED`.
   - Tras un privacy correcto la VM queda RUNNING para withdraw-purge, y withdraw-purge no la detiene.
   - El `finally` de `series` hace stop y `guard()`, y guard borra la reserva.
   - El paso en primer plano (`i7_phase_b stimulus`) llama antes `switch_profile`, que empieza con `operator.stop()`.
   - `series` tampoco aplica `switch_profile('stimulus')`.
   - Ningún test cubre la secuencia withdraw-purge seguida de series.
2. **privacy-09.** El escáner del invitado (`i7_privacy.host_sources`) no completó cuatro de siete dominios en el despliegue real, aunque pasaba con los dobles locales.
   - Solo registra el tipo de excepción, así que la causa exacta no se puede leer.
   - [DECLARADO, sin verificar] Candidatas: `docker ps` o `docker inspect` fallido, `UpperDir` ausente, o un montaje RW que no es directorio. En backups, un archivo que desaparece durante el recorrido o un socket.
3. **Transporte.** Un solo bridge SSH fallido durante el sondeo `i7-smoke-observe` aborta el paso completo, sin reintento acotado.
4. **Runbook.** Ordenaba copiar el `browser-validation.json` de la validación de estabilidad del 16/10.
   - Desde 3f12fd9 (05/11) ese recibo no coincide con la fuente del automatizador.
   - El recibo válido es el de la verificación local del commit de la imagen, que es el precedente del intento 8.
   - Se corrige en el runbook junto con este ADR.
5. **Operación.** Claude eligió el camino 8a tras juzgar inocuo que `series` se saltara `switch_profile`. Revisó el perfil y el STOP de 10800 s, pero no vio que también detiene la VM.

## Decisión

[DECLARADO]
- **Reintento de privacy:** se reintentó una sola vez. El primer fallo fue un corte transitorio con evidencia: los SSH siguientes con la clave fijada salieron 0, y la VM quedó en STOP verificado. El paso no tiene guarda de un solo uso.
- **Estímulo pese a INCOMPLETE:** se siguió porque el gate no exige el recibo de privacy y su GO se define por latencia y validez. Privacy INCOMPLETE bloquea igual la entrada de personas.
- **Sin rescate:** no se creó otra reserva, porque superaría las 18,5 h de GPU (ADR-0025), y no se editó la raíz a mano. Se cerró con `close`.
- **Correcciones propuestas**, con TDD y revisión independiente; requieren commit nuevo, imagen nueva e intento 10:
  - detener la VM antes del primer boot de `series` y aplicar el perfil del estímulo, con un test de la secuencia withdraw-purge seguida de series;
  - registrar en privacy el código fijo del error, nunca el texto privado, y tratar sockets y archivos que desaparecen;
  - dar al bridge un reintento acotado solo para errores de transporte idempotentes;
  - adoptar la identidad tras un error de transporte en un create (F-021).
- **Recursos retenidos tras close:** decide Enzo. La lista está en la bitácora privada.

## Consecuencias

- F-020 sigue activa y las entrevistas siguen en NO-GO.
- Hasta corregir stimulus-09, el paso 8a del runbook remite al paso `stimulus` en primer plano.
- El intento 10 necesita un presupuesto aprobado por Enzo.

## Verificación

[VERIFICADO] Recibos privados de la raíz del intento 9 y bitácora privada,
indexadas en `C:\CloudRAG\LEEME_CLOUDRAG.md`. Lectura de GCP antes y después de
`close`: sin reservas y sin VMs encendidas. Este ADR y la corrección del runbook
son solo documentación; `init.sh` dio exit 0 en las tres capas antes del commit.
