# ADR-0030: Resultado del intento 10 y reintento de la recolección idempotente

[DECLARADO] Registrado el 2026-11-13 sobre la imagen de 1154140 y el anexo publicado en ffe3ab0
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Haz lo necesario para cumplir el objetivo... todo depende de ti"; "sigue hasta completar el
objetivo"), dentro del tope de USD 20 + 3 por intento. El cambio toca `scripts/`, así que exige
commit, suites, `local-readiness`, raíz y cpu-build nuevos (intento 11).

## Resultado del intento 10

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK en 60 min. |
| 2 gpu-restore | Tres rechazos de capacidad L4 liberados (ADR-0026), cada 30 min. Capacidad en la cuarta ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en ffe3ab0 y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY` sin bloqueos. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**: 0 aciertos y todos los dominios completos. Las correcciones de ADR-0029 quedan validadas en la nube. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | Paso `stimulus` en primer plano. Boot 1 COMPLETE. Fallo terminal en el boot 2 (stimulus-10). Guard detuvo la VM y borró la reserva 1. |
| 8b gate | No corrió: exige `stimulus-acceptance.json` ACCEPTED. |
| 9 close | `CLOSED`: VM detenida y verificada, reserva cerrada, temporales CPU ausentes. |

Gasto: USD 6,17 (exposición conservadora, no factura). La latencia del despliegue sigue sin medir.

Antes del estímulo, el lanzador se negó a correr por una tarea programada `CloudRAG-I7-*` que
quedaba del intento 9. Se exportó su XML al archivo privado y se retiró; no hubo gasto.

## Problema (stimulus-10)

[VERIFICADO en recibos] En el boot 2, `collect('stimulus')` descargó la prueba, la verificó, la
guardó en local y la releyó. Después el `bridge` de `stimulus-ack` lanzó `TransportUnavailable`.

- Una llamada `bridge` es una cadena: claves de host del invitado, autenticación con un comando
  fijo, `--dry-run`, `describe` y, al final, OpenSSH nativo por IAP con el comando real.
- Todas las etapas previas salieron con exit 0. Solo la conexión final salió con 255 a los 14 s.
  OpenSSH da 255 cuando falla el túnel o la conexión, normalmente antes del comando remoto.
- ADR-0029 reintenta solo los sondeos de lectura. El ACK tenía un único intento, así que un corte
  transitorio de IAP terminó un estímulo de doce boots. El gate tiene la misma recolección al final
  de un intento terminal que no se repite.

[VERIFICADO en código] En el invitado, la recolección es idempotente:

- `stimulus-evidence`, `gate-evidence` y `gate-terminal-evidence` solo leen y devuelven la prueba.
- `stimulus-ack`, `gate-ack` y `gate-finalized` comparan el hash con `result.json` o `terminal.json`
  y reescriben el mismo archivo de acuse con el mismo contenido. Un hash distinto se rechaza.

## Decisión

[DECLARADO] `RETRYABLE_BRIDGE_OPERATIONS` suma las seis operaciones de recolección idempotente:
`stimulus-evidence`, `stimulus-ack`, `gate-evidence`, `gate-ack`, `gate-terminal-evidence` y
`gate-finalized`. Usan los mismos dos reintentos de ADR-0029, tras 15 s y 45 s.

- `stimulus-start` y `gate-start` registran un trabajo y rechazan un segundo registro, así que
  siguen con un único intento.
- Las demás operaciones con efectos (`stop`, `restore`, `export`, `i7-privacy`, `i7-contexts`)
  tampoco cambian.

## Límites conocidos

[DECLARADO]
- Si el ACK llegó al invitado y solo se perdió la respuesta, el reintento reescribe el mismo acuse.
  No hay daño.
- Tres fallos de transporte seguidos en la misma operación siguen terminando el paso, igual que en
  ADR-0029.
- Un corte durante `stimulus-start` o `gate-start` sigue siendo terminal. Volver a ese paso exige
  revisión humana del registro en el invitado.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_attempt11_fixes.py`: 22 tests.
  - RED: 13 fallos (los reintentos y el conjunto exacto).
  - Los 9 restantes ya pasaban: 7 de idempotencia del invitado, que documentan lo que hace seguro
    el reintento, y 2 de arranques que siguen sin reintento.
  - GREEN: todos pasan.
- `tests/test_study_i7_attempt10_fixes.py` deja de fijar `stimulus-ack` y `gate-finalized` como
  operaciones sin reintento.
