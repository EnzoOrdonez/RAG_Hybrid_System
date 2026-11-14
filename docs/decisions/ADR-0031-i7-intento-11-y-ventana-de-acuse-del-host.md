# ADR-0031: Resultado del intento 11 y ventana de acuse del host

[DECLARADO] Registrado el 2026-11-14 sobre la imagen de 54e0518 y el anexo publicado en 83036b6
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Haz lo necesario para cumplir el objetivo... todo depende de ti"; "sigue hasta completar el
objetivo"), dentro del tope de USD 20 + 3 por intento. El cambio toca `scripts/`, así que exige
commit, suites, `local-readiness`, raíz y cpu-build nuevos (intento 12).

## Resultado del intento 11

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK en 50 min. |
| 2 gpu-restore | Dos rechazos de capacidad L4 liberados (ADR-0026). Capacidad en la tercera ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en 83036b6 y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY` sin bloqueos. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**: 0 aciertos y los siete dominios completos. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | Paso `stimulus` en primer plano. Boots 1 a 4 COMPLETE. Fallo terminal en el boot 5. Guard detuvo la VM y borró la reserva 1. |
| 8b gate | No corrió: exige `stimulus-acceptance.json` ACCEPTED. |
| 9 close | `CLOSED`: VM detenida y verificada, reserva cerrada, temporales CPU ausentes. |

Gasto: USD 6,91 (exposición conservadora, no factura). La latencia del despliegue sigue sin medir.

## Problema

[VERIFICADO en recibos] El boot 5 repitió el patrón del boot 2 del intento 10.
- `collect('stimulus')` descargó la prueba, la verificó y la guardó en local.
- Después, `bridge(stimulus-ack)` lanzó `TransportUnavailable` en los tres intentos de ADR-0030.

[VERIFICADO en código] La causa no es un corte transitorio de IAP:
- Al terminar el trabajo, `stimulus_host.main` daba al dueño 60 s para descargar, verificar y
  acusar la prueba.
- Después, la unidad salía y su `ExecStopPost=/usr/sbin/shutdown -h now` apagaba la VM.
- Los reintentos de ADR-0030 llegaban a una VM apagándose o ya detenida.

[VERIFICADO en recibos] El tiempo real del dueño:
- En los boots 1 a 4, de guardar la prueba a cerrar el acuse pasaron 34-38 s. Se mide con la hora
  del archivo privado y la `recorded_utc` del recibo.
- Antes de eso van la detección por `stimulus-status` y la descarga. Cada `bridge` encadena varias
  llamadas de gcloud.
- El total ronda los 60-75 s, así que cada boot jugaba una carrera contra el apagado.
- ADR-0030 atribuyó el fallo del intento 10 a un corte del túnel. Este ADR corrige ese diagnóstico.
  Su reintento sigue siendo útil para cortes reales.

[VERIFICADO en código] En el gate, `i7_gate_host.main` esperaba hasta 600 s el acuse final. Al verlo,
salía de inmediato hacia el apagado, así que la respuesta del `bridge(gate-finalized)` también
podía perderse.

## Decisión

[DECLARADO] `host_runtime.await_owner_ack` concentra la espera de los dos hosts:
- **Ventana:** `OWNER_ACK_WINDOW_S = 600` s para el estímulo, igual que la del gate. Sigue acotada
  por el límite independiente de cada trabajo (7200 s y 9000 s) y por `RuntimeMaxSec` de la unidad.
- **Gracia:** tras ver el acuse, el host espera `POST_ACK_GRACE_S = 30` s antes de salir, para que la
  respuesta del `bridge` salga antes del apagado. El dueño detiene la VM en cuanto recibe el acuse.
- `stimulus_host.await_download_ack` e `i7_gate_host.await_finalization_ack` usan esa función.
- No cambian la VM (`--max-run-duration` de 10 800 s), la unidad ni el cálculo de reservas
  (`native_stop_s + 60` en `start`).

## Límites conocidos

[DECLARADO]
- Si el dueño desaparece, la VM corre como mucho 10 min más antes del apagado propio. Son unos
  USD 0,12 a precio de lista, y el guard del dueño sigue activo.
- Un dueño que tarde más de 10 min en recolectar sigue perdiendo el boot.
- La causa queda verificada por el código y los tiempos de los recibos, no por registros del
  invitado. El disco se conserva.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_attempt12_fixes.py`: 7 tests con reloj falso.
  - Cubren la ventana compartida de 600 s, un dueño que acusa a los 75 s y la gracia tras el acuse.
  - También cubren el apagado sin acuse y el respeto del límite independiente, en el estímulo y en
    el gate.
  - RED: 7 fallos. GREEN: todos pasan, junto con las suites del estímulo, el gate, el runtime del
    host y los intentos 10 y 11.
