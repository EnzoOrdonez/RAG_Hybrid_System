# ADR-0029: Correcciones tras el intento 9 (privacy-09, stimulus-09 y transporte)

[DECLARADO] Aceptado el 2026-11-13 sobre 313e5c3 (`fix/interview-readiness`). Tarea F-020.
Enzo delegó la decisión el 13/11 ("Haz lo necesario para cumplir el objetivo... todo depende de
ti"). El cambio toca `scripts/`, así que exige commit, suites, `local-readiness`, raíz y cpu-build
nuevos (intento 10).

## Problema

[VERIFICADO] ADR-0028 dejó tres defectos que impedían el GO o los controles:

1. **privacy-09.** El escáner del invitado no completaba caddy, app, runner ni backups.
2. **stimulus-09.** La serie programada arrancaba sin detener la VM.
3. **Transporte.** Un solo bridge SSH fallido en un sondeo abortaba el paso completo. Un estímulo
   de doce boots o un gate sondean miles de veces, así que un corte de IAP terminaba el intento sin
   repetición posible.

[VERIFICADO en un clon del disco real] Para privacy-09 se clonó el disco GPU de la raíz 2 del
intento 9. El escáner se corrió en una VM CPU temporal, sin IP pública ni cuenta de servicio, y la
salida se leyó por el puerto serie.

| Hallazgo | Detalle |
|---|---|
| Docker | 29.1.3 con el almacén de imágenes containerd (`io.containerd.snapshotter.v1`, overlayfs). |
| Capa escribible | `docker inspect` da `GraphDriver: null` en los 84 contenedores retenidos, de ahí `CONTAINER_WRITABLE_LAYER_UNRESOLVED`. |
| containerd | Con contenedores detenidos no existe el contenedor containerd. El snapshot sí existe, con el Id de docker como clave. |
| backups | Hay 10 sockets Unix en `boots/<id>/sockets/`; abrirlos da `ENXIO`. |
| Enlaces | Las capas escribibles tienen 477 enlaces simbólicos. Todos son bibliotecas NVIDIA inyectadas por el container toolkit. |

## Decisión

[DECLARADO]

1. **privacy-09** (`i7_privacy.py`):
   - Sin `UpperDir`, la capa escribible se resuelve con `ctr -n moby snapshots --snapshotter overlayfs mounts <dir> <Id>`. Debe salir exactamente un `upperdir`; si no, el dominio queda incompleto.
   - `tree_sources` lee archivos regulares y miembros tar.
   - Un enlace simbólico se escanea como su texto, sin seguirlo, sea archivo o directorio.
   - Sockets, FIFOs y whiteouts de overlay (dispositivo de caracteres 0:0) no guardan bytes: se cuentan vacíos.
   - Fallan cerrados: cualquier otro archivo especial, una raíz ausente o enlazada, un directorio ilegible y un miembro tar enlazado.
   - `scan` añade `error_code` a un dominio incompleto: el código fijo del operador o el nombre errno, nunca el texto de la excepción ni rutas.
2. **stimulus-09:**
   - `dispatch('series')` aplica `switch_profile('stimulus')`, que detiene la VM y fija el perfil, antes de la serie.
   - Antes de detener nada, la serie programada y el paso en primer plano rechazan un boot en INTENT sin COMPLETE (`TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW`). Así una serie reanudada no mata un boot en curso.
3. **Transporte:**
   - `policy.TransportUnavailable(OperatorError)`, con el mismo mensaje, para tres casos de `compute ssh`: el timeout, el rechazo del canal RPC privado y la salida 255.
   - `bridge` reintenta dos veces, tras 15 s y 45 s, solo los sondeos de lectura: `stimulus-status`, `gate-status`, `i7-smoke-observe`, `preflight` e `i7-preparation-check`.
   - Las operaciones con efectos conservan un único intento.

## Límites conocidos

[DECLARADO]
- No se escanea el destino de un enlace que apunta fuera de las raíces escaneadas. Los datos escritos en tiempo de ejecución viven en la capa escribible o en montajes RW, y esos sí se escanean.
- No se escanean nombres de archivos, directorios ni miembros tar, ni archivos comprimidos anidados. Esto ya era así antes de este ADR.
- Hay una ventana entre `lstat` y `open`. Con contenedores detenidos casi no se alcanza, y en el peor caso el escáner se cuelga, sin dar un PASS falso.
- Un 255 puede venir del comando remoto. Solo se reintentan sondeos de lectura, así que no hay daño.
- Los reintentos de `preflight` pueden alargar la espera de READY más allá de 900 s. El criterio de READY lo mide el invitado y no cambia.
- `switch_profile` sigue fuera del `finally` de `series`: si falla, la reserva no se libera sola. Ese caso va a `close`.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_attempt10_fixes.py`: 40 tests.
  - RED: primero un ImportError; luego 21 fallos por comportamiento, y 2 más tras la revisión.
  - GREEN: los 40 pasan.
- 11 mutantes detectados, con restauración byte a byte.
- Revisión independiente (ecc:code-reviewer): sin hallazgos CRITICAL ni HIGH.
  - Los dos MEDIUM se aplicaron: la prueba previa de boots a medias y los tests de enlace de directorio, miembro tar enlazado y directorio ilegible.
  - Los LOW quedan en los límites conocidos.
- El `i7_privacy.py` final sobre otro clon del disco real deja los cinco dominios locales completos: caddy 70, app 1457, runner 490, journald 1 y backups 426 archivos, con 0 aciertos.
- Los recursos de diagnóstico se borraron.
