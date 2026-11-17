# ADR-0024: Incidente cpu-build-06 y espera del arranque del invitado antes del attach

[DECLARADO] Aceptado el 2026-10-31 para corrección local sobre
92fadfef1a4bc17eba2a8873e16feb4224086fb6. Tarea F-020. El push queda autorizado
de antemano por Enzo solo si el diff stat coincide con el que revisó, las suites
dan exit 0 y es un fast-forward desde 92fadfe sin force. Ningún cpu-build sin un
OK separado. Complementa [ADR-0023](ADR-0023-i7-cpu-build-04.md).

## Incidente

[VERIFICADO por recibos] El intento 5c de la Fase B (período bfa955e8c72b) terminó
con exit 1 a las 22:55:46Z:
- el build dentro de la VM dio PASS y `clean_snapshot` creó el clon de servicio;
- `instances start`, solo con el disco de arranque, terminó a las 22:53:56Z, y
  el `attach-disk` del clon a las 22:54:05Z, que es el orden de ADR-0023;
- `rpc(mount)` devolvió exit 1 después de unos 4 s de ejecución remota, sin
  stdout. El stderr es privado.

Evidencia privada: `C:/CloudRAG/i7-preparacion-20261031c-92fadfe/incident-cpu-build-06/`.

## Sonda de solo lectura (con OK de Enzo; tres arranques breves de la VM)

[VERIFICADO]
- `journalctl --list-boots`: el arranque del fallo empezó a las 22:54:10Z,
  **después** de que terminara el attach. `instances start` responde cuando la
  instancia pasa a RUNNING, antes de que arranque el invitado, así que el kernel
  encontró el clon ya conectado. En la práctica fue un attach durante el arranque.
- En ese arranque, `/boot` se montó desde el clon: al apagar aparece
  `EXT4-fs (sdb16): unmounting`, porque fstab monta por `LABEL=BOOT`.
- Con el arranque terminado (`systemctl is-system-running` = running) y el attach
  hecho a unos 53 s de uptime:
  - `/`, `/boot` y `/boot/efi` siguieron en `sda`;
  - la guarda real dio `GUARD_OK`;
  - `mount -o ro,noload` del clon dio exit 0 y `umount` también.
- Después del attach, `/dev/disk/by-label/{BOOT,UEFI,cloudimg-rootfs}` apuntan al
  clon. Ningún montaje cambia.

[SUPUESTO, consistente con los 4 s y con el journal] La guarda
`ROOT_ON_SERVICE_CLONE` de ADR-0023 detectó `/boot` en el clon y abortó antes de
montar. El defecto es determinista: el attach llegó antes de que arrancara el
invitado.

## Decisión

[DECLARADO] `clean_snapshot` sigue este orden: `instances start` →
`wait_guest_booted` → `attach-disk` del clon → `rpc(mount)`. Para que el `finally`
también detenga la VM si la espera vence, el `try` empieza ahora antes del
`start`.

Señal de "arranque terminado":
- **Guest attribute `hostkeys/`: rechazada [VERIFICADO].** Las claves sobreviven a
  stop/start. En el 5c, `get-guest-attributes` (recibo 0160) devolvió claves a
  las 22:54:06Z, antes de que arrancara el kernel (22:54:10Z), así que no prueban
  un arranque nuevo.
- **Elegida:** la operación `boot-status` de `i7_service_disk.py`, que ejecuta
  `systemctl is-system-running --wait` en el invitado.
  - Viaja por el mismo `rpc` privado que el mount: claves del host fijadas desde
    la API, `-batch`, verificación de pines y OpenSSH fijado. No se abre ninguna
    conexión sin claves fijadas.
  - Se usa este `rpc` porque `windows_ssh.native_argv` solo acepta comandos
    remotos que empiezan con `sudo -n`.
  - sshd arranca después de `local-fs.target`.
  - `running` y `degraded` cuentan como arranque terminado; `degraded` sale con
    exit 1 de systemctl y por eso se lee el stdout.
- **Límite:**
  - host: `GUEST_BOOT_WAIT_S` = 300 s en total;
  - claves pendientes (`ReadyPending`): se reintenta cada 15 s;
  - invitado: `wait_s` acotado a 1–300 y `timeout` del subprocess;
  - cualquier otro estado lanza `GUEST_BOOT_WAIT_TIMEOUT` sin hacer el attach.
- Se mantienen la guarda `ROOT_ON_SERVICE_CLONE` y la espera `by-id` de ADR-0023.

[DECLARADO] Regla operativa: la VM CPU nunca se arranca con el clon conectado. Sigue
valiendo el análisis de ADR-0023: ningún paso posterior de `clean_snapshot`
reinicia la VM. La VM detenida del 5c tiene el clon conectado y no se arranca a
mano.

## Verificación

[VERIFICADO] `tests/test_study_i7_clean_clone_order.py`:
- **En 92fadfe:** fallan 4 tests nuevos, entre ellos el de orden: el código
  anterior hace el attach sin esperar.
- **Con el cambio:** pasan todos.
- **Mutantes en copia, fuera del worktree:**
  - M1 (attach antes de la espera): 2 fallos;
  - M2 (sin límite de tiempo): 1 fallo, por el centinela de bucle infinito.

[DECLARADO] Consecuencias:
- el commit nuevo exige suites Windows/Linux/I7, un `local-readiness.json` nuevo,
  el push condicionado y una raíz nueva con `--check`;
- los recursos del 5c no se tocan hasta que el siguiente intento esté en marcha;
- F-020 sigue `active`, `passes=false`.
