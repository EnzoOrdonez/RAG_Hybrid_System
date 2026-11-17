# ADR-0023: Incidente cpu-build-04 y clon de servicio conectado en caliente

[DECLARADO] Aceptado el 2026-10-31 para corrección local sobre
421695c349c1d9eb3ebe01881885d54ffe115201. Tarea F-020. El push queda autorizado
de antemano por Enzo solo si el diff coincide con el que revisó, las suites dan
exit 0 y es un fast-forward desde 421695c sin force. Ningún build ni gasto sin un
OK separado. Complementa [ADR-0022](ADR-0022-i7-cpu-build-02.md).

## Incidente

[VERIFICADO por recibos] El intento 4 de la Fase B (período 5e9d9dac3ca7) terminó
con exit 1 a las 15:06:05Z:
- el build dentro de la VM dio PASS, con su evidencia descargada y verificada;
- `clean_snapshot` creó el snapshot de staging y el disco de servicio, conectó el
  clon a la VM CPU detenida y la arrancó;
- la primera llamada `rpc(mount)` por SSH pinneado devolvió un código distinto de
  0, sin stdout. El stderr es privado por diseño y no se guarda.

El intento 1 (468eb25) siguió la misma secuencia y el mount pasó.
Evidencia privada: `C:/CloudRAG/i7-preparacion-20261031-421695c/incident-cpu-build-04/`.

## Sonda de solo lectura (con OK de Enzo; VM arrancada unos 3 min)

[VERIFICADO]
- El disco de arranque (`sda`) y el clon (`sdb`) tienen la misma UUID ext4, la
  misma PARTUUID y la misma LABEL (`cloudimg-rootfs`) en la raíz. Lo mismo pasa
  con las particiones 14, 15 (UEFI) y 16 (BOOT).
- El kernel arranca con `root=PARTUUID=…` y fstab monta por `LABEL=`. Ninguno de
  los dos identifica un único disco cuando hay dos conectados.
- En el arranque de la sonda, `/` se montó desde `sdb1`, que es el clon.
- En el arranque del fallo, `/boot` era `sdb16`, y
  `mnt-cloudrag-i7-service.mount` llegó a estar montado.
- `blkid` informa `TYPE="ext4"`, así que no es xfs y no hace falta `nouuid`.

[SUPUESTO, consistente con el journal y el código] Montar el clon, que ya servía
de raíz, en un directorio de ese mismo sistema de archivos deja el mismo
`st_dev`. Por eso `Path.is_mount()` dio False y `verify_mount` lanzó
`OWN_SECONDARY_SERVICE_MOUNT_REQUIRED`.

El defecto es determinista y el resultado depende del orden en que se enumeran
los discos. Cuando gana el clon, el arranque escribe además en el disco que debía
quedar limpio.

## Decisión

[DECLARADO]
- **Orden** (`i7_cpu_pipeline.clean_snapshot`): primero `instances start` con solo
  el disco de arranque, después `instances attach-disk` del clon en caliente y,
  al final, el primer SSH de mount.
- **Invitado** (`i7_service_disk`):
  - `wait_target` ejecuta `udevadm settle` y espera, con un límite de 60 s, al
    symlink `by-id`; si vence, lanza `SERVICE_DISK_BY_ID_TIMEOUT`;
  - `refuse_root_on_clone` lanza `ROOT_ON_SERVICE_CLONE` si `/`, `/boot` o
    `/boot/efi` están en un dispositivo del clon, comparando `st_dev` con `st_rdev`;
  - la guarda corre antes del mount (`mount_service`) y después, como primer
    chequeo de `verify_mount`. Así aplica a mount, archive, sanitize,
    export-images y unmount.

[VERIFICADO por lectura] Ningún paso posterior de `clean_snapshot` vuelve a
arrancar la VM CPU:
- después del mount solo hay archive, export-images, sanitize, unmount, stop,
  describe, el snapshot clean y el stop del `finally`;
- `cleanup_cpu` exige TERMINATED;
- los demás `instances start` son anteriores (`build()`) o de otras VMs.

Riesgo que queda: si algo falla después del attach, la VM queda detenida con el
clon conectado. Solo un arranque manual reproduciría la ambigüedad.

## Verificación

[VERIFICADO] `tests/test_study_i7_clean_clone_order.py`:
- **En 421695c:** 5 fallos y 1 pase (el verificador con el mutante invertido). El
  test de orden falla en `start < attach < ssh`, porque el código anterior es el
  mutante real.
- **Con el cambio:** pasan los 6, más `test_study_i7_phase_b.py`.

[DECLARADO] Consecuencias:
- el commit nuevo exige nuevas suites Windows/Linux/I7, un `local-readiness.json`
  nuevo, el push condicionado y una raíz nueva con `--check`;
- la raíz preparada `iteration7-faseB-20261031b-421695c` queda obsoleta y no se
  borra;
- el intento 4 se cierra recurso por recurso, con OK, después de que arranque el
  intento 5;
- F-020 sigue `active`, `passes=false`.
