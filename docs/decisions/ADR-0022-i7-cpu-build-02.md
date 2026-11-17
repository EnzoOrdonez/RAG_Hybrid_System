# ADR-0022: Incidente cpu-build-02 y nombres por período

[DECLARADO] Aceptado el 2026-10-29 para corrección local sobre
741f66ec3ac80afb564df5070c6e66ca5dd4ca5c. Tarea F-020. No autoriza push, build
ni gasto: cada uno requiere un OK separado de Enzo. Complementa
[ADR-0021](ADR-0021-i7-gpu-restore-01.md).

## Incidente

[VERIFICADO por recibos] El intento 2 de la Fase B (root de período 09e7ae37e3b4)
lanzó cpu-build a las 19:00:31Z, que terminó con exit 1 a las 19:19:45Z:
`ValueError: Build manifest commit differs` (`cpu_build.py`, `collect`).

Causa:
- el prefijo técnico era `iteration4/iteration5/<nombre del directorio>/<label>`;
- en I7 ese directorio siempre se llama `cpu`, así que los intentos 1 y 2 compartían `iteration4/iteration5/cpu/finali7`;
- el primer sondeo leyó el manifest del intento 1, con commit 468eb25.

[VERIFICADO por lectura de GCP] La subida del intento 2 no sobrescribió nada:
- `create_technical` usa `ifGenerationMatch=0`;
- los 67 objetos del prefijo son de 2026-10-29T00:42Z, es decir, del intento 1.

La VM CPU se apagó sola a las 19:31:32Z.
Evidencia privada: `C:/CloudRAG/i7-preparacion-20261029-741f66e/incident-cpu-build-02/`.

[VERIFICADO] Los recursos CPU también tenían nombre y marcador fijos:
- nombres: `cloudrag-i5-restore-i7cpu01`, `-boot` e `-iap`;
- marcador: `CloudRAG-I5-restore-cpu-i7cpu01`.

Por eso un resto de otro intento podía chocar, y el marcador no probaba el período.
Con OK de Enzo se borraron uno a uno:
- la VM, el disco de 180 GB y la regla IAP del intento 2;
- antes de cada borrado se exigió un describe con el id registrado en el root, el marcador y una creación posterior al inicio del reloj.

Quedó solo el snapshot limpio del intento 1, `cloudrag-i7-clean-867c595a9e19`.

## Decisión

[DECLARADO] Todo nombre que escriben las etapas vivas lleva el período:
- **Prefijo técnico del build:** `iteration4/iteration5/i7-<period_id[:12]>/finali7` (`i7_cpu_pipeline.build_scope`).
  - `cpu_build.Build` recibe `scope` opcional.
  - Sin `scope` conserva el nombre del paquete, así que I5 no cambia.
- **CPU:** label `i7cpu-<period_id[:8]>` (`i7_cpu_pipeline.cpu_label`).
  - La VM, el disco y la regla quedan como `cloudrag-i5-restore-i7cpu-<p8>`.
  - El marcador queda como `CloudRAG-I5-restore-cpu-i7cpu-<p8>`.
- **Limpieza:** `cleanup_cpu` exige el nombre exacto del período en vez del prefijo `cloudrag-i5-restore-i7cpu`.

El resto ya llevaba el período, el boot o el id de VM, o vive en discos y roots nuevos:
- snapshots `stage`/`service`/`clean`, la VM GPU y la reserva;
- `periods/<period_id>` y `failed-boots/<instance_id>`;
- la tarea programada y la unidad de gate.

## Verificación

[VERIFICADO] `tests/test_study_i7_period_names.py`:
- **En 741f66e:** 7 fallos y 2 pases, los de la guarda heredada y del CPU del propio período.
- **Con el cambio:** 149 pruebas pasan en los módulos afectados.

Mutantes, aplicados en una copia y no en el worktree:
- prefijo compartido restaurado: 1 fallo;
- label CPU fijo `i7cpu01`: 9 fallos;
- limpieza por prefijo: 1 fallo.

Todos se restauraron con el mismo SHA-256.

[DECLARADO] Consecuencias:
- el commit nuevo exige nuevas suites Windows/Linux/I7, un `local-readiness.json` nuevo, push con OK, root nuevo y `--check`;
- las rutas privadas antiguas no se renombran;
- F-020 sigue `active`, `passes=false`.
