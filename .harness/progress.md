[Claude] | 2026-11-13 | rama `fix/interview-readiness` @ base `a81038c` | tarea F-020 | modo: ejecución (intento 9 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2191 pasan, 0 fallan, 6 omitidas) | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0028, adr-log, runbook Fase B (recibo del navegador y aviso en 8a), F-020 | Sin verificar: latencia del despliegue (el gate no corrió)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 9 es `83382fb`; el anexo publicado y anclado está en `a81038c`.
  - `main` sigue detrás de esta rama.
- **Intento 9, del 13/11** (raíz `C:\CloudRAG\iteration9-faseB-20261113-83382fb`, cerrada; detalle en ADR-0028):
  - **Pasos 1 a 7:** cpu-build, gpu-restore (L4 en la 7.ª ronda), ready, tls-three, contexts, anexo, smoke, restore y withdraw-purge salieron OK.
  - **privacy: INCOMPLETE.** 0 aciertos, pero el escáner del invitado no completó 4 de 7 dominios (privacy-09).
  - **Estímulo programado (8a):** falló en el boot 1 (stimulus-09) y guard borró la reserva.
  - **gate:** no corrió.
  - **close:** CLOSED.
  - **Gasto:** ≤ USD 10 del tope de USD 20 + 3.
- **Bitácora privada:** `C:\CloudRAG\i9-preparacion-20261113-83382fb\BITACORA_NOCTURNA.md`, indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas.
  - Quedan detenidos y con costo de disco la raíz 1 abortada (VM CPU, dos discos y snapshot stage) y la raíz 2 (VM GPU, su disco y el snapshot limpio).
  - También quedan los snapshots previos y la IP estática.
  - Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (desde 83382fb, sin PR): Ruff en 9 scripts, medición de F-016 y la tarea F-021.

## Qué sigue (cada paso con su OK)

1. **Recursos retenidos en GCP:** Enzo decide qué borrar.
   - Las raíces 1 y 2 no sirven para el intento 10, que necesita imagen nueva.
   - El snapshot `i6-cierre` es la base del cpu-build y no se borra.
2. **Correcciones con TDD** y revisión independiente, en una rama desde esta:
   - **stimulus-09:** `series` debe detener la VM y aplicar el perfil del estímulo antes del primer boot, con un test de withdraw-purge seguido de series;
   - **privacy-09:** registrar el código fijo del error y tratar sockets y archivos que desaparecen;
   - reintento acotado del bridge ante errores de transporte;
   - F-021.
3. **Intento 10** con presupuesto nuevo de Enzo: commit, suites, `local-readiness`, raíz y cpu-build nuevos.
   - Mientras stimulus-09 siga abierto, el estímulo corre por el paso `stimulus` en primer plano.
4. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
5. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** el 13/11 hubo 6 rechazos en 3 h. La zona sigue fijada por Enzo (ADR-0026).
- **Comandos `i7` sobre raíces de 83382fb:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Este traspaso toca `.harness/`, así que para operar esas raíces hace falta un worktree en `a81038c`.
- **RAM local:** con el navegador personal abierto, la guarda de 1,5 GB libres frenó lanzamientos 11 veces.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
