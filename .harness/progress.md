[Claude] | 2026-11-13 | rama `fix/interview-readiness` @ base `ffe3ab0` | tarea F-020 | modo: ejecución (intentos 10 y 11 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0030 (resultado del intento 10 y reintento de la recolección idempotente), lifecycle.py, tests, adr-log, F-020 | Sin verificar: latencia del despliegue (el gate no corrió)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 10 fue `1154140` y su anexo `ffe3ab0`. El commit de ADR-0030 es la imagen del intento 11.
  - `main` sigue detrás de esta rama.
- **Intento 10, del 13/11** (raíz `C:\CloudRAG\iteration10-faseB-20261113-1154140`, cerrada; detalle en ADR-0030):
  - **Pasos 1 a 7:** todos en exit 0. L4 en la 4.ª ronda de capacidad.
  - **privacy: PASS.** 0 aciertos y todos los dominios completos; las correcciones de ADR-0029 funcionan en la nube.
  - **Estímulo (8a), paso en primer plano:** boot 1 COMPLETE; el boot 2 terminó por un solo corte del túnel IAP en `stimulus-ack` (stimulus-10).
  - **gate:** no corrió. **close:** CLOSED. **Gasto:** USD 6,17.
- **Corrección ADR-0030:** `bridge` reintenta también la recolección idempotente (`*-evidence`, `*-ack`, `gate-terminal-evidence`, `gate-finalized`). TDD con 22 tests.
- **Bitácora privada:** `C:\CloudRAG\i10-preparacion-20261113\BITACORA_I10.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas. Quedan detenidos y con costo de disco la VM GPU y el disco del intento 10, los snapshots previos y la IP estática. Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Intento 11** sobre la imagen de ADR-0030, con el mismo tope de USD 20 + 3 por intento y el control delegado por Enzo:
   evidencia local nueva, raíz nueva, cpu-build, capacidad, tramo 1, anexo, tramo 2a, estímulo en primer plano, gate y close.
2. **Recursos retenidos en GCP:** Enzo decide qué borrar. El snapshot `i6-cierre` es la base del cpu-build y no se borra.
3. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13/11 (intentos 9 y 10). La zona sigue fijada por Enzo (ADR-0026).
- **Transporte IAP:** los cortes transitorios existen. ADR-0029 y ADR-0030 cubren los sondeos y la recolección; un corte en `stimulus-start` o `gate-start` sigue siendo terminal.
- **Comandos `i7` sobre una raíz:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Un traspaso que toca `.harness/` va después del close.
- **RAM local:** la guarda de 1,5 GB libres frena lanzamientos con el navegador personal abierto.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS en la misma raíz y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
