[Claude] | 2026-11-14 | rama `fix/interview-readiness` @ base `83036b6` | tarea F-020 | modo: ejecución (intentos 11 y 12 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0031 (resultado del intento 11 y ventana de acuse del host), host_runtime.py, stimulus_host.py, i7_gate_host.py, tests, adr-log, F-020 | Sin verificar: latencia del despliegue (el gate no corrió)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 11 fue `54e0518` y su anexo `83036b6`. El commit de ADR-0031 es la imagen del intento 12.
  - `main` sigue detrás de esta rama.
- **Intento 11, del 13 al 14/11** (raíz `C:\CloudRAG\iteration11-faseB-20261113-54e0518`, cerrada; detalle en ADR-0031):
  - **Pasos 1 a 7:** todos en exit 0. L4 en la 3.ª ronda de capacidad.
  - **privacy: PASS.** 0 aciertos y los siete dominios completos, igual que en el intento 10.
  - **Estímulo (8a), paso en primer plano:** boots 1 a 4 COMPLETE; el boot 5 perdió el acuse aun con los reintentos de ADR-0030.
  - **gate:** no corrió. **close:** CLOSED. **Gasto:** USD 6,91.
- **Causa y corrección (ADR-0031):**
  - El host de estímulo daba al dueño solo 60 s antes de `shutdown -h now`, y la recolección del dueño ronda los 60-75 s. Esto corrige el diagnóstico de ADR-0030.
  - `host_runtime.await_owner_ack`: ventana de 600 s compartida por estímulo y gate, y 30 s de gracia tras el acuse. TDD con 7 tests.
- **Bitácora privada:** `C:\CloudRAG\i11-preparacion-20261113\BITACORA_I11.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas. Quedan detenidas, con costo de disco, las VMs GPU y los discos de los intentos 10 y 11, los snapshots previos y la IP estática. Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Intento 12** sobre la imagen de ADR-0031, con el mismo tope de USD 20 + 3 por intento y el control delegado por Enzo:
   evidencia local nueva, raíz nueva, cpu-build, capacidad, tramo 1, anexo, tramo 2a, estímulo en primer plano, gate y close.
2. **Recursos retenidos en GCP:** Enzo decide qué borrar. El snapshot `i6-cierre` es la base del cpu-build y no se borra.
3. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13 y 14/11. La zona sigue fijada por Enzo (ADR-0026).
- **Recolección al final de cada trabajo:** la ventana de ADR-0031 cubre la latencia medida. Un dueño que tarde más de 10 min sigue perdiendo el boot.
- **Transporte IAP:** un corte en `stimulus-start` o `gate-start` sigue siendo terminal (ADR-0030).
- **Comandos `i7` sobre una raíz:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Un traspaso que toca `.harness/` va en el commit de imagen o después del close.
- **IP estática:** queda IN_USE en la VM GPU detenida tras cada close. Hay que desligarla antes del gpu-restore siguiente.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS en la misma raíz y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
