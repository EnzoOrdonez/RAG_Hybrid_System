[Claude] | 2026-11-14 | rama `fix/interview-readiness` @ base `d053c4b` | tarea F-020 | modo: ejecución (intentos 12 y 13 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0032 (resultado del intento 12 y nombre de reserva ligado a la VM), i7.py, tests, adr-log, F-020 | Sin verificar: latencia del despliegue (el gate no midió) y reutilización del nombre de reserva en la nube

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 12 fue `d2c2845` y su anexo `d053c4b`. El commit de ADR-0032 es la imagen del intento 13.
  - `main` sigue detrás de esta rama.
- **Intento 12, del 14/11** (raíz `C:\CloudRAG\iteration12-faseB-20261114-d2c2845`, cerrada; detalle en ADR-0032):
  - **Pasos 1 a 7:** todos en exit 0. L4 en la 1.ª ronda de capacidad. **privacy: PASS.**
  - **Estímulo (8a): ACCEPTED 12/12**, el primero de la Fase B. Valida la ventana de acuse de ADR-0031.
  - **gate:** 5 rechazos de capacidad liberados en `reserve(5)`; reserva 2 a las 14:37Z. Después, fallo terminal sin medir: `SPECIFIC_RESERVATION_NOT_BOUND`. El intento único quedó consumido.
  - **close:** CLOSED. **Gasto:** USD 7,95.
- **Causa y corrección (ADR-0032):**
  - La VM GPU queda ligada por nombre a la reserva 1, y `reserve()` daba otro nombre a la reserva 2. Google Cloud no re-liga una VM a otra reserva específica.
  - Con la VM creada, `reserve()` reutiliza el nombre que la VM consume y lo verifica antes de cualquier efecto pagado o intento del gate. TDD con 3 tests.
- **Bitácora privada:** `C:\CloudRAG\i12-preparacion-20261114\BITACORA_I12.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas. Quedan detenidas, con costo de disco, las VMs GPU y los discos de los intentos 10, 11 y 12, los snapshots previos y la IP estática (RESERVED, desligada). Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Intento 13** sobre la imagen de ADR-0032, con el mismo tope de USD 20 + 3 por intento y el control delegado por Enzo:
   evidencia local nueva, raíz nueva, cpu-build, capacidad, tramo 1, anexo, tramo 2a, estímulo en primer plano, gate y close.
2. **Recursos retenidos en GCP:** Enzo decide qué borrar. El snapshot `i6-cierre` es la base del cpu-build y no se borra.
3. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13 y 14/11, también para la reserva 2 del gate. La zona sigue fijada por Enzo (ADR-0026).
- **Reutilización del nombre de reserva (ADR-0032):** no se ha probado en la nube. Si Google Cloud lo rechaza, la intención queda UNKNOWN y el gate no escribe su intento.
- **Recolección al final de cada trabajo:** la ventana de ADR-0031 cubre la latencia medida. Un dueño que tarde más de 10 min sigue perdiendo el boot.
- **Transporte IAP:** un corte en `stimulus-start` o `gate-start` sigue siendo terminal (ADR-0030).
- **Comandos `i7` sobre una raíz:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Un traspaso que toca `.harness/` va en el commit de imagen o después del close.
- **IP estática:** queda IN_USE en la VM GPU detenida tras cada close. Hay que desligarla antes del gpu-restore siguiente.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS en la misma raíz y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
