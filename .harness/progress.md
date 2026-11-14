[Claude] | 2026-11-14 | rama `fix/interview-readiness` @ base `1349b20` | tarea F-020 | modo: ejecución (intentos 13 y 14 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0033 (resultado del intento 13 y sondeos largos tolerantes), lifecycle.py, tests, adr-log, F-020 | Sin verificar: latencia del despliegue (el gate no se lanzó) y el calendario largo frente a un corte real en la nube

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 13 fue `06ec707` y su anexo `1349b20`. El commit de ADR-0033 es la imagen del intento 14.
  - `main` sigue detrás de esta rama.
- **Intento 13, del 14/11** (raíz `C:\CloudRAG\iteration13-faseB-20261114-06ec707`, cerrada; detalle en ADR-0033):
  - **Pasos 1 a 7:** todos en exit 0. L4 en la 1.ª ronda de capacidad. **privacy: PASS.**
  - **Estímulo (8a):** boots 1 y 2 COMPLETE; fallo terminal en el boot 3. El SSH por IAP a la VM se cortó unos 5,5 min mientras la API de GCP respondía, y `stimulus-status` se rindió tras 15 s y 45 s.
  - **gate:** no se lanzó, porque exige un estímulo ACCEPTED.
  - **close:** CLOSED. **Gasto:** USD 6,31.
- **Corrección (ADR-0033):** solo `stimulus-status` y `gate-status` reintentan con (15, 45, 60, 60, 60, 60, 60) s. Empiezan con el calendario corto y luego prueban cada 60 s, para recolectar dentro de la ventana de acuse de 600 s. `preflight` y la recolección conservan (15, 45). TDD y revisión independiente.
- **Bitácora privada:** `C:\CloudRAG\i13-preparacion-20261114\BITACORA_I13.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas. Quedan detenidas, con costo de disco, las VMs GPU y los discos de los intentos 10 a 13, los snapshots previos y la IP estática (RESERVED, desligada). Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Intento 14** sobre la imagen de ADR-0033, con el mismo tope de USD 20 + 3 por intento y el control delegado por Enzo:
   evidencia local nueva, raíz nueva, cpu-build, capacidad, tramo 1, anexo, tramo 2a, estímulo en primer plano, gate y close.
2. **Recursos retenidos en GCP:** Enzo decide qué borrar. El snapshot `i6-cierre` es la base del cpu-build y no se borra.
3. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13 y 14/11, también para la reserva 2 del gate. La zona sigue fijada por Enzo (ADR-0026).
- **Reutilización del nombre de reserva (ADR-0032):** no se ha probado en la nube porque el intento 13 no llegó al gate. Si Google Cloud lo rechaza, la intención queda UNKNOWN y el gate no escribe su intento.
- **Transporte IAP:** cortes de minutos con la API respondiendo (intento 13). Los sondeos largos toleran ahora unos 23 min. Un corte en `stimulus-start`, `gate-start` o en otra operación con efectos sigue siendo terminal (ADR-0030).
- **Recolección al final de cada trabajo:** la ventana de ADR-0031 cubre la latencia medida. Un dueño que tarde más de 10 min sigue perdiendo el boot.
- **Comandos `i7` sobre una raíz:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Un traspaso que toca `.harness/` va en el commit de imagen o después del close.
- **IP estática:** queda IN_USE en la VM GPU detenida tras cada close. Hay que desligarla antes del gpu-restore siguiente.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS en la misma raíz y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
