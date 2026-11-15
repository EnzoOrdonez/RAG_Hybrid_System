[Claude] | 2026-11-15 | rama `fix/interview-readiness` @ base `996db48` | tarea F-020 | modo: ejecución (intento 14 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh antes del commit)
Cambios: ADR-0034 (resultado del intento 14, reloj de tramos archivados y lecturas de mantenimiento), i7_gate_host.py, lifecycle.py, tests, adr-log, F-020 | Sin verificar: un gate completo con la corrección en la nube

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 14 fue `4ef8d35` y su anexo `996db48`. El commit de ADR-0034 es la imagen del intento 15.
  - `main` sigue detrás de esta rama.
- **Intento 14, del 14 y 15/11** (raíz `C:\CloudRAG\iteration14-faseB-20261114-4ef8d35`, cerrada; detalle en ADR-0034):
  - **Pasos 1 a 7:** todos en exit 0. L4 en la 1.ª ronda. **privacy: PASS.**
  - **Estímulo (8a):** **ACCEPTED** 12/12, 144 llamadas.
  - **gate (8b):** cinco rechazos de capacidad liberados y reserva 2 en la ronda 5. Corrió 140/140 llamadas y luego quedó `INCOMPLETE_TERMINAL` por un defecto de la verificación del invitado. Sin GO ni NO-GO.
  - **close:** CLOSED. **Gasto:** USD 9,39.
- **Corrección (ADR-0034):**
  - `verify` evalúa los tramos de admisión y de ventanas con el reloj de su propia última muestra, no con el de cierre. El lease de Ollama dura unos 30 min y el gate unos 62.
  - `inventory` y `download-disk` reintentan (15, 45) s: la limpieza técnica se cortó en `download-disk` por transporte.
  - TDD y revisión independiente.
- **Bitácora privada:** `C:\CloudRAG\i14-preparacion-20261114\BITACORA_I14.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close:** sin VMs encendidas ni reservas. Quedan detenidas, con costo de disco, las VMs GPU y los discos de los intentos 10 a 14, los snapshots previos y la IP estática (RESERVED, desligada). Decide Enzo.
- **Sesiones sintéticas del gate del intento 14** (códigos P9000xx, no son personas): sin purgar en el disco GPU y el bucket de ese período. `technical-cycle` exige otro boot pagado. Decide Enzo.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Intento 15** sobre la imagen de ADR-0034, con el mismo tope de USD 20 + 3 por intento y el control delegado por Enzo:
   evidencia local nueva, raíz nueva, cpu-build, capacidad, tramo 1, anexo, tramo 2a, estímulo, gate y close.
2. **Recursos retenidos en GCP y sesiones sintéticas:** Enzo decide. El snapshot `i6-cierre` es la base del cpu-build y no se borra.
3. Después del GO y de privacy PASS: aprobación ética escrita y sesiones dentro del tope de USD 100.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13 a 15/11, también para la reserva 2 del gate. La zona sigue fijada por Enzo (ADR-0026).
- **Gate completo:** el intento 14 fue el primero en llegar a la verificación final. La corrección no se ha probado en la nube.
- **Transporte IAP:** cortes de minutos con la API respondiendo (intentos 13 y 14). Los sondeos largos toleran unos 23 min. Un corte en `stimulus-start`, `gate-start` o en otra operación con efectos sigue siendo terminal (ADR-0030).
- **Recolección al final de cada trabajo:** la ventana de ADR-0031 cubre la latencia medida. Un dueño que tarde más de 10 min sigue perdiendo el boot.
- **Comandos `i7` sobre una raíz:** `verify_tool_source` solo admite cambios en `docs/` y `output/audit/` sobre la imagen. Un traspaso que toca `.harness/` va en el commit de imagen o después del close.
- **IP estática:** queda IN_USE en la VM GPU detenida tras cada close. Hay que desligarla antes del gpu-restore siguiente.
- **Entrevistas:** en NO-GO hasta el GO del gate, privacy PASS en la misma raíz y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
