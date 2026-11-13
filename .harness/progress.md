[Claude] | 2026-11-13 | rama `fix/interview-readiness` @ base `42e10830` | tarea F-020 | modo: corrección local tras el intento 8
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh sobre el árbol con el cambio, antes del commit)
Cambios: guest_bridge.py, i7_cpu_pipeline.py, 2 archivos de tests, ADR-0027, F-020 | Sin verificar: build y nube del intento 9

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos 2 a 6 están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`, base `42e1083` = `origin/fix/interview-readiness`.
  El cambio de ADR-0027 se publica en esta rama con el OK de Enzo del 13/11. `main` (`8fdaa65`) va 40 commits detrás de esta rama.
- **Intento 8** (raíz `C:\CloudRAG\iteration8-faseB-20261109-86143c7`, código `1b07520`, 10 y 11/11):
  - pasos 1 a 4b, smoke y restore OK; el smoke confirmó el arreglo del relé (`86143c7`);
  - privacy falló por privacy-08 y close por close-08 (ADR-0027); withdraw-purge OK;
  - estímulo y gate no corrieron. Gasto del intento: USD 9,46.
- **GCP después de la limpieza del 13/11** (con OK, registro en `C:\CloudRAG\cleanup-20261113`):
  - sin VM, discos ni reservas;
  - quedan los snapshots `cloudrag-i6-cierre-disco-20261016` (base del cpu-build, no borrar), final03 y original I2-I3;
  - la IP en RESERVED, dos buckets y dos reglas de firewall;
  - costo en reposo estimado: ~USD 16 al mes.
- Las raíces de los intentos 7 y 8 quedan abandonadas: no se cierran con `close`.

## Qué sigue (cada paso con su OK)

1. Commit y push de ADR-0027: autorizados por Enzo el 13/11.
2. **Presupuesto del intento 9:** Enzo aprobó el 13/11 un tope de USD 20 + 3 de cierre (ADR-0027). El lanzamiento espera su aviso.
3. Antes de lanzar: preflight local (`i7_phase_b --check`) con la raíz nueva y la cuota SSD libre.
4. Intento 9 con raíz nueva (`i7_config`): cpu-build (imagen nueva, porque el arreglo de privacy corre en el invitado),
   gpu-restore, ready, tls-three, contexts, anexo publicado con OK, smoke, restore, privacy (IP real de origen de Enzo),
   withdraw-purge, estímulo 12/12 y gate. El GO del gate exige p95 de 60 s o menos.
5. Después del GO: aprobación ética escrita y sesiones dentro del tope de USD 100 del estudio.
6. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde.

## Riesgos y bloqueos vigentes

- Cuota SSD: el intento 9 suma 360 GB en el cpu-build y 180 GB en gpu-restore. El límite es 1000 GB y hoy no hay discos.
- Capacidad L4 en `us-central1-a`: el 10/11 hubo `ZONE_RESOURCE_POOL_EXHAUSTED` y errores de backend en la reserva (ADR-0026).
- Intermitencia F-016: `f7309a0` amplió el presupuesto de preparación del worker de 3 a 30 s. No reapareció en el init.sh del 13/11.
- privacy pide la IP pública real de origen de Enzo por entrada privada: el paso no se puede automatizar sin él.
- Entrevistas en NO-GO hasta el GO del gate y la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
