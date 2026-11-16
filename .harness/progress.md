[Claude] | 2026-11-15 | rama `fix/interview-readiness` @ `603fa49` (+ este commit de harness) | tarea F-020 | modo: ejecución (intento 15 de la Fase B, control delegado por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2303 pasan, 0 fallan, 6 omitidas) | 3 docs=exit 0 (init.sh sobre el árbol final)
Cambios: ADR-0035 e informe de sesiones sintéticas en b4a290e y 603fa49 (solo docs/); F-020 y este archivo en el commit siguiente | Sin verificar: factura real de GCP (solo el ledger del operador)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 15 es `b5e54f1` y su anexo `1a4ead8`.
  - `main` sigue detrás de esta rama.
- **Intento 15, del 15/11** (raíz `C:\CloudRAG\iteration15b-faseB-20261115-b5e54f1`, cerrada; detalle en ADR-0035):
  - **Pasos 1 a 7:** todos en exit 0, tras 5 rechazos de capacidad L4 liberados. **privacy: PASS.**
  - **Estímulo (8a):** **ACCEPTED** 12/12, 144 llamadas.
  - **gate (8b):** reserva 2 al primer intento. **GO:** 120 válidas, 0 fallos ni inválidas; p95 hybrid 34,88 s y no_rag 27,64 s (límite 60 s).
  - **close:** CLOSED. IP estática desligada y RESERVED. **Gasto:** USD 9,46 (ledger del operador, no factura).
  - **Limpieza técnica incompleta:** `purge-study` falló en `maintenance` por transporte IAP, después del GO.
- **Paso E de F-020 cumplido.** F-020 sigue `active`: faltan F (ensayo P998) y G (sesiones).
- **Informe de las sesiones sintéticas:** [i7-gate-sesiones-sinteticas-i15.md](../docs/reports/i7-gate-sesiones-sinteticas-i15.md).
- **Bitácora privada:** `C:\CloudRAG\i15b-preparacion-20261115\BITACORA_I15.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **GCP tras el close y el retiro del 16/11** (borrados con OK de Enzo: la VM y el disco del intento 14 y los snapshots limpios de los intentos 10 a 14; log privado en la carpeta de preparación). No hay VMs encendidas ni reservas. Quedan, con unos USD 37 al mes de almacenamiento:
  - la VM GPU detenida del intento 15, con su disco de 180 GB, que usará la fase humana;
  - el snapshot limpio del intento 15;
  - los tres snapshots base (`i6-cierre`, `i5-final03`, `original-i2-i3`), que no se borran;
  - la IP estática RESERVED sin usuarios.
- **Sesiones sintéticas** (códigos P9000xx, no son personas):
  - del gate del intento 15: sin purgar en su disco GPU y en el bucket del período;
  - del intento 14: solo en el bucket, porque su disco se borró.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Enzo decide sobre las sesiones sintéticas:** purgarlas con `technical-cycle` (un boot pagado; cotizar antes).
2. **Paso F:** aprobación ética escrita, `human-phase-open`, ensayo P998 con Enzo y calendario de sesiones dentro del tope de USD 100.
3. **Paso G:** invitaciones y sesiones, con respaldo verificado por hash entre sesiones.
4. Integrar `fix/interview-readiness` en `main` mediante PR, con CI verde. Antes, decidir desde qué checkout corre la fase humana (abajo).

## Riesgos y bloqueos vigentes

- **Checkout de la fase humana:** todo comando `i7` sobre la raíz del GO llama `verify_tool_source`. Falla con `OPERATOR_SOURCE_COMMIT_DIFFERS` si el diff contra `b5e54f1` toca algo fuera de `docs/` y `output/audit/`. Este commit toca `.harness/`, así que la fase humana corre desde un checkout de `603fa49` (por ejemplo, un worktree propio) o de un commit posterior que cumpla esa regla.
- **Código del operador congelado para esta raíz:** cualquier arreglo en `scripts/`, `src/` o `config/` invalida el uso de la raíz del GO y exige otra Fase B.
- **Transporte IAP:** cortes de minutos con la API respondiendo (intentos 13 a 15). Una operación con efectos sigue teniendo un solo intento: un corte en `maintenance`, `stimulus-start` o `gate-start` es terminal para esa operación (ADR-0030, ADR-0035).
- **Arranque en frío:** la primera respuesta hybrid tras un boot llegó a 47,4 s. Calentar antes de cada participante.
- **Capacidad L4 en `us-central1-a`:** rechazos de 1 a 3 h los días 13 a 15/11. La zona sigue fijada por Enzo (ADR-0026).
- **IP estática:** queda IN_USE en la VM GPU tras cada boot. Desligarla al cerrar.
- **Entrevistas:** en NO-GO hasta la aprobación ética escrita y `human-phase-open`.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
