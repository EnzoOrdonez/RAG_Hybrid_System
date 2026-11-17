[Claude] | 2026-11-17 | rama `fix/interview-readiness` @ `449465c` (+ este commit de harness) | tarea F-020 | modo: ejecución (ensayo P998 y cierre técnico, decisiones delegadas por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2303 pasan, 0 fallan, 6 omitidas) | 3 docs=exit 0 (init.sh sobre el árbol con la nota del runbook)
Cambios: runbook, ADR-0036 y adr-log en 449465c (solo docs/); F-020 y este archivo en el commit siguiente | Sin verificar: evento `_preparation` del ensayo en la VM; factura real de GCP (solo el ledger del operador)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Rama:** `fix/interview-readiness`, worktree `.worktrees/i7-human-ledger-ready`.
  - La imagen del intento 15 es `b5e54f1` y su anexo `1a4ead8`.
  - `main` sigue detrás de esta rama. El 17/11 se abrió un PR hacia `main`, sin merge.
- **Fase B de I7 cerrada con GO** (intento 15, 15/11, ADR-0035): 120 válidas, 0 fallos ni inválidas; p95 hybrid 34,88 s y no_rag 27,64 s (límite 60 s).
- **Paso F técnico cumplido** (ensayo P998, 16-17/11, ADR-0036):
  - 44 rondas de capacidad L4 y 2 reservas. La ventana del 16/11 venció sin recorrido.
  - El 17/11: READY 03:08Z, invitación registrada sin persistir el token y recorrido sintético de Enzo hasta la pantalla de cierre. `stop` y `guard` verificados.
  - **Gasto del operador:** USD 13,06 en total (Fase B 9,46 y P998 3,60). Exposición abierta 0.
- **Hallazgo de «Preparar sesión»:** la preparación tarda unos 52 s sin aviso de espera. Un segundo clic crea otra ejecución de Streamlit (`fastReruns`) que choca con `FileLock(timeout=0)` y muestra un error genérico. Mitigación en el paso 7 de [STUDY_DEPLOYMENT_RUNBOOK.md](../docs/STUDY_DEPLOYMENT_RUNBOOK.md), sin cambio de código.
- **Decisiones delegadas por Enzo el 17/11** (ADR-0036):
  - la purga de las sesiones sintéticas (intentos 14 y 15 y P998) se aplaza al cierre del estudio;
  - se conservan los recursos GCP de abajo;
  - el identificador del proyecto GCP se queda en `scripts/` y en docs hasta cerrar el estudio, sin reescribir historial.
- **GCP:** no hay VMs encendidas ni reservas. Quedan, con unos USD 37 al mes de almacenamiento:
  - la VM GPU detenida del intento 15, con su disco de 180 GB, que usará la fase humana;
  - el snapshot limpio del intento 15;
  - los tres snapshots base (`i6-cierre`, `i5-final03`, `original-i2-i3`), que no se borran;
  - la IP estática, IN_USE en la VM detenida (cuesta lo mismo que RESERVED).
- **Auditoría RAG de Fable** (rama `fable/rag-auditoria`, PR encadenado sobre esta rama, sin merge):
  - `exp20_rerank_arms` firmada el 16/11: con el mismo reranker, el híbrido supera al denso en +0,016 de NDCG@5 (n.s.); la ganancia procede del reranker.
  - B1, B2 y B6 aplicadas; B3, B4, B5, B7 y B8 sin aplicar, con test xfail y declaradas como límites (ADR-0037 en esa rama).
  - Entradas de ledger de exp20 y de la fe de erratas S6 («42 multi-cloud queries» son 25) pegadas con OK de Enzo; nota de erratas en el README.
  - Enzo aceptó el 17/11 las ocho frases propuestas (S1, S5, S7c, S11a, S11c, S13, S6 y S12) para el A.3 y la defensa. El camera-ready certificado no se toca.
- **Bitácora privada:** `C:\CloudRAG\i15b-preparacion-20261115\BITACORA_I15.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Enzo:** revisar y fusionar el PR de esta rama a `main` y después el de Fable.
2. **Enzo:** aplicar en el A.3 y en la defensa las ocho frases aceptadas.
3. **Paso F restante:** aprobación ética escrita (Enzo). Sin ella, la fase humana sigue en NO-GO.
4. **Paso G:** invitaciones y sesiones, con respaldo verificado por hash entre sesiones y dentro del tope de USD 100.
   - Cada sesión necesita un driver de capacidad y al coordinador presente al READY.
   - «Preparar sesión» se pulsa una sola vez, antes de entregar el control.
5. **Al cerrar el estudio:** purgar las sesiones sintéticas (y leer el evento `_preparation` de P998), pasar el identificador del proyecto a configuración, revisar los recursos GCP y corregir el aviso de espera de la preparación con una imagen nueva.

## Riesgos y bloqueos vigentes

- **Checkout de la fase humana:** todo comando `i7` sobre la raíz del GO llama `verify_tool_source`. Falla con `OPERATOR_SOURCE_COMMIT_DIFFERS` si el diff contra `b5e54f1` toca algo fuera de `docs/` y `output/audit/`. Esta rama ya toca `.harness/`, así que la fase humana corre desde el worktree `.worktrees/i7-human-phase-603fa49` (detached en `603fa49`) o desde un commit posterior que cumpla esa regla.
- **Código del operador congelado para esta raíz:** cualquier arreglo en `scripts/`, `src/` o `config/` invalida el uso de la raíz del GO y exige otra Fase B.
- **Transporte IAP:** cortes de minutos con la API respondiendo (intentos 13 a 15). Una operación con efectos sigue teniendo un solo intento: un corte en `maintenance`, `stimulus-start` o `gate-start` es terminal para esa operación (ADR-0030, ADR-0035).
- **Capacidad L4 en `us-central1-a`:** 2 reservas en 44 rondas entre el 16 y el 17/11. La zona sigue fijada por Enzo (ADR-0026).
- **Ventana tras el READY:** la VM se apaga sola unas 2 h 40 min después. Si el recorrido no empieza a tiempo, se pierde el boot.
- **Arranque en frío:** la primera respuesta hybrid tras un boot llegó a 47,4 s. La preparación por sesión calienta ambos sistemas.
- **Entrevistas:** en NO-GO hasta la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
