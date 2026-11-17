[Claude] | 2026-11-17 | rama `fix/f023-cierre-codigo` (sobre `fable/rag-auditoria` y `fix/interview-readiness`) | tareas F-020, F-022 y F-023 | modo: ejecución (cierre técnico, decisiones delegadas por Enzo)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 | 3 docs=exit 0 (init.sh sobre la punta de esta rama antes de fusionar; cifras en el PR)
Cambios: aviso de «Preparar sesión» (F-023), decisiones menores de la auditoría (ADR-0037), A.3 V2.21 fuera del repo | Sin verificar: evento `_preparation` del ensayo en la VM; factura real de GCP (solo el ledger del operador)

# Progreso: estado para arranque en frío

Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar; las
secciones de los intentos anteriores están en el historial de git de este archivo.

## Dónde estoy

- **Integración del 17/11:** Enzo aprobó fusionar la cadena de PR con init.sh y CI en verde: #9 (`fix/f023-cierre-codigo`) en `fable/rag-auditoria`, #8 en `fix/interview-readiness` y #7 en `main`. Sin push directo a `main`.
- **Fase B de I7 cerrada con GO** (intento 15, 15/11, ADR-0035): 120 válidas, 0 fallos ni inválidas; p95 hybrid 34,88 s y no_rag 27,64 s (límite 60 s). Imagen `b5e54f1`, anexo `1a4ead8`.
- **Paso F técnico cumplido** (ensayo P998, 16-17/11, ADR-0036): recorrido sintético de Enzo hasta la pantalla de cierre; `stop` y `guard` verificados. Gasto del operador USD 13,06; exposición abierta 0.
- **«Preparar sesión» (F-023):** el código de `main` muestra un aviso de espera y, ante una ejecución concurrente, un mensaje de preparación en curso en vez del error genérico. La imagen desplegada `b5e54f1` no lo incluye: para la fase humana sigue la nota del paso 7 de [STUDY_DEPLOYMENT_RUNBOOK.md](../docs/STUDY_DEPLOYMENT_RUNBOOK.md) (un solo clic).
- **Auditoría RAG de Fable (F-022, ADR-0037):** exp20 firmada (con el mismo reranker, híbrido − denso = +0,016 de NDCG@5, n.s.); B1, B2 y B6 aplicadas; B3, B4, B5, B7 y B8 sin aplicar, como límites; ledgers de exp20, de la fe de erratas S6 y de las 399 respuestas vacías de exp12; nota de erratas en el README.
- **A.3 (fuera del repo):** V2.21 junto a la V2.20 en la carpeta de entregables de Seminario II, con S1, S5, S7c, S11c, S12, S13 y el aviso del Flag 33 (versión ENTREGA y versión con cambios resaltados). Notas para la defensa en la carpeta `Defensa` de esos entregables.
- **Decisiones de Enzo del 17/11** (ADR-0036 y ADR-0037): purga de sintéticas y liberación de recursos al cierre del estudio; el identificador del proyecto GCP pasa a configuración también entonces; el aviso a LACCI o IEEE de la errata queda documentado fuera del repo para que decida él.
- **GCP:** sin VMs encendidas ni reservas. Quedan, con unos USD 37 al mes: la VM GPU detenida del intento 15 con su disco de 180 GB, su snapshot limpio, los tres snapshots base (no se borran) y la IP estática, IN_USE en la VM detenida.
- **Bitácora privada:** `C:\CloudRAG\i15b-preparacion-20261115\BITACORA_I15.md` (estado en `ESTADO.json`), indexada en `C:\CloudRAG\LEEME_CLOUDRAG.md`.
- **Rama aparte `chore/f009-ruff-scanner`** (sin PR): Ruff en 9 scripts, medición de F-016 y F-021.

## Qué sigue

1. **Enzo:** revisar la V2.21 del A.3 (versión con cambios resaltados) antes de entregarla. Conviene actualizar también su declaración de uso de IA (menciona herramientas hasta octubre y no la auditoría de Fable 5.2 ni este cierre).
2. **Paso F restante:** aprobación ética escrita (Enzo). Sin ella, la fase humana sigue en NO-GO.
3. **Paso G:** invitaciones y sesiones con personas desde el worktree fijado `.worktrees/i7-human-phase-603fa49`, dentro del tope de USD 100. Cada sesión necesita un driver de capacidad y al coordinador presente al READY; «Preparar sesión» se pulsa una sola vez.
4. **Al cerrar el estudio:** purgar las sesiones sintéticas (y leer el evento `_preparation` de P998), pasar el identificador del proyecto a configuración, liberar los recursos GCP y, si hay otra Fase B, desplegar el aviso de F-023.

## Riesgos y bloqueos vigentes

- **Checkout de la fase humana:** todo comando `i7` sobre la raíz del GO llama `verify_tool_source`. Falla con `OPERATOR_SOURCE_COMMIT_DIFFERS` si el diff contra `b5e54f1` toca algo fuera de `docs/` y `output/audit/`. `main` ya lo hace, así que la fase humana corre desde el worktree `.worktrees/i7-human-phase-603fa49` (detached en `603fa49`).
- **Código del operador congelado para esta raíz:** cualquier arreglo en `scripts/`, `src/` o `config/` invalida el uso de la raíz del GO y exige otra Fase B.
- **Transporte IAP:** cortes de minutos con la API respondiendo (intentos 13 a 15). Una operación con efectos tiene un solo intento (ADR-0030, ADR-0035).
- **Capacidad L4 en `us-central1-a`:** 2 reservas en 44 rondas entre el 16 y el 17/11. La zona sigue fijada por Enzo (ADR-0026).
- **Ventana tras el READY:** la VM se apaga sola unas 2 h 40 min después.
- **Arranque en frío:** la primera respuesta hybrid tras un boot llegó a 47,4 s. La preparación por sesión calienta ambos sistemas.
- **Entrevistas:** en NO-GO hasta la aprobación ética escrita.

## Cómo verificar en frío

```bash
PYTHON=<ruta a .venv-app/Scripts/python.exe> bash init.sh   # Git Bash; objetivo: exit 0 en las tres capas
```
