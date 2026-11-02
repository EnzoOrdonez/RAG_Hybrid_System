```
[Claude Code, auditor de nube] | 2026-11-02 | rama `claude/review-i8` desde `origin/fix/i7-attempt8` @ `3bb9ca8` | tarea: auditoría del intento 8 (sin F-NNN nueva; contexto F-020) | modo: sondeo
Capas (PLAYWRIGHT_BROWSERS_PATH sin definir): ver el cierre del traspaso en el chat (exit de las tres capas tras el commit)
Cambios: docs/review/2026-11-02-i8/**, tests/test_review_i8_findings.py, tests/test_review_i8_mutants.py | Sin verificar: GCP, Docker, Windows, PowerShell, GPU, Ollama, Streamlit real en la VM
```

# Auditoría del intento 8: diff b470d3e..3bb9ca8

- **Base verificada:** HEAD 3bb9ca8 con padre b470d3e. El diff toca 10 archivos y coincide con el encargo. `origin/fix/interview-readiness` está en b470d3e.
- **Detalle de la auditoría:** [ANEXOS.md](ANEXOS.md) recoge la redacción con Chromium real, la tabla de timeouts, los pines de rama, la mutación y las regresiones.
- **Hallazgos:** [findings.json](findings.json).

## 1. Veredicto: GO condicionado para el cpu-build con 3bb9ca8

El código del cpu-build no tiene ningún defecto nuevo que bloquee. Con el estado actual el cpu-build **no arranca**:
`i7_cpu_pipeline.py:46` corta con `PUBLISHED_I7_SOURCE_REQUIRED` antes de pagar, porque el remoto sigue en b470d3e.

Para un GO tienen que cumplirse estas condiciones:

1. **Push fast-forward de `origin/fix/interview-readiness` de b470d3e a 3bb9ca8.**
   - Es posible: `merge-base --is-ancestor` da OK.
   - Necesita el OK de Enzo (ADR-0013).
2. **En el checkout del operador, la rama local `fix/interview-readiness` debe estar en 3bb9ca8.** Lo recomendable es hacer checkout de esa rama y `merge --ff-only`.
   - Quedarse en fix/i7-attempt8 con la rama local vieja construye b470d3e después de pagar (I8-01).
3. **Raíz nueva creada con `i7_config` estando HEAD en 3bb9ca8, y ningún commit entre la creación de la raíz y el cpu-build.**
   - `:45` exige igualdad exacta.
   - **El commit de esta auditoría no debe fusionarse en esa rama antes del build:** toca `tests/`.
4. **Árbol limpio en scripts, src y config** (`i7_workflows.py:42-45`).
5. **Recibo de readiness nuevo** sobre los bytes de 3bb9ca8, con las suites de Windows, Linux e I7 (`i7_readiness.py:17-25`). El recibo del 16/10 que cita el runbook ya no vale.

**Recomendado antes del smoke, no del cpu-build:** los parches 002 (I8-02) y 003 (I8-03).
Aplicarlos cambia scripts/ y exige commit, raíz y build nuevos, así que conviene decidirlo **antes** del cpu-build.

## 2. Hallazgos que sobrevivieron a la refutación

| id | sev | archivo:línea | resumen | reproductor |
|---|---|---|---|---|
| I8-01 | S1 | `i7_cpu_pipeline.py:42-46,60,90-91` | La guarda mira la rama remota, pero el clon usa la rama local y se hace después de `admit_paid`. Si la rama local está obsoleta se construye el commit viejo; si no existe, rc 128 con `cpu/` ya creado. Es anterior a 3bb9ca8 y la rama del intento 8 lo activa | `test_i8_01_*` (xfail) |
| I8-02 | S2 | `i7_smoke.py:35`, `i7_phase_b.py:247` | `start('smoke')` está fuera del try. Un timeout o Ctrl+C en `instances start` (timeout=300) deja la VM arrancando sin `stop()`. Lo acota el STOP nativo de 3 h | `test_i8_02_*` (xfail) |
| I8-03 | S2 | `i7_browser.py:213-214,178-180,348` | El volcado escribe la IP pública real: `<IP>.sslip.io` aparece en `console.json` (ws.url y consola) y en `aria.yaml` (/url). Token, contraseña y canario sí se redactan | `test_i8_03_*` (xfail) |
| I8-04 | S3 | `i7_smoke.py:48-51`, `i7_phase_b.py:251-254` | Patrón B-2: si falla el `stop()` del except, se tapa la causa, que queda solo en `__context__` | `test_i8_04_*` (xfail) |
| I8-05 | S3 | `deployment.py:17-18` | `smoke_participant` no tiene ninguna vía soportada para fijarse; obliga a editar installation.json a mano (ADR-0026) | lectura |
| I8-06 | S3 | `i7_browser.py:180,346-348` | Para diagnosticar F-020 faltan en el volcado `sync.requests`, `sync.completed` y un snapshot de `observe()` | lectura |
| I8-07 | S3 | `i7_browser.py:256-258,121,91-95` | Ninguna operación queda en 600000 ms, pero los presupuestos se suman: «Probar consulta» puede tardar hasta 3300 s. Varias llamadas no tienen tope | ANEXOS §B |
| I8-08 | S3 | `STUDY_I7_RUNBOOK_FASE_B.md:9,41-43` | El runbook contradice fix/i7-attempt8 y cita un `$evidence` y un `$run` obsoletos | lectura |

**Refutados o sin hallazgo:**
- §2.1, acceso: no hay debilitamiento.
  - Los códigos reales son P01-P24.
  - `smoke_participant()` solo se consulta con purpose=smoke.
  - rehearsal sigue en P998.
  - `register_invitation_hash` no cambia.
  - El valor configurado llega igual al operador, al seed, al env y a observe/status.
- §2.7: sin regresiones. La suite completa queda en exit 0, y los tests de la auditoría anterior dan lo mismo.
- KeyboardInterrupt durante `browse`: el smoke sí para la VM (verificado).

**Respecto a la auditoría anterior:** F-01, B-3, B-4 y los archivos de un solo intento siguen igual; 3bb9ca8 ni los arregla ni los agrava.
- `gate-cohort-intent.json` se sigue escribiendo antes de `run(smoke)`.
- El smoke no crea archivos nuevos de un solo intento.
- Sí consume la invitación del código en el disco del periodo: un segundo smoke en la misma raíz falla en `invite`, después de pagar el arranque. La salida es I8-05.

## 3. No verificado
- GCP real, incluido el comportamiento de `instances start` tras matar gcloud.
- Docker y la imagen.
- Windows y PowerShell (rutas `C:/`, ACL de `root/private`).
- GPU y Ollama.
- El recorrido Streamlit completo en el navegador: los 11 tests de Playwright se omiten sin el runtime fijado (R-08).
- La causa de F-020.
- Los mutantes M31 y M32 (recorrido real).

## 4. Parches propuestos, sin aplicar (`patches/`)

Se probaron en un worktree desechable: solo cambian los xfail estrictos de I8-01, I8-02 e I8-03, que pasan a XPASS. El resto del subconjunto I7 sigue en verde.

- **001:** comprueba que `refs/heads/fix/interview-readiness` local == HEAD junto a `:46`, antes de pagar. Ningún archivo del pin de receta cambia.
- **002:** mueve `start('smoke')` dentro del try en el smoke y en privacy.
- **003:** añade el host `.sslip.io` y su IP a los secretos de `run()`.
