# Decisiones de nivel 1 (2026-11-02)

- **D-01** Rama: el prompt pide `claude/review-2026-11-02`; el entorno asignó `claude/blissful-euler-0a3xkp`
  (rama nueva, igual a `main` @ `8fdaa65`, con la instrucción de no publicar en otra). Se trabaja en la asignada:
  cumple el aislamiento pedido y no toca `main` ni `fix/interview-readiness`.
- **D-02** Intérprete: el contenedor trae 3.13 sin dependencias. Se creó un venv 3.14.6 fuera del repo
  (scratchpad) con `uv pip sync --torch-backend cpu --require-hashes requirements-app.txt`, igual que el CI.
  Desvío declarado: 3.14.6 Linux frente a 3.14.x Windows del dueño.
- **D-03** Línea base con `env -u PLAYWRIGHT_BROWSERS_PATH`: el contenedor define esa variable para su propio
  Chromium y `init.sh` la toma como señal de runtime fijado de I7 (ver hallazgo sobre `init.sh`). Se registran ambas
  corridas.
- **D-04** Tests reproductores sobre la zona congelada: se añaden en `tests/review_2026_11_02/` marcados
  `xfail(strict=True)` con el id del hallazgo. Fallan hoy (documentan el defecto), mantienen `init.sh` en verde y,
  al aplicar el parche, pasan a XPASS y obligan a retirar la marca.
- **D-05** Subagentes de pasada: solo lectura sobre el repo; escriben en el scratchpad. La integración en
  `docs/review/` la hace el agente principal.
- **D-06** Compuerta de push (ADR-0013): el usuario eligió "esperar al cierre" (AskUserQuestion, 2026-11-02).
  No se publica nada hasta un único OK final. El hook que avisa de commits sin publicar se atiende con esa decisión.
