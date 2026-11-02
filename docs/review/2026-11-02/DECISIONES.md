# Decisiones de nivel 1
- Rama `claude/review-2026-11-02` desde b470d3e; no se usa `claude/blissful-euler-0a3xkp` salvo para leer su findings.json como pistas.
- Entorno: Python 3.14.6 (uv) en venv del scratchpad con `requirements-app.txt` + índice CPU de PyTorch; el `python` del PATH (3.13) no tiene pytest.
- Los tests reproductores van en `tests/test_review_2026_11_02_*.py`; los que demuestran un defecto aún no corregido se marcan `xfail(strict=True)` para no romper la capa 2 y para que pasen a XPASS (fallo) en cuanto se aplique el parche.
- Severidad de F-01: se mantiene S1 frente al S2 del refutador, porque la única salida sin start() técnico es editar la raíz a mano (prohibido por ADR-0026); queda registrado como desacuerdo.
- Los reproductores de A y B que fijan el defecto (aserción de comportamiento actual) se incorporan tal cual, con docstring que exige invertirlos al corregir; los de F-01, A-1 y R-xx usan xfail estricto.
- (seguimiento) Decisiones de Enzo: publicar solo esta rama, no aplicar los parches 001-003 y F-01 = S1.
- (seguimiento) Cierre por inventario: la lógica va en `closeout/closeout.py`, probado con un gcloud falso; `closeout.ps1` es solo un lanzador. Motivo: aquí no hay PowerShell, y ejecutar un `pwsh` descargado fue denegado por la política de permisos. Un script PowerShell entero habría salido sin ninguna prueba.
- (seguimiento) «Línea desactualizada de CLAUDE.md»: tomo como candidata `CLAUDE.md:37` («Linux o CI») con evidencia; queda pendiente de que Enzo confirme que es esa.
- (seguimiento) Parches 004 (tornado) y 005 (R-05) se añaden como documentación, igual de no aplicados.
