# Decisiones de nivel 1
- Rama `claude/review-2026-11-02` desde b470d3e; no se usa `claude/blissful-euler-0a3xkp` salvo para leer su findings.json como pistas.
- Entorno: Python 3.14.6 (uv) en venv del scratchpad con `requirements-app.txt` + índice CPU de PyTorch; el `python` del PATH (3.13) no tiene pytest.
- Los tests reproductores van en `tests/test_review_2026_11_02_*.py`; los que demuestran un defecto aún no corregido se marcan `xfail(strict=True)` para no romper la capa 2 y para que pasen a XPASS (fallo) en cuanto se aplique el parche.
