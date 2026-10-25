#!/usr/bin/env bash
# Verificación en tres capas. El trabajo solo se da por terminado si las tres salen con exit 0.
#   Capa 1  secretos      scripts/scan_secrets.py contra secrets_baseline.json
#   Capa 2  tests CPU     pytest -m "not slow and not gpu"
#   Capa 3  docs/harness  scripts/check_docs.py
# Uso: bash init.sh           (Linux, macOS, CI o Git Bash en Windows; no WSL)
#      PYTHON=/ruta/al/python bash init.sh
# Intérprete: $PYTHON si está definido; si no, .venv-app (entorno congelado que exigen los
# tests de I7); si no, python. Ver docs/guides/reproduce.md §0. No ejecuta modelos, no escribe evidencia, no usa red.
set -u
cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv-app/Scripts/python.exe ]; then PY=".venv-app/Scripts/python.exe"   # Windows: lo exige la suite de I7
elif [ -x .venv-app/bin/python ]; then PY=".venv-app/bin/python"
else PY="python"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1

# Chromium fijado de I7 (F-012): si no esta definida y existe en esta maquina, se usa; CI y otras maquinas no cambian.
if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
  for d in /c/CloudRAG/i7-browser-runtime C:/CloudRAG/i7-browser-runtime; do
    if [ -d "$d" ]; then export PLAYWRIGHT_BROWSERS_PATH="C:/CloudRAG/i7-browser-runtime"; break; fi
  done
fi
# Con el runtime definido, los tests de navegador de I7 son obligatorios: no se omiten en silencio.
if [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then export CLOUDRAG_REQUIRE_I7_BROWSER=1; fi

declare -a NAMES=("1 secretos" "2 tests CPU" "3 docs/harness")
declare -a CMDS=(
  "$PY scripts/scan_secrets.py --baseline secrets_baseline.json"
  "$PY -m pytest tests/ -q -m \"not slow and not gpu\""
  "$PY scripts/check_docs.py"
)
declare -a CODES=()

for i in "${!CMDS[@]}"; do
  echo "== Capa ${NAMES[$i]}: ${CMDS[$i]}"
  eval "${CMDS[$i]}"
  CODES[$i]=$?
done

echo "== Resumen"
fail=0
for i in "${!CMDS[@]}"; do
  echo "Capa ${NAMES[$i]}: exit ${CODES[$i]}"
  [ "${CODES[$i]}" -ne 0 ] && fail=1
done
[ "$fail" -eq 0 ] && echo "init.sh: exit 0 en las tres capas" || echo "init.sh: FALLO, no traspasar"
exit "$fail"
