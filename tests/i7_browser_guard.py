"""Guard de CI para los tests de navegador de I7 (Playwright + Chromium fijado).

El CI solo instala ``requirements-app.txt``; ``playwright`` vive en ``requirements-i7-browser.txt``
(dependencia del dueno) y el Chromium 1208 se instala aparte. Sin ellos, estos tests no pueden
correr, y eso no es un defecto del codigo.

* Si falta el modulo o el binario fijado y la corrida E2E no se exige: se omiten con motivo.
* Si el binario fijado existe, la funcion no hace nada y los tests se ejecutan (y fallan si algo
  esta mal). Omitirlos con el binario instalado esta prohibido.
* ``CLOUDRAG_REQUIRE_I7_BROWSER=1`` convierte la omision en un fallo. ``init.sh`` lo exporta cuando
  ``PLAYWRIGHT_BROWSERS_PATH`` esta definida, de modo que en la maquina del dueno nunca se omiten
  en silencio.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRE_ENV = "CLOUDRAG_REQUIRE_I7_BROWSER"
SKIP_REASON = "Playwright/pinned Chromium not available in CPU-only environment"
_EXECUTABLE = {"win32": "chrome-win64/chrome.exe", "linux": "chrome-linux64/chrome"}


def _has_playwright() -> bool:
    try:
        return importlib.util.find_spec("playwright") is not None
    except (ImportError, ValueError):
        return False


def pinned_executable() -> Path | None:
    """Ruta esperada del Chromium fijado bajo PLAYWRIGHT_BROWSERS_PATH, o None."""
    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    relative = _EXECUTABLE.get(sys.platform)
    if not browsers or relative is None:
        return None
    pin = json.loads((ROOT / "config/i7-browser-pin.json").read_text(encoding="utf-8"))
    return Path(browsers) / f"chromium-{pin['chromium_revision']}" / relative


def missing_reason(*, need_binary: bool = True) -> str | None:
    if not _has_playwright():
        return "playwright module not installed"
    if need_binary:
        executable = pinned_executable()
        if executable is None or not executable.is_file():
            return "pinned Chromium not installed under PLAYWRIGHT_BROWSERS_PATH"
    return None


def require_i7_browser(*, need_binary: bool = True) -> None:
    """Omite (o falla si se exige) cuando faltan Playwright o el Chromium fijado."""
    reason = missing_reason(need_binary=need_binary)
    if reason is None:
        return
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason}, but {REQUIRE_ENV}=1 requires the local E2E run", pytrace=False)
    pytest.skip(SKIP_REASON)
