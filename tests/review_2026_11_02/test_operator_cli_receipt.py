"""Reproductor de la auditoría 2026-11-02: R-07 (S2), `run` sombreado en scripts/study_operator/cli.py.

cli.py:67 liga `run` al directorio de la corrida y cli.py:93 lo religa a stimulus_owner.run; cli.py:120
falla con TypeError después de ejecutar la acción remota, sin recibo. Zona congelada: xfail(strict=True)
hasta aplicar docs/review/2026-11-02/patches/004-*.patch (D-04).
"""
import json
import sys
import types
from pathlib import Path

import pytest

import scripts.study_operator.cli as cli


@pytest.mark.parametrize("operation", ["stimulus-start", "stimulus-status", "stimulus-collect"])
@pytest.mark.xfail(strict=True, reason="R-07 (S2): cli.py:93 sombrea `run`; save_state(run/'receipt.json') lanza TypeError")
def test_stimulus_operation_writes_receipt(tmp_path, monkeypatch, operation):
    monkeypatch.chdir(tmp_path)                      # en Linux la raíz 'C:/CloudRAG/...' resuelve bajo cwd
    root = Path("C:/CloudRAG/operator-iteration5").resolve()
    root.mkdir(parents=True)
    (root / "installation.json").write_text(json.dumps({"gcloud": "gcloud", "project": "p"}))
    calls = []

    class Operator:
        def __init__(self, *args):
            self.state = {}

    owner = types.ModuleType("scripts.study_operator.stimulus_owner")
    owner.run = lambda op, name, boot_index=None: calls.append(name) or {"status": "STIMULUS_DONE"}
    monkeypatch.setitem(sys.modules, "scripts.study_operator.stimulus_owner", owner)
    monkeypatch.setattr(cli, "require_limited", lambda: None)
    monkeypatch.setattr(cli, "Cloud", lambda *a, **k: object())
    monkeypatch.setattr(cli, "Operator", Operator)
    argv = [operation] + (["--boot-index", "1"] if operation == "stimulus-start" else [])
    try:
        assert cli.main(argv) == 0
    finally:
        assert calls == [operation]                  # la acción remota sí se ejecutó
    assert list(root.glob("runs/*/receipt.json"))    # ...y debe quedar su recibo
