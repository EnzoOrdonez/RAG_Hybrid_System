"""Unit tests for scripts/check_project_map.py (F-017)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_project_map", ROOT / "scripts" / "check_project_map.py")
check_project_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_project_map)


def _repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_cited_paths_extracts_backticked_repo_paths_with_lines():
    text = "Ver `src/mod.py:2`, `docs/` y `C:/CloudRAG/x.md`; no `pytest -q` ni `exp12`."
    assert check_project_map.cited_paths(text) == [("src/mod.py", 2), ("docs/", None)]


def test_existing_paths_and_lines_pass(tmp_path):
    root = _repo(tmp_path)
    assert check_project_map.check("`src/mod.py:3` y `docs/`", root) == []


def test_missing_path_fails(tmp_path):
    root = _repo(tmp_path)
    errors = check_project_map.check("`src/inventado.py`", root)
    assert errors == ["src/inventado.py: no existe"]


def test_line_beyond_end_of_file_fails(tmp_path):
    root = _repo(tmp_path)
    errors = check_project_map.check("`src/mod.py:9`", root)
    assert errors == ["src/mod.py:9: el archivo tiene 3 líneas"]


def test_line_range_checks_last_line(tmp_path):
    root = _repo(tmp_path)
    assert check_project_map.check("`src/mod.py:1-3`", root) == []
    assert check_project_map.check("`src/mod.py:2-4`", root) == ["src/mod.py:4: el archivo tiene 3 líneas"]


def test_main_exit_code(tmp_path):
    root = _repo(tmp_path)
    good = tmp_path / "good.md"
    good.write_text("`src/mod.py:1`\n", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("`src/nada.py`\n", encoding="utf-8")
    assert check_project_map.main(["--map", str(good), "--root", str(root)]) == 0
    assert check_project_map.main(["--map", str(bad), "--root", str(root)]) == 1


def test_real_project_map_passes():
    assert check_project_map.main([]) == 0
