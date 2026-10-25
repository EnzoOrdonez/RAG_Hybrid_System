"""Unit tests for scripts/check_docs.py (layer 3 of init.sh)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts" / "check_docs.py")
check_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docs)


def _feature(**overrides):
    feat = {
        "id": "F-001",
        "category": "docs",
        "description": "d",
        "steps": ["s"],
        "status": "passing",
        "passes": True,
    }
    feat.update(overrides)
    return feat


def test_valid_feature_list_passes():
    assert check_docs.validate_features({"features": [_feature()]}) == []


def test_passes_must_match_status():
    errors = check_docs.validate_features({"features": [_feature(status="active", passes=True)]})
    assert any("contradicts" in e for e in errors)


def test_unknown_status_and_extra_key_rejected():
    assert check_docs.validate_features({"features": [_feature(status="done", passes=False)]})
    extra = _feature()
    extra["evidence"] = "x"
    assert check_docs.validate_features({"features": [extra]})


def test_duplicate_ids_rejected():
    errors = check_docs.validate_features({"features": [_feature(), _feature()]})
    assert any("duplicate" in e for e in errors)


def test_clean_root_flags_stray_markdown(tmp_path):
    for name in ("CLAUDE.md", "README.md", "NOTES.md"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    assert check_docs.check_clean_root(tmp_path) == ["root Markdown outside the allowed set: NOTES.md"]


def test_claude_md_line_limit(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("x\n" * 99, encoding="utf-8")
    assert check_docs.check_claude_md(tmp_path) == []
    (tmp_path / "CLAUDE.md").write_text("x\n" * 100, encoding="utf-8")
    assert check_docs.check_claude_md(tmp_path)


def test_broken_link_detected(tmp_path):
    (tmp_path / "README.md").write_text("[a](missing.md) [b](https://x.org)\n", encoding="utf-8")
    errors = check_docs.check_links(tmp_path)
    assert errors == ["README.md: broken link -> missing.md"]
