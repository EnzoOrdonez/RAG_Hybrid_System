"""El guard omite en un entorno CPU-only y jamas omite si el Chromium fijado existe."""

import sys

import pytest

from tests import i7_browser_guard as guard


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.delenv(guard.REQUIRE_ENV, raising=False)


def _install_fake_chromium(tmp_path, monkeypatch):
    executable = tmp_path / "chromium-1208" / guard._EXECUTABLE[sys.platform]
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))


def test_skips_when_playwright_is_missing(clean_env, monkeypatch):
    monkeypatch.setattr(guard, "_has_playwright", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="not available in CPU-only environment"):
        guard.require_i7_browser()


def test_skips_when_module_exists_but_pinned_chromium_does_not(clean_env, monkeypatch):
    monkeypatch.setattr(guard, "_has_playwright", lambda: True)
    with pytest.raises(pytest.skip.Exception):
        guard.require_i7_browser()


@pytest.mark.skipif(sys.platform not in guard._EXECUTABLE, reason="plataforma sin binario fijado")
def test_never_skips_when_the_pinned_binary_exists(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "_has_playwright", lambda: True)
    _install_fake_chromium(tmp_path, monkeypatch)
    guard.require_i7_browser()  # no lanza: los tests se ejecutan
    # ... ni siquiera con la exigencia activa
    monkeypatch.setenv(guard.REQUIRE_ENV, "1")
    guard.require_i7_browser()


def test_requirement_turns_the_skip_into_a_failure(clean_env, monkeypatch):
    monkeypatch.setattr(guard, "_has_playwright", lambda: False)
    monkeypatch.setenv(guard.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="requires the local E2E run"):
        guard.require_i7_browser()


def test_module_only_check_does_not_need_the_binary(clean_env, monkeypatch):
    monkeypatch.setattr(guard, "_has_playwright", lambda: True)
    guard.require_i7_browser(need_binary=False)
    monkeypatch.setattr(guard, "_has_playwright", lambda: False)
    with pytest.raises(pytest.skip.Exception):
        guard.require_i7_browser(need_binary=False)
