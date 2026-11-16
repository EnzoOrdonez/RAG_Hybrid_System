"""Auditoría Fable (H-09): robustez de la caché de `LLMManager` y del log de reintentos.

Clase A: nada de esto cambia el texto generado, la clave de caché ni el contenido del archivo.
Lo que cambia es qué pasa cuando la escritura falla a medias, cuando la caché no se puede leer
y qué dice el log en el último intento. Sin red y sin modelos: Ollama es un doble.
"""

import json
import logging
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import llm_manager as llm_module  # noqa: E402
from src.generation.llm_manager import LLMManager  # noqa: E402

ENTRY = {"text": "respuesta", "tokens_input": 3, "tokens_output": 5}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Gestor con la caché redirigida a tmp_path y sin tocar `.env` ni `data/llm_cache`."""
    monkeypatch.setattr(LLMManager, "_load_env", lambda self: None)
    m = LLMManager(provider="ollama", model="fake-model", cache_enabled=True)
    m.cache_path = tmp_path / "fake-model_cache.json"
    m._cache = {}
    return m


def _torn_write(real_write_text):
    """Doble de `Path.write_text` que escribe la mitad y falla, como un corte a media escritura."""
    def torn(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("disco lleno")
    return torn


def test_a_failed_write_leaves_the_previous_cache_file_intact(manager, monkeypatch):
    """Un corte a media escritura no debe dejar un JSON truncado en el lugar de la caché."""
    manager._cache = {"k1": ENTRY}
    manager._save_cache()
    before = manager.cache_path.read_bytes()

    monkeypatch.setattr(Path, "write_text", _torn_write(Path.write_text))
    manager._cache = {"k1": ENTRY, "k2": ENTRY}
    manager._save_cache()  # no debe propagar: el fallo de caché nunca tumba una generación

    assert manager.cache_path.read_bytes() == before
    assert json.loads(manager.cache_path.read_text(encoding="utf-8")) == {"k1": ENTRY}


def test_a_failed_write_leaves_no_temporary_file_behind(manager, monkeypatch):
    manager._cache = {"k1": ENTRY}
    manager._save_cache()
    monkeypatch.setattr(Path, "write_text", _torn_write(Path.write_text))
    manager._save_cache()

    assert sorted(p.name for p in manager.cache_path.parent.iterdir()) == [manager.cache_path.name]


def test_saved_bytes_are_identical_to_the_legacy_serialisation(manager, tmp_path):
    """Equivalencia: mismo contenido que `write_text(json.dumps(indent=2, ensure_ascii=False))`."""
    manager._cache = {"k1": ENTRY, "ñ": {"text": "año — «cita»", "tokens_input": 0, "tokens_output": 0}}
    manager._save_cache()

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(manager._cache, indent=2, ensure_ascii=False), encoding="utf-8")
    assert manager.cache_path.read_bytes() == legacy.read_bytes()


def test_an_unreadable_cache_is_reported_not_silently_dropped(manager, caplog):
    manager.cache_path.write_text('{"k1": {"text": "cort', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=llm_module.__name__):
        loaded = manager._load_cache()

    assert loaded == {}  # el comportamiento no cambia: se sigue con caché vacía
    assert any(manager.cache_path.name in r.getMessage() for r in caplog.records), \
        "una caché ilegible debe dejar un aviso con el nombre del archivo"


def test_the_last_failed_attempt_does_not_announce_a_retry(manager, monkeypatch, caplog):
    fake = types.ModuleType("ollama")

    def chat(**kwargs):
        raise RuntimeError("boom")

    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    manager.cache_enabled = False
    manager.max_retries = 2

    with caplog.at_level(logging.WARNING, logger=llm_module.__name__):
        response = manager.generate("p", system_prompt="s")

    assert response.error and response.text == ""
    assert sleeps == [2], "el reintento y su espera no cambian"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("retrying in" in w for w in warnings) == 1
    assert "no retries left" in warnings[-1]


def _counting_ollama(monkeypatch, content):
    """Doble de Ollama que devuelve siempre `content` y cuenta las llamadas."""
    fake = types.ModuleType("ollama")
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": content}, "prompt_eval_count": 7, "eval_count": 0}

    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    return calls


def test_a_non_empty_answer_is_generated_once_and_then_served_from_cache(manager, monkeypatch):
    """Caracterización: la caché es lo que hace reproducible una corrida ya generada."""
    calls = _counting_ollama(monkeypatch, "S3 stores objects in buckets.")

    first = manager.generate("p", system_prompt="s", config_name="cfg")
    second = manager.generate("p", system_prompt="s", config_name="cfg")

    assert len(calls) == 1
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.text == first.text and second.latency_ms == 0.0


@pytest.mark.xfail(strict=True, reason="FABLE-B8: pendiente de decisión de Enzo")
def test_an_empty_answer_is_not_frozen_into_the_cache(manager, monkeypatch):
    """H-08: `generate()` cachea hoy una respuesta vacía y no vuelve a preguntar al modelo.

    `generate_stream()` ya la descarta (`if self.cache_enabled and text`).
    """
    calls = _counting_ollama(monkeypatch, "")

    first = manager.generate("p", system_prompt="s", config_name="cfg")
    second = manager.generate("p", system_prompt="s", config_name="cfg")

    assert first.text == "" and first.error is None
    assert len(calls) == 2, "la segunda llamada debe volver al modelo, no a la caché"
    assert second.from_cache is False
