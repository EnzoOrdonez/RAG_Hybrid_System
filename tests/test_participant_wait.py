"""Participant wait policy: actual SDK configuration and client-side elapsed feedback."""
from types import SimpleNamespace

import pytest

from src.generation.llm_manager import LLMManager


def test_read_override_preserves_other_http_budgets():
    seen = {}
    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def chat(self, **kwargs):
            return kwargs
    manager = LLMManager('ollama', 'fake', cache_enabled=False,
                         enforce_timeout=True, timeout=60, read_timeout=180)
    manager._ollama_chat(SimpleNamespace(Client=Client), options={})
    timeout = seen['timeout']
    assert (timeout.read, timeout.connect, timeout.write, timeout.pool) == (180, 5, 60, 60)


@pytest.mark.parametrize('mode,read', [('participant', 180), ('development', None)])
def test_loader_applies_read_override_only_to_participants(monkeypatch, mode, read):
    from src.ui.components import index_loader
    import src.pipeline.rag_pipeline as rag
    import src.generation.llm_manager as llm
    monkeypatch.setenv('CLOUDRAG_MODE', mode)
    monkeypatch.setattr(llm, 'LLMManager', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(rag, 'RAGPipeline', lambda **kw: SimpleNamespace(**kw))
    index_loader.load_pipeline.clear()
    try:
        pipeline = index_loader.load_pipeline('hybrid', _hybrid_index=object())
        assert pipeline.llm_manager.read_timeout == read
        assert pipeline.llm_manager.timeout == 60
    finally:
        index_loader.load_pipeline.clear()


def test_feedback_uses_persisted_elapsed_and_monotonic_browser_clock():
    from src.ui.components.wait_feedback import wait_html
    markup = wait_html(120.5)
    assert 'performance.now()' in markup
    assert '120.5' in markup
    assert 'setInterval' in markup
    assert 'fetch(' not in markup  # no server refresh or inference from the clock
    assert 'Recargar la página no cancela la consulta.' in markup
    with pytest.raises(ValueError):
        wait_html(float('nan'))


def test_feedback_handles_server_clock_rollback_without_negative_display():
    from src.ui.components.wait_feedback import wait_html
    assert 'const initial = 0.0;' in wait_html(-5)
