import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_embedding_cache_cannot_reuse_changed_text(tmp_path, monkeypatch):
    from src.embedding.embedding_manager import EmbeddingManager
    manager = EmbeddingManager(cache_dir=tmp_path, device="cpu")
    calls = []
    def embed(texts):
        calls.append(texts)
        return np.ones((len(texts), 1024), dtype=np.float32) * len(calls)
    monkeypatch.setattr(manager, "embed_documents", embed)
    first, _ = manager.embed_and_cache(["old"], ["c1"], "adaptive", 500)
    second, _ = manager.embed_and_cache(["new"], ["c1"], "adaptive", 500)
    assert len(calls) == 2
    assert not np.array_equal(first, second)
    manager.embed_and_cache(["new"], ["c1"], "adaptive", 500)
    assert len(calls) == 2


def test_load_rejects_missing_chunk_map_and_mismatched_ids(tmp_path):
    from src.embedding.index.hybrid_index import HybridIndex
    index = HybridIndex(SimpleNamespace(get_dimension=lambda: 2, get_model_name=lambda: "fake"), indices_dir=tmp_path)
    index.faiss_index = SimpleNamespace(load=lambda path: None, chunk_ids=["a"], index=SimpleNamespace(ntotal=1, d=2), dimension=2)
    index.bm25_index = SimpleNamespace(load=lambda path: None, chunk_ids=["b"], corpus_tokens=[["text"]])
    with pytest.raises((ValueError, FileNotFoundError)):
        index.load()
    (tmp_path / "chunk_map_fake_adaptive_500.json").write_text(json.dumps({"a": {"chunk_id": "a", "text": "text"}}))
    with pytest.raises(ValueError, match="IDs"):
        index.load()


def test_nli_prediction_failure_is_reported_as_mixed(monkeypatch):
    from src.generation.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()
    detector._nli_model = SimpleNamespace(predict=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failure")))
    detector._nli_available = True
    monkeypatch.setattr(detector, "_extract_claims", lambda text: ["AWS S3 stores objects in buckets."])
    report = detector.check("answer", [{"chunk_id": "c1", "text": "AWS S3 stores objects in buckets."}])
    assert report.method == "mixed"
    assert report.claim_details[0].verification_method == "keyword_fallback"
    assert report.claim_details[0].verification_error == "RuntimeError"


def test_deployment_ollama_client_gets_timeout_and_context(monkeypatch):
    import sys
    from src.generation.llm_manager import LLMManager
    calls = {}
    class Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
        def chat(self, **kwargs):
            calls["chat"] = kwargs
            return {"message": {"content": "ok"}}
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Client))
    manager = LLMManager("ollama", "fake", cache_enabled=False, enforce_timeout=True, num_ctx=4096)
    assert manager.generate("prompt").text == "ok"
    assert calls["client"]["timeout"].read == 60
    assert calls["chat"]["options"]["num_ctx"] == 4096


def test_deployment_rechecks_model_digest_before_each_generation(monkeypatch):
    import sys
    from src.generation.llm_manager import LLMManager, LLMError
    state = {"digest": "a" * 64, "calls": 0}
    class Client:
        def __init__(self, **kwargs):
            pass
        def list(self):
            return {"models": [{"model": "fake", "digest": state["digest"]}]}
        def chat(self, **kwargs):
            state["calls"] += 1
            return {"message": {"content": "ok"}}
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Client))
    manager = LLMManager("ollama", "fake", cache_enabled=False, enforce_timeout=True,
                         expected_model_digest="a" * 64)
    assert manager.generate("first").text == "ok"
    state["digest"] = "b" * 64
    with pytest.raises(LLMError):
        manager.generate("second")
    assert state["calls"] == 1
