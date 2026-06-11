"""
Benchmark-parity guard for the demo (F6) changes to llm_manager.

The demo added keep_alive and generate_stream. These tests prove the
BENCHMARK path is byte-identical to the pre-demo behavior:
  1. generate() without keep_alive sends ollama.chat EXACTLY the legacy
     kwargs: model, messages, options={temperature, num_predict, seed} —
     no keep_alive key, no extra options.
  2. The cache key function is unchanged for fixed inputs (frozen hash).
  3. generate() with keep_alive (demo) forwards it as a top-level kwarg
     without touching options.

Run: python -m pytest tests/test_benchmark_parity.py -q
"""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.llm_manager import LLMManager  # noqa: E402


def _manager():
    m = LLMManager(provider="ollama", model="fake-model", cache_enabled=False)
    return m


def _capture_chat(monkeypatch_calls):
    fake = types.ModuleType("ollama")

    def chat(**kwargs):
        monkeypatch_calls.append(kwargs)
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    fake.chat = chat
    sys.modules["ollama"] = fake


def test_benchmark_call_is_legacy_identical():
    calls = []
    _capture_chat(calls)
    m = _manager()
    resp = m.generate("p", system_prompt="s", max_tokens=1024,
                      temperature=0.0, config_name="RAG Hibrido Propuesto")
    assert resp.text == "ok" and not resp.error
    assert len(calls) == 1
    kw = calls[0]
    # Exactly the legacy kwargs — nothing demo-related leaks in.
    assert set(kw.keys()) == {"model", "messages", "options"}
    assert kw["options"] == {"temperature": 0.0, "num_predict": 1024, "seed": 42}
    assert kw["messages"] == [{"role": "system", "content": "s"},
                              {"role": "user", "content": "p"}]


def test_demo_keep_alive_is_top_level_and_options_untouched():
    calls = []
    _capture_chat(calls)
    m = _manager()
    m.generate("p", max_tokens=512, temperature=0.0,
               config_name="demo", keep_alive="30m")
    kw = calls[0]
    assert kw["keep_alive"] == "30m"
    assert kw["options"] == {"temperature": 0.0, "num_predict": 512, "seed": 42}


def test_cache_key_frozen():
    # Frozen expected value: sha256 of the legacy key layout. If this test
    # fails, demo changes altered the cache key and benchmark cache hits
    # would silently change — that is a regression.
    import hashlib
    expected = hashlib.sha256(
        "cfg||prompt||sys||0.0||seed=42||max_tokens=1024".encode()
    ).hexdigest()
    got = LLMManager._cache_key("prompt", "sys", 0.0, "cfg", 42, 1024)
    assert got == expected


def test_stream_cache_is_demo_namespaced(tmp_path):
    # Streaming entries live under "demo::<key>": a benchmark entry with the
    # SAME parameters must NOT be served to the stream path, and vice versa.
    calls = []
    _capture_chat(calls)
    m = LLMManager(provider="ollama", model="fake-model", cache_enabled=True)
    m.cache_path = tmp_path / "cache.json"  # keep test writes out of data/llm_cache
    base_key = m._cache_key("p", "s", 0.0, "demo", 42, 512)
    m._cache = {base_key: {"text": "benchmark-answer",
                           "tokens_input": 1, "tokens_output": 1}}
    # benchmark entry present, demo entry absent -> stream must NOT hit it
    # (it would call the network; our fake chat is non-streaming, so instead
    # we pre-seed the demo entry and assert it is the one served).
    m._cache["demo::" + base_key] = {"text": "demo-answer",
                                     "tokens_input": 1, "tokens_output": 1}
    chunks = list(m.generate_stream("p", system_prompt="s", max_tokens=512,
                                    temperature=0.0, config_name="demo"))
    assert chunks == ["demo-answer"]
    assert calls == []  # served from the demo namespace, no network call


def test_benchmark_cache_never_reads_demo_entries(tmp_path):
    # generate() must ignore demo:: entries even when only those exist.
    calls = []
    _capture_chat(calls)
    m = LLMManager(provider="ollama", model="fake-model", cache_enabled=True)
    m.cache_path = tmp_path / "cache.json"  # keep test writes out of data/llm_cache
    base_key = m._cache_key("p", "s", 0.0, "cfg", 42, 1024)
    m._cache = {"demo::" + base_key: {"text": "demo-answer",
                                      "tokens_input": 1, "tokens_output": 1}}
    resp = m.generate("p", system_prompt="s", max_tokens=1024,
                      temperature=0.0, config_name="cfg")
    assert resp.text == "ok"            # generated, NOT the demo entry
    assert not resp.from_cache
    assert len(calls) == 1              # network call happened
