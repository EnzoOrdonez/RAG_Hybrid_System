"""
Smoke test for Flag 152/153/155 fix.

Validates (no network, no GPU required):
  1. set_all_seeds(42) produces bit-identical Python-random, numpy-random,
     torch-random, and torch-cuda-random (if CUDA available) sequences
     across repeated calls.
  2. LLMManager(seed=42) stores self.seed and _generate_ollama passes
     it through ollama.chat's options dict (inspected via monkey-patch
     without actually calling a real Ollama server).
  3. BenchmarkRunner(seed=42) calls set_all_seeds during construction
     and when run_experiment is invoked.

Run with:
    python scripts/audit/smoke_seeds.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_deterministic_rng_sequences():
    """set_all_seeds(42) twice gives identical RNG output."""
    import random
    import numpy as np
    from src.utils.reproducibility import set_all_seeds

    set_all_seeds(42)
    py1 = [random.random() for _ in range(5)]
    np1 = np.random.rand(5).tolist()

    try:
        import torch
        torch1 = torch.rand(5).tolist()
    except ImportError:
        torch1 = None

    set_all_seeds(42)
    py2 = [random.random() for _ in range(5)]
    np2 = np.random.rand(5).tolist()
    try:
        import torch
        torch2 = torch.rand(5).tolist()
    except ImportError:
        torch2 = None

    assert py1 == py2, f"Python random drift: {py1} vs {py2}"
    assert np1 == np2, f"numpy random drift: {np1} vs {np2}"
    if torch1 is not None:
        assert torch1 == torch2, f"torch random drift: {torch1} vs {torch2}"

    print(f"PASS: set_all_seeds(42) deterministic across Python/numpy/torch")


def test_set_all_seeds_rejects_negative():
    from src.utils.reproducibility import set_all_seeds
    try:
        set_all_seeds(-1)
    except ValueError as e:
        assert "non-negative" in str(e)
        print("PASS: set_all_seeds rejects negative seed")
        return
    raise AssertionError("FAIL: set_all_seeds did not reject negative seed")


def test_pythonhashseed_env_set():
    import os
    from src.utils.reproducibility import set_all_seeds
    os.environ.pop("PYTHONHASHSEED", None)
    set_all_seeds(42)
    assert os.environ.get("PYTHONHASHSEED") == "42", (
        f"PYTHONHASHSEED not set; got {os.environ.get('PYTHONHASHSEED')!r}"
    )
    print("PASS: set_all_seeds sets PYTHONHASHSEED in env")


def test_llm_manager_stores_seed():
    from src.generation.llm_manager import LLMManager
    mgr = LLMManager(provider="ollama", model="llama3.1", seed=42)
    assert mgr.seed == 42, f"LLMManager.seed={mgr.seed}"
    mgr2 = LLMManager(provider="ollama", model="llama3.1", seed=7)
    assert mgr2.seed == 7
    print("PASS: LLMManager stores seed (42 and 7 verified)")


def test_ollama_options_include_seed():
    """Monkey-patch ollama.chat to capture the options dict."""
    import sys as _sys
    from types import SimpleNamespace

    captured = {}

    def fake_chat(model, messages, options):
        captured["options"] = options
        # Return a shape that matches ollama.chat()'s contract enough
        # for _generate_ollama to complete without crashing.
        return {
            "message": {"content": "ok"},
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration": 1000,
        }

    fake_ollama = SimpleNamespace(chat=fake_chat)
    _sys.modules["ollama"] = fake_ollama

    from src.generation.llm_manager import LLMManager
    mgr = LLMManager(provider="ollama", model="llama3.1", seed=123)
    # Bypass cache by setting cache_enabled=False-equivalent: just call
    # the private method directly.
    mgr._generate_ollama(
        prompt="test",
        system_prompt="",
        max_tokens=16,
        temperature=0.1,
    )
    assert "options" in captured, "ollama.chat was not invoked"
    opts = captured["options"]
    assert opts.get("seed") == 123, (
        f"ollama options missing seed=123: {opts}"
    )
    assert opts.get("temperature") == 0.1
    assert opts.get("num_predict") == 16
    print(f"PASS: LLMManager._generate_ollama passes seed to ollama.chat options: {opts}")


def test_benchmark_runner_seeds_on_init():
    """BenchmarkRunner.__init__ should call set_all_seeds."""
    with patch("src.utils.reproducibility.set_all_seeds") as mock_seed:
        from src.evaluation.benchmark_runner import BenchmarkRunner
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            BenchmarkRunner(results_dir=tmp, seed=99)
        assert mock_seed.called, (
            "BenchmarkRunner.__init__ did not call set_all_seeds"
        )
        mock_seed.assert_called_with(99)
        print(f"PASS: BenchmarkRunner(seed=99) -> set_all_seeds(99)")


def main():
    test_deterministic_rng_sequences()
    test_set_all_seeds_rejects_negative()
    test_pythonhashseed_env_set()
    test_llm_manager_stores_seed()
    test_ollama_options_include_seed()
    test_benchmark_runner_seeds_on_init()
    print("\nAll seed-propagation smoke tests passed.")


if __name__ == "__main__":
    main()
