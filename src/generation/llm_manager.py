"""
LLM Manager - Unified interface for multiple LLM providers.

Supports:
  - Ollama (local): llama3.1, mistral, qwen2.5, phi3-mini
  - OpenAI API: gpt-4o-mini
  - Anthropic API: claude-sonnet

Features:
  - Retry with exponential backoff (3 attempts, delay 2^n seconds)
  - Timeout of 60 seconds per request
  - Logging of each call: model, tokens in/out, latency, success/error
  - Disk cache: data/llm_cache/{model}_cache.json
  - Graceful handling if Ollama is not running or API keys missing
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class LLMResponse(BaseModel):
    """Structured response from any LLM provider."""
    text: str
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    from_cache: bool
    error: Optional[str] = None


class LLMError(Exception):
    """Raised when LLM call fails after all retries."""
    pass


# ============================================================
# Model configurations
# ============================================================

LLM_CONFIGS = {
    # Local (Ollama) - FREE, for development and most experiments
    # GPU: RTX 3060 Laptop 6GB VRAM. Only models <= 5GB VRAM.
    "llama3.1": {
        "provider": "ollama",
        "model": "llama3.1:8b-instruct-q4_K_M",
        "vram_gb": 5,
        "description": "Meta Llama 3.1 8B - good quality/speed balance",
    },
    "mistral": {
        "provider": "ollama",
        "model": "mistral:7b-instruct-v0.3-q4_K_M",
        "vram_gb": 5,
        "description": "Mistral 7B - good at following instructions",
    },
    "qwen2.5": {
        "provider": "ollama",
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "vram_gb": 5,
        "description": "Qwen 2.5 7B - strong reasoning",
    },
    "phi3-mini": {
        "provider": "ollama",
        "model": "phi3:3.8b-mini-instruct-4k-q4_K_M",
        "vram_gb": 3,
        "description": "Phi-3 Mini 3.8B - lightweight, fits in 6GB VRAM",
    },
    # APIs - COST MONEY, use only for final evaluation
    "gpt4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "description": "GPT-4o Mini - commercial reference",
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "description": "Claude Sonnet - commercial reference",
    },
}


class LLMManager:
    """Unified LLM manager with caching, retry, and multi-provider support."""

    def __init__(
        self,
        provider: str,
        model: str,
        cache_enabled: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.provider = provider
        self.model = model
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        self.timeout = timeout

        # Resolve from config shortnames
        if model in LLM_CONFIGS:
            cfg = LLM_CONFIGS[model]
            self.provider = cfg["provider"]
            self.model = cfg["model"]

        # Cache directory
        self.cache_dir = PROJECT_ROOT / "data" / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_safe = self.model.replace("/", "_").replace(":", "_")
        self.cache_path = self.cache_dir / f"{model_safe}_cache.json"
        self._cache = self._load_cache()

        # Load env vars
        self._load_env()

    def _load_env(self):
        """Load .env file if exists."""
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            self.cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save cache: %s", e)

    @staticmethod
    def _cache_key(prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate deterministic cache key."""
        content = f"{prompt}||{system_prompt}||{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    # ============================================================
    # Main generate method
    # ============================================================

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate response. Temperature=0.1 for experimental reproducibility.
        Uses cache if enabled. Retries 3 times with exponential backoff.
        """
        # Check cache
        if self.cache_enabled:
            key = self._cache_key(prompt, system_prompt, temperature)
            if key in self._cache:
                cached = self._cache[key]
                logger.info("Cache hit for %s (key=%s...)", self.model, key[:8])
                return LLMResponse(
                    text=cached["text"],
                    model=self.model,
                    provider=self.provider,
                    tokens_input=cached.get("tokens_input", 0),
                    tokens_output=cached.get("tokens_output", 0),
                    latency_ms=0.0,
                    from_cache=True,
                )

        # Generate with retry
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()

                if self.provider == "ollama":
                    response = self._generate_ollama(
                        prompt, system_prompt, max_tokens, temperature
                    )
                elif self.provider == "openai":
                    response = self._generate_openai(
                        prompt, system_prompt, max_tokens, temperature
                    )
                elif self.provider == "anthropic":
                    response = self._generate_anthropic(
                        prompt, system_prompt, max_tokens, temperature
                    )
                else:
                    raise LLMError(f"Unknown provider: {self.provider}")

                elapsed_ms = (time.perf_counter() - start) * 1000
                response.latency_ms = elapsed_ms

                # Log the call
                logger.info(
                    "LLM call: model=%s, tokens_in=%d, tokens_out=%d, "
                    "latency=%.0fms, attempt=%d",
                    self.model,
                    response.tokens_input,
                    response.tokens_output,
                    elapsed_ms,
                    attempt + 1,
                )

                # Cache the response
                if self.cache_enabled and not response.error:
                    key = self._cache_key(prompt, system_prompt, temperature)
                    self._cache[key] = {
                        "text": response.text,
                        "tokens_input": response.tokens_input,
                        "tokens_output": response.tokens_output,
                    }
                    self._save_cache()

                return response

            except LLMError:
                raise  # Don't retry config errors (missing keys, wrong model)
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        # All retries exhausted
        error_msg = f"All {self.max_retries} retries failed: {last_error}"
        logger.error(error_msg)
        return LLMResponse(
            text="",
            model=self.model,
            provider=self.provider,
            tokens_input=0,
            tokens_output=0,
            latency_ms=0.0,
            from_cache=False,
            error=error_msg,
        )

    # ============================================================
    # Provider-specific implementations
    # ============================================================

    def _generate_ollama(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate via Ollama (local)."""
        try:
            import ollama
        except ImportError:
            raise LLMError(
                "ollama package not installed. Run: pip install ollama"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
        except Exception as e:
            error_str = str(e)
            if "connection" in error_str.lower() or "refused" in error_str.lower():
                raise LLMError(
                    "Ollama is not running. Start it with 'ollama serve' in another terminal. "
                    "If not installed: https://ollama.com/download"
                )
            if "not found" in error_str.lower() or "pull" in error_str.lower():
                raise LLMError(
                    f"Model {self.model} not downloaded. Run: ollama pull {self.model}"
                )
            raise

        text = response.get("message", {}).get("content", "")
        tokens_in = response.get("prompt_eval_count", 0) or 0
        tokens_out = response.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            model=self.model,
            provider="ollama",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=0.0,
            from_cache=False,
        )

    def _generate_openai(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate via OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY not set in .env file. "
                "API models are optional - use Ollama local models for development."
            )

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return LLMResponse(
            text=text,
            model=self.model,
            provider="openai",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=0.0,
            from_cache=False,
        )

    def _generate_anthropic(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Generate via Anthropic API."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY not set in .env file. "
                "API models are optional - use Ollama local models for development."
            )

        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        tokens_in = response.usage.input_tokens if response.usage else 0
        tokens_out = response.usage.output_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            model=self.model,
            provider="anthropic",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=0.0,
            from_cache=False,
        )

    # ============================================================
    # Utility
    # ============================================================

    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            if self.provider == "ollama":
                import ollama
                models = ollama.list()
                model_names = []
                # Handle different ollama SDK versions
                if hasattr(models, "models"):
                    model_names = [m.model for m in models.models]
                elif isinstance(models, dict):
                    model_names = [m.get("name", "") for m in models.get("models", [])]
                # Check if our model (or its base name) is in the list
                base = self.model.split(":")[0]
                return any(base in n for n in model_names)
            elif self.provider == "openai":
                return bool(os.getenv("OPENAI_API_KEY"))
            elif self.provider == "anthropic":
                return bool(os.getenv("ANTHROPIC_API_KEY"))
        except Exception:
            return False
        return False

    def clear_cache(self):
        """Clear the LLM response cache."""
        self._cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Cache cleared for %s", self.model)

    @staticmethod
    def list_available() -> dict:
        """List all configured models and their availability."""
        result = {}
        for name, cfg in LLM_CONFIGS.items():
            mgr = LLMManager(cfg["provider"], name)
            result[name] = {
                "provider": cfg["provider"],
                "model": cfg["model"],
                "description": cfg["description"],
                "available": mgr.is_available(),
            }
        return result
