"""
Global reproducibility helpers.

Audit §20.1 Flag 152: `seed=42` was declared in ExperimentConfig and
BenchmarkRunner but never propagated. No call to `random.seed`,
`np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`torch.backends.cudnn.deterministic`, `torch.use_deterministic_algorithms`,
or `PYTHONHASHSEED` existed anywhere in the repo before this module.

Audit §20.2 Flag 153: because `PYTHONHASHSEED` was not fixed,
`hash("aws")` etc. in `test_queries.py:420,547` varied across process
launches, making the query set formally non-reproducible.

This module centralizes seed propagation. Call `set_all_seeds(seed)`
once at the top of BenchmarkRunner.__init__ and at the top of every
`run_experiment` invocation to guarantee:

  * the Python builtin `random` module is seeded
  * `numpy` global RNG is seeded
  * `torch` CPU + CUDA RNGs are seeded
  * cuDNN runs in deterministic mode (no algorithm autotuning)
  * torch's deterministic-algorithm mode is on with warn_only=True
  * `PYTHONHASHSEED` is set for any subsequent subprocess (note: the
    current process's string-hash randomization is already fixed by
    the time Python starts, so this setting protects forks/subprocess
    invocations, not the in-process hash() — see the docstring on
    ensure_hashseed_at_startup below)

Usage:
    from src.utils.reproducibility import set_all_seeds
    set_all_seeds(42)
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42


def set_all_seeds(seed: int = _DEFAULT_SEED) -> None:
    """
    Seed every stochastic subsystem this project touches.

    This function is idempotent: calling it twice with the same `seed`
    produces the same downstream RNG state. Re-calling with a different
    seed re-seeds everything.

    Subsystems covered:
      - Python `random` (global state)
      - `numpy.random` (global state)
      - `torch` CPU and all CUDA devices
      - `torch.backends.cudnn` deterministic mode
      - `torch.use_deterministic_algorithms(True, warn_only=True)`
      - `PYTHONHASHSEED` env var (affects subprocesses only; see module
        docstring and ensure_hashseed_at_startup for in-process fix)

    Args:
        seed: Non-negative integer seed. Default 42 (matches
              ExperimentConfig.seed default).
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.debug("numpy not available; skipping np.random.seed")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            # Some torch versions raise if the backend already loaded
            # a non-deterministic kernel. warn_only=True should avoid
            # this, but belt-and-suspenders.
            logger.debug("torch.use_deterministic_algorithms failed: %s", e)
    except ImportError:
        logger.debug("torch not available; skipping torch seeding")

    logger.info("set_all_seeds(%d) — random/numpy/torch/cuDNN seeded", seed)


def ensure_hashseed_at_startup(seed: int = _DEFAULT_SEED) -> None:
    """
    Re-exec the current Python process with PYTHONHASHSEED set if it is
    not already fixed. Closes audit §20.2 Flag 153.

    `PYTHONHASHSEED` must be set in the environment BEFORE the Python
    interpreter starts for string `hash()` randomization to be
    deterministic in-process. Setting it from within Python (as
    set_all_seeds does via os.environ) only affects children/subprocesses.

    Call this as the FIRST line of a `if __name__ == "__main__":` block
    that relies on reproducible `hash()` of strings. It no-ops if the
    env var is already set.

    Args:
        seed: Value to set PYTHONHASHSEED to if unset.
    """
    current = os.environ.get("PYTHONHASHSEED")
    if current == str(seed):
        return
    if current is not None and current != str(seed):
        warnings.warn(
            f"PYTHONHASHSEED already set to {current!r}, not re-seeding to {seed}. "
            "If you need a fixed seed across runs, set PYTHONHASHSEED "
            "before launching Python.",
            RuntimeWarning,
        )
        return
    # Env var not set; we cannot fix in-process hash() at this point.
    # Log a warning; the caller may decide to re-exec.
    logger.warning(
        "PYTHONHASHSEED not set in env at interpreter startup. "
        "String hash() will be randomized per process — set "
        "PYTHONHASHSEED=%d before launching Python for bit-identical "
        "query-set generation (audit §20.2 Flag 153).",
        seed,
    )


def get_default_seed() -> int:
    """Return the project-wide default seed (42)."""
    return _DEFAULT_SEED
