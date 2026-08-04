"""Resolve a Hugging Face model ID to its local snapshot under `data/models/`, if present.

Why this exists. The scoring scripts load their verifiers from explicit `data/models/<name>`
paths, but the DEPLOYMENT path -- `RAGPipeline`, hence the Streamlit demo and the survey
config -- loaded by Hugging Face ID (`BAAI/bge-large-en-v1.5`,
`cross-encoder/ms-marco-MiniLM-L-12-v2`). Measured 2026-08-04: neither is in any HF cache on
this machine (only `nli-deberta-v3-small` and `flan-t5-base` are), while both sit complete
under `data/models/` with a manifest recording revision and per-file sha256. So the deployed
pipeline could not start at all without re-downloading ~1.3 GB, and under the phase's declared
environment (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) it failed outright.

This is a LOADING fix, not a model change: same weights, same revision, read from disk instead
of from the network. Callers keep passing the HF ID and get the local snapshot when it exists,
so nothing changes on a machine that has the models cached.

The mapping is by directory name, derived from the ID's last path segment, so a new model
dropped into `data/models/` is picked up with no code change and no second list to drift.
"""

from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_MODELS_DIR = PROJECT_ROOT / "data" / "models"

# A sentence-transformers / transformers snapshot is only usable if the weights are there.
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def local_snapshot(model_id: str) -> Union[Path, None]:
    """Path to the local snapshot for `model_id`, or None if it is not usable.

    `BAAI/bge-large-en-v1.5` -> `data/models/bge-large-en-v1.5` when that directory holds
    weights. A directory that exists but is empty (an interrupted download) returns None, so a
    half-finished snapshot never silently shadows a working cache entry.
    """
    if not model_id:
        return None
    name = model_id.rstrip("/").split("/")[-1]
    d = LOCAL_MODELS_DIR / name
    if d.is_dir() and any((d / f).exists() for f in _WEIGHT_FILES):
        return d
    return None


def resolve(model_id: str) -> str:
    """`model_id` unchanged, or the local snapshot path when one is usable."""
    p = local_snapshot(model_id)
    return str(p) if p else model_id
