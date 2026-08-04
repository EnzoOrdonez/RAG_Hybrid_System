"""The deployed pipeline must start from the models on disk, not from the network.

Measured 2026-08-04: `RAGPipeline` -- the Streamlit demo and the survey config -- loaded
`BAAI/bge-large-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-12-v2` by Hugging Face ID, and
NEITHER is in any HF cache on this machine (only `nli-deberta-v3-small` and `flan-t5-base`
are). Both sit complete under `data/models/` with a manifest recording revision and per-file
sha256. So the demo could not start without re-downloading ~1.3 GB, and under the phase's
declared environment (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) it failed outright with
`OSError: We couldn't connect to 'https://huggingface.co'`.

The scoring scripts never hit this because they load from explicit `data/models/` paths. The
asymmetry between the measured path and the deployed path is the same shape as the NLI test
that loads a hub ID while the detector has a local fallback.

The load-bearing test here is the last one: the deployed retrieval must return exp18's
`baseline_repro_ids` EXACTLY. Preferring a local snapshot is only safe if it is the same model
that built the index, and nothing short of reproducing signed output shows that.

Run: pytest tests/test_local_model_resolution.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import local_models as LM  # noqa: E402

EXP18 = PROJECT_ROOT / "experiments" / "results" / "exp18_evidence_ceiling"
DEPLOYED_MODELS = ["BAAI/bge-large-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-12-v2"]


@pytest.mark.parametrize("model_id", DEPLOYED_MODELS)
def test_the_deployed_models_resolve_to_local_snapshots(model_id):
    p = LM.local_snapshot(model_id)
    if p is None:
        pytest.skip(f"{model_id} not downloaded to data/models/")
    assert p.is_dir()
    assert LM.resolve(model_id) == str(p)


def test_an_unknown_id_is_left_alone():
    """Callers keep passing HF IDs; only a usable local copy changes the source."""
    assert LM.resolve("some/model-we-never-downloaded") == "some/model-we-never-downloaded"
    assert LM.local_snapshot("some/model-we-never-downloaded") is None


def test_empty_and_missing_inputs_are_safe():
    assert LM.local_snapshot("") is None
    assert LM.resolve("") == ""


def test_a_directory_without_weights_does_not_shadow_the_cache(tmp_path, monkeypatch):
    """An interrupted download must not silently become the model that gets loaded."""
    monkeypatch.setattr(LM, "LOCAL_MODELS_DIR", tmp_path)
    (tmp_path / "half-downloaded").mkdir()
    (tmp_path / "half-downloaded" / "config.json").write_text("{}", encoding="utf-8")
    assert LM.local_snapshot("org/half-downloaded") is None

    (tmp_path / "half-downloaded" / "model.safetensors").write_bytes(b"\x00")
    assert LM.local_snapshot("org/half-downloaded") is not None


def test_mapping_is_derived_from_the_id_not_from_a_second_list():
    """A hardcoded ID->dir table is the duplication that caused this phase's silent defects.

    Checked on executable tokens only: the module docstring names both models on purpose, to
    say which ones were missing from the cache.
    """
    from conftest import code_only
    src = code_only((PROJECT_ROOT / "src" / "utils" / "local_models.py").read_text(encoding="utf-8"))
    for model_id in DEPLOYED_MODELS:
        assert model_id not in src, f"{model_id} is hardcoded; derive the dir from the ID"


@pytest.mark.parametrize("rel", ["src/embedding/embedding_manager.py",
                                 "src/reranking/cross_encoder_reranker.py"])
def test_both_production_loaders_route_through_the_resolver(rel):
    src = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    assert "from src.utils.local_models import resolve" in src, rel
    assert "resolve(full_name)" in src, f"{rel} still loads the raw HF id"


@pytest.mark.slow
@pytest.mark.needs_artifacts
def test_deployed_retrieval_reproduces_exp18_ids_exactly():
    """The one that makes the change safe: same model => same retrieval as the signed run."""
    ids_path = EXP18 / "retrieval_ids.json"
    if not ids_path.exists():
        pytest.skip("exp18 retrieval ids not present")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]

    from src.pipeline.pipeline_config import get_config
    from src.pipeline.rag_pipeline import RAGPipeline, load_hybrid_index

    pipeline = RAGPipeline(config=get_config("hybrid"), hybrid_index=load_hybrid_index())
    for qid in sorted(ids)[:3]:
        got = [c.get("chunk_id") or c.get("id")
               for c in pipeline.query(ids[qid]["question"]).retrieved_chunks][:5]
        assert got == ids[qid]["baseline_repro_ids"], (
            f"{qid}: deployed retrieval no longer matches exp18. The local snapshot is NOT the "
            f"model that built the index — do not ship this.")
