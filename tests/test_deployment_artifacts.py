import json

import pytest

from src.utils.deployment_artifacts import INDEX_FILES, MODEL_DIRS, build_manifest, verify_manifest


@pytest.fixture
def bundle(tmp_path):
    for name in INDEX_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / INDEX_FILES[1]).write_text(json.dumps({"chunk_ids": ["c1"], "total_vectors": 1}))
    (tmp_path / INDEX_FILES[3]).write_text(json.dumps({"c1": {"chunk_id": "c1", "text": "evidence"}}))
    for name in MODEL_DIRS:
        path = tmp_path / "data/models" / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"synthetic weights; never loaded")
    manifest = tmp_path / "bundle.json"
    manifest.write_text(json.dumps(build_manifest(tmp_path)))
    return tmp_path, manifest


def test_manifest_checks_all_bytes_without_model_loading(bundle):
    root, manifest = bundle
    assert len(verify_manifest(root, manifest)) == 64
    (root / INDEX_FILES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(root, manifest)


def test_manifest_rejects_missing_snapshot(bundle):
    root, manifest = bundle
    (root / "data/models" / MODEL_DIRS[0] / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_manifest(root, manifest)


def test_manifest_rejects_added_traversal_path(bundle):
    root, manifest = bundle
    data = json.loads(manifest.read_text())
    data["files"]["../elsewhere"] = "a" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_manifest(root, manifest)
