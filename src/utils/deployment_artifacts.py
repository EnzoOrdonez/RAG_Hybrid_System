"""Hash and verify an existing deployment bundle, without loading any model."""

import hashlib
import json
from pathlib import Path

INDEX_FILES = (
    "data/indices/faiss_bge-large_adaptive_500.index",
    "data/indices/faiss_bge-large_adaptive_500.mapping.json",
    "data/indices/bm25_adaptive_500.pkl",
    "data/indices/chunk_map_bge-large_adaptive_500.json",
    "data/evaluation/test_queries.json",
)
MODEL_DIRS = ("bge-large-en-v1.5", "ms-marco-MiniLM-L-12-v2", "nli-deberta-v3-small")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Invalid artifact path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Artifact path must be relative and confined")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes root")
    return path


def build_manifest(root):
    root = Path(root).resolve()
    paths = list(INDEX_FILES)
    for name in MODEL_DIRS:
        directory = root / "data/models" / name
        if not (directory / "config.json").is_file() or not any((directory / w).is_file() for w in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"Incomplete local snapshot: {name}")
        paths.extend(p.relative_to(root).as_posix() for p in sorted(directory.rglob("*")) if p.is_file())
    files = {name: digest(_inside(root, name)) for name in paths}
    return {"schema_version": 1, "files": files}


def verify_manifest(root, manifest_path):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported artifact manifest")
    actual = build_manifest(root)
    if set(manifest["files"]) != set(actual["files"]):
        raise ValueError("Deployment bundle file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        _inside(root, name)
        if actual["files"][name] != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    mapping = json.loads((root / INDEX_FILES[1]).read_text(encoding="utf-8"))
    chunks = json.loads((root / INDEX_FILES[3]).read_text(encoding="utf-8"))
    ids = mapping["chunk_ids"]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(chunks) or mapping["total_vectors"] != len(ids):
        raise ValueError("FAISS mapping and chunk map are inconsistent")
    return digest(manifest_path)
