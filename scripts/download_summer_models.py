"""One-time supervised download of the retrieval-stack models (summer phase).

Approved by Enzo 2026-07-22 (ledger: paper/summer_ablation_log.md, entrada 0):
the HF cache was purged ~2026-06-30, leaving the machine unable to run dense/
hybrid retrieval, reranking, or oracle scoring. This script snapshots the three
missing models into data/models/ (durable, like the NLI pair) and writes a
provenance manifest (repo revision + per-file sha256) for the ledger.

MUST run with HF_HUB_OFFLINE **unset** (the one sanctioned online session).
After it finishes, re-freeze HF_HUB_OFFLINE=1 for all experiment work.

Usage:
  python scripts/download_summer_models.py
"""

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

if os.environ.get("HF_HUB_OFFLINE") == "1":
    sys.exit("HF_HUB_OFFLINE=1 is set - unset it for this one sanctioned download session.")

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "models"

# Weight-file preferences: skip formats transformers/sentence-transformers
# never load here; fall back to *.bin only when a repo has no safetensors.
IGNORE_ALWAYS = ["*.onnx", "*.h5", "*.msgpack", "*.tflite", "openvino*", "*.ot",
                 "onnx/*", "*.tar.gz"]

MODELS = {
    "BAAI/bge-large-en-v1.5": "bge-large-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-large": "bge-reranker-large",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    manifest = {"date": date.today().isoformat(),
                "purpose": "summer phase Tier B retrieval ablation (ledger entrada 0)",
                "models": {}}
    for repo_id, local_name in MODELS.items():
        info = api.model_info(repo_id)
        siblings = [s.rfilename for s in info.siblings]
        has_safetensors = any(f.endswith(".safetensors") for f in siblings)
        ignore = list(IGNORE_ALWAYS) + (["*.bin"] if has_safetensors else [])
        target = DEST / local_name
        print(f"\n=== {repo_id} -> {target}  (revision {info.sha[:12]}, "
              f"{'safetensors' if has_safetensors else 'bin'})")
        snapshot_download(repo_id=repo_id, revision=info.sha,
                          local_dir=str(target), ignore_patterns=ignore)
        files = {}
        for p in sorted(target.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                files[str(p.relative_to(target)).replace("\\", "/")] = {
                    "sha256": sha256_of(p), "bytes": p.stat().st_size}
        manifest["models"][repo_id] = {
            "local_dir": f"data/models/{local_name}",
            "revision": info.sha,
            "files": files,
            "total_bytes": sum(f["bytes"] for f in files.values()),
        }
        print(f"    {len(files)} files, "
              f"{manifest['models'][repo_id]['total_bytes'] / 1e9:.2f} GB")
    out = DEST / "summer_models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {out}")
    print("Re-freeze the environment now: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
