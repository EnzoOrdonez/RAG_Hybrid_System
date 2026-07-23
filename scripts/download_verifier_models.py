"""One-time supervised download of the Tier 3 alternative faithfulness verifiers.

Approved by Enzo 2026-07-23 (ledger summer_ablation_log entrada 4):
  * cross-encoder/nli-deberta-v3-large  — 3-class NLI, drop-in third verifier
  * vectara/hallucination_evaluation_model (HHEM-2.1) — grounding specialist,
    ORTHOGONAL family; ships custom code (trust_remote_code) so keep the .py.

Snapshots into data/models/ with a SHA256+bytes provenance manifest for the
ledger, mirroring download_summer_models.py. MUST run with HF_HUB_OFFLINE unset;
re-freeze offline afterwards.

Usage: python scripts/download_verifier_models.py
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
IGNORE_ALWAYS = ["*.onnx", "*.h5", "*.msgpack", "*.tflite", "openvino*", "*.ot",
                 "onnx/*", "*.tar.gz"]

MODELS = {
    "cross-encoder/nli-deberta-v3-large": "nli-deberta-v3-large",
    "vectara/hallucination_evaluation_model": "hhem-2.1",
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
                "purpose": "Tier 3 alternative verifiers (ledger entrada 4)",
                "models": {}}
    for repo_id, local_name in MODELS.items():
        info = api.model_info(repo_id)
        siblings = [s.rfilename for s in info.siblings]
        has_safetensors = any(f.endswith(".safetensors") for f in siblings)
        # HHEM ships custom *.py code required at load; never ignore .py
        ignore = list(IGNORE_ALWAYS) + (["*.bin"] if has_safetensors else [])
        target = DEST / local_name
        print(f"\n=== {repo_id} -> {target}  (rev {info.sha[:12]}, "
              f"{'safetensors' if has_safetensors else 'bin'})")
        snapshot_download(repo_id=repo_id, revision=info.sha,
                          local_dir=str(target), ignore_patterns=ignore)
        files = {}
        for p in sorted(target.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                files[str(p.relative_to(target)).replace("\\", "/")] = {
                    "sha256": sha256_of(p), "bytes": p.stat().st_size}
        manifest["models"][repo_id] = {
            "local_dir": f"data/models/{local_name}", "revision": info.sha,
            "files": files, "total_bytes": sum(f["bytes"] for f in files.values())}
        print(f"    {len(files)} files, "
              f"{manifest['models'][repo_id]['total_bytes'] / 1e9:.2f} GB")
    out = DEST / "verifier_models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {out}\nRe-freeze: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
