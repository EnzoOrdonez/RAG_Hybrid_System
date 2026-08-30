"""Tier 3 · Block B — score the negative-control pairs with every verifier.

Scores negative_control_pairs.json (400 claims x 5 random chunks) with the three
NLI verifiers (small/base/large -> [contr,ent,neut] per claim-chunk) and HHEM
(grounding prob per claim-chunk). Persists raw scores so the false-contradicted /
false-grounded base rates (Block B selection criterion) are a CPU re-aggregation.

Runs one verifier per invocation (--verifier) to keep only one model resident
(6 GB VRAM). Output merges into experiments/results/exp15_ablation_nli/
negative_control_scores.json under the verifier key.

Usage:
  python scripts/score_negative_control.py --verifier small|base|large
  python scripts/score_negative_control.py --verifier hhem
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"
PAIRS = OUT / "negative_control_pairs.json"
SCORES = OUT / "negative_control_scores.json"


def score_nli(verifier, pairs):
    from sentence_transformers import CrossEncoder
    import torch
    local = ROOT / "data" / "models" / f"nli-deberta-v3-{verifier}"
    name = str(local) if local.exists() else f"cross-encoder/nli-deberta-v3-{verifier}"
    model = CrossEncoder(name, max_length=512)
    if torch.cuda.is_available():
        model.model.half()
    flat, spans = [], []
    for p in pairs:
        spans.append(len(flat))
        for txt in p["random_chunk_texts"]:
            flat.append((txt, p["claim"]))
    preds = model.predict(flat, batch_size=64, show_progress_bar=False, apply_softmax=True)
    out = []
    for i, p in enumerate(pairs):
        start = spans[i]
        k = len(p["random_chunk_texts"])
        out.append([[float(x[0]), float(x[1]), float(x[2])]
                    for x in preds[start:start + k]])
    return {"schema": "[contr,ent,neut] per claim-chunk", "scores": out}


def score_hhem(pairs):
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location(
        "rg", ROOT / "scripts" / "rescore_grounding_exp15.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    model = rg.load_hhem()
    out = []
    for p in pairs:
        pairs_in = [(txt[:1500], p["claim"]) for txt in p["random_chunk_texts"]]
        with torch.no_grad():
            s = model.predict(pairs_in)
        out.append([round(float(x), 5) for x in s])
    return {"schema": "grounding p(consistent) per claim-chunk", "scores": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True,
                    choices=["small", "base", "large", "hhem"])
    args = ap.parse_args()
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
    merged = json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else {
        "source": "negative_control_pairs.json", "n": len(pairs), "verifiers": {}}
    result = score_hhem(pairs) if args.verifier == "hhem" else score_nli(args.verifier, pairs)
    merged["verifiers"][args.verifier] = result
    SCORES.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"scored {len(pairs)} claims x5 chunks with {args.verifier} -> "
          f"negative_control_scores.json ({list(merged['verifiers'])})")


if __name__ == "__main__":
    main()
