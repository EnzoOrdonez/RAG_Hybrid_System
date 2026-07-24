"""exp15 Tier 3 · Block C — HHEM-2.1 grounding scoring over exp12 answers.

HHEM (vectara/hallucination_evaluation_model) is a GROUNDING verifier from an
ORTHOGONAL family to the deberta NLI trio: it emits a single consistency
probability p(consistent | premise=chunk, hypothesis=claim), no contradiction
channel. This is the strongest evidence on whether the small-vs-base
disagreement (kappa 0.32) is an NLI-family artifact vs a real signal.

Reads the SAME claims_extraction.json + exp12 answers (read-only) as the NLI
scorers; scores every genuine (claim, chunk) pair; persists the raw per-
(config, query, claim, chunk) grounding probs (gzip) so the tau sweep is CPU.
Decision rule (binary, no contradicted class): supported iff
max_chunk p_consistent > tau (default 0.5); else unsupported. faithfulness =
supported / genuine. vacuous (genuine==0) handled as in the NLI path.

HHEM ships custom code (trust_remote_code) and builds its T5 backbone from the
foundation config; we point that at the local data/models/flan-t5-base
(config+tokenizer only, no foundation weights needed — HHEM's own safetensors
populate the backbone).

Outputs (experiments/results/exp15_ablation_nli/):
  grounding_probs__hhem.json.gz, faithfulness_rows__hhem.json  (rows @ tau 0.5)

Usage: python scripts/rescore_grounding_exp15.py [--tau 0.5] [--max-queries N]
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector, classify_artifact)

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
HHEM_LOCAL = PROJECT_ROOT / "data" / "models" / "hhem-2.1"
FOUNDATION_LOCAL = PROJECT_ROOT / "data" / "models" / "flan-t5-base"
EXP12_DIR = PROJECT_ROOT / "experiments/results/exp12_matrix"
OUT_DIR = PROJECT_ROOT / "experiments/results/exp15_ablation_nli"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"


def load_hhem():
    """Load HHEM with its foundation config/tokenizer pinned to the local dir."""
    import importlib.util
    import torch
    from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification

    # import the custom classes from the snapshot
    spec = importlib.util.spec_from_file_location(
        "modeling_hhem_v2", HHEM_LOCAL / "modeling_hhem_v2.py")
    mod = importlib.util.module_from_spec(spec)
    # its relative import `from .configuration_hhem_v2 import HHEMv2Config` needs a package;
    # load the config module first under the expected name
    cfg_spec = importlib.util.spec_from_file_location(
        "configuration_hhem_v2", HHEM_LOCAL / "configuration_hhem_v2.py")
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    sys.modules["configuration_hhem_v2"] = cfg_mod
    # patch relative import target
    import types
    pkg = types.ModuleType("hhem_pkg")
    sys.modules["modeling_hhem_v2"] = mod
    mod.__package__ = ""
    # rewrite the relative import by injecting HHEMv2Config into module globals pre-exec
    src = (HHEM_LOCAL / "modeling_hhem_v2.py").read_text(encoding="utf-8").replace(
        "from .configuration_hhem_v2 import HHEMv2Config",
        "from configuration_hhem_v2 import HHEMv2Config")
    exec(compile(src, str(HHEM_LOCAL / "modeling_hhem_v2.py"), "exec"), mod.__dict__)

    HHEMv2Config = cfg_mod.HHEMv2Config
    HHEMv2Config.foundation = str(FOUNDATION_LOCAL)  # pin offline
    config = HHEMv2Config()
    model = mod.HHEMv2ForSequenceClassification(config)
    # load HHEM weights: safetensors keys are prefixed "t5." (the full
    # HHEMv2ForSequenceClassification state, whose submodule is self.t5), so we
    # load into `model`, NOT model.t5 (that mismatched -> strict=False dropped
    # every weight -> random T5 -> garbage scores; caught by the controlled
    # sanity test sky-blue->high / sky-red->low).
    from safetensors.torch import load_file
    state = load_file(str(HHEM_LOCAL / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    if n_loaded < 100:
        raise SystemExit(f"HHEM load FAILED: only {n_loaded}/{len(state)} tensors matched "
                         f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    if torch.cuda.is_available():
        model = model.half().cuda()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__smoke{args.max_queries}" if args.max_queries else ""

    model = load_hhem()
    det = HallucinationDetector(use_nli=False)
    results = json.loads((EXP12_DIR / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    probs_out = {"model": "vectara/hallucination_evaluation_model (HHEM-2.1)",
                 "score": "p(consistent | premise=chunk, hypothesis=claim)",
                 "source": "exp12_matrix/results.json (read-only)",
                 "generated_by": "scripts/rescore_grounding_exp15.py", "configs": {}}
    rows_out = {"model": "hhem-2.1", "tau": args.tau, "rule": "supported iff max_chunk p>tau",
                "generated_by": "scripts/rescore_grounding_exp15.py", "configs": {}}
    # resume: per-config checkpoint (the full run is ~3 h on 6 GB)
    part_path = OUT_DIR / f"grounding_probs__hhem{suffix}.partial.json.gz"
    rows_part = OUT_DIR / f"faithfulness_rows__hhem{suffix}.partial.json"
    if not args.max_queries and part_path.exists():
        with gzip.open(part_path, "rt", encoding="utf-8") as f:
            probs_out["configs"] = json.load(f).get("configs", {})
        if rows_part.exists():
            rows_out["configs"] = json.loads(rows_part.read_text(encoding="utf-8")).get("configs", {})
        print(f"resuming, {len(probs_out['configs'])} configs done", flush=True)
    t0 = time.time()

    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            if cname in probs_out["configs"]:
                continue
            rows = results[cname]["results"]
            if args.max_queries:
                rows = [r for r in rows
                        if (r.get("hallucination_metrics") or {}).get("method") == "nli"][:args.max_queries]
            cfg_probs, cfg_rows = {}, {}
            pairs, spans = [], []
            for r in rows:
                answer = r.get("answer") or ""
                claims = det._extract_claims(answer) if answer.strip() else []
                genuine = [c for c in claims if not classify_artifact(c)]
                n_art = len(claims) - len(genuine)
                cids = [cid for cid in r["retrieved_ids"] if cid in chunk_map]
                # truncate premise to ~1500 chars (~375 tok) so premise+claim+prompt
                # stays under flan-t5's 512-token window: avoids the 664>512 warning,
                # the batch-of-long-sequences OOM (10.4 GiB on 6 GB), and the ~1.9 h/config
                # slowdown. Chunks are size-500 so most are already short.
                texts = [chunk_map[cid]["text"][:1500] for cid in cids]
                if not claims or not texts:
                    cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                               "genuine": 0, "supported": 0, "unsupported": 0,
                                               "faithfulness": (1.0 if genuine == [] and claims
                                                                else None),
                                               "method": "vacuous" if (claims and not genuine) else "none"}
                    continue
                if not genuine:
                    cfg_probs[r["query_id"]] = []
                    cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                               "genuine": 0, "supported": 0, "unsupported": 0,
                                               "faithfulness": 1.0, "method": "vacuous"}
                    continue
                spans.append((r["query_id"], genuine, len(texts), n_art, len(pairs)))
                pairs.extend((t, cl) for cl in genuine for t in texts)  # (premise, hypothesis)
            scores = []
            if pairs:
                import torch
                B = 16  # T5 on 6 GB: 64 OOMs on batches of long sequences
                for i in range(0, len(pairs), B):
                    with torch.no_grad():
                        s = model.predict(pairs[i:i + B])
                    scores.extend(float(x) for x in s)
            for qid, genuine, k, n_art, start in spans:
                q_probs, sup = [], 0
                for ci in range(len(genuine)):
                    sc_chunks = scores[start + ci * k: start + (ci + 1) * k]
                    q_probs.append([round(x, 5) for x in sc_chunks])
                    if max(sc_chunks) > args.tau:
                        sup += 1
                g = len(genuine)
                cfg_probs[qid] = q_probs
                cfg_rows[qid] = {"total_claims": g + n_art, "not_a_claim": n_art, "genuine": g,
                                 "supported": sup, "unsupported": g - sup,
                                 "faithfulness": round(sup / g, 4), "method": "nli"}
            probs_out["configs"][cname] = cfg_probs
            rows_out["configs"][cname] = cfg_rows
            if not args.max_queries:
                with gzip.open(part_path, "wt", encoding="utf-8") as f:
                    json.dump(probs_out, f)
                rows_part.write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
            print(f"  {cname}: {len(cfg_rows)} responses, {len(pairs)} pairs "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with gzip.open(OUT_DIR / f"grounding_probs__hhem{suffix}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(probs_out, f)
    (OUT_DIR / f"faithfulness_rows__hhem{suffix}.json").write_text(
        json.dumps(rows_out, indent=1), encoding="utf-8")
    if not args.max_queries:
        part_path.unlink(missing_ok=True)
        rows_part.unlink(missing_ok=True)
    print(f"wrote grounding_probs__hhem{suffix}.json.gz + faithfulness_rows__hhem{suffix}.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
