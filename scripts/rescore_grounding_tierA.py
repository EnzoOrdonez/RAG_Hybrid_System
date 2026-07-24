"""exp15 Tier A · HHEM-2.1 grounding scoring over the 5 ablation arms (read-only).

Triangulates the Tier A NLI null: Tier 3 showed NLI is the noisy instrument that
MASKS the granite retrieval->faithfulness effect that HHEM reveals. So before
calling the context-ablation null "robust", re-score the same 5 arms with the
clean grounding instrument (HHEM). Reuses load_hhem() and the IDENTICAL scoring
rule as scripts/rescore_grounding_exp15.py (tau 0.5, max_chunk p>tau, premise
truncated to 1500 chars, batch 16, same vacuous/decline handling) so the arm
numbers are directly comparable to the exp12 HHEM numbers.

Source: experiments/results/exp15_ablation_tierA/results.json (read-only).
Out:    experiments/results/exp15_ablation_tierA/{grounding_probs__hhem.json.gz,
        faithfulness_rows__hhem.json}
Usage:  python scripts/rescore_grounding_tierA.py [--tau 0.5]
Env:    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
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
from scripts.rescore_grounding_exp15 import load_hhem  # noqa: E402

DEFAULT_EXP_DIR = PROJECT_ROOT / "experiments/results/exp15_ablation_tierA"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--exp-dir", default=str(DEFAULT_EXP_DIR),
                    help="results dir with results.json (default: Tier A)")
    args = ap.parse_args()
    EXP_DIR = Path(args.exp_dir)

    model = load_hhem()
    det = HallucinationDetector(use_nli=False)
    results = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    probs_out = {"model": "vectara/hallucination_evaluation_model (HHEM-2.1)",
                 "score": "p(consistent | premise=chunk, hypothesis=claim)",
                 "source": "exp15_ablation_tierA/results.json (read-only)",
                 "generated_by": "scripts/rescore_grounding_tierA.py", "configs": {}}
    rows_out = {"model": "hhem-2.1", "tau": args.tau, "rule": "supported iff max_chunk p>tau",
                "generated_by": "scripts/rescore_grounding_tierA.py", "configs": {}}
    t0 = time.time()

    cnames = list(results.keys())
    for cname in cnames:
        rows = results[cname]["results"]
        cfg_probs, cfg_rows = {}, {}
        pairs, spans = [], []
        for r in rows:
            answer = r.get("answer") or ""
            claims = det._extract_claims(answer) if answer.strip() else []
            genuine = [c for c in claims if not classify_artifact(c)]
            n_art = len(claims) - len(genuine)
            cids = [cid for cid in r["retrieved_ids"] if cid in chunk_map]
            texts = [chunk_map[cid]["text"][:1500] for cid in cids]
            if not claims or not texts:
                cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                           "genuine": 0, "supported": 0, "unsupported": 0,
                                           "faithfulness": (1.0 if genuine == [] and claims else None),
                                           "method": "vacuous" if (claims and not genuine) else "none"}
                continue
            if not genuine:
                cfg_probs[r["query_id"]] = []
                cfg_rows[r["query_id"]] = {"total_claims": len(claims), "not_a_claim": n_art,
                                           "genuine": 0, "supported": 0, "unsupported": 0,
                                           "faithfulness": 1.0, "method": "vacuous"}
                continue
            spans.append((r["query_id"], genuine, len(texts), n_art, len(pairs)))
            pairs.extend((t, cl) for cl in genuine for t in texts)
        scores = []
        if pairs:
            import torch
            B = 16
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
        print(f"  {cname}: {len(cfg_rows)} responses, {len(pairs)} pairs "
              f"({time.time()-t0:.0f}s)", flush=True)

    with gzip.open(EXP_DIR / "grounding_probs__hhem.json.gz", "wt", encoding="utf-8") as f:
        json.dump(probs_out, f)
    (EXP_DIR / "faithfulness_rows__hhem.json").write_text(
        json.dumps(rows_out, indent=1), encoding="utf-8")
    print(f"wrote grounding_probs__hhem.json.gz + faithfulness_rows__hhem.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
