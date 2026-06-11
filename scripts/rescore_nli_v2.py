"""
Second-verifier re-score of exp12 faithfulness + claim-format ablation (F3b, N5).

Mirror of the retrieval multi-oracle (N2): if the faithfulness ordering holds
under two NLI verifiers, the claim is defensible; if it collapses, the metric
is an instrument artifact. Motivated by q085 (28/28 procedural claims
"contradicted" at prob ~0.99 by the small verifier) and the scenario-
insensitive ~27-39 % contradiction band.

Arms (responses are NEVER regenerated; claims re-extracted deterministically):
  base      cross-encoder/nli-deberta-v3-base, claims as produced by the
            runtime extractor (identical procedure: softmax, TRUE rule,
            ENT>0.7 & ENT>CONTR -> supported; CONTR>0.7 -> contradicted).
  small_noheader
            runtime small verifier, but claims with the synthetic
            "Header: " prefix stripped -> isolates the format artifact
            (extractor rewrites bullets as "<header>: <content>.") from
            verifier capacity.

Outputs (NEW files; historical exp12 artifacts untouched):
  experiments/results/exp12_matrix/faithfulness_rescore__nli-base.json
  experiments/results/exp12_matrix/faithfulness_rescore__small-noheader.json
  output/audit/claim_audit_sample_scores_v2.json   (per-claim labels of the
      50-claim human sample under both arms, for kappa)

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 (deberta-v3-base
must be pre-downloaded; the one-time download is the single allowed exception).
Usage: python scripts/rescore_nli_v2.py [--arm base|small_noheader|both]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
ENT_T = 0.7
CONTR_T = 0.7
HEADER_RE = re.compile(r"^[^:\n]{1,80}:\s+")

# Local snapshot: the HF download path is blocked by TLS interception on this
# box (CRYPT_E_NO_REVOCATION_CHECK), so the one-time download was done via
# curl --ssl-no-revoke into data/models/. Same artifact as the hub repo.
_BASE_LOCAL = PROJECT_ROOT / "data" / "models" / "nli-deberta-v3-base"

ARMS = {
    "base": {"model": str(_BASE_LOCAL) if _BASE_LOCAL.exists()
             else "cross-encoder/nli-deberta-v3-base", "strip_header": False,
             "out": "faithfulness_rescore__nli-base.json"},
    "small_noheader": {"model": "cross-encoder/nli-deberta-v3-small", "strip_header": True,
                       "out": "faithfulness_rescore__small-noheader.json"},
}


def true_rule(scores):
    """[contr, ent, neutral] rows -> (status, score, idx) per the runtime rule."""
    best_ent, bei, best_contr, bci = -1.0, -1, -1.0, -1
    for i, s in enumerate(scores):
        c, e = float(s[0]), float(s[1])
        if e > best_ent:
            best_ent, bei = e, i
        if c > best_contr:
            best_contr, bci = c, i
    if best_ent > ENT_T and best_ent > best_contr:
        return "supported", best_ent, bei
    if best_contr > CONTR_T:
        return "contradicted", best_contr, bci
    return ("unsupported", best_ent, bei) if best_ent >= best_contr else ("unsupported", best_contr, bci)


def run_arm(arm_name, cfg, results, chunk_map, sample_claims):
    """Score one arm.

    Pairs are pooled per CONFIG into a single predict() call (the original
    per-response calls paid ~189 launch overheads per config and crawled
    under GPU contention), and each finished config is checkpointed to
    <out>.partial.json so a killed run resumes without rework.
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(cfg["model"], max_length=512)
    # fp16 inference: ~2x on the laptop 3060, where fp32 with ~500-token
    # premises runs 25-50 min/config (5-10 h/arm). DeBERTa-v3 is fp16-safe;
    # the precision is documented as part of this verifier's spec. Threshold
    # flips from ~1e-3 prob shifts are part of instrument-2's definition.
    import torch
    if torch.cuda.is_available():
        model.model.half()
    det = HallucinationDetector(use_nli=False)  # extractor only

    out_path = PROJECT_ROOT / "experiments/results/exp12_matrix" / cfg["out"]
    part_path = out_path.with_suffix(".partial.json")
    out = {"arm": arm_name, "verifier": cfg["model"], "strip_header": cfg["strip_header"],
           "procedure": "softmax + TRUE rule, ENT/CONTR thresholds 0.7, max over 5 chunks",
           "generated": "2026-06-11", "configs": {}}
    sample_labels = {}
    claim_mismatch = 0
    if part_path.exists():
        prev = json.loads(part_path.read_text(encoding="utf-8"))
        out["configs"] = prev.get("configs", {})
        sample_labels = prev.get("_sample_labels", {})
        claim_mismatch = prev.get("claim_count_mismatches", 0)
        print(f"[{arm_name}] resuming, {len(out['configs'])} configs done", flush=True)
    t0 = time.time()

    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            if cname in out["configs"]:
                continue
            # Pool every (chunk, claim) pair of the config.
            pairs, spans = [], []  # spans: (query_id, claims, k_chunks, start)
            for r in results[cname]["results"]:
                hm = r.get("hallucination_metrics") or {}
                if hm.get("method") != "nli" or not (hm.get("total_claims") or 0):
                    continue
                claims = det._extract_claims(r["answer"])
                if len(claims) != hm.get("total_claims"):
                    claim_mismatch += 1
                chunks = [chunk_map[cid] for cid in r["retrieved_ids"] if cid in chunk_map]
                texts = [c["text"] for c in chunks]
                if not texts or not claims:
                    continue
                scored = [HEADER_RE.sub("", c) if cfg["strip_header"] else c for c in claims]
                spans.append((r["query_id"], claims, len(texts), len(pairs)))
                pairs.extend((t, cl) for cl in scored for t in texts)
            preds = model.predict(pairs, batch_size=64,
                                  show_progress_bar=False, apply_softmax=True)
            rows_out = {}
            for qid, claims, k, start in spans:
                agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
                for ci, orig_claim in enumerate(claims):
                    row = preds[start + ci * k: start + (ci + 1) * k]
                    status, score, _ = true_rule(row)
                    agg[status] += 1
                    key = (cname, qid, orig_claim)
                    if key in sample_claims:
                        sample_labels[sample_claims[key]] = {
                            "label": status, "score": round(float(score), 4)}
                total = len(claims)
                rows_out[qid] = {"total_claims": total, **agg,
                                 "faithfulness": round(agg["supported"] / total, 4)}
            out["configs"][cname] = rows_out
            out["claim_count_mismatches"] = claim_mismatch
            out["_sample_labels"] = sample_labels
            part_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"  [{arm_name}] {cname}: {len(rows_out)} responses, "
                  f"{len(pairs)} pairs ({time.time()-t0:.0f}s elapsed)", flush=True)

    out["n_responses"] = sum(len(v) for v in out["configs"].values())
    out["claim_count_mismatches"] = claim_mismatch
    out.pop("_sample_labels", None)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    part_path.unlink(missing_ok=True)
    print(f"[{arm_name}] wrote {out_path} ({time.time()-t0:.0f}s, "
          f"{claim_mismatch} claim-count mismatches)", flush=True)
    return sample_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="both", choices=["base", "small_noheader", "both"])
    args = ap.parse_args()

    results = json.loads(
        (PROJECT_ROOT / "experiments/results/exp12_matrix/results.json")
        .read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(
        (PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json")
        .read_text(encoding="utf-8"))

    # 50-claim human sample -> claim-level kappa under each arm.
    sample_claims = {}
    sample_csv = PROJECT_ROOT / "output/audit/claim_audit_sample.csv"
    if sample_csv.exists():
        import csv
        with open(sample_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter=";"):
                claim = row["claim"].replace(" ⏎ ", "\n")
                sample_claims[(row["config"], row["query_id"], claim)] = row["idx"]
        print(f"Loaded {len(sample_claims)} sampled claims for kappa")

    all_labels = {}
    arms = ["base", "small_noheader"] if args.arm == "both" else [args.arm]
    for a in arms:
        all_labels[a] = run_arm(a, ARMS[a], results, chunk_map, sample_claims)

    if sample_claims:
        lab_path = PROJECT_ROOT / "output/audit/claim_audit_sample_scores_v2.json"
        existing = json.loads(lab_path.read_text(encoding="utf-8")) if lab_path.exists() else {}
        existing.update({a: all_labels[a] for a in arms})
        lab_path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        print("wrote", lab_path)


if __name__ == "__main__":
    main()
