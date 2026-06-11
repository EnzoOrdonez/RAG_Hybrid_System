"""
Stratified claim sample for HUMAN audit of the NLI verifier (Fase 3a, ledger N5).

Why: exp12 persisted only aggregate claim counts per response, and the external
audit (2026-06-11) flagged the contradicted side as a likely instrument
artifact (q085 granite-hibrido: 28/28 procedural claims "contradicted" at
prob ~0.99). This script regenerates claim-level detail DETERMINISTICALLY
(same extractor + same cached cross-encoder/nli-deberta-v3-small + softmax +
TRUE rule + 0.7 thresholds) and exports a 50-claim stratified sample for
manual judgment:

  20 contradicted (q085 contributes 3) + 20 unsupported + 10 supported,
  stratified across the 4 models and the 3 RAG scenarios.

Guard: for every sampled response the regenerated aggregate counts must match
the persisted hallucination_metrics; mismatching responses are reported and
replaced by the next candidate, so every exported claim provably reproduces
the persisted measurement.

Output: output/audit/claim_audit_sample.csv (;-separated) and .md, with empty
`juicio_humano` / `comentario` columns for Enzo. Historical experiment files
are never touched.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42.
Usage: python scripts/build_claim_audit_sample.py
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402

SEED = 42
MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
# stratum -> (target_n, per-model quota)
PLAN = {"contradicted": (20, 5), "unsupported": (20, 5), "supported": (10, 2)}
PINNED = [("hibrido | granite4.1-8b", "q085", "contradicted", 3)]  # config, qid, status, n_claims
CHUNK_MAP = PROJECT_ROOT / "data" / "indices" / "chunk_map_bge-large_adaptive_500.json"
OUT_DIR = PROJECT_ROOT / "output" / "audit"


def main():
    rng = random.Random(SEED)
    results = json.loads(
        (PROJECT_ROOT / "experiments/results/exp12_matrix/results.json")
        .read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    det = HallucinationDetector(use_nli=True)
    assert det.nli_model is not None, "NLI model required (cached, offline)"

    # Candidate responses per stratum: persisted aggregates tell us which
    # responses CONTAIN claims of each status.
    field = {"contradicted": "contradicted_claims",
             "unsupported": "unsupported_claims",
             "supported": "supported_claims"}
    by_row = {}   # (config, qid) -> row
    cands = {st: {m: [] for m in MODELS} for st in PLAN}
    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            for r in results[cname]["results"]:
                hm = r.get("hallucination_metrics") or {}
                if hm.get("method") != "nli":
                    continue
                by_row[(cname, r["query_id"])] = r
                for st in PLAN:
                    if (hm.get(field[st]) or 0) > 0:
                        cands[st][m].append((cname, r["query_id"]))
    for st in cands:
        for m in MODELS:
            rng.shuffle(cands[st][m])

    rescore_cache = {}
    mismatches = []

    def claim_details(cname, qid):
        """Regenerate claim details; None if aggregates mismatch persisted."""
        key = (cname, qid)
        if key in rescore_cache:
            return rescore_cache[key]
        r = by_row[key]
        hm = r["hallucination_metrics"]
        chunks = [chunk_map[cid] for cid in r["retrieved_ids"] if cid in chunk_map]
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        claims = det._extract_claims(r["answer"])
        details = det._nli_matching(claims, texts, ids)
        agg = {
            "total": len(details),
            "supported": sum(1 for d in details if d.status == "supported"),
            "contradicted": sum(1 for d in details if d.status == "contradicted"),
            "unsupported": sum(1 for d in details if d.status == "unsupported"),
        }
        ok = (agg["total"] == hm.get("total_claims")
              and agg["supported"] == hm.get("supported_claims")
              and agg["contradicted"] == hm.get("contradicted_claims")
              and agg["unsupported"] == hm.get("unsupported_claims"))
        if not ok:
            mismatches.append((cname, qid, agg,
                               {k: hm.get(f"{k}_claims" if k != "total" else "total_claims")
                                for k in agg}))
        rescore_cache[key] = details if ok else None
        return rescore_cache[key]

    sample = []

    def take(cname, qid, status, n, stratum_label):
        details = claim_details(cname, qid)
        if details is None:
            return 0
        pool = [d for d in details if d.status == status]
        rng.shuffle(pool)
        taken = 0
        for d in pool[:n]:
            chunk = chunk_map.get(d.evidence_chunk_id or "", {})
            sample.append({
                "stratum": stratum_label,
                "config": cname,
                "query_id": qid,
                "question": by_row[(cname, qid)]["question"],
                "claim": d.claim_text,
                "nli_label": d.status,
                "nli_score": d.nli_score,
                "best_chunk_id": d.evidence_chunk_id or "",
                "best_chunk_source": (f"{chunk.get('cloud_provider','')}/"
                                      f"{chunk.get('service_name','')} :: "
                                      f"{chunk.get('heading_path','')}") if chunk else "",
                "best_chunk_text": chunk.get("text", "") if chunk else "",
                "juicio_humano": "",
                "comentario": "",
            })
            taken += 1
        return taken

    # Pinned q085 first.
    for cname, qid, status, n in PINNED:
        got = take(cname, qid, status, n, status)
        print(f"pinned {qid}: {got}/{n} {status} claims")

    # Stratified fill: per model quota, rotating scenarios via the shuffled list.
    for st, (target, quota) in PLAN.items():
        for m in MODELS:
            need = quota
            if st == "contradicted" and m == "granite4.1-8b":
                need = max(0, quota - 3)  # q085 already contributed 3
            for cname, qid in cands[st][m]:
                if need <= 0:
                    break
                if any(s["config"] == cname and s["query_id"] == qid and s["stratum"] == st
                       for s in sample):
                    continue
                need -= take(cname, qid, st, 1, st)
        have = sum(1 for s in sample if s["stratum"] == st)
        # Top-up across models if a quota could not be met.
        if have < target:
            flat = [xy for m in MODELS for xy in cands[st][m]]
            rng.shuffle(flat)
            for cname, qid in flat:
                if have >= target:
                    break
                if any(s["config"] == cname and s["query_id"] == qid and s["stratum"] == st
                       for s in sample):
                    continue
                have += take(cname, qid, st, 1, st)
        print(f"stratum {st}: {have}/{target}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "stratum", "config", "query_id", "question", "claim",
            "nli_label", "nli_score", "best_chunk_id", "best_chunk_source",
            "best_chunk_text", "juicio_humano", "comentario"]

    def csv_escape(v):
        v = str(v).replace("\r", " ").replace("\n", " ⏎ ")
        return f'"{v}"' if ";" in v or '"' in v else v

    csv_lines = [";".join(cols)]
    for i, s in enumerate(sample, 1):
        csv_lines.append(";".join(csv_escape(x) for x in
                                  [i] + [s[c] for c in cols[1:]]))
    (OUT_DIR / "claim_audit_sample.csv").write_text(
        "\n".join(csv_lines), encoding="utf-8-sig")

    md = ["# Muestra de auditoría del verificador NLI — exp12_matrix (Fase 3a, N5)",
          "",
          f"50 claims estratificados (seed={SEED}); etiquetas regeneradas determinísticamente "
          "y verificadas contra los agregados persistidos. Completar `juicio_humano` "
          "(correcto / incorrecto / dudoso) y `comentario`.",
          ""]
    for i, s in enumerate(sample, 1):
        md += [f"## {i}. [{s['stratum']}] {s['config']} — {s['query_id']} "
               f"(score {s['nli_score']:.3f})",
               f"**Pregunta:** {s['question']}",
               "",
               f"**Claim:** {s['claim']}",
               "",
               f"**Mejor chunk** ({s['best_chunk_source']}):",
               "",
               "> " + (s["best_chunk_text"][:600].replace("\n", "\n> ")
                       + ("…" if len(s["best_chunk_text"]) > 600 else "")),
               "",
               f"**Etiqueta NLI:** `{s['nli_label']}`  |  **Juicio humano:** ______  |  "
               f"**Comentario:** ______",
               ""]
    (OUT_DIR / "claim_audit_sample.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nSample: {len(sample)} claims -> {OUT_DIR}")
    print(f"Rescore guard: {len(rescore_cache) - len(mismatches)}/{len(rescore_cache)} "
          f"responses reproduce persisted aggregates exactly")
    for cname, qid, agg, hm in mismatches:
        print(f"  MISMATCH {cname} {qid}: regenerated={agg} persisted={hm}")


if __name__ == "__main__":
    main()
