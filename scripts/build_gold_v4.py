"""Tier 3 · Block D — extend the human gold set to N~=200 (150 new claims).

Selects 150 NEW genuine claims (disjoint from the 50 in claim_audit_sample_v3),
oversampling the cells that carry the small-vs-base disagreement signal so the
human labels let us (a) estimate kappa(verifier, human) with CI half-width <=0.1
and (b) discriminate candidate verifiers by accuracy. Strata (seed 42):

  disagreement     50  — small_label != base_label (where the instruments split)
  near_threshold   40  — best_ent(small) in [0.65, 0.75] (fragile decisions)
  false_contr      30  — small=contradicted conf>=0.9 AND base=supported (q085 family)
  random_anchor    30  — unstratified, for unbiased marginal accuracy

The human judges BLIND: the template shows the claim + its best-evidence chunk +
an empty `juicio_humano` (correcto / incorrecto / dudoso). Verifier labels are
NOT shown (anchoring bias); each verifier is scored against the human afterward
by a post-hoc join on (config, query_id, claim). Strata + verifier labels are
recorded in a SIDE json for analysis, never in the annotator-facing file.

Outputs:
  output/audit/claim_audit_sample_v4.csv / .md   (annotator-facing, blind)
  output/audit/claim_audit_sample_v4_meta.json   (strata + per-verifier labels)

Usage: python scripts/build_gold_v4.py
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import csv
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"
AUDIT = ROOT / "output" / "audit"
CHUNK_MAP = ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
SEED = 42
STRATA_N = {"disagreement": 50, "near_threshold": 40, "false_contr": 30, "random_anchor": 30}

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402


def load_probs(tag):
    with gzip.open(OUT / f"nli_probs__{tag}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["configs"]


def genuine(meta):
    return [c for c, a in zip(meta["claims"], meta["artifact"]) if not a]


def decide(chunk_probs):
    contr = [p[0] for p in chunk_probs]
    ent = [p[1] for p in chunk_probs]
    st, _, idx = decide_nli_status(contr, ent, 0.7, 0.7, variant="vb_agree")
    return st, max(ent), max(contr), idx


def main():
    rng = random.Random(SEED)
    probs = {t: load_probs(t) for t in ("small", "base")}
    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    # existing 50 to exclude (by config, qid, claim-prefix)
    seen = set()
    with (AUDIT / "claim_audit_sample_v3.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            seen.add((row["config"], row["query_id"], row["claim"][:40]))

    # build the candidate pool with stratum flags
    pool = []
    for cfg in sorted(probs["small"]):
        for qid in sorted(probs["small"][cfg]):
            gc = genuine(claims[cfg][qid])
            cids = claims[cfg][qid]["chunk_ids"]
            for i, (cp_s, cp_b) in enumerate(zip(probs["small"][cfg][qid], probs["base"][cfg][qid])):
                claim = gc[i] if i < len(gc) else ""
                if (cfg, qid, claim[:40]) in seen:
                    continue
                ls, es, cs, bi_s = decide(cp_s)
                lb, eb, cb, _ = decide(cp_b)
                pool.append({
                    "config": cfg, "query_id": qid, "claim_idx": i, "claim": claim,
                    "chunk_ids": cids, "best_ent_idx_small": bi_s,
                    "s_label": ls, "s_ent": es, "s_contr": cs,
                    "b_label": lb, "b_ent": eb,
                    "disagreement": ls != lb,
                    "near_threshold": 0.65 <= es <= 0.75,
                    "false_contr": ls == "contradicted" and cs >= 0.9 and lb == "supported",
                })

    chosen, used = [], set()

    def pick(pred, n, stratum):
        cand = [p for p in pool if pred(p) and (p["config"], p["query_id"], p["claim_idx"]) not in used]
        rng.shuffle(cand)
        for p in cand[:n]:
            p2 = dict(p, stratum=stratum)
            chosen.append(p2)
            used.add((p["config"], p["query_id"], p["claim_idx"]))
        return min(n, len(cand))

    got = {}
    got["false_contr"] = pick(lambda p: p["false_contr"], STRATA_N["false_contr"], "false_contr")
    got["near_threshold"] = pick(lambda p: p["near_threshold"], STRATA_N["near_threshold"], "near_threshold")
    got["disagreement"] = pick(lambda p: p["disagreement"], STRATA_N["disagreement"], "disagreement")
    got["random_anchor"] = pick(lambda p: True, STRATA_N["random_anchor"], "random_anchor")

    # best-evidence chunk = small's argmax-entailment chunk (the evidence the
    # instrument keyed on); human may mark 'dudoso' if that chunk is insufficient.
    cols = ["idx", "config", "query_id", "question", "claim",
            "best_chunk_id", "best_chunk_source", "best_chunk_text",
            "juicio_humano", "comentario"]

    def esc(v):
        v = str(v).replace("\r", " ").replace("\n", " ⏎ ")
        return '"' + v.replace('"', '""') + '"' if ('"' in v or ";" in v) else v

    rows_csv = [";".join(cols)]
    meta = []
    for idx, p in enumerate(chosen, 1):
        bi = p["best_ent_idx_small"]
        cid = p["chunk_ids"][bi] if 0 <= bi < len(p["chunk_ids"]) else ""
        ch = chunk_map.get(cid, {})
        src = f"{ch.get('cloud_provider','')}/{ch.get('service_name','')} :: {ch.get('heading_path','')}"
        # question text: pull from any config's answer set is overkill; use chunk_map-free
        question = ""  # annotator judges claim-vs-chunk; question optional
        rows_csv.append(";".join(esc(x) for x in
            [idx, p["config"], p["query_id"], question, p["claim"],
             cid, src, ch.get("text", "")[:800], "", ""]))
        meta.append({"idx": idx, "stratum": p["stratum"], "config": p["config"],
                     "query_id": p["query_id"], "claim_idx": p["claim_idx"],
                     "s_label": p["s_label"], "b_label": p["b_label"],
                     "s_ent": round(p["s_ent"], 4), "s_contr": round(p["s_contr"], 4)})

    (AUDIT / "claim_audit_sample_v4.csv").write_text("\n".join(rows_csv), encoding="utf-8-sig")
    md = ["# Gold v4 — auditoría humana del verificador (Tier 3, N≈200)", "",
          f"{len(chosen)} claims NUEVOS (seed {SEED}), disjuntos de los 50 de v3. "
          "**Juicio ciego**: no se muestran las etiquetas de los verificadores para evitar "
          "anclaje. Completar `juicio_humano` con: `correcto` (el claim está respaldado por el "
          "chunk), `incorrecto` (contradicho o no respaldado), `dudoso` (evidencia insuficiente).",
          "", f"Estratos (ocultos al anotador): {got}", ""]
    for idx, p in enumerate(chosen, 1):
        bi = p["best_ent_idx_small"]
        cid = p["chunk_ids"][bi] if 0 <= bi < len(p["chunk_ids"]) else ""
        ch = chunk_map.get(cid, {})
        md += [f"## {idx}. {p['config']} — {p['query_id']}",
               f"**Claim:** {p['claim']}", "",
               f"**Mejor evidencia** ({ch.get('cloud_provider','')}/{ch.get('service_name','')}):", "",
               "> " + ch.get("text", "")[:800].replace("\n", "\n> "), "",
               "**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______", ""]
    (AUDIT / "claim_audit_sample_v4.md").write_text("\n".join(md), encoding="utf-8")
    (AUDIT / "claim_audit_sample_v4_meta.json").write_text(
        json.dumps({"seed": SEED, "strata_target": STRATA_N, "strata_got": got,
                    "n": len(chosen), "rows": meta}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"gold v4: {len(chosen)} claims -> claim_audit_sample_v4.{{csv,md}} + _meta.json")
    print(f"strata: {got}")


if __name__ == "__main__":
    main()
