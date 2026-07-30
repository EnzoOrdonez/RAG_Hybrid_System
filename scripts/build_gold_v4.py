"""Tier 3 · Block D — extend the human gold set to N~=200 (150 new claims).

Selects 150 NEW genuine claims (disjoint from the 50 in claim_audit_sample_v3),
oversampling the cells that carry the small-vs-base disagreement signal so the
human labels let us (a) estimate kappa(verifier, human) with CI half-width <=0.1
and (b) discriminate candidate verifiers by accuracy. Strata (seed 42):

  disagreement     50  — small_label != base_label (where the instruments split)
  near_threshold   40  — best_ent(small) in [0.65, 0.75] (fragile decisions)
  false_contr      30  — small=contradicted conf>=0.9 AND base=supported (q085 family)
  random_anchor    30  — unstratified, for unbiased marginal accuracy

TWO-STAGE EVIDENCE DESIGN (2026-07-30, Enzo)
--------------------------------------------
The instruments do NOT all see the same evidence: NLI `vb_agree` reads all 5 context
chunks (contradiction needs >=2 to agree) and HHEM scores `max_chunk` over all 5
(premise truncated to 1500 chars, rescore_grounding_exp15.py:151). Showing the human a
single chunk therefore biases kappa(human, HHEM) DOWNWARD by construction — the human
says "not supported" precisely when the evidence sits in a chunk they were not shown —
and that bias falls on the very decision the gold exists to arbitrate (level NLI 0.30
vs HHEM 0.55). Showing all 5 chunks for all 150 claims closes the confound but costs
~5.8-9.0 h of extra reading (the 150 claims span 139 distinct contexts, so grouping
does not compress it). Chosen design instead:

  Stage A  all 150 claims, 1 chunk (small's argmax-entailment) @800 chars  -> kappa
  Stage B  a 50-claim stratified subsample OF stage A, all 5 chunks @1500  -> bias

Stage B measures the confound directly (how many judgements FLIP once the full evidence
is visible), so the stage-A kappa can be bias-corrected instead of merely carrying a
caveat. Stage B must be annotated AFTER stage A is finished and never shows the
stage-A answer; the carry-over risk (recalling one's own earlier judgement) is real and
is recorded as a limitation.

The human judges BLIND: the template shows the question + the claim + the evidence +
an empty `juicio_humano` (correcto / incorrecto / dudoso). Verifier labels are NOT
shown (anchoring bias), and stage B does NOT mark which chunk the instrument keyed on
(that would steer attention straight to it); the argmax index lives only in the side
json. Each verifier is scored against the human afterward by a post-hoc join on
(config, query_id, claim_idx). The question text IS shown — a claim carrying a pronoun
or an implicit subject is not judgeable without it.

Outputs:
  output/audit/claim_audit_sample_v4.csv / .md           (stage A, blind)
  output/audit/claim_audit_sample_v4_stageB.csv / .md    (stage B, blind)
  output/audit/claim_audit_sample_v4_meta.json           (strata + per-verifier labels)

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
QUERIES = ROOT / "data/evaluation/test_queries.json"
SEED = 42
STRATA_N = {"disagreement": 50, "near_threshold": 40, "false_contr": 30, "random_anchor": 30}
STAGE_A_CHARS = 800     # single argmax chunk, as originally designed
STAGE_B_N = 50          # subsample of stage A re-judged against the full context
STAGE_B_CHARS = 1500    # == HHEM's premise truncation -> parity with the instrument

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


def load_questions():
    """query_id -> question text (so the annotator can judge context-dependent claims)."""
    q = json.loads(QUERIES.read_text(encoding="utf-8"))
    if isinstance(q, dict):
        q = q.get("queries", q)
    return {r["query_id"]: r["question"] for r in q}


def stage_b_subsample(chosen, rng):
    """Proportional (largest-remainder) subsample of STAGE_B_N across the strata.

    Proportional rather than uniform so stage B estimates the evidence bias on the
    same stratum mix stage A is made of; otherwise the correction it yields would not
    transfer back to the stage-A kappa.
    """
    by_stratum = {}
    for p in chosen:
        by_stratum.setdefault(p["stratum"], []).append(p)
    quota = {k: STAGE_B_N * len(v) / len(chosen) for k, v in by_stratum.items()}
    alloc = {k: int(v) for k, v in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - alloc[k], reverse=True)[
            : STAGE_B_N - sum(alloc.values())]:
        alloc[k] += 1
    assert sum(alloc.values()) == STAGE_B_N
    picked = []
    for k in sorted(alloc):  # sorted -> stable iteration order
        pool = sorted(by_stratum[k], key=lambda p: (p["config"], p["query_id"], p["claim_idx"]))
        picked += rng.sample(pool, alloc[k])
    # present in a shuffled order so stage-B position carries no stratum signal
    rng.shuffle(picked)
    return picked, alloc


def main():
    rng = random.Random(SEED)
    probs = {t: load_probs(t) for t in ("small", "base")}
    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    questions = load_questions()

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
        question = questions.get(p["query_id"], "")
        rows_csv.append(";".join(esc(x) for x in
            [idx, p["config"], p["query_id"], question, p["claim"],
             cid, src, ch.get("text", "")[:STAGE_A_CHARS], "", ""]))
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
               f"**Pregunta:** {questions.get(p['query_id'], '')}", "",
               f"**Claim:** {p['claim']}", "",
               f"**Mejor evidencia** ({ch.get('cloud_provider','')}/{ch.get('service_name','')}):", "",
               "> " + ch.get("text", "")[:STAGE_A_CHARS].replace("\n", "\n> "), "",
               "**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______", ""]
    (AUDIT / "claim_audit_sample_v4.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------------------------------------------------------- stage B
    # Same claims, full context. Chunks are listed in CONTEXT order (the order the
    # generator saw them), never sorted by entailment and never flagged, so the file
    # leaks nothing about which chunk the instrument keyed on.
    stage_b, b_alloc = stage_b_subsample(chosen, rng)
    b_keys = {(p["config"], p["query_id"], p["claim_idx"]) for p in stage_b}
    stage_a_idx = {(m["config"], m["query_id"], m["claim_idx"]): m["idx"] for m in meta}

    b_cols = ["idx", "stage_a_idx", "config", "query_id", "question", "claim",
              "evidence_all_chunks", "juicio_humano", "comentario"]
    b_csv = [";".join(b_cols)]
    b_md = ["# Gold v4 · ETAPA B — mismo claim, CONTEXTO COMPLETO (5 chunks)", "",
            f"{len(stage_b)} claims (submuestreo proporcional por estrato de los {len(chosen)} de la "
            f"etapa A, seed {SEED}). **Rellenar SOLO después de terminar la etapa A**, y sin "
            "consultar lo que respondiste allí.", "",
            "Objetivo: los verificadores puntúan contra los **5** chunks del contexto (HHEM "
            f"`max_chunk`, premisa truncada a {STAGE_B_CHARS} chars; NLI `vb_agree` exige ≥2 chunks). "
            "La etapa A te mostró **uno**. Comparando tu juicio aquí con el de allí medimos cuánto "
            "sesga eso la κ, en vez de solo declararlo como limitación.", "",
            "Mismo criterio: `correcto` (el claim está respaldado por ALGUNO de los chunks), "
            "`incorrecto` (contradicho, o no respaldado por ninguno), `dudoso` (evidencia "
            "insuficiente). Los chunks van en el orden en que los vio el modelo.", ""]
    for bidx, p in enumerate(stage_b, 1):
        key = (p["config"], p["query_id"], p["claim_idx"])
        parts, md_ev = [], []
        for n, cid in enumerate(p["chunk_ids"], 1):
            ch = chunk_map.get(cid, {})
            head = f"{ch.get('cloud_provider','')}/{ch.get('service_name','')} :: {ch.get('heading_path','')}"
            body = ch.get("text", "")[:STAGE_B_CHARS]
            parts.append(f"[E{n}] {head}\n{body}")
            md_ev += [f"**[E{n}]** ({head})", "", "> " + body.replace("\n", "\n> "), ""]
        b_csv.append(";".join(esc(x) for x in
            [bidx, stage_a_idx[key], p["config"], p["query_id"],
             questions.get(p["query_id"], ""), p["claim"], "\n\n".join(parts), "", ""]))
        b_md += [f"## B{bidx}. {p['config']} — {p['query_id']}",
                 f"**Pregunta:** {questions.get(p['query_id'], '')}", "",
                 f"**Claim:** {p['claim']}", "", "**Contexto completo (5 chunks):**", ""] + md_ev + [
                 "**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  "
                 "**Comentario:** ______", ""]
    (AUDIT / "claim_audit_sample_v4_stageB.csv").write_text("\n".join(b_csv), encoding="utf-8-sig")
    (AUDIT / "claim_audit_sample_v4_stageB.md").write_text("\n".join(b_md), encoding="utf-8")

    for m in meta:
        m["stage_b"] = (m["config"], m["query_id"], m["claim_idx"]) in b_keys
    (AUDIT / "claim_audit_sample_v4_meta.json").write_text(
        json.dumps({"seed": SEED, "strata_target": STRATA_N, "strata_got": got,
                    "n": len(chosen), "stage_a_chars": STAGE_A_CHARS,
                    "stage_b_n": len(stage_b), "stage_b_chars": STAGE_B_CHARS,
                    "stage_b_strata": b_alloc, "rows": meta}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"gold v4 etapa A: {len(chosen)} claims -> claim_audit_sample_v4.{{csv,md}}")
    print(f"gold v4 etapa B: {len(stage_b)} claims x 5 chunks -> claim_audit_sample_v4_stageB.{{csv,md}}")
    print(f"strata A: {got}")
    print(f"strata B: {b_alloc}")


if __name__ == "__main__":
    main()
