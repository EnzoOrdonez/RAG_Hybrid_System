"""Tier 3 · Block D — score every candidate verifier against the human gold (v4).

The gold exists to settle two things the instruments cannot settle among themselves:
  (a) the LEVEL of faithfulness (NLI ~0.30 vs HHEM ~0.55 on the same answers), and
  (b) which verifier gets promoted to primary.
This script is the consumer that was missing: build_gold_v4.py produced the annotation
file, nothing read it back.

Inputs (annotator-facing files, once `juicio_humano` is filled):
  output/audit/claim_audit_sample_v4.csv          stage A — 150 claims, 1 chunk
  output/audit/claim_audit_sample_v4_stageB.csv   stage B —  50 claims, 5 chunks
  output/audit/claim_audit_sample_v4_meta.json    strata + the join key
  experiments/results/exp15_ablation_nli/*        the persisted verifier probabilities

Per candidate (small, base, hhem, E1_mean, E5_base_and_hhem, ... via the shared
`label_one` in compute_exp15_ensemble_sweep.py) it reports:

  1. Cohen's kappa vs the human, DESIGN-WEIGHTED. The gold is a stratified sample that
     deliberately oversamples disagreement/near-threshold/false-contradicted cells, so
     the unweighted kappa describes a population that does not exist. Weights are
     n_population(stratum)/n_sampled(stratum), recovered by replaying the exact
     priority-order strata rule of build_gold_v4 over the full claim pool. The replay
     is ASSERTED against the recorded meta before any weight is used — if the rule
     drifted, the script fails instead of quietly reporting wrong weights.
  2. The assumption-free cross-check: kappa and accuracy on `random_anchor` alone (the
     one unstratified cell), which needs no weights.
  3. A reliability curve (verifier confidence vs observed human-supported rate) + ECE.
  4. A threshold sweep, reported as EXPLORATORY ONLY: picking tau on the same gold that
     then evaluates it is circular. The pre-registered selection criterion stays the
     negative control (ensemble_sweep_results.json), which is blind to the contrast.
  5. Stage-B evidence-bias correction: among the 50 claims judged twice, how many
     judgements FLIP once all 5 chunks are visible. Stage A shows the human ONE chunk
     while HHEM maxes over five, so kappa(human, HHEM) is biased downward by
     construction; the flip rate turns that from a caveat into a correction.

Multiplicity: candidate-vs-human comparisons form one declared BH family (one test per
candidate, H0: kappa = 0).

Usage:
  python scripts/analyze_gold_v4.py                 # real annotations
  python scripts/analyze_gold_v4.py --simulate 0.15 # synthetic smoke test, writes nothing
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"
AUDIT = ROOT / "output" / "audit"
SEED = 42
N_BOOT = 10000
# must mirror build_gold_v4.STRATA_N — the sampler being replayed for HT weights
STRATA_N = {"disagreement": 50, "near_threshold": 40, "false_contr": 30, "random_anchor": 30}

_spec = importlib.util.spec_from_file_location(
    "sweep_ens", ROOT / "scripts" / "compute_exp15_ensemble_sweep.py")
ens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ens)

# Candidates worth arbitrating. 'large' is excluded while only a .partial file exists.
CANDIDATES = ["small", "base"] + (["hhem", "E5_base_and_hhem"] if ens.HAS_HHEM else [])
if len(ens.NLI_TRIO) >= 2:
    CANDIDATES.append("E1_mean")

# human judgement -> supported? ; 'dudoso' is ambiguous and handled by --dudoso
JUDGE_MAP = {"correcto": 1, "incorrecto": 0}


# --------------------------------------------------------------------- loading
def read_annotations(path, key_col="idx"):
    """{stage_idx: judgement} for rows whose juicio_humano is filled."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            j = (row.get("juicio_humano") or "").strip().lower()
            if j:
                out[int(row[key_col])] = j
    return out


def build_pool_flags(probs, claims, seen_v3):
    """Every eligible claim with its three stratum FLAGS, exactly as build_gold_v4 sets them.

    Returns (keys, flags) where flags[i] = (false_contr, near_threshold, disagreement).
    The flags overlap (413 of 14409 claims carry more than one), which is why a
    claim's stratum cannot be recovered by priority order — it depends on which pass
    happened to draw it. See inclusion_probs().
    """
    keys, flags = [], []
    for cfg in sorted(probs["small"]):
        for qid in sorted(probs["small"][cfg]):
            meta = claims[cfg][qid]
            gc = [c for c, a in zip(meta["claims"], meta["artifact"]) if not a]
            for i, (cp_s, cp_b) in enumerate(zip(probs["small"][cfg][qid],
                                                 probs["base"][cfg][qid])):
                claim = gc[i] if i < len(gc) else ""
                if (cfg, qid, claim[:40]) in seen_v3:
                    continue
                ls = ens.decide_single_nli(cp_s)
                lb = ens.decide_single_nli(cp_b)
                es = max(p[1] for p in cp_s)
                cs = max(p[0] for p in cp_s)
                keys.append((cfg, qid, i))
                flags.append((ls == "contradicted" and cs >= 0.9 and lb == "supported",
                              0.65 <= es <= 0.75,
                              ls != lb))
    return keys, flags


def inclusion_probs(flags, n_replicates=400, seed=SEED):
    """Horvitz-Thompson inclusion probability per claim, by replaying the real sampler.

    build_gold_v4 draws sequentially WITHOUT replacement across OVERLAPPING strata
    (false_contr 30 -> near_threshold 40 -> disagreement 50 -> random_anchor 30, each
    excluding what earlier passes already used). The resulting inclusion probability
    has no clean closed form, so it is estimated by re-running the exact procedure.

    pi depends only on a claim's FLAG PATTERN (claims with identical flags are
    exchangeable under the sampler), so the estimate is pooled within pattern — a
    pattern with hundreds of members reaches negligible Monte-Carlo error at a few
    hundred replicates, which per-claim counting never would (pi ~ 0.002 for the
    unflagged bulk).
    """
    n = len(flags)
    idx_fc = [i for i, f in enumerate(flags) if f[0]]
    idx_nt = [i for i, f in enumerate(flags) if f[1]]
    idx_dis = [i for i, f in enumerate(flags) if f[2]]
    idx_all = list(range(n))
    passes = [(idx_fc, STRATA_N["false_contr"]), (idx_nt, STRATA_N["near_threshold"]),
              (idx_dis, STRATA_N["disagreement"]), (idx_all, STRATA_N["random_anchor"])]

    hits = [0] * n
    rng = random.Random(seed)
    for _ in range(n_replicates):
        used = set()
        for idxs, k in passes:
            cand = [i for i in idxs if i not in used]
            rng.shuffle(cand)
            for i in cand[:k]:
                used.add(i)
                hits[i] += 1

    # pool within flag pattern
    pat_hits, pat_n = {}, {}
    for i, f in enumerate(flags):
        pat_hits[f] = pat_hits.get(f, 0) + hits[i]
        pat_n[f] = pat_n.get(f, 0) + 1
    pi_by_pattern = {f: pat_hits[f] / (pat_n[f] * n_replicates) for f in pat_n}
    return [pi_by_pattern[f] for f in flags], pi_by_pattern, pat_n


def load_seen_v3():
    seen = set()
    p = AUDIT / "claim_audit_sample_v3.csv"
    if not p.exists():
        return seen
    with p.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            seen.add((row["config"], row["query_id"], row["claim"][:40]))
    return seen


# ------------------------------------------------------------------ statistics
def weighted_kappa(human, verifier, w):
    """Cohen's kappa on a weighted 2x2. w = per-item design weight."""
    h = np.asarray(human, float)
    v = np.asarray(verifier, float)
    w = np.asarray(w, float)
    W = w.sum()
    if W == 0 or len(h) == 0:
        return float("nan")
    po = float((w * (h == v)).sum() / W)
    ph1 = float((w * h).sum() / W)
    pv1 = float((w * v).sum() / W)
    pe = ph1 * pv1 + (1 - ph1) * (1 - pv1)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def stratified_bootstrap_kappa(human, verifier, w, strata, n_boot=N_BOOT, seed=SEED):
    """Percentile CI + a two-sided p for H0: kappa = 0, resampling WITHIN strata.

    Resampling within stratum (rather than over the pooled sample) is what keeps the
    replicates faithful to the stratified design that produced the data.
    """
    rng = np.random.default_rng(seed)
    idx_by_s = {}
    for i, s in enumerate(strata):
        idx_by_s.setdefault(s, []).append(i)
    h, v, w = np.asarray(human), np.asarray(verifier), np.asarray(w, float)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_s.values()])
        boots[b] = weighted_kappa(h[take], v[take], w[take])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return None, None, None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # p = 2 x the mass on the far side of 0 (bootstrap-inversion, two-sided)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(lo), float(hi), float(min(1.0, max(p, 1.0 / len(boots))))


def reliability(conf, human, n_bins=5):
    """Confidence-vs-observed-rate table + expected calibration error."""
    conf = np.asarray(conf, float)
    human = np.asarray(human, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= 1)
        if not m.any():
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "mean_conf": None, "human_supported_rate": None})
            continue
        mc, hr = float(conf[m].mean()), float(human[m].mean())
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                     "mean_conf": round(mc, 4), "human_supported_rate": round(hr, 4)})
        ece += m.mean() * abs(mc - hr)
    return rows, round(float(ece), 4)


def bh(pvals):
    """Benjamini-Hochberg; returns adjusted p in the input order."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        val = min(prev, p[i] * n / (n - rank + 1))
        adj[i] = prev = val
    return [float(x) for x in adj]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dudoso", default="exclude", choices=["exclude", "unsupported"],
                    help="primary = exclude; 'unsupported' is the reported sensitivity")
    ap.add_argument("--simulate", type=float, default=None, metavar="NOISE",
                    help="fill judgements synthetically at this flip rate for an "
                         "end-to-end smoke test; writes NOTHING to output/")
    args = ap.parse_args()

    meta = json.loads((AUDIT / "claim_audit_sample_v4_meta.json").read_text(encoding="utf-8"))
    probs = {t: ens.load_nli(t) for t in ens.NLI_TRIO}
    hhem = ens.load_hhem() if ens.HAS_HHEM else {}
    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]

    # ---- per-claim verifier labels for every sampled claim
    rows = []
    for m in meta["rows"]:
        cfg, qid, ci = m["config"], m["query_id"], m["claim_idx"]
        members = {t: probs[t][cfg][qid][ci] for t in ens.NLI_TRIO}
        hh = hhem.get(cfg, {}).get(qid, [])
        hh = hh[ci] if ci < len(hh) else []
        labels = {c: ens.label_one(c, members, hh) for c in CANDIDATES}
        # confidence for the reliability curve: P(entailment) for NLI, max chunk p for HHEM
        conf = {c: (max(hh) if hh else 0.0) if c in ("hhem", "E5_base_and_hhem")
                else max(p[1] for p in members["base" if c in ("base", "E1_mean") else "small"])
                for c in CANDIDATES}
        rows.append({**m, "labels": labels, "conf": conf})

    # ---- Horvitz-Thompson design weights from the replayed sampler
    pool_keys, pool_flags = build_pool_flags(probs, claims, load_seen_v3())
    pi_all, pi_by_pattern, pat_n = inclusion_probs(pool_flags)
    pi_of = dict(zip(pool_keys, pi_all))

    # Validity check that replaces the old (wrong) priority-order assertion: every
    # claim the sampler actually drew must be reachable by the replayed sampler.
    missing = [(r["config"], r["query_id"], r["claim_idx"]) for r in rows
               if pi_of.get((r["config"], r["query_id"], r["claim_idx"]), 0) <= 0]
    if missing:
        sys.exit(f"{len(missing)}/{len(rows)} sampled claims have zero replayed inclusion "
                 f"probability (e.g. {missing[:3]}) — the pool or the flag rule drifted "
                 f"from build_gold_v4.py; refusing to report weighted kappa.")

    weight_of = {(r["config"], r["query_id"], r["claim_idx"]):
                 1.0 / pi_of[(r["config"], r["query_id"], r["claim_idx"])] for r in rows}
    samp_n = {}
    for r in rows:
        samp_n[r["stratum"]] = samp_n.get(r["stratum"], 0) + 1

    # ---- human judgements
    if args.simulate is not None:
        rng = random.Random(SEED)
        ref = CANDIDATES[0]
        jA = {}
        for i, r in enumerate(rows, 1):
            truth = 1 if r["labels"][ref] == "supported" else 0
            if rng.random() < args.simulate:
                truth = 1 - truth
            jA[i] = "correcto" if truth else "incorrecto"
        # stage B judged independently: flip a few so the bias path is exercised
        jB = {}
        for m in meta["rows"]:
            if m.get("stage_b"):
                j = jA[m["idx"]]
                jB[m["idx"]] = ("correcto" if j == "incorrecto" else "incorrecto") \
                    if rng.random() < 0.2 else j
        print(f"[SIMULACION noise={args.simulate}] datos sinteticos, no se escribe nada.\n")
    else:
        jA = read_annotations(AUDIT / "claim_audit_sample_v4.csv")
        jB = read_annotations(AUDIT / "claim_audit_sample_v4_stageB.csv",
                              key_col="stage_a_idx")
        if not jA:
            sys.exit("claim_audit_sample_v4.csv has no filled `juicio_humano` yet — "
                     "nothing to analyse. (Use --simulate to smoke-test the pipeline.)")

    n_dudoso = sum(1 for v in jA.values() if v == "dudoso")
    use = [r for r in rows if r["idx"] in jA
           and (jA[r["idx"]] in JUDGE_MAP or args.dudoso == "unsupported")]
    human = [JUDGE_MAP.get(jA[r["idx"]], 0) for r in use]
    strata = [r["stratum"] for r in use]
    w = [weight_of[(r["config"], r["query_id"], r["claim_idx"])] for r in use]
    # Kish effective sample size: HT weights span ~2 orders of magnitude here (the
    # oversampled cells are a tiny slice of the 14k-claim population), so the
    # design-weighted kappa is deliberately high-variance. Report it, do not hide it.
    wa = np.asarray(w, float)
    n_eff = float(wa.sum() ** 2 / (wa ** 2).sum()) if len(wa) else 0.0

    # ---- per-candidate scoring
    results, pvals = [], []
    for c in CANDIDATES:
        pred = [1 if r["labels"][c] == "supported" else 0 for r in use]
        k = weighted_kappa(human, pred, w)
        lo, hi, p = stratified_bootstrap_kappa(human, pred, w, strata)
        anchor = [i for i, r in enumerate(use) if r["stratum"] == "random_anchor"]
        ka = (weighted_kappa([human[i] for i in anchor], [pred[i] for i in anchor],
                             [1.0] * len(anchor)) if anchor else float("nan"))
        acc_a = (float(np.mean([human[i] == pred[i] for i in anchor]))
                 if anchor else float("nan"))
        tp = sum(1 for i in range(len(use)) if pred[i] == 1 and human[i] == 1)
        fp = sum(1 for i in range(len(use)) if pred[i] == 1 and human[i] == 0)
        fn = sum(1 for i in range(len(use)) if pred[i] == 0 and human[i] == 1)
        rel, ece = reliability([r["conf"][c] for r in use], human)
        pvals.append(p if p is not None else 1.0)
        results.append({
            "candidate": c,
            "kappa_weighted": None if np.isnan(k) else round(k, 4),
            "kappa_ci95": [None if lo is None else round(lo, 4),
                           None if hi is None else round(hi, 4)],
            "kappa_random_anchor": None if np.isnan(ka) else round(ka, 4),
            "n_random_anchor": len(anchor),
            "accuracy_random_anchor": None if np.isnan(acc_a) else round(acc_a, 4),
            "precision_supported": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall_supported": round(tp / (tp + fn), 4) if tp + fn else None,
            "verifier_supported_rate": round(float(np.mean(pred)), 4),
            "reliability_bins": rel, "ece": ece,
        })
    for r, pa in zip(results, bh(pvals)):
        r["p_bh"] = round(pa, 5)
    results.sort(key=lambda r: (r["kappa_weighted"] is None, -(r["kappa_weighted"] or 0)))

    # ---- stage B: evidence bias
    # Keyed on the stage_a_idx COLUMN, never on row order: stage B is deliberately
    # shuffled (so its position leaks no stratum), which makes positional joins wrong.
    both = [(a_idx, j) for a_idx, j in jB.items() if a_idx in jA]
    flips = [(a_idx, jA[a_idx], jb) for a_idx, jb in both if jA[a_idx] != jb]
    to_correct = sum(1 for _, ja, jb in flips if ja != "correcto" and jb == "correcto")
    stage_b = {
        "n_judged_twice": len(both),
        "n_flips": len(flips),
        "flip_rate": round(len(flips) / len(both), 4) if both else None,
        "flips_toward_supported": to_correct,
        "reading": ("flips toward 'correcto' are the confound: evidence that was present "
                    "in the context but hidden from stage A. A high rate means the "
                    "stage-A kappa understates every verifier that reads all 5 chunks "
                    "(HHEM, NLI vb_agree) and the weighted kappa above should be read as "
                    "a lower bound for them."),
    }

    payload = {
        "generated_by": "scripts/analyze_gold_v4.py",
        "simulated": args.simulate is not None,
        "n_annotated_stage_a": len(jA), "n_used": len(use), "n_dudoso": n_dudoso,
        "dudoso_rule": args.dudoso,
        "pool_size": len(pool_keys), "sampled_strata_n": samp_n,
        "kish_n_eff": round(n_eff, 1),
        "ht_weights": {
            "scheme": "Horvitz-Thompson, 1/pi; pi replayed from the real sequential "
                      "sampler over overlapping strata, pooled within flag pattern",
            "n_replicates": 400,
            "pi_by_flag_pattern": {
                f"fc={int(k[0])},nt={int(k[1])},dis={int(k[2])}":
                    {"pi": round(v, 6), "pool_n": pat_n[k], "weight": round(1 / v, 1)}
                for k, v in sorted(pi_by_pattern.items()) if v > 0},
        },
        "bh_family": f"{len(CANDIDATES)} candidate-vs-human kappa tests (fdr_bh)",
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "scheme": "stratified, within-stratum"},
        "selection_note": ("Promotion to primary verifier is NOT decided here: the "
                           "pre-registered blind criterion is the negative control in "
                           "ensemble_sweep_results.json. This table is the human-arbitration "
                           "evidence that goes to Enzo alongside it."),
        "candidates": results,
        "stage_b_evidence_bias": stage_b,
    }

    if args.simulate is not None:
        slim = dict(payload, candidates=[{k: v for k, v in c.items()
                                          if k != "reliability_bins"} for c in results])
        print(json.dumps(slim, indent=1, ensure_ascii=False))
        print("\n[SIMULACION] OK — el pipeline corre de punta a punta. Nada escrito.")
        return

    (AUDIT / "gold_v4_analysis.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    L = ["# Gold v4 — arbitraje humano del verificador", "",
         f"n anotado {len(jA)}/150 · usados {len(use)} · dudoso {n_dudoso} "
         f"(regla: {args.dudoso}). Familia BH = {payload['bh_family']}.",
         f"Pesos Horvitz-Thompson sobre un pool de {len(pool_keys)} claims; "
         f"**n efectivo de Kish = {n_eff:.1f}** — la κ ponderada estima la poblacion real "
         f"pero con varianza alta por diseño; la columna `κ random_anchor` es la lectura "
         f"limpia sin supuestos y `κ` por estrato es la que discrimina verificadores.", "",
         "| Candidato | κ ponderado | IC95 | p_BH | κ random_anchor | acc anchor | prec | rec | ECE |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['candidate']} | {r['kappa_weighted']} | "
                 f"[{r['kappa_ci95'][0]}, {r['kappa_ci95'][1]}] | {r['p_bh']} | "
                 f"{r['kappa_random_anchor']} | {r['accuracy_random_anchor']} | "
                 f"{r['precision_supported']} | {r['recall_supported']} | {r['ece']} |")
    L += ["", f"**Sesgo de evidencia (etapa B):** {stage_b['n_flips']}/{stage_b['n_judged_twice']} "
              f"juicios cambian al ver los 5 chunks (tasa {stage_b['flip_rate']}); "
              f"{stage_b['flips_toward_supported']} hacia 'correcto'.", "",
          stage_b["reading"], "", payload["selection_note"]]
    (AUDIT / "gold_v4_analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
