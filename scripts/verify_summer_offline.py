"""Offline verification of every summer-phase headline number (no GPU, no LLM).

Companion to scripts/verify_v4_offline.py, which does the same for the signed v4/N9
figures. This one covers what the summer phase added: Tier A (exp15_ablation_tierA),
exp16 (anchored decoding) and exp17 (provider-balanced cross-cloud).

The phase was designed so the GPU pays exactly once per verifier and everything
downstream is pure CPU re-aggregation of the persisted probabilities. This script is
the proof of that claim: it re-derives the numbers from the raw probs and demands they
match the committed artifacts.

  1. NLI re-aggregation   nli_probs__{small,base}.json.gz + claims_extraction.json
                          -> decide_nli_status(vb_agree, 0.7/0.7) -> faithfulness_rows
  2. HHEM re-aggregation  grounding_probs__hhem.json.gz -> max_chunk p > 0.5 -> rows
  3. Paired statistics    rows -> Wilcoxon/d_z/bootstrap/BH -> arm_stats__*.json
  4. Instrument load      the HHEM baseline level must land in 0.40-0.55. This is the
                          guard the HHEM loading bug of ledger entry 6 defeated: a
                          silently mis-loaded model scored ~0.04 and still "ran".
  5. Declared BH family   the family in each artifact must equal its contrast count
                          (the defect ledger entry 9 retracted, and entry 15 fixed).

Reads only; writes a report under output/audit/. Exit 0 = all checks passed.

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
  python scripts/verify_summer_offline.py
"""

import gzip
import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "results"
REPORT = ROOT / "output" / "audit" / f"summer_offline_check_{date.today().isoformat()}.md"

TOL = 1e-9
ENT_T = CONTR_T = 0.7
HHEM_TAU = 0.5

# (experiment dir, anchor arm) — the three consumers of the paired harness
EXPERIMENTS = [
    ("exp15_ablation_tierA", "baseline_repro"),
    ("exp16_anchored_decoding", "baseline_repro"),
    ("exp17_crosscloud_balanced", "baseline"),
]

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "arm_stats", ROOT / "scripts" / "compute_tierA_arm_stats.py")
arm_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm_stats)

from src.evaluation.statistical_analysis import (  # noqa: E402
    paired_comparison, cohens_d, apply_multiple_comparison_correction)

failures, lines = [], []


def log(s=""):
    print(s)
    lines.append(s)


def check(name, ok, detail=""):
    log(f"- [{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1 + 2. rows
def recompute_nli_rows(exp_dir, verifier):
    """faithfulness rows re-derived from the persisted probs (pure CPU)."""
    probs = load_gz(exp_dir / f"nli_probs__{verifier}.json.gz")["configs"]
    out = {}
    for cname, per_q in probs.items():
        rows = {}
        for qid, claim_probs in per_q.items():
            if not claim_probs:          # vacuous: no genuine claim to score
                continue
            agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
            for cp in claim_probs:
                st, _, _ = decide_nli_status([p[0] for p in cp], [p[1] for p in cp],
                                             ENT_T, CONTR_T, variant="vb_agree", margin=0.0)
                agg[st] += 1
            g = len(claim_probs)
            rows[qid] = round(agg["supported"] / g, 4)
        out[cname] = rows
    return out


def recompute_hhem_rows(exp_dir):
    probs = load_gz(exp_dir / "grounding_probs__hhem.json.gz")["configs"]
    out = {}
    for cname, per_q in probs.items():
        rows = {}
        for qid, claim_scores in per_q.items():
            if not claim_scores:
                continue
            sup = sum(1 for sc in claim_scores if sc and max(sc) > HHEM_TAU)
            rows[qid] = round(sup / len(claim_scores), 4)
        out[cname] = rows
    return out


def verify_rows(exp, exp_dir, verifier):
    stored_path = (exp_dir / "faithfulness_rows__hhem.json" if verifier == "hhem"
                   else exp_dir / f"faithfulness_rows__{verifier}__vb_agree.json")
    if not stored_path.exists():
        check(f"{exp}/{verifier}: rows present", False, f"missing {stored_path.name}")
        return
    stored = json.loads(stored_path.read_text(encoding="utf-8"))["configs"]
    recomputed = (recompute_hhem_rows(exp_dir) if verifier == "hhem"
                  else recompute_nli_rows(exp_dir, verifier))

    n_cmp, bad = 0, []
    for cname, rows in recomputed.items():
        for qid, f in rows.items():
            got = (stored.get(cname, {}).get(qid) or {}).get("faithfulness")
            if got is None:              # decline / no-evidence rows carry no probs
                continue
            n_cmp += 1
            if abs(float(got) - f) > TOL:
                bad.append((cname, qid, got, f))
    check(f"{exp}/{verifier}: {n_cmp} faithfulness cells re-aggregated from raw probs",
          not bad and n_cmp > 0, f"{len(bad)} mismatches {bad[:3]}" if bad else "exact")


# ------------------------------------------------------------ 3. paired stats
def verify_arm_stats(exp, exp_dir, baseline_arm, verifier):
    art = exp_dir / f"arm_stats__{verifier}.json"
    if not art.exists():
        check(f"{exp}/{verifier}: arm_stats present", False, "missing")
        return
    stored = json.loads(art.read_text(encoding="utf-8"))
    rows_path = (exp_dir / "faithfulness_rows__hhem.json" if verifier == "hhem"
                 else exp_dir / f"faithfulness_rows__{verifier}__vb_agree.json")
    cfgs = json.loads(rows_path.read_text(encoding="utf-8"))["configs"]
    baseline, arm_order, _ = arm_stats.derive_layout(exp_dir, baseline_arm)

    recomputed, pvals = [], []
    for arm in arm_order:
        if arm not in cfgs:
            continue
        b, a, _, _, _ = arm_stats.pair(cfgs[baseline], cfgs[arm])
        pc = paired_comparison(b, a)
        d_z, _ = cohens_d(b, a)
        _, _, mdiff = arm_stats.paired_bootstrap_meandiff(b, a)
        pvals.append(pc["p_value"])
        recomputed.append({"arm": arm.split(" | ")[0], "p": pc["p_value"],
                           "d_z": d_z, "diff": mdiff, "n": pc["n"]})
    p_bh, _ = apply_multiple_comparison_correction(pvals, method="fdr_bh")
    for r, pb in zip(recomputed, p_bh):
        r["p_bh"] = float(pb)

    bad = []
    for r, s in zip(recomputed, stored["contrasts"]):
        for key, skey in (("p", "p_value"), ("p_bh", "p_bh"), ("d_z", "cohens_d_z"),
                          ("diff", "mean_diff_arm_minus_base"), ("n", "n_paired")):
            got, want = r[key], s[skey]
            if abs(round(float(got), 5 if "p" in key else 4) - float(want)) > 1e-4:
                bad.append((r["arm"], skey, got, want))
    check(f"{exp}/{verifier}: {len(recomputed)} paired contrasts recomputed "
          f"(Wilcoxon/d_z/BH)", not bad, f"{bad[:3]}" if bad else "exact")

    # declared BH family must equal the contrast count (ledger entries 9 + 15)
    declared = int(re.match(r"\s*(\d+)", stored["bh_family"]).group(1))
    check(f"{exp}/{verifier}: declared BH family == contrast count",
          declared == len(stored["contrasts"]),
          f"declares {declared}, has {len(stored['contrasts'])}")
    check(f"{exp}/{verifier}: declared anchor == actual anchor",
          stored["baseline"] == baseline_arm,
          f"declares {stored['baseline']}, is {baseline_arm}")

    n_sig = sum(1 for c in stored["contrasts"] if c["sig_bh"])
    log(f"    · {n_sig}/{len(stored['contrasts'])} significativos (BH) · "
        f"nivel ancla {stored['baseline_mean_faithfulness']}")
    return stored


# --------------------------------------------------------------------- main
def main():
    log("# Verificación offline — cifras de la fase de verano (Tier A / exp16 / exp17)")
    log(f"Fecha: {date.today().isoformat()} · solo lectura sobre experiments/ · sin GPU ni LLM")
    log("")
    log("Re-deriva cada cifra desde las probabilidades persistidas: si el pase GPU se hizo "
        "una vez y todo lo demás es re-agregación CPU (el diseño de la fase), esto debe dar "
        "exacto.")
    log("")

    for exp, baseline_arm in EXPERIMENTS:
        exp_dir = RESULTS / exp
        if not exp_dir.exists():
            check(f"{exp}: presente", False, "directorio ausente")
            continue
        log(f"## {exp}")
        for verifier in ("small", "base", "hhem"):
            verify_rows(exp, exp_dir, verifier)
            stored = verify_arm_stats(exp, exp_dir, baseline_arm, verifier)

            # instrument-load guard: a mis-loaded HHEM scored ~0.04 and still "ran"
            if verifier == "hhem" and stored:
                lvl = stored["baseline_mean_faithfulness"]
                ok = lvl is not None and 0.40 <= lvl <= 0.55
                check(f"{exp}: nivel HHEM del ancla en rango (carga verificada)", ok,
                      f"{lvl}" if ok else
                      f"{lvl} fuera de 0.40-0.55 => el modelo NO cargó bien")
        log("")

    log("## Resultado")
    if failures:
        log(f"**{len(failures)} verificación(es) FALLARON:**")
        for f in failures:
            log(f"  - {f}")
    else:
        log("**Todas las verificaciones pasaron.** Las cifras titulares de la fase de verano "
            "se reproducen desde los artefactos committeados, sin GPU.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreporte -> {REPORT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
