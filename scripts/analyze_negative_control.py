"""Tier 3 · Block B — negative-control base rates (construct-validity readout).

CPU over negative_control_scores.json. For each NLI verifier reports the
false-contradicted rate (fraction of 400 random claim x 5-random-chunk tuples
labeled CONTRADICTED) under v0 and vb_agree; for HHEM the false-grounded rate
(labeled SUPPORTED) across a tau grid. A verifier that flags unrelated text as
contradicted/grounded is failing construct validity — this is the pre-registered,
scenario-blind selection criterion for Block B.

Output: experiments/results/exp15_ablation_nli/negative_control_rates.json + .md
Usage: python scripts/analyze_negative_control.py
Env: HF_HUB_OFFLINE=1 PYTHONHASHSEED=42
"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402


def nli_label(chunks, variant):
    st_, _, _ = decide_nli_status([c[0] for c in chunks], [c[1] for c in chunks],
                                  0.7, 0.7, variant=variant)
    return st_


def main():
    nc = json.loads((OUT / "negative_control_scores.json").read_text(encoding="utf-8"))
    n, V = nc["n"], nc["verifiers"]
    rep = {"n_random_pairs": n,
           "criterion": "lower = better construct validity (unrelated text should NOT be "
                        "contradicted/grounded)", "nli_false_contradicted": {}, "hhem_false_grounded": {}}
    for v in ("small", "base", "large"):
        if v not in V:
            continue
        sc = V[v]["scores"]
        rep["nli_false_contradicted"][v] = {
            variant: round(sum(1 for i in range(n) if nli_label(sc[i], variant) == "contradicted") / n, 4)
            for variant in ("v0", "vb_agree")}
    if "hhem" in V:
        sc = V["hhem"]["scores"]
        rep["hhem_false_grounded"] = {
            f"tau_{tau}": round(sum(1 for i in range(n) if max(sc[i]) > tau) / n, 4)
            for tau in (0.5, 0.8, 0.9, 0.95, 0.99)}
        allh = [s for row in sc for s in row]
        rep["hhem_random_score_dist"] = {"mean": round(st.mean(allh), 4),
                                         "median": round(st.median(allh), 4),
                                         "max": round(max(allh), 4)}
    (OUT / "negative_control_rates.json").write_text(
        json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")

    md = ["# Tier 3 · control negativo — validez de constructo", "",
          f"{n} pares aleatorios (claim × 5 chunks NO relacionados). Un verificador NO debería "
          "etiquetar texto no relacionado como contradicted (NLI) / grounded (HHEM).", "",
          "## NLI: tasa falso-contradicted (menor=mejor)", "",
          "| verificador | v0 | vb_agree |", "|---|---|---|"]
    for v, d in rep["nli_false_contradicted"].items():
        md.append(f"| {v} | {d['v0']} | {d['vb_agree']} |")
    md += ["", "## HHEM: tasa falso-grounded por τ (menor=mejor)", "",
           "| τ | " + " | ".join(k for k in rep["hhem_false_grounded"]) + " |",
           "|" + "---|" * (len(rep["hhem_false_grounded"]) + 1),
           "| rate | " + " | ".join(str(x) for x in rep["hhem_false_grounded"].values()) + " |"]
    if "hhem_random_score_dist" in rep:
        md += ["", f"HHEM score en pares aleatorios: mean {rep['hhem_random_score_dist']['mean']}, "
               f"median {rep['hhem_random_score_dist']['median']}"]
    (OUT / "negative_control_rates.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
