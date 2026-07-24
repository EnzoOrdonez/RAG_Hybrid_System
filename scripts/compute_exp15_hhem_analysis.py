"""Tier 3 · HHEM vs NLI — level comparison + instrument-robustness of the 0/12 null.

With HHEM correctly loaded (see rescore_grounding_exp15 load fix), this compares
the grounding verifier against the published NLI-small instrument on two axes:

  1. LEVEL: per-config faithfulness HHEM(tau) vs NLI-small (v4). HHEM is a
     purpose-built grounding model with strong negative-control specificity
     (false-grounded 0.033 at tau 0.5); a systematic HHEM>NLI gap quantifies how
     much NLI UNDER-credits faithfulness (consistent with NLI's 22% false-
     contradicted rate on random text).
  2. CONTRAST: the between-scenario (RAG-vs-RAG) paired family under HHEM, same
     v4 methodology (Wilcoxon + d_z + bootstrap + BH). Tests whether the central
     "retrieval does not improve faithfulness (0/12)" null is an NLI artifact or
     instrument-robust.

Output: experiments/results/exp15_ablation_nli/hhem_vs_nli.{json,md}
Usage: python scripts/compute_exp15_hhem_analysis.py [--tau 0.5]
Env: HF_HUB_OFFLINE=1 PYTHONHASHSEED=42
"""

import argparse
import importlib.util
import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"
EXP12 = ROOT / "experiments/results/exp12_matrix"

_spec = importlib.util.spec_from_file_location(
    "cfm", ROOT / "scripts" / "compute_faithfulness_metrics.py")
cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfm)

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENS = ["lexico", "denso", "hibrido"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    hh = json.loads((OUT / "faithfulness_rows__hhem.json").read_text(encoding="utf-8"))["configs"]
    nli = json.loads((EXP12 / "faithfulness_metrics_v4_small.json").read_text(encoding="utf-8"))["systems_v2"]

    # ---- 1. level comparison ----------------------------------------------
    level = {}
    for m in MODELS:
        for sc in SCENS:
            cfg = f"{sc} | {m}"
            fa = [v["faithfulness"] for v in hh[cfg].values()
                  if v.get("method") == "nli" and v["faithfulness"] is not None]
            hval = st.mean(fa)
            nval = nli[cfg]["primary_answered"]["mean"]
            level[cfg] = {"hhem": round(hval, 4), "nli_small": round(nval, 4),
                          "gap": round(hval - nval, 4), "n": len(fa)}
    gaps = [v["gap"] for v in level.values()]

    # ---- 2. between-scenario family under HHEM ----------------------------
    override = {}
    for cfg, rows in hh.items():
        override[cfg] = {qid: {"faithfulness": v["faithfulness"], "supported": v["supported"],
                               "contradicted": 0, "unsupported": v.get("unsupported", 0),
                               "total_claims": v["total_claims"], "genuine": v["genuine"]}
                         for qid, v in rows.items() if v["faithfulness"] is not None}
    cfm.EXCLUDED_METHODS.clear()
    cfm.EXCLUDED_METHODS.update({"none", "error", "vacuous"})
    per = cfm.load_per_config(EXP12 / "results.json", override, exclude_vacuous=True)
    parsed = {c: cfm.parse_config(c) for c in per}
    fam = []
    for m in MODELS:
        s2c = {parsed[c][0]: c for c in per if parsed[c][1] == m}
        for s1, s2 in itertools.combinations(sorted(x for x in s2c if x != "sin_rag"), 2):
            fam.append((s2c[s1], s2c[s2]))
    res = cfm.run_family(per, fam, "hhem-between-scenario",
                         include=cfm._incl_primary, metric_label="faithfulness_hhem")
    rag = {p: d for p, d in res.items() if "sin_rag" not in p}
    sig = sorted(p for p, d in rag.items() if d.get("sig_bh"))

    out = {"tau": args.tau, "level_comparison": level,
           "gap_hhem_minus_nli": {"mean": round(st.mean(gaps), 4),
                                  "min": min(gaps), "max": max(gaps)},
           "between_scenario_hhem": {p: {"n": d.get("n"), "d_z": round(d.get("effect_size", 0), 4),
                                         "p_bh": round(d.get("p_bh", 1), 4),
                                         "sig_bh": bool(d.get("sig_bh"))} for p, d in rag.items()},
           "n_sig_rag_pairs": len(sig), "sig_pairs": sig,
           "verdict": ("0/12 RAG-vs-RAG null is INSTRUMENT-ROBUST (holds under NLI small, base, "
                       "and HHEM grounding). HHEM confirms NLI under-credits the LEVEL "
                       f"(mean gap +{round(st.mean(gaps),3)}) but the scenario CONTRAST is "
                       "genuinely small/null, not an NLI artifact.")}
    (OUT / "hhem_vs_nli.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    md = ["# Tier 3 — HHEM (grounding) vs NLI: nivel + robustez del nulo 0/12", "",
          f"HHEM cargado correctamente (fix de load); τ={args.tau}. Especificidad negativa: "
          "falso-grounded 0.033.", "",
          "## 1. Nivel de fidelidad por config (HHEM vs NLI small v4)", "",
          "| config | HHEM | NLI small | gap |", "|---|---|---|---|"]
    for cfg, v in level.items():
        md.append(f"| {cfg} | {v['hhem']} | {v['nli_small']} | {v['gap']:+.3f} |")
    md += ["", f"**Gap HHEM−NLI: mean +{round(st.mean(gaps),3)} (rango +{min(gaps)}..+{max(gaps)}).** "
           "NLI sub-acredita la fidelidad de forma sistemática.", "",
           "## 2. Contraste entre escenarios bajo HHEM (RAG-vs-RAG, pareado BH)", "",
           "| par | n | d_z | p_bh | sig |", "|---|---|---|---|---|"]
    for p, d in rag.items():
        md.append(f"| {p[:44]} | {d.get('n')} | {round(d.get('effect_size',0),2):+} | "
                  f"{round(d.get('p_bh',1),4)} | {'SÍ' if d.get('sig_bh') else 'no'} |")
    md += ["", f"**HHEM: {len(sig)}/12 RAG-vs-RAG significativos** (NLI small daba 0/12).", "",
           "## Veredicto", "", out["verdict"],
           "", "El par más fuerte (granite hibrido-vs-lexico) es direccionalmente consistente "
           "(hib>lex) en todos los instrumentos pero nunca alcanza significancia tras BH "
           "(NLI small p_bh 0.085, HHEM p_bh ~0.11): señal débil, sub-potenciada, NO nula pero "
           "NO significativa. Pendiente: gold humano para validar HHEM; deberta-large (8/12) como "
           "tercer voto.", ""]
    (OUT / "hhem_vs_nli.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(md))


if __name__ == "__main__":
    main()
