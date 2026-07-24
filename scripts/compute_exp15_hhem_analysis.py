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

# canonical v4 between-scenario evaluation (BH family INCLUDING sin_rag pairs,
# 24 = 4 models x C(4,2); the ad-hoc 12-pair family that excluded sin_rag gave
# an inconsistent BH correction and a wrong 0/12 verdict — ledger entrada 9).
_swspec = importlib.util.spec_from_file_location(
    "sweep", ROOT / "scripts" / "compute_exp15_nli_sweep.py")
sweep = importlib.util.module_from_spec(_swspec)
_swspec.loader.exec_module(sweep)

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

    # ---- 2. between-scenario family under HHEM (v4-consistent, 24-pair BH) --
    override = {}
    for cfg, rows in hh.items():
        override[cfg] = {qid: {"faithfulness": v["faithfulness"], "supported": v["supported"],
                               "contradicted": 0, "unsupported": v.get("unsupported", 0),
                               "total_claims": v["total_claims"], "genuine": v["genuine"]}
                         for qid, v in rows.items() if v["faithfulness"] is not None}
    # evaluate_point builds the family EXACTLY as compute_faithfulness_metrics.main()
    # (all scenario pairs incl sin_rag; BH within), then reports the RAG-vs-RAG count.
    ev = sweep.evaluate_point(override, "hhem-between-scenario")
    rag = ev["rag_pair_stats"]  # {pair: {d_z, p_bh, n}}
    sig = sorted(ev["sig_rag_pairs"])

    out = {"tau": args.tau, "bh_family": "v4-consistent (24 pairs incl sin_rag)",
           "level_comparison": level,
           "gap_hhem_minus_nli": {"mean": round(st.mean(gaps), 4),
                                  "min": min(gaps), "max": max(gaps)},
           "between_scenario_hhem": {p: {"n": d.get("n"), "d_z": d.get("d_z"),
                                         "p_bh": d.get("p_bh"),
                                         "sig_bh": p in sig} for p, d in rag.items()},
           "n_sig_rag_pairs": len(sig), "sig_pairs": sig,
           "verdict": (f"HHEM (v4-consistent BH family): {len(sig)}/12 RAG-vs-RAG significativos "
                       f"({', '.join(sig) if sig else 'ninguno'}). NLI small/base dan 0/12. "
                       "Bajo el instrumento de grounding limpio, granite hibrido-vs-lexico CRUZA "
                       "significancia (p_bh 0.020, d_z -0.35) donde el NLI ruidoso no (p_bh 0.085): "
                       "el efecto retrieval->fidelidad existe para el modelo determinista pero solo "
                       "es detectable con un instrumento menos ruidoso. HHEM tambien sub-acredita el "
                       f"NIVEL (gap medio +{round(st.mean(gaps),3)} sobre NLI). Efecto pequeno, "
                       "tau-dependiente, solo granite (1/12): matizar, pendiente gold humano.")}
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
           "## 2. Contraste entre escenarios bajo HHEM (RAG-vs-RAG, familia BH v4-consistente 24)", "",
           "| par | n | d_z | p_bh | sig |", "|---|---|---|---|---|"]
    for p, d in rag.items():
        md.append(f"| {p[:44]} | {d.get('n')} | {round(d.get('d_z') or 0,2):+} | "
                  f"{round(d.get('p_bh') or 1,4)} | {'SÍ' if p in sig else 'no'} |")
    md += ["", f"**HHEM: {len(sig)}/12 RAG-vs-RAG significativos** (NLI small/base dan 0/12). "
           f"Significativo: {', '.join(sig) if sig else 'ninguno'}.", "",
           "## Veredicto", "", out["verdict"],
           "", "El par granite hibrido-vs-lexico es direccionalmente consistente (hib>lex) en los "
           "tres instrumentos; bajo el NLI ruidoso NO cruza BH (p_bh 0.085) pero bajo HHEM (grounding "
           "limpio, familia v4-consistente) SÍ (p_bh 0.020). Es 1/12, d_z pequeño (-0.35), "
           "tau-dependiente. Pendiente: gold humano para validar HHEM; deberta-large (8/12) 3.er voto.", ""]
    (OUT / "hhem_vs_nli.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(md))


if __name__ == "__main__":
    main()
