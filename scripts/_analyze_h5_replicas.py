"""
H5 (N9) — análisis offline de las réplicas de exp14: ¿los 2 pares entre-modelos
supervivientes de v4 son estables frente al no-determinismo run-a-run?

Lee experiments/results/exp14_h5_replicas/replicas_checkpoint.json (salida de
run_h5_replicas.py) y reporta, por par:

  1. Estabilidad del TEXTO: % de (query, modelo) cuyas 3 réplicas son
     byte-idénticas (greedy estable) y % con al menos una divergencia.
  2. Estabilidad de la MÉTRICA: faithfulness por réplica bajo la convención v4
     (se excluyen method none/error, respuestas pure_decline y filas con
     supported==0 y total_claims==0 — proxy de vacuas del runtime, que reporta
     1.0 con 0 claims efectivos); media por celda por réplica r1/r2/r3.
  3. Estabilidad del CONTRASTE: d_z pareado del par para cada combinación de
     réplicas (r_i del modelo A vs r_j del modelo B, 9 combinaciones) + Wilcoxon
     p crudo — rango de d_z y de p como banda de sensibilidad run-a-run.
  4. Veredicto sugerido por par: ESTABLE (todas las combinaciones mantienen el
     signo y |d_z| dentro de ±0,15 del original) / SENSIBLE (si no).

No modifica nada; imprime el reporte y escribe
experiments/results/exp14_h5_replicas/analysis.json.

Usage: python scripts/_analyze_h5_replicas.py
"""

import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_faithfulness_metrics import classify_response  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "exp14_h5_replicas"
# d_z originales bajo v4 (referencia para el veredicto de estabilidad)
ORIGINAL_DZ = {
    "A_denso_granite_vs_mistral": +0.415,   # small, p_bh=0.0136, n=75
    "B_lexico_gemma_vs_granite": -0.531,    # base,  p_bh=0.0383, n=42
}
DZ_TOL = 0.15


def v4_eligible(rec):
    """Convención v4 sobre el registro del runtime (proxy):
    método NLI, no pure_decline, y con señal (total_claims>0)."""
    if rec.get("method") not in (None, "nli"):
        return False
    if rec.get("class_v2") == "pure_decline":
        return False
    if not rec.get("total_claims"):
        return False  # 0 claims efectivos -> faithfulness 1.0 vacuo del runtime
    return True


def dz(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def main():
    data = json.loads((OUT_DIR / "replicas_checkpoint.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    rows = data["results"]
    print(f"registros: {len(rows)}")

    # (pair, model_label, qid) -> {rep: rec}
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["pair_id"], r["model_label"], r["query_id"])][r["replica"]] = r

    analysis = {}
    for pid, pmeta in meta["pairs"].items():
        scenario = pmeta["scenario"]
        labels = [t.replace("/", "-").replace(":", "-") for t in pmeta["models"]]
        la, lb = labels
        reps = sorted({r["replica"] for r in rows if r["pair_id"] == pid})
        qids = sorted({q for (p, _m, q) in cell if p == pid})

        # 1) estabilidad de texto por modelo
        text_stab = {}
        for lab in labels:
            same = diff = 0
            for q in qids:
                answers = {rep: cell[(pid, lab, q)][rep]["answer"]
                           for rep in reps if rep in cell[(pid, lab, q)]}
                if len(answers) < 2:
                    continue
                vals = list(answers.values())
                (same, diff) = (same + 1, diff) if all(v == vals[0] for v in vals) \
                    else (same, diff + 1)
            text_stab[lab] = {"identical": same, "divergent": diff}

        # 2) media v4 por celda por réplica
        cell_means = {}
        for lab in labels:
            cell_means[lab] = {}
            for rep in reps:
                vals = [cell[(pid, lab, q)][rep]["faithfulness"] for q in qids
                        if rep in cell[(pid, lab, q)]
                        and v4_eligible(cell[(pid, lab, q)][rep])]
                cell_means[lab][f"r{rep}"] = {
                    "mean": float(np.mean(vals)) if vals else None, "n": len(vals)}

        # 3) d_z por combinación de réplicas (pareado por query, elegible en ambos)
        combos = {}
        from scipy.stats import wilcoxon
        for ra, rb in product(reps, reps):
            a, b = [], []
            for q in qids:
                xa = cell[(pid, la, q)].get(ra)
                xb = cell[(pid, lb, q)].get(rb)
                if xa and xb and v4_eligible(xa) and v4_eligible(xb):
                    a.append(xa["faithfulness"])
                    b.append(xb["faithfulness"])
            if len(a) < 3:
                continue
            # convención del par original: d_z de (modelo_a - modelo_b)? Los
            # originales son "a vs b" con d_z = mean(b_scores - a_scores) en
            # compare_systems(scores_a=a, scores_b=b) -> aquí replicamos igual.
            d = dz(a, b)
            try:
                p = float(wilcoxon(a, b, alternative="two-sided").pvalue) \
                    if any(abs(x - y) > 1e-12 for x, y in zip(a, b)) else 1.0
            except ValueError:
                p = 1.0
            combos[f"r{ra}xr{rb}"] = {"d_z": round(d, 3), "p_raw": round(p, 4),
                                      "n": len(a)}

        dzs = [c["d_z"] for c in combos.values()]
        ref = ORIGINAL_DZ.get(pid)
        stable = bool(dzs) and ref is not None and all(
            np.sign(x) == np.sign(ref) or x == 0 for x in dzs) and all(
            abs(x - ref) <= DZ_TOL + abs(ref) * 0 for x in dzs)
        analysis[pid] = {
            "scenario": scenario, "models": labels, "n_queries": len(qids),
            "text_stability": text_stab, "cell_means_v4_by_replica": cell_means,
            "dz_by_replica_combo": combos,
            "dz_range": [min(dzs), max(dzs)] if dzs else None,
            "dz_original_v4": ref,
            "verdict": "ESTABLE" if stable else "SENSIBLE",
            "verdict_rule": f"signo constante y |d_z - original| <= {DZ_TOL} en las 9 combinaciones",
        }
        print(f"\n== {pid} ({scenario}) n={len(qids)}")
        print("  texto:", text_stab)
        print("  d_z original:", ref, "| rango réplicas:",
              analysis[pid]["dz_range"], "->", analysis[pid]["verdict"])

    (OUT_DIR / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito: {OUT_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
