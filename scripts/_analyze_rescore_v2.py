"""
Verifier-agreement analysis for the F3b re-score (ledger N5).

Inputs (read-only):
  experiments/results/exp12_matrix/results.json                  (verifier 1: small, persisted)
  experiments/results/exp12_matrix/faithfulness_rescore__nli-base.json
  experiments/results/exp12_matrix/faithfulness_rescore__small-noheader.json
  output/audit/claim_audit_sample.csv + claim_audit_sample_scores_v2.json

Reports, per arm:
  - Tabla 6 under the arm's verifier (published + v2-primary denominators)
  - contradiction band per model (micro %, RAG scenarios)
  - Spearman of the 12 RAG config means vs verifier 1
  - per-model scenario ordering under both denominators
  - Cohen's kappa (3 classes) vs verifier 1 on the 50-claim human sample
  - q085 hibrido|granite4.1-8b under the arm

Output: console + output/audit/rescore_v2_summary.md
Usage: python scripts/_analyze_rescore_v2.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_faithfulness_metrics import classify_response  # noqa: E402

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
SCENARIOS = ["lexico", "denso", "hibrido"]
EXP = PROJECT_ROOT / "experiments/results/exp12_matrix"
ARM_FILES = {"base": "faithfulness_rescore__nli-base.json",
             "small_noheader": "faithfulness_rescore__small-noheader.json"}


def cohen_kappa(y1, y2, labels):
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    results = json.loads((EXP / "results.json").read_text(encoding="utf-8"))["configs"]

    # Verifier-1 per-query metrics + v2 class per answer.
    v1 = {}
    for m in MODELS:
        for sc in SCENARIOS:
            cname = f"{sc} | {m}"
            for r in results[cname]["results"]:
                hm = r.get("hallucination_metrics") or {}
                if hm.get("method") != "nli" or not (hm.get("total_claims") or 0):
                    continue
                v1[(cname, r["query_id"])] = {
                    "faith": float(hm["faithfulness"]),
                    "contr": hm.get("contradicted_claims") or 0,
                    "supp": hm.get("supported_claims") or 0,
                    "total": hm.get("total_claims") or 0,
                    "class": classify_response(r.get("answer")),
                }

    lines = ["# Acuerdo entre verificadores NLI — exp12_matrix (F3b, N5)", ""]

    def emit(s=""):
        print(s)
        lines.append(s)

    # 50-claim sample labels.
    sample_rows = []
    csv_path = PROJECT_ROOT / "output/audit/claim_audit_sample.csv"
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            sample_rows.append(row)
    arm_labels = json.loads(
        (PROJECT_ROOT / "output/audit/claim_audit_sample_scores_v2.json")
        .read_text(encoding="utf-8"))

    for arm, fname in ARM_FILES.items():
        path = EXP / fname
        if not path.exists():
            emit(f"\n[{arm}] {fname} not found — skipped")
            continue
        arm_data = json.loads(path.read_text(encoding="utf-8"))
        emit(f"\n## Arm `{arm}` — verificador {arm_data['verifier']}, "
             f"strip_header={arm_data['strip_header']}")
        emit(f"claim-count mismatches vs persistidos: {arm_data['claim_count_mismatches']}")

        # Tabla 6 per denominator + contradiction band.
        emit(f"\n| config | pub v1 | pub {arm} | prim v1 | prim {arm} | contr% v1 | contr% {arm} |")
        emit("|---|---|---|---|---|---|---|")
        means_v1_pub, means_arm_pub, means_v1_prim, means_arm_prim = [], [], [], []
        for m in MODELS:
            for sc in SCENARIOS:
                cname = f"{sc} | {m}"
                rows = arm_data["configs"].get(cname, {})
                pv1 = [v1[(cname, q)]["faith"] for q in rows if (cname, q) in v1]
                pa = [rows[q]["faithfulness"] for q in rows if (cname, q) in v1]
                prim_idx = [q for q in rows if (cname, q) in v1
                            and v1[(cname, q)]["class"] != "pure_decline"]
                qv1 = [v1[(cname, q)]["faith"] for q in prim_idx]
                qa = [rows[q]["faithfulness"] for q in prim_idx]
                c_v1 = 100 * sum(v1[(cname, q)]["contr"] for q in rows if (cname, q) in v1) \
                    / max(1, sum(v1[(cname, q)]["total"] for q in rows if (cname, q) in v1))
                c_a = 100 * sum(rows[q]["contradicted"] for q in rows) \
                    / max(1, sum(rows[q]["total_claims"] for q in rows))
                means_v1_pub.append(np.mean(pv1)); means_arm_pub.append(np.mean(pa))
                means_v1_prim.append(np.mean(qv1)); means_arm_prim.append(np.mean(qa))
                emit(f"| {cname} | {np.mean(pv1):.3f} | {np.mean(pa):.3f} | "
                     f"{np.mean(qv1):.3f} | {np.mean(qa):.3f} | {c_v1:.1f} | {c_a:.1f} |")

        rho_pub = spearmanr(means_v1_pub, means_arm_pub)
        rho_prim = spearmanr(means_v1_prim, means_arm_prim)
        emit(f"\nSpearman 12 configs (publicada): rho={rho_pub.statistic:.3f} p={rho_pub.pvalue:.4f}")
        emit(f"Spearman 12 configs (primaria v2): rho={rho_prim.statistic:.3f} p={rho_prim.pvalue:.4f}")

        # Scenario ordering per model (primary denominator).
        emit("\nOrden de escenarios por modelo (primaria):")
        for mi, m in enumerate(MODELS):
            tri_v1 = means_v1_prim[mi * 3:(mi + 1) * 3]
            tri_a = means_arm_prim[mi * 3:(mi + 1) * 3]
            o = lambda t: ">".join(s for s, _ in
                                   sorted(zip(SCENARIOS, t), key=lambda x: -x[1]))
            emit(f"  {m:<22} v1: {o(tri_v1):<24} {arm}: {o(tri_a)}")

        # Kappa on the 50-claim sample.
        labels = arm_labels.get(arm, {})
        y1 = [r["nli_label"] for r in sample_rows if r["idx"] in labels]
        y2 = [labels[r["idx"]]["label"] for r in sample_rows if r["idx"] in labels]
        kap = cohen_kappa(y1, y2, ["supported", "contradicted", "unsupported"])
        emit(f"\nMuestra de 50 claims: matched={len(y1)}  kappa(v1 vs {arm})={kap:.3f}")
        conf = {}
        for a, b in zip(y1, y2):
            conf[(a, b)] = conf.get((a, b), 0) + 1
        emit("confusión (v1 -> arm): " + ", ".join(
            f"{a}->{b}: {n}" for (a, b), n in sorted(conf.items())))

        q85 = arm_data["configs"].get("hibrido | granite4.1-8b", {}).get("q085")
        if q85:
            emit(f"\nq085 bajo {arm}: {q85}  (v1: 28/28 contradicted, faith 0.0)")

    out = PROJECT_ROOT / "output/audit/rescore_v2_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
