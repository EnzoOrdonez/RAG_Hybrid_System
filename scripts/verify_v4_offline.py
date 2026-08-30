"""Phase 0 (summer) — offline verification of the v4 (N9) citable figures.

Recomputes, IN MEMORY, the v4 faithfulness aggregates and paired statistics
from the signed inputs (exp12_matrix/results.json + faithfulness_rescore_v3
JSONs) by importing functions from scripts/compute_faithfulness_metrics.py
(never running its main(), which writes into the signed exp12 dir), and
checks retrieval/exp13 headline numbers for consistency against the signed
metric JSONs. Writes NOTHING outside output/audit/ and stdout.

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
  python scripts/verify_v4_offline.py
Exit code 0 = every check passed; 1 = at least one mismatch (details on stdout
and in output/audit/phase0_verification_summer_<date>.md).
"""

import importlib.util
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP12 = ROOT / "experiments" / "results" / "exp12_matrix"
EXP11 = ROOT / "experiments" / "results" / "exp11_retrieval194_fullrerank"
EXP13 = ROOT / "experiments" / "results" / "exp13_expansion"
REPORT = ROOT / "output" / "audit" / f"v4_offline_check_{date.today().isoformat()}.md"

TOL = 1e-9        # exact-recompute tolerance (faithfulness cells, d_z, p_bh)
TOL_DOC = 5e-5    # documented-4-decimals tolerance (NDCG headline numbers)

failures = []
lines = []          # report lines


def log(s=""):
    print(s)
    lines.append(s)


def check(name, ok, detail=""):
    status = "OK " if ok else "FAIL"
    log(f"- [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# Import compute_faithfulness_metrics as a module (main() is __main__-guarded)
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "cfm", ROOT / "scripts" / "compute_faithfulness_metrics.py")
cfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfm)

log(f"# Verificación Fase 0 (verano) — cifras v4 desde JSONs firmados")
log(f"Fecha: {date.today().isoformat()} · script: scratchpad/phase0_verify_v4.py (solo lectura sobre experiments/)")
log()

# ---------------------------------------------------------------------------
# 1. Faithfulness v4: full in-memory recompute, both verifiers
# ---------------------------------------------------------------------------
results_path = EXP12 / "results.json"

for tag, rescore_name in (("v4_small", "faithfulness_rescore_v3__small__vb_agree.json"),
                          ("v4", "faithfulness_rescore_v3__base__vb_agree.json")):
    log(f"## Fidelidad {tag} (rescore: {rescore_name})")
    stored = json.loads((EXP12 / f"faithfulness_metrics_{tag}.json").read_text(encoding="utf-8"))
    src = json.loads((EXP12 / rescore_name).read_text(encoding="utf-8"))
    override = src.get("configs", src)

    # replicate --exclude-vacuous: main() mutates the module-level set
    cfm.EXCLUDED_METHODS.clear()
    cfm.EXCLUDED_METHODS.update({"none", "error", "vacuous"})

    per_config = cfm.load_per_config(results_path, override, exclude_vacuous=True)
    configs = list(per_config.keys())
    parsed = {c: cfm.parse_config(c) for c in configs}
    scenarios = sorted({parsed[c][0] for c in configs})
    models = sorted({parsed[c][1] for c in configs})

    # -- 16 primary_answered cells --------------------------------------------
    n_cells_ok = 0
    for c in configs:
        recs = [v for v in per_config[c].values()
                if v["method"] not in cfm.EXCLUDED_METHODS and v["faithfulness"] is not None]
        vals = [r["faithfulness"] for r in recs if cfm._incl_primary(r)]
        mean = float(np.mean(vals)) if vals else 0.0
        n = len(vals)
        st = stored["systems_v2"][c]["primary_answered"]
        ok = abs(mean - st["mean"]) < TOL and n == st["n"]
        if ok:
            n_cells_ok += 1
        else:
            check(f"celda {tag} '{c}'", False,
                  f"recomputado {mean:.6f}(n={n}) vs almacenado {st['mean']:.6f}(n={st['n']})")
    check(f"{tag}: celdas primary_answered", n_cells_ok == len(configs),
          f"{n_cells_ok}/{len(configs)} exactas (tol {TOL})")

    # -- families: replicate main()'s pair construction -----------------------
    fam_b_pairs = []
    for m in models:
        cfgs_m = [c for c in configs if parsed[c][1] == m]
        scen_to_cfg = {parsed[c][0]: c for c in cfgs_m}
        for s1, s2 in itertools.combinations(sorted(scen_to_cfg), 2):
            fam_b_pairs.append((scen_to_cfg[s1], scen_to_cfg[s2]))
    fam_c_pairs = []
    for s in scenarios:
        if s == "sin_rag":
            continue
        cfgs_s = [c for c in configs if parsed[c][0] == s]
        model_to_cfg = {parsed[c][1]: c for c in cfgs_s}
        for m1, m2 in itertools.combinations(sorted(model_to_cfg), 2):
            fam_c_pairs.append((model_to_cfg[m1], model_to_cfg[m2]))

    fam_b = cfm.run_family(per_config, fam_b_pairs, f"b-{tag}",
                           include=cfm._incl_primary, metric_label="faithfulness_answered")
    fam_c = cfm.run_family(per_config, fam_c_pairs, f"c-{tag}",
                           include=cfm._incl_primary, metric_label="faithfulness_answered")

    for fam_name, fam, stored_key in (
            ("between_scenario", fam_b, "faithfulness_answered__between_scenario"),
            ("between_model", fam_c, "faithfulness_answered__between_model")):
        st_fam = stored["statistical_tests_v2"][stored_key]
        check(f"{tag}/{fam_name}: mismo conjunto de pares", set(fam) == set(st_fam),
              f"{len(fam)} pares")
        n_pair_ok = 0
        for pair, r in fam.items():
            s = st_fam.get(pair)
            if s is None:
                continue
            ok = (abs(r.get("effect_size", 0) - s.get("effect_size", 0)) < TOL
                  and abs(r.get("p_bh", 1) - s.get("p_bh", 1)) < TOL
                  and r.get("n") == s.get("n")
                  and bool(r.get("sig_bh")) == bool(s.get("sig_bh")))
            if ok:
                n_pair_ok += 1
            else:
                check(f"{tag}/{fam_name} par '{pair}'", False,
                      f"d_z {r.get('effect_size'):.6f}/{s.get('effect_size'):.6f} "
                      f"p_bh {r.get('p_bh'):.6g}/{s.get('p_bh'):.6g} "
                      f"n {r.get('n')}/{s.get('n')} sig {r.get('sig_bh')}/{s.get('sig_bh')}")
        check(f"{tag}/{fam_name}: pares exactos (d_z, p_bh, n, sig_bh)",
              n_pair_ok == len(st_fam), f"{n_pair_ok}/{len(st_fam)}")

    # -- headline claims ------------------------------------------------------
    rag_pairs_sig = [p for p, r in fam_b.items()
                     if "sin_rag" not in p and r.get("sig_bh")]
    rag_pairs_all = [p for p in fam_b if "sin_rag" not in p]
    check(f"{tag}: 0/12 RAG-vs-RAG significativos",
          len(rag_pairs_all) == 12 and not rag_pairs_sig,
          f"{len(rag_pairs_sig)}/{len(rag_pairs_all)} sig")
    model_sig = sorted(p for p, r in fam_c.items() if r.get("sig_bh"))
    log(f"  - entre-modelos sig ({tag}): {model_sig if model_sig else 'ninguno'} "
        f"de {len(fam_c)} pares")
    stored["_model_sig"] = model_sig  # stash for cross-verifier robustness check
    if tag == "v4_small":
        small_sig = model_sig
    else:
        base_sig = model_sig
    log()

# cross-verifier robustness: 1/18 + 1/18, disjoint -> 0/18 robust
check("0/18 robusto entre verificadores (pares sig disjuntos)",
      len(small_sig) == 1 and len(base_sig) == 1 and not set(small_sig) & set(base_sig),
      f"small={small_sig} base={base_sig}")

# -- Granite headline cells + CSV tabla6 cross-check (small/citable) ----------
stored_small = json.loads((EXP12 / "faithfulness_metrics_v4_small.json").read_text(encoding="utf-8"))
gran = {s: stored_small["systems_v2"][f"{s} | granite4.1-8b"]["primary_answered"]
        for s in ("lexico", "denso", "hibrido")}
expected_gran = {"lexico": (0.235, 75), "denso": (0.247, 85), "hibrido": (0.299, 87)}
ok = all(abs(gran[s]["mean"] - m) < 5e-4 and gran[s]["n"] == n
         for s, (m, n) in expected_gran.items())
check("Granite 0.235(75)/0.247(85)/0.299(87) (RESULTADOS_RESUMEN Tabla 6 v4)", ok,
      "; ".join(f"{s}={gran[s]['mean']:.3f}({gran[s]['n']})" for s in gran))

csv_path = ROOT / "output" / "tables" / "nota3" / "tabla6_fidelidad_v4__exp12_matrix.csv"
csv_rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
scen_map = {"RAG l": "lexico", "RAG d": "denso", "RAG h": "hibrido", "Sin R": "sin_rag"}
model_cols = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
n_csv_ok, n_csv = 0, 0
for row in csv_rows[1:]:
    cells = row.split(";")
    scen = scen_map.get(cells[0][:5])
    if scen is None:
        continue
    for mi, cell in enumerate(cells[1:5]):
        val = float(cell.split(" ")[0].replace(",", ".").rstrip("*"))
        n_par = int(cell.split("(")[1].rstrip(")"))
        st = stored_small["systems_v2"][f"{scen} | {model_cols[mi]}"]["primary_answered"]
        n_csv += 1
        if abs(round(st["mean"], 3) - val) < 5e-4 and st["n"] == n_par:
            n_csv_ok += 1
        else:
            check(f"CSV tabla6 {scen}/{model_cols[mi]}", False,
                  f"csv {val}({n_par}) vs json {st['mean']:.3f}({st['n']})")
check("CSV tabla6_v4 == JSON v4_small (16 celdas)", n_csv_ok == n_csv, f"{n_csv_ok}/{n_csv}")
log()

# ---------------------------------------------------------------------------
# 2. Retrieval exp11 — consistency (full recompute impossible offline)
# ---------------------------------------------------------------------------
log("## Retrieval exp11 (consistencia; recomputación completa imposible offline)")
expected_ndcg = {
    "bge-reranker-indep": {"RAG Lexico (BM25)": 0.4421, "RAG Semantico (Dense)": 0.6237,
                           "RAG Hibrido Propuesto": 0.7405, "RAG Hibrido (pre-rerank RRF)": 0.6026},
    "ms-marco-circular": {"RAG Lexico (BM25)": 0.5516, "RAG Semantico (Dense)": 0.6494,
                          "RAG Hibrido Propuesto": 0.9948, "RAG Hibrido (pre-rerank RRF)": 0.6681},
}
for label, exp_vals in expected_ndcg.items():
    rm = json.loads((EXP11 / f"retrieval_metrics__{label}.json").read_text(encoding="utf-8"))
    check(f"exp11/{label}: total_queries=194", rm["total_queries"] == 194)
    circ_expected = label == "ms-marco-circular"
    check(f"exp11/{label}: oracle_is_circular={circ_expected}",
          bool(rm["oracle_is_circular"]) == circ_expected, f"oracle={rm['oracle_model']}")
    n_ok = sum(1 for sysname, v in exp_vals.items()
               if abs(rm["systems"][sysname]["ndcg@5_mean"] - v) < TOL_DOC)
    detail = "; ".join(f"{s.split('(')[0].strip()}={rm['systems'][s]['ndcg@5_mean']:.4f}"
                       for s in exp_vals)
    check(f"exp11/{label}: 4 NDCG@5 vs RESULTADOS_RESUMEN", n_ok == 4, detail)
log("- Nota: los scores por par (query, chunk) del oráculo NO están persistidos en exp11;")
log("  la recomputación completa del NDCG queda bloqueada hasta la descarga de modelos (decisión")
log("  tomada: snapshot a data\\models). Cerrable después con recompute a dir NUEVO, nunca in-place.")
log()

# ---------------------------------------------------------------------------
# 3. exp13 expansion — ON≈OFF from stored metrics
# ---------------------------------------------------------------------------
log("## exp13 expansión (25 q cross-cloud)")
r13 = json.loads((EXP13 / "results.json").read_text(encoding="utf-8"))
check("exp13: 25 queries", r13.get("num_queries") == 25 or len(
    next(iter(r13["configs"].values()))["results"]) == 25)
rm13_path = next(EXP13.glob("retrieval_metrics__*bge*.json"), None)
if rm13_path:
    rm13 = json.loads(rm13_path.read_text(encoding="utf-8"))
    vals = {k: v.get("ndcg@5_mean") for k, v in rm13["systems"].items()}
    log(f"  - NDCG@5 ({rm13_path.name}): " +
        "; ".join(f"{k}={v:.3f}" for k, v in vals.items() if v is not None))
    offv = [v for k, v in vals.items() if "off" in k.lower()]
    onv = [v for k, v in vals.items() if "on" in k.lower() and "off" not in k.lower()]
    check("exp13: NDCG off≈0.852 / on≈0.820",
          bool(offv and onv) and abs(offv[0] - 0.852) < 5e-3 and abs(onv[0] - 0.820) < 5e-3,
          f"off={offv} on={onv}")
fm13_path = EXP13 / "faithfulness_metrics_v2.json"
if fm13_path.exists():
    fm13 = json.loads(fm13_path.read_text(encoding="utf-8"))
    prim = {k: v["primary_answered"]["mean"] for k, v in fm13["systems_v2"].items()}
    log("  - fidelidad primary (v2): " + "; ".join(f"{k}={v:.3f}" for k, v in prim.items()))
    offf = [v for k, v in prim.items() if "off" in k.lower()]
    onf = [v for k, v in prim.items() if "on" in k.lower() and "off" not in k.lower()]
    # RESULTADOS_RESUMEN cita dos versiones: v1 0,175≈0,174 (línea 50, N4) y la
    # re-medición v2 0,285 vs 0,324 n=10 n.s. (línea 208, N7). Verificamos la v2 vigente.
    check("exp13: fidelidad v2 off≈0.285 / on≈0.324 (RESUMEN §8, N7)",
          bool(offf and onf) and abs(offf[0] - 0.285) < 5e-3 and abs(onf[0] - 0.324) < 5e-3,
          f"off={offf} on={onf}")
log()

# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------
log("## Veredicto")
if failures:
    log(f"**{len(failures)} FALLOS**: " + "; ".join(failures))
else:
    log("**TODO CUADRA** — cifras v4 reproducidas en memoria desde JSONs firmados; "
        "ningún archivo de experiments/ modificado.")
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReporte: {REPORT}")
sys.exit(1 if failures else 0)
