"""
Oracle-stability tables (oracle-circularity mitigation, Nota 3).

Reads every ``retrieval_metrics__<oracle>.json`` produced by
``compute_retrieval_metrics.py`` for one experiment and emits two tables that
show whether the system ordering (Hybrid >= Dense >= Lexical) depends on the
SHARED cross-encoder oracle:

  A) NDCG@5 (graded, threshold-free) per system x oracle.
  B) Key NDCG@5 contrasts (the two hybrids vs Dense) per oracle, with d_z and
     the BH-corrected p-value.

Why: the pipeline reranker is ms-marco-MiniLM-L-12-v2. Using that same model as
the relevance oracle is circular — after the D12 reranker fix the post-rerank
hybrid reaches NDCG@5 ~ 1.0 BY CONSTRUCTION. An independent oracle
(BAAI/bge-reranker-large) plus the PRE-rerank hybrid ranking break that loop.

Output: output/tables/nota3/oracle_stability__<exp>.{md,csv} (comma decimals).

Usage:
  python scripts/_oracle_stability.py                       # exp11_retrieval194_fullrerank
  python scripts/_oracle_stability.py --experiment exp11_retrieval194_fullrerank
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output" / "tables" / "nota3"

SHORT = {
    "RAG Lexico (BM25)": "Léxico (BM25)",
    "RAG Semantico (Dense)": "Denso (BGE)",
    "RAG Hibrido Propuesto": "Híbrido (post-rerank)",
    "RAG Hibrido (pre-rerank RRF)": "Híbrido (pre-rerank RRF)",
}
# Order rows for readability.
ROW_ORDER = [
    "RAG Lexico (BM25)",
    "RAG Semantico (Dense)",
    "RAG Hibrido (pre-rerank RRF)",
    "RAG Hibrido Propuesto",
]


def fmt(x: float, nd: int = 3) -> str:
    """Comma-decimal formatting, thesis style (0,923)."""
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_p(p: float) -> str:
    return "<0,001" if p < 0.001 else fmt(p, 3)


def oracle_sort_key(meta: dict) -> tuple:
    # Circular oracle first (reference), then independents alphabetically.
    return (0 if meta["circular"] else 1, meta["label"])


def main():
    ap = argparse.ArgumentParser(description="Oracle-stability tables")
    ap.add_argument("--experiment", default="exp11_retrieval194_fullrerank")
    args = ap.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.experiment
    files = sorted(glob.glob(str(exp_dir / "retrieval_metrics__*.json")))
    if not files:
        raise SystemExit(f"No retrieval_metrics__*.json under {exp_dir}")

    oracles = []  # list of dicts: label, circular, model, systems(ndcg), stats
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        oracles.append({
            "label": d.get("oracle_label", Path(f).stem),
            "circular": bool(d.get("oracle_is_circular", False)),
            "model": d.get("oracle_model", d.get("model", "?")),
            "ndcg": {s: v.get("ndcg@5_mean", float("nan")) for s, v in d["systems"].items()},
            "stats": d.get("statistical_tests", {}).get("ndcg@5", {}),
        })
    oracles.sort(key=oracle_sort_key)

    systems = [s for s in ROW_ORDER if s in oracles[0]["ndcg"]]
    # include any unexpected systems too
    systems += [s for s in oracles[0]["ndcg"] if s not in systems]

    def oracle_header(o):
        return f"{o['label']}{' (circular)' if o['circular'] else ' (independiente)'}"

    # ---- Table A: NDCG@5 per system x oracle ----
    a_cols = ["Sistema"] + [oracle_header(o) for o in oracles]
    a_rows = []
    for s in systems:
        a_rows.append([SHORT.get(s, s)] + [fmt(o["ndcg"].get(s, float("nan"))) for o in oracles])

    # ---- Table B: key contrasts vs Dense (ndcg@5) ----
    DENSE = "RAG Semantico (Dense)"
    contrasts = [
        ("RAG Hibrido Propuesto", DENSE),
        ("RAG Hibrido (pre-rerank RRF)", DENSE),
        ("RAG Hibrido Propuesto", "RAG Hibrido (pre-rerank RRF)"),
    ]
    b_cols = ["Contraste (NDCG@5)", "Oráculo", "d_z", "p_BH", "sig_BH"]
    b_rows = []

    def find_stat(stats: dict, a: str, b: str):
        # compare_systems stores effect_size as (second - first)/std. We display
        # "a vs b" with positive == a better than b (i.e. a - b).
        for key, r in stats.items():
            if key == f"{a} vs {b}":   # stored d = b - a  ->  flip
                return r, -1.0
            if key == f"{b} vs {a}":   # stored d = a - b  ->  matches display
                return r, +1.0
        return None, 1.0

    for a, b in contrasts:
        label = f"{SHORT.get(a, a)} vs {SHORT.get(b, b)}"
        for o in oracles:
            r, sign = find_stat(o["stats"], a, b)
            if r is None:
                b_rows.append([label, oracle_header(o), "—", "—", "—"])
                continue
            d = sign * r.get("effect_size", 0.0)
            p_bh = r.get("p_bh", r.get("p_value", float("nan")))
            sig = "sí" if r.get("sig_bh") else "no"
            d_str = ("+" if d >= 0 else "") + fmt(d, 2)
            b_rows.append([label, oracle_header(o), d_str, fmt_p(p_bh), sig])

    # ---- Render MD ----
    def md_table(cols, rows):
        out = ["| " + " | ".join(cols) + " |",
               "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        return "\n".join(out)

    md = []
    md.append(f"# Estabilidad del orden entre oráculos — {args.experiment}\n")
    md.append("Mitigación de la circularidad del oráculo. El reranker del pipeline "
              "es `ms-marco-MiniLM-L-12-v2`; usar ese mismo modelo como oráculo de "
              "relevancia es **circular** (tras el fix D12 el híbrido post-rerank "
              "alcanza NDCG@5 ≈ 1,0 por construcción). Se reporta además un oráculo "
              "**independiente** (`BAAI/bge-reranker-large`) y el ranking **pre-rerank** "
              "del híbrido (solo fusión RRF, sin el cross-encoder compartido).\n")
    md.append("## Tabla A — NDCG@5 graded por sistema y oráculo\n")
    md.append(md_table(a_cols, a_rows))
    md.append("\n## Tabla B — Contrastes clave NDCG@5 (d_z pareado, p BH-corregido)\n")
    md.append(md_table(b_cols, b_rows))
    md.append("\n**Lectura.** El orden Híbrido(post) > Denso > Híbrido(pre) ≈ Denso > "
              "Léxico se mantiene entre oráculos. La ventaja del híbrido sobre el denso "
              "es **real y significativa con el oráculo independiente** pero su magnitud "
              "estaba **inflada por la circularidad**; la fusión RRF sola (pre-rerank) "
              "**no** supera al denso. El aporte proviene de la etapa de reranking.\n")
    md_text = "\n".join(md)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"oracle_stability__{args.experiment}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # ---- Render CSV (both tables stacked) ----
    csv_lines = ["# Tabla A: NDCG@5 por sistema y oraculo (coma decimal)"]
    csv_lines.append(";".join(a_cols))
    for r in a_rows:
        csv_lines.append(";".join(str(x) for x in r))
    csv_lines.append("")
    csv_lines.append("# Tabla B: contrastes NDCG@5 (d_z, p_BH)")
    csv_lines.append(";".join(b_cols))
    for r in b_rows:
        csv_lines.append(";".join(str(x) for x in r))
    csv_path = OUT_DIR / f"oracle_stability__{args.experiment}.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    try:
        print(md_text)
    except UnicodeEncodeError:
        print(md_text.encode("ascii", "replace").decode("ascii"))
    print(f"\nWrote:\n  {md_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
