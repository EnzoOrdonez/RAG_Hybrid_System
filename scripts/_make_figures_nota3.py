"""
Figuras finales Nota 3 (I6, N7) — generadas por script, reproducibles.

Fuentes (solo lectura): los JSON/CSV v2 ya versionados. Salida:
output/figures/nota3/*.png (300 dpi, etiquetas en español, estilo sobrio —
el caption va en el documento, no en la figura).

  f1_retrieval_ndcg_oraculos.png   NDCG@5 por sistema × oráculo (N2)
  f2_fidelidad_v2.png              faithfulness_answered escenario × modelo (+n)
  f2_fidelidad_v4.png              ídem con la métrica v4 (N9, sin filas vacuas) — CITABLE
  f3_census_declinacion_v2.png     % pure/hedged/answered apilado por config
  f4_latencia_gen_p50.png          gen p50 (s) por modelo, híbrido (qwen = cota sup.)

Usage: python scripts/_make_figures_nota3.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures" / "nota3"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["granite4.1-8b", "gemma4-e4b", "mistral-7b-instruct", "qwen3.5-9b"]
MODEL_DISP = {"granite4.1-8b": "Granite 4.1 8B", "gemma4-e4b": "Gemma 4 E4B",
              "mistral-7b-instruct": "Mistral 7B", "qwen3.5-9b": "Qwen 3.5 9B"}
SCEN = ["lexico", "denso", "hibrido"]
SCEN_DISP = {"lexico": "Léxico (BM25)", "denso": "Denso (BGE)", "hibrido": "Híbrido"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


def f1_retrieval():
    base = ROOT / "experiments/results/exp11_retrieval194_fullrerank"
    sys_order = [("RAG Lexico (BM25)", "Léxico\n(BM25)"),
                 ("RAG Semantico (Dense)", "Denso\n(BGE)"),
                 ("RAG Hibrido (pre-rerank RRF)", "Híbrido\npre-rerank"),
                 ("RAG Hibrido Propuesto", "Híbrido\npost-rerank")]
    oracles = [("bge-reranker-indep", "Oráculo independiente (bge)", "#2b6cb0"),
               ("ms-marco-circular", "Oráculo circular (ms-marco)", "#c05621")]
    vals = {}
    for okey, _, _ in oracles:
        d = json.loads((base / f"retrieval_metrics__{okey}.json").read_text(encoding="utf-8"))
        vals[okey] = [d["systems"][s]["ndcg@5_mean"] for s, _ in sys_order]
    x = range(len(sys_order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (okey, olabel, color) in enumerate(oracles):
        xs = [xi + (i - 0.5) * w for xi in x]
        bars = ax.bar(xs, vals[okey], width=w, label=olabel, color=color)
        for b, v in zip(bars, vals[okey]):
            ax.annotate(f"{v:.3f}".replace(".", ","), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in sys_order])
    ax.set_ylabel("NDCG@5 (graded)")
    ax.set_ylim(0, 1.12)
    ax.axhline(0.995, ls=":", lw=0.8, color="#c05621")
    ax.annotate("0,995 ≈ techo por construcción (circularidad, N2)",
                (0.02, 0.97), xycoords="axes fraction", fontsize=8.5, color="#c05621")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    fig.savefig(OUT / "f1_retrieval_ndcg_oraculos.png", dpi=300)
    plt.close(fig)


def _load_v2():
    return json.loads((ROOT / "experiments/results/exp12_matrix/faithfulness_metrics_v2.json")
                      .read_text(encoding="utf-8"))["systems_v2"]


def _load_v4():
    return json.loads((ROOT / "experiments/results/exp12_matrix/faithfulness_metrics_v4_small.json")
                      .read_text(encoding="utf-8"))["systems_v2"]


def f2_fidelidad(sys2=None, tag="v2"):
    if sys2 is None:
        sys2 = _load_v2()
    colors = {"lexico": "#a0aec0", "denso": "#4a8fd4", "hibrido": "#2b6cb0"}
    x = range(len(MODELS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, sc in enumerate(SCEN):
        vals, ns = [], []
        for m in MODELS:
            cell = sys2[f"{sc} | {m}"]["primary_answered"]
            vals.append(cell["mean"])
            ns.append(cell["n"])
        xs = [xi + (i - 1) * w for xi in x]
        bars = ax.bar(xs, vals, width=w, label=SCEN_DISP[sc], color=colors[sc])
        for b, v, n in zip(bars, vals, ns):
            ax.annotate(f"{v:.2f}".replace(".", ",") + f"\n(n={n})",
                        (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_DISP[m] for m in MODELS])
    ax.set_ylabel(f"Fidelidad NLI en contestadas (primaria {tag})")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / f"f2_fidelidad_{tag}.png", dpi=300)
    plt.close(fig)


def f3_census():
    sys2 = _load_v2()
    labels, pure, hedged, answered = [], [], [], []
    for m in MODELS:
        for sc in SCEN:
            cen = sys2[f"{sc} | {m}"]["decline_census_v2"]
            n = max(1, sys2[f"{sc} | {m}"]["n_total"] - cen["(empty)"])
            labels.append(f"{MODEL_DISP[m].split()[0]}·{SCEN_DISP[sc].split()[0].rstrip(' (')}")
            pure.append(100 * cen["pure_decline"] / n)
            hedged.append(100 * cen["hedged_partial"] / n)
            answered.append(100 * cen["answered"] / n)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x, answered, label="Contestada", color="#2f855a")
    ax.bar(x, hedged, bottom=answered, label="Parcial con hedge", color="#ecc94b")
    bottoms = [a + h for a, h in zip(answered, hedged)]
    ax.bar(x, pure, bottom=bottoms, label="Declinación pura", color="#c53030")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("% de respuestas (clasificador v2, N5)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.tight_layout()
    fig.savefig(OUT / "f3_census_declinacion_v2.png", dpi=300)
    plt.close(fig)


def f4_latencia():
    csv = (ROOT / "output/tables/nota3/latency__exp12_matrix.csv").read_text(encoding="utf-8")
    p50 = {}
    for line in csv.splitlines()[1:]:
        parts = line.split(";")
        if len(parts) < 4 or not parts[0].startswith("hibrido"):
            continue
        model = parts[0].split("|")[1].strip()
        p50[model] = float(parts[2].replace(",", "."))
    vals = [p50[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(range(len(MODELS)), vals, color="#4a5568", width=0.55)
    for i, (b, v) in enumerate(zip(bars, vals)):
        note = " *" if MODELS[i].startswith("qwen") else ""
        ax.annotate(f"{v:.1f} s".replace(".", ",") + note,
                    (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISP[m] for m in MODELS])
    ax.set_ylabel("Generación p50 (s) — escenario híbrido")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.annotate("* qwen3.5: cota superior (contención de GPU, N6)",
                (0.02, 0.95), xycoords="axes fraction", fontsize=8.5, color="#4a5568")
    fig.tight_layout()
    fig.savefig(OUT / "f4_latencia_gen_p50.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    f1_retrieval()
    f2_fidelidad()
    f2_fidelidad(_load_v4(), tag="v4")
    f3_census()
    f4_latencia()
    print("Figuras escritas en", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name, f"({p.stat().st_size // 1024} KB)")
