"""
Per-stage latency p50/p95 per (scenario, model) from an exp12-style results.json.

Excludes from_cache=True rows (cached generations report latency_ms=0) and
error rows. Emits output/tables/nota3/latency__<exp>.{csv,md} (comma decimals).

Usage:
  python scripts/_latency_p50p95.py [exp12_matrix]
"""
import json
import os
import sys

import numpy as np

exp = sys.argv[1] if len(sys.argv) > 1 else "exp12_matrix"
base = os.path.join("experiments", "results", exp)
out_dir = os.path.join("output", "tables", "nota3")
os.makedirs(out_dir, exist_ok=True)

data = json.loads(open(os.path.join(base, "results.json"), encoding="utf-8").read())


def fmt(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


rows = []  # (config, n, gen_p50, gen_p95, nli_p50, nli_p95, total_p50, total_p95, toks_p50)
for cname, c in data["configs"].items():
    gen, nli, tot, toks = [], [], [], []
    for r in c["results"]:
        if r.get("from_cache") or r.get("error"):
            continue
        lat = r.get("latency", {})
        # N8: only record latencies that are actually present; a missing key must
        # NOT inject a phantom 0 ms that deflates the reported p50/p95 table.
        if lat.get("generation_ms") is not None:
            gen.append(lat["generation_ms"])
        if lat.get("hallucination_check_ms") is not None:
            nli.append(lat["hallucination_check_ms"])
        if lat.get("total_ms") is not None:
            tot.append(lat["total_ms"])
        tps = r.get("cost_proxy", {}).get("tok_per_s")
        if tps:
            toks.append(tps)
    if not gen:
        continue

    def p(a, q):
        return float(np.percentile(a, q)) if a else 0.0
    rows.append((cname, len(gen),
                 p(gen, 50) / 1000, p(gen, 95) / 1000,
                 p(nli, 50), p(nli, 95),
                 p(tot, 50) / 1000, p(tot, 95) / 1000,
                 p(toks, 50)))

rows.sort()
cols = ["Config (escenario | modelo)", "n", "gen p50 (s)", "gen p95 (s)",
        "NLI p50 (ms)", "NLI p95 (ms)", "total p50 (s)", "total p95 (s)", "tok/s p50"]

# CSV
csv = [";".join(cols)]
for r in rows:
    csv.append(";".join([r[0], str(r[1])] + [fmt(v) for v in r[2:]]))
open(os.path.join(out_dir, f"latency__{exp}.csv"), "w", encoding="utf-8").write("\n".join(csv))

# MD
md = [f"# Latencias por etapa (p50/p95) — {exp}\n",
      "Excluye respuestas cacheadas (`from_cache=True`, latency=0) y errores. "
      "Generación y total en segundos; verificación NLI en ms.\n",
      "**Nota qwen3.5 (N6):** sus corridas (06-09 → 06-11) compartieron GPU con un "
      "segundo proceso durante parte de la ventana; sus latencias se reportan como "
      "**cota superior**. No se re-muestrea: qwen3.5 es el modelo más lento con o sin "
      "contención y mayor precisión no cambia ninguna conclusión (decisión N6).\n",
      "| " + " | ".join(cols) + " |",
      "|" + "|".join(["---"] * len(cols)) + "|"]
for r in rows:
    md.append("| " + " | ".join([r[0], str(r[1])] + [fmt(v) for v in r[2:]]) + " |")
md_text = "\n".join(md)
open(os.path.join(out_dir, f"latency__{exp}.md"), "w", encoding="utf-8").write(md_text)

# Console (ascii-safe)
print(f"{'config':<30} {'n':>4} {'gen_p50s':>9} {'gen_p95s':>9} {'nli_p50ms':>10} {'tok/s':>7}")
print("-" * 74)
for r in rows:
    print(f"{r[0][:30]:<30} {r[1]:>4} {r[2]:>9.1f} {r[3]:>9.1f} {r[4]:>10.0f} {r[8]:>7.1f}")
print(f"\nWrote output/tables/nota3/latency__{exp}.{{csv,md}}")
