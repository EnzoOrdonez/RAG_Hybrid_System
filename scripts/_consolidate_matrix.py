"""
Fold run_generation_matrix checkpoints into a single results.json.

Standalone version of the runner's final consolidation step, so a partially
completed matrix (interrupted run) can be analyzed without waiting for the full
run to finish and reach its own consolidation. Idempotent; the resumed run will
overwrite results.json with the complete set at its end.

Usage:
  python scripts/_consolidate_matrix.py [exp12_matrix]
"""
import glob
import json
import os
import sys
from datetime import datetime

exp = sys.argv[1] if len(sys.argv) > 1 else "exp12_matrix"
base = os.path.join("experiments", "results", exp)
configs = {}
partial = []
for f in sorted(glob.glob(os.path.join(base, "checkpoint__*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    rs = d["results"]
    cname = d["config_name"]
    scen, _, lab = cname.partition(" | ")
    n = len(rs)
    if n < 194:
        partial.append((cname, n))
    configs[cname] = {
        "total_queries": n,
        "errors": sum(1 for r in rs if r.get("error")),
        "scenario": scen.strip(),
        "model": lab.strip(),
        "results": rs,
    }

payload = {
    "experiment_id": exp,
    "consolidated_from_checkpoints": True,
    "complete": len(partial) == 0 and len(configs) > 0,
    "timestamp": datetime.now().isoformat(),
    "configs": configs,
}
out = os.path.join(base, "results.json")
open(out, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"Consolidated {len(configs)} configs -> {out}")
if partial:
    print("PARTIAL configs (n<194):", partial)
