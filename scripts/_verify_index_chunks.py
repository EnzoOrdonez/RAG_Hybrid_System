"""TEMP: verify new index (faiss + bm25) chunk_ids all have chunk files.
Index/chunk mismatch -> STOP. Not committed."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data" / "indices"
CHUNK_DIR = ROOT / "data" / "chunks" / "adaptive" / "size_500"

faiss_map = json.loads((IDX / "faiss_bge-large_adaptive_500.mapping.json").read_text(encoding="utf-8"))
print("faiss mapping type:", type(faiss_map).__name__)
if isinstance(faiss_map, dict):
    print("  dict keys sample:", list(faiss_map.keys())[:3])
    # could be {idx: chunk_id} or {chunk_id: idx} or {'chunk_ids': [...]}
    if "chunk_ids" in faiss_map:
        faiss_ids = list(faiss_map["chunk_ids"])
    else:
        vals = list(faiss_map.values())
        keys = list(faiss_map.keys())
        # heuristic: UUIDs are 36-char strings with dashes
        if isinstance(vals[0], str) and len(vals[0]) == 36:
            faiss_ids = vals
        else:
            faiss_ids = keys
elif isinstance(faiss_map, list):
    faiss_ids = list(faiss_map)
else:
    print("UNEXPECTED faiss mapping structure"); sys.exit(2)

print("faiss_ids count:", len(faiss_ids))
print("faiss_ids sample:", faiss_ids[:3])

# chunk_map
cmap = json.loads((IDX / "chunk_map_bge-large_adaptive_500.json").read_text(encoding="utf-8"))
print("chunk_map type:", type(cmap).__name__, "size:", len(cmap))
cmap_ids = set(cmap.keys()) if isinstance(cmap, dict) else set()
print("chunk_map sample id:", next(iter(cmap_ids)) if cmap_ids else None)

# bm25 pkl
sys.path.insert(0, str(ROOT))
bm25_ids = None
try:
    raw = pickle.loads((IDX / "bm25_adaptive_500.pkl").read_bytes())
    print("bm25 pkl top type:", type(raw).__name__)
    if isinstance(raw, dict):
        print("  bm25 dict keys:", list(raw.keys()))
        for k in ("chunk_ids", "ids", "doc_ids"):
            if k in raw:
                bm25_ids = list(raw[k]); break
    else:
        for attr in ("chunk_ids", "ids", "doc_ids"):
            if hasattr(raw, attr):
                bm25_ids = list(getattr(raw, attr)); break
except Exception as e:
    print("bm25 raw pickle load failed:", repr(e))
print("bm25_ids count:", len(bm25_ids) if bm25_ids is not None else None)

# count chunk files
chunk_files = {p.stem for p in CHUNK_DIR.glob("*.json")}
print("chunk files on disk:", len(chunk_files))

# consistency: all faiss ids -> chunk file?
faiss_set = set(faiss_ids)
missing_faiss = faiss_set - chunk_files
print("faiss ids WITHOUT chunk file:", len(missing_faiss))
if missing_faiss:
    print("  examples:", list(missing_faiss)[:5])

if bm25_ids is not None:
    bm25_set = set(bm25_ids)
    print("bm25 ids WITHOUT chunk file:", len(bm25_set - chunk_files))
    print("faiss vs bm25 same set:", faiss_set == bm25_set,
          "| only_faiss:", len(faiss_set - bm25_set), "only_bm25:", len(bm25_set - faiss_set))

print("chunk_map vs faiss same set:", cmap_ids == faiss_set,
      "| only_cmap:", len(cmap_ids - faiss_set), "only_faiss:", len(faiss_set - cmap_ids))

# verify one chunk file readable + has text
sample_id = faiss_ids[0]
sp = CHUNK_DIR / f"{sample_id}.json"
if sp.exists():
    d = json.loads(sp.read_text(encoding="utf-8"))
    print("sample chunk", sample_id, "has text:", bool(d.get("text")), "len:", len(d.get("text", "")))

VERDICT = "OK" if not missing_faiss else "MISMATCH"
print("VERDICT:", VERDICT)
