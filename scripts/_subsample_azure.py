"""
Fase 2.5 — Sub-sample Azure Functions (10681->2500) and Blob Storage (3841->1500)
with stratified sampling by section_hierarchy[0]. Rebuilds chunk_map,
embeddings .npy, FAISS, BM25 from the surviving subset. Deletes dropped
chunk JSONs from data/chunks/adaptive/size_500/ after backup.

Reproducible via np.random.seed(42). Backups in data/{indices,embeddings,chunks}/backup_pre_subsample/.
"""

import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CHUNK_MAP_PATH = ROOT / "data" / "indices" / "chunk_map_bge-large_adaptive_500.json"
EMB_PATH = ROOT / "data" / "embeddings" / "bge-large_adaptive_500.npy"
IDS_PATH = ROOT / "data" / "embeddings" / "bge-large_adaptive_500_ids.npy"
FAISS_PATH = ROOT / "data" / "indices" / "faiss_bge-large_adaptive_500.index"
BM25_PATH = ROOT / "data" / "indices" / "bm25_adaptive_500.pkl"
CHUNKS_DIR = ROOT / "data" / "chunks" / "adaptive" / "size_500"

BACKUP_INDICES = ROOT / "data" / "indices" / "backup_pre_subsample"
BACKUP_EMB = ROOT / "data" / "embeddings" / "backup_pre_subsample"
BACKUP_CHUNKS = ROOT / "data" / "chunks" / "backup_pre_subsample" / "adaptive" / "size_500"

TARGETS = {"Azure Functions": 2500, "Blob Storage": 1500}
SEED = 42

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def stratified_sample(chunks: list, target: int, seed: int) -> tuple:
    """Stratified random sample by section_hierarchy[0].

    Returns (survivors, allocs_dict).
    """
    by_section = defaultdict(list)
    for c in chunks:
        sec = c.get("section_hierarchy") or [""]
        sec0 = sec[0] if sec else "<no_section>"
        by_section[sec0].append(c)

    service_total = len(chunks)
    allocs = {}
    for sec, chs in by_section.items():
        raw = target * len(chs) / service_total
        allocs[sec] = max(1, round(raw))

    diff = target - sum(allocs.values())
    if diff > 0:
        for sec in sorted(by_section, key=lambda s: -len(by_section[s])):
            if diff == 0:
                break
            if allocs[sec] < len(by_section[sec]):
                allocs[sec] += 1
                diff -= 1
    elif diff < 0:
        for sec in sorted(by_section, key=lambda s: allocs[s]):
            if diff == 0:
                break
            if allocs[sec] > 1:
                allocs[sec] -= 1
                diff += 1

    rng = np.random.default_rng(seed)
    survivors = []
    for sec, chs in by_section.items():
        n = allocs[sec]
        if n >= len(chs):
            survivors.extend(chs)
        else:
            idx = rng.choice(len(chs), size=n, replace=False)
            survivors.extend(chs[i] for i in idx)
    return survivors, allocs


def main() -> int:
    log("Loading chunk_map ...")
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    log(f"  {len(chunk_map)} chunks")

    log("Loading embeddings + ids ...")
    embeddings = np.load(EMB_PATH)
    ids = np.load(IDS_PATH)
    log(f"  embeddings shape={embeddings.shape} | ids shape={ids.shape}")
    if not (len(embeddings) == len(ids) == len(chunk_map)):
        raise SystemExit(
            f"Alignment failure: embeddings={len(embeddings)}, ids={len(ids)}, chunk_map={len(chunk_map)}"
        )

    log("Creating backups ...")
    BACKUP_INDICES.mkdir(parents=True, exist_ok=True)
    BACKUP_EMB.mkdir(parents=True, exist_ok=True)
    BACKUP_CHUNKS.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data" / "indices").iterdir():
        if f.is_file():
            shutil.copy2(f, BACKUP_INDICES / f.name)
    for f in (ROOT / "data" / "embeddings").glob("*.npy"):
        shutil.copy2(f, BACKUP_EMB / f.name)
    log(f"  indices + embeddings backed up")

    drop_ids: set = set()
    diversity_stats = {}
    for service, target in TARGETS.items():
        svc_chunks = [
            c for c in chunk_map.values()
            if c.get("cloud_provider") == "azure" and c.get("service_name") == service
        ]
        log(f"  {service}: {len(svc_chunks)} chunks -> target {target}")
        if len(svc_chunks) <= target:
            log("    SKIP (already <= target)")
            continue

        def _sec0(c):
            sh = c.get("section_hierarchy") or [""]
            return sh[0] if sh else ""

        pre_sections = len(set(_sec0(c) for c in svc_chunks))
        survivors, allocs = stratified_sample(svc_chunks, target, seed=SEED)
        post_sections = len(set(_sec0(c) for c in survivors))

        surv_ids = {c["chunk_id"] for c in survivors}
        svc_ids = {c["chunk_id"] for c in svc_chunks}
        new_drops = svc_ids - surv_ids
        drop_ids |= new_drops
        diversity_stats[service] = (pre_sections, post_sections, len(survivors), len(new_drops))
        log(f"    survivors={len(survivors)} dropped={len(new_drops)} sections pre={pre_sections} post={post_sections}")

    log(f"Total drop set: {len(drop_ids)} chunk_ids")

    log("Backing up dropped chunk JSONs ...")
    backed = 0
    missing = 0
    for did in drop_ids:
        src = CHUNKS_DIR / f"{did}.json"
        if src.exists():
            shutil.copy2(src, BACKUP_CHUNKS / src.name)
            backed += 1
        else:
            missing += 1
    log(f"  backed_up={backed} missing={missing} -> {BACKUP_CHUNKS}")

    log("Building surviving subset ...")
    surviving_ids_list = [cid for cid in chunk_map if cid not in drop_ids]
    log(f"  {len(surviving_ids_list)} surviving chunks")

    id_to_pos = {str(s): pos for pos, s in enumerate(ids)}
    surviving_positions = np.array(
        [id_to_pos[cid] for cid in surviving_ids_list], dtype=np.int64
    )
    new_embeddings = embeddings[surviving_positions]
    new_ids = ids[surviving_positions]
    log(f"  new_embeddings shape={new_embeddings.shape}")

    log("Saving new embeddings + ids ...")
    np.save(EMB_PATH, new_embeddings)
    np.save(IDS_PATH, new_ids)

    log("Saving new chunk_map ...")
    new_chunk_map = {cid: chunk_map[cid] for cid in surviving_ids_list}
    CHUNK_MAP_PATH.write_text(
        json.dumps(new_chunk_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("Rebuilding FAISS index ...")
    from src.embedding.index.faiss_index import FaissIndex
    faiss = FaissIndex(dimension=int(new_embeddings.shape[1]))
    faiss.build_index(new_embeddings, list(surviving_ids_list))
    faiss.save(str(FAISS_PATH))

    log("Rebuilding BM25 index ...")
    from src.embedding.index.bm25_index import BM25Index
    bm25 = BM25Index(k1=1.2, b=0.75)
    texts = [new_chunk_map[cid]["text"] for cid in surviving_ids_list]
    bm25.build_index(texts, list(surviving_ids_list))
    bm25.save(str(BM25_PATH))

    log("Deleting dropped chunk JSONs from data/chunks/ ...")
    deleted = 0
    for did in drop_ids:
        src = CHUNKS_DIR / f"{did}.json"
        if src.exists():
            src.unlink()
            deleted += 1
    log(f"  {deleted} files deleted")

    log("Updating corpus_stats.json ...")
    by_prov: Counter = Counter()
    by_svc: dict = defaultdict(Counter)
    for c in new_chunk_map.values():
        p = c.get("cloud_provider", "unknown")
        s = c.get("service_name", "unknown")
        by_prov[p] += 1
        by_svc[p][s] += 1

    proc_dir = ROOT / "data" / "processed"
    proc_by = {
        d.name: sum(1 for _ in d.glob("*.json"))
        for d in proc_dir.iterdir() if d.is_dir()
    }
    stats = {
        "rebuild_date": "2026-05-22",
        "subsample_date": "2026-05-26",
        "rebuild_note": (
            "Corpus rebuilt removing k8s/cncf; Phase 2.5 stratified sub-sample "
            "of Azure Functions (10681->2500) and Blob Storage (3841->1500) "
            "for vendor balance. Sampling by section_hierarchy[0] with seed=42."
        ),
        "providers": {
            p: {
                "processed_documents": proc_by.get(p, 0),
                "chunks": by_prov[p],
                "services_count": len(by_svc[p]),
                "services": dict(by_svc[p].most_common()),
            }
            for p in sorted(by_prov.keys())
        },
        "total_documents": sum(proc_by.values()),
        "total_chunks": len(new_chunk_map),
        "chunking_config": {"strategy": "adaptive", "size": 500, "overlap": 50},
        "embedding": {"model": "BAAI/bge-large-en-v1.5", "dimension": 1024},
        "index_files": {
            "faiss": "data/indices/faiss_bge-large_adaptive_500.index",
            "bm25": "data/indices/bm25_adaptive_500.pkl",
            "chunk_map": "data/indices/chunk_map_bge-large_adaptive_500.json",
        },
        "subsample_diversity": {
            svc: {
                "pre_sections": pre,
                "post_sections": post,
                "post_chunks": n,
                "dropped": dropped,
            }
            for svc, (pre, post, n, dropped) in diversity_stats.items()
        },
    }
    (ROOT / "data" / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("Verification checks ...")
    checks = []
    checks.append((
        "len(chunk_map) == len(embeddings) == len(ids)",
        len(new_chunk_map) == len(new_embeddings) == len(new_ids),
        f"{len(new_chunk_map)} / {len(new_embeddings)} / {len(new_ids)}",
    ))

    try:
        import faiss as faiss_lib
        loaded = faiss_lib.read_index(str(FAISS_PATH))
        checks.append((
            "FAISS total_vectors == len(chunk_map)",
            loaded.ntotal == len(new_chunk_map),
            f"{loaded.ntotal}",
        ))
    except Exception as e:
        checks.append(("FAISS read-back", False, str(e)))

    bm25_loaded = BM25Index.load(str(BM25_PATH))
    checks.append((
        "BM25 docs == len(chunk_map)",
        len(bm25_loaded.chunk_ids) == len(new_chunk_map),
        f"{len(bm25_loaded.chunk_ids)}",
    ))

    svc_counts = Counter()
    for c in new_chunk_map.values():
        svc_counts[c["service_name"]] += 1
    af = svc_counts.get("Azure Functions", 0)
    bs = svc_counts.get("Blob Storage", 0)
    checks.append(("Azure Functions in [2490, 2510]", 2490 <= af <= 2510, str(af)))
    checks.append(("Blob Storage in [1490, 1510]", 1490 <= bs <= 1510, str(bs)))

    aws_count = sum(1 for c in new_chunk_map.values() if c["cloud_provider"] == "aws")
    gcp_count = sum(1 for c in new_chunk_map.values() if c["cloud_provider"] == "gcp")
    checks.append(("AWS chunks unchanged (6366)", aws_count == 6366, str(aws_count)))
    checks.append(("GCP chunks unchanged (8509)", gcp_count == 8509, str(gcp_count)))

    remaining_files = len(list(CHUNKS_DIR.glob("*.json")))
    checks.append((
        "chunk JSONs in data/chunks/ == len(chunk_map)",
        remaining_files == len(new_chunk_map),
        f"{remaining_files} files",
    ))

    new_ids_set = set(str(s) for s in new_ids)
    sample = list(new_chunk_map.keys())[:5]
    sample_ok = all(cid in new_ids_set for cid in sample)
    checks.append((
        "5 random chunk_map IDs present in ids.npy",
        sample_ok,
        f"sample={sample[:3]}...",
    ))

    # Diversity sanity
    for svc, (pre, post, n, _dropped) in diversity_stats.items():
        ok = post >= pre // 2
        checks.append((
            f"{svc}: post_sections >= pre/2",
            ok,
            f"pre={pre} post={post}",
        ))

    all_pass = True
    print()
    for desc, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{marker}] {desc}  ({detail})")

    log(f"DONE. Total time={time.time() - t0:.1f}s  all_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
