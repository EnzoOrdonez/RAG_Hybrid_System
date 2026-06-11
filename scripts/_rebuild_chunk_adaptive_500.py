"""
One-shot rebuild of chunks/adaptive/size_500/ from data/processed/
(post-k8s/cncf removal). Used during the 2026-05-21 corpus rebuild.

Mirrors `resolve_gaps.run_chunking` but only for adaptive/500 (the
config used by exp3, exp5, exp6, exp8, exp8b, exp9). The 14 other
(strategy, size) combinations are deferred.
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.adaptive_chunker import AdaptiveChunker  # noqa: E402
from src.ingestion.ingestion_pipeline import IngestionPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main() -> int:
    pipeline = IngestionPipeline()
    docs = pipeline.load_processed_documents()
    by_provider: Counter = Counter()
    for d in docs:
        by_provider[getattr(d, "cloud_provider", "?")] += 1
    log.info("Loaded %d processed documents", len(docs))
    for prov, n in by_provider.most_common():
        log.info("  %s: %d docs", prov, n)

    output_dir = PROJECT_ROOT / "data" / "chunks" / "adaptive" / "size_500"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Chunk output dir: %s", output_dir)

    chunker = AdaptiveChunker(chunk_size=500, overlap=50)
    start = time.time()
    total_chunks = 0
    by_provider_chunks: Counter = Counter()
    errors = 0
    for doc in docs:
        try:
            chunks = chunker.chunk_document(doc)
            for chunk in chunks:
                chunk_path = output_dir / f"{chunk.chunk_id}.json"
                chunk_path.write_text(
                    chunk.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                total_chunks += 1
                prov = getattr(chunk, "cloud_provider", "?")
                by_provider_chunks[prov] += 1
        except Exception as e:
            errors += 1
            log.debug("Chunking error for %s: %s", doc.doc_id, e)

    elapsed = time.time() - start
    log.info("Chunking complete: %d chunks in %.1fs (%d errors)", total_chunks, elapsed, errors)
    for prov, n in by_provider_chunks.most_common():
        log.info("  %s: %d chunks", prov, n)
    log.info("k8s present: %s | cncf present: %s",
             by_provider_chunks.get("kubernetes", 0) or by_provider_chunks.get("k8s", 0),
             by_provider_chunks.get("cncf", 0))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
