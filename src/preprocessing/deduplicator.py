"""
Deduplicator - 3-level deduplication for cloud documentation.
1. Intra-document: duplicate paragraphs within same doc
2. Intra-provider: near-duplicate docs within same provider (MinHash)
3. Cross-provider: conceptually similar docs across providers (mark only)
"""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.ingestion.doc_parser import Document

logger = logging.getLogger(__name__)


class Deduplicator:
    """Multi-level deduplication for cloud documentation corpus."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        dedup_cfg = self.config.get("deduplication", {})

        self.intra_doc_enabled = dedup_cfg.get("intra_document", {}).get("enabled", True)
        self.intra_prov_enabled = dedup_cfg.get("intra_provider", {}).get("enabled", True)
        self.cross_prov_enabled = dedup_cfg.get("cross_provider", {}).get("enabled", True)

        self.intra_prov_threshold = dedup_cfg.get("intra_provider", {}).get(
            "similarity_threshold", 0.9
        )
        self.cross_prov_threshold = dedup_cfg.get("cross_provider", {}).get(
            "similarity_threshold", 0.85
        )
        self.num_perm = dedup_cfg.get("intra_provider", {}).get("num_perm", 128)

    def deduplicate(
        self, documents: List[Document], report: bool = False
    ) -> Tuple[List[Document], Dict]:
        """Run all levels of deduplication.

        Returns:
            Tuple of (deduplicated documents, report dict)
        """
        stats = {
            "input_documents": len(documents),
            "intra_document": {"paragraphs_removed": 0},
            "intra_provider": {"duplicates_found": 0, "documents_removed": 0},
            "cross_provider": {"equivalents_marked": 0},
        }

        # Level 1: Intra-document paragraph dedup
        if self.intra_doc_enabled:
            para_removed = 0
            for doc in documents:
                removed = self._dedup_intra_document(doc)
                para_removed += removed
            stats["intra_document"]["paragraphs_removed"] = para_removed
            logger.info("Intra-document: removed %d duplicate paragraphs", para_removed)

        # Level 2: Intra-provider near-duplicate removal
        if self.intra_prov_enabled:
            documents, intra_stats = self._dedup_intra_provider(documents)
            stats["intra_provider"] = intra_stats
            logger.info(
                "Intra-provider: found %d duplicates, removed %d docs",
                intra_stats["duplicates_found"],
                intra_stats["documents_removed"],
            )

        # Level 3: Cross-provider marking (does NOT remove, only marks)
        if self.cross_prov_enabled:
            cross_stats = self._mark_cross_provider(documents)
            stats["cross_provider"] = cross_stats
            logger.info(
                "Cross-provider: marked %d equivalent groups",
                cross_stats["equivalents_marked"],
            )

        stats["output_documents"] = len(documents)

        if report:
            self._print_report(stats)

        return documents, stats

    def _dedup_intra_document(self, doc: Document) -> int:
        """Remove duplicate paragraphs within a single document."""
        paragraphs = doc.content.split("\n\n")
        seen: Set[str] = set()
        unique = []
        removed = 0

        for para in paragraphs:
            stripped = para.strip()
            if not stripped or len(stripped) < 30:
                unique.append(para)
                continue
            h = hashlib.md5(stripped.encode()).hexdigest()
            if h in seen:
                removed += 1
                continue
            seen.add(h)
            unique.append(para)

        doc.content = "\n\n".join(unique)
        doc.word_count = len(doc.content.split())
        doc.char_count = len(doc.content)
        return removed

    def _dedup_intra_provider(
        self, documents: List[Document]
    ) -> Tuple[List[Document], Dict]:
        """Remove near-duplicate documents within each provider using MinHash."""
        stats = {"duplicates_found": 0, "documents_removed": 0, "details": {}}

        # Group by provider
        by_provider: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            by_provider[doc.cloud_provider].append(doc)

        keep_ids: Set[str] = set()
        remove_ids: Set[str] = set()

        for provider, provider_docs in by_provider.items():
            if len(provider_docs) < 2:
                for d in provider_docs:
                    keep_ids.add(d.doc_id)
                continue

            # Build MinHash signatures
            try:
                from datasketch import MinHash, MinHashLSH

                lsh = MinHashLSH(threshold=self.intra_prov_threshold, num_perm=self.num_perm)
                minhashes = {}

                for doc in provider_docs:
                    mh = MinHash(num_perm=self.num_perm)
                    # Shingle the content (3-word shingles)
                    words = doc.content.lower().split()
                    for i in range(len(words) - 2):
                        shingle = " ".join(words[i:i+3])
                        mh.update(shingle.encode("utf-8"))
                    minhashes[doc.doc_id] = mh
                    try:
                        lsh.insert(doc.doc_id, mh)
                    except ValueError:
                        pass  # Duplicate key

                # Find duplicates
                provider_dupes = 0
                processed = set()
                for doc in provider_docs:
                    if doc.doc_id in processed or doc.doc_id in remove_ids:
                        continue
                    results = lsh.query(minhashes[doc.doc_id])
                    if len(results) > 1:
                        # Keep the longest document
                        group = [d for d in provider_docs if d.doc_id in results]
                        group.sort(key=lambda d: d.word_count, reverse=True)
                        keep_ids.add(group[0].doc_id)
                        for d in group[1:]:
                            remove_ids.add(d.doc_id)
                            provider_dupes += 1
                        for d in group:
                            processed.add(d.doc_id)
                    else:
                        keep_ids.add(doc.doc_id)
                        processed.add(doc.doc_id)

                stats["details"][provider] = provider_dupes
                stats["duplicates_found"] += provider_dupes

            except ImportError:
                logger.warning(
                    "datasketch not installed, skipping MinHash dedup for %s",
                    provider,
                )
                for d in provider_docs:
                    keep_ids.add(d.doc_id)

        # Filter documents
        result = [d for d in documents if d.doc_id not in remove_ids]
        stats["documents_removed"] = len(documents) - len(result)
        return result, stats

    def _mark_cross_provider(self, documents: List[Document]) -> Dict:
        """Mark conceptually equivalent documents across providers.
        Does NOT remove - only adds metadata.
        """
        stats = {"equivalents_marked": 0, "groups": []}

        # Group by service_name-like concept using simple title similarity
        by_title: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            # Normalize title for grouping
            key = self._normalize_title(doc.title, doc.cloud_provider)
            if key:
                by_title[key].append(doc)

        for key, group in by_title.items():
            providers = set(d.cloud_provider for d in group)
            if len(providers) >= 2:
                stats["equivalents_marked"] += 1
                doc_ids = [d.doc_id for d in group]
                stats["groups"].append({
                    "topic": key,
                    "providers": sorted(providers),
                    "doc_ids": doc_ids,
                })

        return stats

    def _normalize_title(self, title: str, provider: str) -> str:
        """Normalize a title for cross-provider comparison."""
        if not title:
            return ""
        t = title.lower().strip()
        # Remove provider-specific prefixes
        for prefix in [
            "amazon ", "aws ", "azure ", "google cloud ", "cloud ",
            "microsoft ", "gcp ",
        ]:
            if t.startswith(prefix):
                t = t[len(prefix):]
        # Remove common suffixes
        for suffix in [
            " documentation", " user guide", " developer guide",
            " overview", " introduction", " getting started",
        ]:
            if t.endswith(suffix):
                t = t[:-len(suffix)]
        return t.strip()

    def _print_report(self, stats: Dict):
        """Print deduplication report."""
        print("\n=== Deduplication Report ===")
        print(f"Input documents:  {stats['input_documents']}")
        print(f"Output documents: {stats['output_documents']}")
        print(f"\nIntra-document:")
        print(f"  Paragraphs removed: {stats['intra_document']['paragraphs_removed']}")
        print(f"\nIntra-provider:")
        print(f"  Duplicates found: {stats['intra_provider']['duplicates_found']}")
        print(f"  Documents removed: {stats['intra_provider']['documents_removed']}")
        if "details" in stats["intra_provider"]:
            for prov, count in stats["intra_provider"]["details"].items():
                print(f"    {prov}: {count} duplicates")
        print(f"\nCross-provider:")
        print(f"  Equivalent groups marked: {stats['cross_provider']['equivalents_marked']}")

    def process_directory(self, input_dir: Path, report: bool = False) -> Dict:
        """Process all documents in a directory."""
        input_dir = Path(input_dir)
        documents = []
        json_files = list(input_dir.rglob("*.json"))

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                documents.append(Document(**data))
            except Exception as e:
                logger.warning("Failed to load %s: %s", jf, e)

        if not documents:
            logger.warning("No documents found in %s", input_dir)
            return {}

        deduped, stats = self.deduplicate(documents, report=report)

        # Save deduped documents back
        # Remove originals that were deduped
        kept_ids = {d.doc_id for d in deduped}
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("doc_id") not in kept_ids:
                    jf.unlink()
            except Exception:
                pass

        # Save updated documents
        for doc in deduped:
            # Find the right provider subdirectory
            prov_dir = input_dir / doc.cloud_provider
            if not prov_dir.exists():
                prov_dir = input_dir
            out_path = prov_dir / f"{doc.doc_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

        # Save stats
        stats_path = input_dir.parent / "dedup_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

        return stats


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Deduplicate documents")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--report", action="store_true", help="Print report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    deduplicator = Deduplicator(config)
    deduplicator.process_directory(Path(args.input), report=args.report)


if __name__ == "__main__":
    main()
