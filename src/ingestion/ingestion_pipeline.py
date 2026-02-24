"""
Ingestion Pipeline - Orchestrates crawling, parsing, and initial processing.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.ingestion.doc_parser import Document

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion process."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(__file__).parent.parent.parent
        with open(self.project_root / config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(
            self.project_root / "config/cloud_services.yaml", encoding="utf-8"
        ) as f:
            self.cloud_config = yaml.safe_load(f)

        self.raw_dir = self.project_root / self.config["paths"]["raw_data"]
        self.processed_dir = self.project_root / self.config["paths"]["processed_data"]
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        providers: Optional[List[str]] = None,
        max_services: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Run the full ingestion pipeline.

        Args:
            providers: List of providers to crawl. None = all.
            max_services: Max services per provider (for testing).
            dry_run: If True, only show what would be crawled.

        Returns:
            Statistics dict.
        """
        all_providers = ["aws", "azure", "gcp", "kubernetes", "cncf"]
        if providers:
            active = [p for p in providers if p in all_providers]
        else:
            active = all_providers

        if dry_run:
            return self._dry_run(active, max_services)

        stats = {"providers": {}, "total_documents": 0}

        for provider in active:
            logger.info("Starting ingestion for %s...", provider)
            docs = self._crawl_provider(provider, max_services)
            self._save_processed(docs, provider)
            stats["providers"][provider] = {
                "documents": len(docs),
                "total_words": sum(d.word_count for d in docs),
                "with_code": sum(1 for d in docs if d.has_code),
                "with_tables": sum(1 for d in docs if d.has_tables),
            }
            stats["total_documents"] += len(docs)
            logger.info(
                "%s: ingested %d documents (%d words)",
                provider,
                len(docs),
                sum(d.word_count for d in docs),
            )

        # Save stats
        stats_path = self.project_root / "data" / "corpus_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Corpus stats saved to %s", stats_path)

        return stats

    def _crawl_provider(
        self, provider: str, max_services: Optional[int]
    ) -> List[Document]:
        """Crawl a single provider."""
        provider_config = self.cloud_config.get(provider, {})
        if not provider_config:
            logger.warning("No config found for provider: %s", provider)
            return []

        github_token = ""
        try:
            from dotenv import dotenv_values
            env = dotenv_values(self.project_root / ".env")
            github_token = env.get("GITHUB_TOKEN", "")
        except ImportError:
            pass

        if provider == "aws":
            from src.ingestion.aws_docs_crawler import AWSDocsCrawler
            crawler = AWSDocsCrawler(
                self.config, self.raw_dir, github_token=github_token
            )
            return crawler.crawl(provider_config, max_services=max_services)
        elif provider == "azure":
            from src.ingestion.azure_docs_crawler import AzureDocsCrawler
            crawler = AzureDocsCrawler(
                self.config, self.raw_dir, github_token=github_token
            )
            return crawler.crawl(provider_config, max_services=max_services)
        elif provider == "gcp":
            from src.ingestion.gcp_docs_crawler import GCPDocsCrawler
            crawler = GCPDocsCrawler(self.config, self.raw_dir)
            return crawler.crawl(provider_config, max_services=max_services)
        elif provider == "kubernetes":
            from src.ingestion.k8s_docs_crawler import K8sDocsCrawler
            crawler = K8sDocsCrawler(self.config, self.raw_dir)
            return crawler.crawl(provider_config, max_services=max_services)
        elif provider == "cncf":
            from src.ingestion.cncf_glossary_crawler import CNCFGlossaryCrawler
            crawler = CNCFGlossaryCrawler(self.config, self.raw_dir)
            return crawler.crawl(provider_config, max_services=max_services)
        else:
            logger.warning("Unknown provider: %s", provider)
            return []

    def _save_processed(self, documents: List[Document], provider: str):
        """Save processed documents as JSON."""
        out_dir = self.processed_dir / provider
        out_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            path = out_dir / f"{doc.doc_id}.json"
            path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    def _dry_run(self, providers: List[str], max_services: Optional[int]) -> Dict:
        """Show what would be crawled without actually doing it."""
        result = {"providers": {}}
        for provider in providers:
            provider_config = self.cloud_config.get(provider, {})
            if not provider_config:
                continue

            services = []
            if "services" in provider_config:
                for cat, svc_list in provider_config["services"].items():
                    for svc in svc_list:
                        services.append({
                            "category": cat,
                            "name": svc["name"],
                            "source": svc.get("github_repo", svc.get("github_path", svc.get("docs_url", ""))),
                        })
            elif "sections" in provider_config:
                for sec in provider_config["sections"]:
                    services.append({
                        "category": "section",
                        "name": sec["name"],
                        "source": sec.get("github_path", sec.get("docs_url", "")),
                    })

            if max_services:
                services = services[:max_services]

            result["providers"][provider] = {
                "source_type": provider_config.get("source_type", "unknown"),
                "services_count": len(services),
                "services": services,
            }

        return result

    def load_processed_documents(
        self, provider: Optional[str] = None
    ) -> List[Document]:
        """Load already-processed documents from disk."""
        documents = []
        if provider:
            dirs = [self.processed_dir / provider]
        else:
            dirs = [d for d in self.processed_dir.iterdir() if d.is_dir()]

        for d in dirs:
            if not d.exists():
                continue
            for json_file in d.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    documents.append(Document(**data))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", json_file, e)

        return documents


def main():
    """CLI entry point for ingestion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run document ingestion pipeline")
    parser.add_argument(
        "--input", type=str, default="data/raw", help="Input directory"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--provider", type=str, nargs="*",
        help="Providers to process (aws, azure, gcp, kubernetes, cncf)",
    )
    parser.add_argument(
        "--services", type=int, default=None,
        help="Max services per provider",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be crawled",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = IngestionPipeline()
    stats = pipeline.run(
        providers=args.provider,
        max_services=args.services,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n=== DRY RUN - What would be crawled ===")
        for prov, info in stats["providers"].items():
            print(f"\n{prov.upper()} ({info['source_type']}):")
            print(f"  Services: {info['services_count']}")
            for svc in info["services"]:
                print(f"    - {svc['name']} [{svc['category']}]: {svc['source']}")
    else:
        print(f"\n=== Ingestion Complete ===")
        print(f"Total documents: {stats['total_documents']}")
        for prov, info in stats["providers"].items():
            print(
                f"  {prov}: {info['documents']} docs, "
                f"{info['total_words']} words, "
                f"{info['with_code']} with code"
            )


if __name__ == "__main__":
    main()
