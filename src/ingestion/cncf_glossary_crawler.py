"""
CNCF Glossary Crawler.
Strategy: Clone cncf/glossary repo - each term is a short .md file.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.doc_parser import DocParser, Document

logger = logging.getLogger(__name__)


class CNCFGlossaryCrawler:
    """Crawls CNCF Glossary from GitHub - ideal for atomic chunks."""

    GITHUB_REPO = "https://github.com/cncf/glossary"

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir) / "cncf"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = DocParser()

    def crawl(
        self, services_config: dict, max_services: Optional[int] = None
    ) -> List[Document]:
        """Crawl CNCF glossary terms."""
        # Clone the repo
        clone_dir = self.output_dir / "_repo"
        if not self._clone_repo(clone_dir):
            logger.error("Failed to clone cncf/glossary repo")
            return []

        documents = []
        sections = services_config.get("sections", [])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                "[magenta]Parsing CNCF glossary terms...", total=None
            )
            for section in sections:
                github_path = section["github_path"]
                content_dir = clone_dir / github_path
                if not content_dir.exists():
                    logger.warning("CNCF path not found: %s", content_dir)
                    continue

                md_files = list(content_dir.glob("*.md"))
                if max_services:
                    md_files = md_files[:max_services]

                progress.update(
                    task,
                    total=len(md_files),
                    description=f"[magenta]CNCF: {len(md_files)} terms",
                )

                out_dir = self.output_dir / "glossary"
                out_dir.mkdir(parents=True, exist_ok=True)

                for md_file in md_files:
                    if md_file.name.startswith("_"):
                        progress.advance(task)
                        continue
                    try:
                        doc = self.parser.parse_file(
                            file_path=md_file,
                            cloud_provider="cncf",
                            service_name="Glossary",
                            service_category="glossary",
                            url_source=f"https://glossary.cncf.io/{md_file.stem}/",
                        )
                        if doc.word_count >= 5:  # Glossary terms can be short
                            doc.doc_type = "glossary"
                            doc_path = out_dir / f"{doc.doc_id}.json"
                            doc_path.write_text(
                                doc.model_dump_json(indent=2), encoding="utf-8"
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.debug("Failed to parse %s: %s", md_file, e)
                    progress.advance(task)

        # Cleanup
        shutil.rmtree(clone_dir, ignore_errors=True)
        logger.info("CNCF: parsed %d glossary terms", len(documents))
        return documents

    def _clone_repo(self, clone_dir: Path) -> bool:
        """Clone cncf/glossary with depth 1."""
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        try:
            result = subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    self.GITHUB_REPO, str(clone_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error("git clone failed: %s", e)
            return False

    def get_stats(self) -> Dict:
        """Return statistics about crawled documents."""
        json_files = list(self.output_dir.rglob("*.json"))
        total_docs = len(json_files)
        total_size = sum(f.stat().st_size for f in json_files)
        return {
            "provider": "cncf",
            "total_documents": total_docs,
            "total_size_bytes": total_size,
            "type": "glossary",
        }
