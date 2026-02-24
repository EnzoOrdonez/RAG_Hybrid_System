"""
Kubernetes Documentation Crawler.
Strategy: Clone kubernetes/website repo (Markdown with Hugo).
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.doc_parser import DocParser, Document

logger = logging.getLogger(__name__)


class K8sDocsCrawler:
    """Crawls Kubernetes documentation from the kubernetes/website GitHub repo."""

    GITHUB_REPO = "https://github.com/kubernetes/website"

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir) / "k8s"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = DocParser()

    def crawl(
        self, services_config: dict, max_services: Optional[int] = None
    ) -> List[Document]:
        """Crawl Kubernetes docs for configured sections."""
        sections = services_config.get("sections", [])
        if max_services:
            sections = sections[:max_services]

        # Clone the repo
        clone_dir = self.output_dir / "_repo"
        if not self._clone_repo(clone_dir):
            logger.error("Failed to clone kubernetes/website repo")
            return []

        documents = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"[yellow]Parsing K8s docs ({len(sections)} sections)...",
                total=len(sections),
            )
            for section in sections:
                section_name = section["name"]
                progress.update(
                    task, description=f"[yellow]K8s: {section_name}"
                )
                try:
                    docs = self._parse_section(clone_dir, section)
                    documents.extend(docs)
                    logger.info(
                        "K8s/%s: parsed %d documents", section_name, len(docs)
                    )
                except Exception as e:
                    logger.error("K8s/%s: parse failed: %s", section_name, e)
                progress.advance(task)

        # Cleanup
        shutil.rmtree(clone_dir, ignore_errors=True)
        return documents

    def _clone_repo(self, clone_dir: Path) -> bool:
        """Clone kubernetes/website with depth 1."""
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        try:
            result = subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    self.GITHUB_REPO, str(clone_dir),
                ],
                capture_output=True, text=True, timeout=180,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error("git clone failed: %s", e)
            return False

    def _parse_section(
        self, clone_dir: Path, section: dict
    ) -> List[Document]:
        """Parse all .md files in a K8s docs section."""
        section_name = section["name"]
        github_path = section["github_path"]
        docs_url = section.get("docs_url", "")

        section_dir = clone_dir / github_path
        if not section_dir.exists():
            logger.warning("Section path not found: %s", section_dir)
            return []

        out_dir = self.output_dir / section_name.lower().replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        documents = []
        md_files = list(section_dir.rglob("*.md"))

        for md_file in md_files:
            # Skip _index.md aggregation files if they're empty
            rel = md_file.relative_to(section_dir)
            if any(p.startswith(".") for p in rel.parts):
                continue
            try:
                doc = self.parser.parse_file(
                    file_path=md_file,
                    cloud_provider="k8s",
                    service_name=section_name,
                    service_category="kubernetes",
                    url_source=docs_url,
                )
                if doc.word_count >= 20:
                    doc_path = out_dir / f"{doc.doc_id}.json"
                    doc_path.write_text(
                        doc.model_dump_json(indent=2), encoding="utf-8"
                    )
                    documents.append(doc)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", md_file, e)

        return documents

    def get_stats(self) -> Dict:
        """Return statistics about crawled documents."""
        json_files = list(self.output_dir.rglob("*.json"))
        total_docs = len(json_files)
        total_size = sum(f.stat().st_size for f in json_files)
        sections = set()
        for f in json_files:
            if f.parent != self.output_dir:
                sections.add(f.parent.name)
        return {
            "provider": "k8s",
            "total_documents": total_docs,
            "total_size_bytes": total_size,
            "sections_crawled": sorted(sections),
        }
