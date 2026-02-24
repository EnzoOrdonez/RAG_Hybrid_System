"""
Azure Documentation Crawler.
Strategy: Clone MicrosoftDocs/azure-docs with sparse checkout.
Alternative: Download via GitHub API for specific directories.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.doc_parser import DocParser, Document

logger = logging.getLogger(__name__)


class AzureDocsCrawler:
    """Crawls Azure documentation from the MicrosoftDocs/azure-docs repo."""

    GITHUB_REPO = "https://github.com/MicrosoftDocs/azure-docs"

    def __init__(self, config: dict, output_dir: Path, github_token: str = ""):
        self.config = config
        self.output_dir = Path(output_dir) / "azure"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.parser = DocParser()
        self.rate_limit = config.get("ingestion", {}).get("rate_limit_seconds", 2.0)
        self.max_retries = config.get("ingestion", {}).get("max_retries", 3)
        self.backoff = config.get("ingestion", {}).get("retry_backoff_factor", 2.0)
        self.timeout = config.get("ingestion", {}).get("request_timeout", 30)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; academic-research-bot)",
            "Accept": "application/vnd.github.v3+json",
        })
        if self.github_token:
            s.headers["Authorization"] = f"token {self.github_token}"
        return s

    def crawl(
        self, services_config: dict, max_services: Optional[int] = None
    ) -> List[Document]:
        """Crawl Azure docs for configured services."""
        documents = []
        all_services = []
        for category, svc_list in services_config.get("services", {}).items():
            for svc in svc_list:
                all_services.append((category, svc))

        if max_services:
            all_services = all_services[:max_services]

        # Collect github_paths for sparse checkout
        github_paths = []
        for _, svc in all_services:
            gp = svc.get("github_path", "")
            if gp:
                github_paths.append(gp)

        # Try sparse clone first
        clone_dir = self.output_dir / "_repo"
        cloned = self._sparse_clone(github_paths, clone_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"[blue]Crawling Azure docs ({len(all_services)} services)...",
                total=len(all_services),
            )
            for category, svc in all_services:
                svc_name = svc["name"]
                progress.update(task, description=f"[blue]Azure: {svc_name}")
                try:
                    if cloned:
                        docs = self._parse_from_clone(
                            clone_dir, category, svc
                        )
                    else:
                        docs = self._crawl_via_api(category, svc)
                    documents.extend(docs)
                    logger.info(
                        "Azure/%s: crawled %d documents", svc_name, len(docs)
                    )
                except Exception as e:
                    logger.error("Azure/%s: crawl failed: %s", svc_name, e)
                progress.advance(task)

        # Cleanup
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)

        return documents

    def _sparse_clone(self, paths: List[str], clone_dir: Path) -> bool:
        """Perform sparse checkout of specific directories."""
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)

        try:
            # Initialize with sparse checkout
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "--filter=blob:none", "--sparse",
                    self.GITHUB_REPO, str(clone_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if not clone_dir.exists():
                return False

            # Set sparse checkout paths
            if paths:
                subprocess.run(
                    ["git", "sparse-checkout", "set"] + paths,
                    cwd=str(clone_dir),
                    capture_output=True, text=True, timeout=60,
                )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Sparse clone failed: %s", e)
            return False

    def _parse_from_clone(
        self, clone_dir: Path, category: str, svc: dict
    ) -> List[Document]:
        """Parse .md files from cloned repo directory."""
        svc_name = svc["name"]
        github_path = svc.get("github_path", "")
        docs_url = svc.get("docs_url", "")
        svc_dir = self.output_dir / svc_name.lower().replace(" ", "_")
        svc_dir.mkdir(parents=True, exist_ok=True)

        search_dir = clone_dir / github_path if github_path else clone_dir
        if not search_dir.exists():
            logger.warning("Path not found in clone: %s", search_dir)
            return self._crawl_via_api(category, svc)

        documents = []
        md_files = list(search_dir.rglob("*.md"))
        for md_file in md_files:
            rel = md_file.relative_to(clone_dir)
            if any(p.startswith(".") for p in rel.parts):
                continue
            try:
                doc = self.parser.parse_file(
                    file_path=md_file,
                    cloud_provider="azure",
                    service_name=svc_name,
                    service_category=category,
                    url_source=docs_url,
                )
                if doc.word_count >= 20:
                    doc_path = svc_dir / f"{doc.doc_id}.json"
                    doc_path.write_text(
                        doc.model_dump_json(indent=2), encoding="utf-8"
                    )
                    documents.append(doc)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", md_file, e)

        return documents

    def _crawl_via_api(self, category: str, svc: dict) -> List[Document]:
        """Download files via GitHub API (rate-limited fallback)."""
        svc_name = svc["name"]
        github_path = svc.get("github_path", "")
        docs_url = svc.get("docs_url", "")
        svc_dir = self.output_dir / svc_name.lower().replace(" ", "_")
        svc_dir.mkdir(parents=True, exist_ok=True)

        if not github_path:
            return []

        api_url = (
            f"https://api.github.com/repos/MicrosoftDocs/azure-docs"
            f"/contents/{github_path}"
        )

        documents = []
        self._crawl_api_dir(
            api_url, svc_name, category, docs_url, svc_dir, documents, depth=0
        )
        return documents

    def _crawl_api_dir(
        self,
        api_url: str,
        svc_name: str,
        category: str,
        docs_url: str,
        svc_dir: Path,
        documents: list,
        depth: int,
    ):
        """Recursively crawl GitHub API directory listing."""
        if depth > 3 or len(documents) >= 100:
            return

        time.sleep(self.rate_limit)
        try:
            resp = self.session.get(api_url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning("API returned %d for %s", resp.status_code, api_url)
                return
            items = resp.json()
        except Exception as e:
            logger.error("API request failed: %s", e)
            return

        if not isinstance(items, list):
            return

        for item in items:
            if item["type"] == "file" and item["name"].endswith(".md"):
                try:
                    time.sleep(0.5)
                    file_resp = self.session.get(
                        item["download_url"], timeout=self.timeout
                    )
                    if file_resp.status_code == 200:
                        doc = self.parser.parse_markdown(
                            file_resp.text,
                            cloud_provider="azure",
                            service_name=svc_name,
                            service_category=category,
                            url_source=docs_url,
                            file_path=item["path"],
                        )
                        if doc.word_count >= 20:
                            doc_path = svc_dir / f"{doc.doc_id}.json"
                            doc_path.write_text(
                                doc.model_dump_json(indent=2), encoding="utf-8"
                            )
                            documents.append(doc)
                except Exception as e:
                    logger.debug("Failed to download %s: %s", item["name"], e)
            elif item["type"] == "dir":
                self._crawl_api_dir(
                    item["url"], svc_name, category, docs_url,
                    svc_dir, documents, depth + 1,
                )

    def get_stats(self) -> Dict:
        """Return statistics about crawled documents."""
        json_files = list(self.output_dir.rglob("*.json"))
        total_docs = len(json_files)
        total_size = sum(f.stat().st_size for f in json_files)
        services = set()
        for f in json_files:
            if f.parent != self.output_dir:
                services.add(f.parent.name)
        return {
            "provider": "azure",
            "total_documents": total_docs,
            "total_size_bytes": total_size,
            "services_crawled": sorted(services),
        }
