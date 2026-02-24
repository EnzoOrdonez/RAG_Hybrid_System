"""
AWS Documentation Crawler.
Strategy: Clone awsdocs GitHub repos (preferred - clean Markdown).
Fallback: Scrape docs.aws.amazon.com with BeautifulSoup.
"""

import logging
import os
import shutil
import subprocess
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.doc_parser import DocParser, Document

logger = logging.getLogger(__name__)


class AWSDocsCrawler:
    """Crawls AWS documentation from GitHub repos or web."""

    GITHUB_ORG = "https://github.com/awsdocs"

    def __init__(self, config: dict, output_dir: Path, github_token: str = ""):
        self.config = config
        self.output_dir = Path(output_dir) / "aws"
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
            "Accept-Language": "en-US,en;q=0.9",
        })
        if self.github_token:
            s.headers["Authorization"] = f"token {self.github_token}"
        return s

    def crawl(
        self, services_config: dict, max_services: Optional[int] = None
    ) -> List[Document]:
        """Crawl AWS docs for all configured services."""
        documents = []
        all_services = []
        for category, svc_list in services_config.get("services", {}).items():
            for svc in svc_list:
                all_services.append((category, svc))

        if max_services:
            all_services = all_services[:max_services]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Crawling AWS docs ({len(all_services)} services)...",
                total=len(all_services),
            )
            for category, svc in all_services:
                svc_name = svc["name"]
                progress.update(task, description=f"[cyan]AWS: {svc_name}")
                try:
                    docs = self._crawl_service(category, svc)
                    documents.extend(docs)
                    logger.info(
                        "AWS/%s: crawled %d documents", svc_name, len(docs)
                    )
                except Exception as e:
                    logger.error("AWS/%s: crawl failed: %s", svc_name, e)
                progress.advance(task)

        return documents

    def _crawl_service(self, category: str, svc: dict) -> List[Document]:
        """Crawl a single AWS service."""
        svc_name = svc["name"]
        github_repo = svc.get("github_repo", "")
        docs_url = svc.get("docs_url", "")
        svc_dir = self.output_dir / svc_name.lower().replace(" ", "_")
        svc_dir.mkdir(parents=True, exist_ok=True)

        if github_repo:
            docs = self._crawl_github_repo(
                repo_name=github_repo,
                svc_name=svc_name,
                category=category,
                docs_url=docs_url,
                output_dir=svc_dir,
            )
            if docs:
                return docs
            logger.warning(
                "AWS/%s: GitHub clone failed, falling back to web", svc_name
            )

        if docs_url:
            return self._crawl_web(
                base_url=docs_url,
                svc_name=svc_name,
                category=category,
                output_dir=svc_dir,
            )
        return []

    def _crawl_github_repo(
        self,
        repo_name: str,
        svc_name: str,
        category: str,
        docs_url: str,
        output_dir: Path,
    ) -> List[Document]:
        """Clone a GitHub repo and parse all .md files."""
        repo_url = f"{self.GITHUB_ORG}/{repo_name}"
        clone_dir = output_dir / "_repo"

        try:
            # Try git clone --depth 1
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning("git clone failed for %s: %s", repo_url, result.stderr)
                return self._download_zip(repo_url, svc_name, category, docs_url, output_dir)
        except FileNotFoundError:
            logger.warning("git not found, trying ZIP download")
            return self._download_zip(repo_url, svc_name, category, docs_url, output_dir)
        except subprocess.TimeoutExpired:
            logger.warning("git clone timed out for %s", repo_url)
            return self._download_zip(repo_url, svc_name, category, docs_url, output_dir)

        # Parse .md files
        documents = []
        md_files = list(clone_dir.rglob("*.md"))
        for md_file in md_files:
            # Skip non-doc files
            rel = md_file.relative_to(clone_dir)
            if any(
                part.startswith(".") or part in ("node_modules", "test", "tests")
                for part in rel.parts
            ):
                continue
            try:
                doc = self.parser.parse_file(
                    file_path=md_file,
                    cloud_provider="aws",
                    service_name=svc_name,
                    service_category=category,
                    url_source=docs_url,
                )
                if doc.word_count >= 20:
                    # Save parsed document
                    doc_path = output_dir / f"{doc.doc_id}.json"
                    doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                    documents.append(doc)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", md_file, e)

        # Clean up cloned repo to save space
        shutil.rmtree(clone_dir, ignore_errors=True)
        return documents

    def _download_zip(
        self,
        repo_url: str,
        svc_name: str,
        category: str,
        docs_url: str,
        output_dir: Path,
    ) -> List[Document]:
        """Download repo as ZIP and extract .md files."""
        zip_url = f"{repo_url}/archive/refs/heads/main.zip"
        try:
            resp = self._request_with_retry(zip_url)
            if resp.status_code != 200:
                # Try master branch
                zip_url = f"{repo_url}/archive/refs/heads/master.zip"
                resp = self._request_with_retry(zip_url)
                if resp.status_code != 200:
                    return []
        except Exception as e:
            logger.error("ZIP download failed for %s: %s", zip_url, e)
            return []

        extract_dir = output_dir / "_zip"
        try:
            with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            logger.error("Bad ZIP for %s", zip_url)
            return []

        documents = []
        for md_file in extract_dir.rglob("*.md"):
            try:
                doc = self.parser.parse_file(
                    file_path=md_file,
                    cloud_provider="aws",
                    service_name=svc_name,
                    service_category=category,
                    url_source=docs_url,
                )
                if doc.word_count >= 20:
                    doc_path = output_dir / f"{doc.doc_id}.json"
                    doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                    documents.append(doc)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", md_file, e)

        shutil.rmtree(extract_dir, ignore_errors=True)
        return documents

    def _crawl_web(
        self,
        base_url: str,
        svc_name: str,
        category: str,
        output_dir: Path,
        max_depth: int = 3,
    ) -> List[Document]:
        """Fallback: scrape docs.aws.amazon.com."""
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        visited = set()
        documents = []
        queue = [(base_url, 0)]

        while queue and len(documents) < 100:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            try:
                time.sleep(self.rate_limit)
                resp = self._request_with_retry(url)
                if resp.status_code != 200:
                    continue
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Parse document
            doc = self.parser.parse_html(
                resp.text,
                cloud_provider="aws",
                service_name=svc_name,
                service_category=category,
                url_source=url,
            )
            if doc.word_count >= 20:
                doc_path = output_dir / f"{doc.doc_id}.json"
                doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                documents.append(doc)

            # Find internal links
            if depth < max_depth:
                parsed_base = urlparse(base_url)
                for link in soup.find_all("a", href=True):
                    href = urljoin(url, link["href"])
                    parsed = urlparse(href)
                    # Stay within same service docs
                    if (
                        parsed.netloc == parsed_base.netloc
                        and parsed.path.startswith(parsed_base.path)
                        and href not in visited
                        and not href.endswith((".png", ".jpg", ".gif", ".svg", ".pdf"))
                    ):
                        queue.append((href.split("#")[0], depth + 1))

        return documents

    def _request_with_retry(self, url: str) -> requests.Response:
        """Make HTTP request with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = self.backoff ** (attempt + 1)
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = self.backoff ** (attempt + 1)
                    logger.warning("Request failed, retry in %ds: %s", wait, e)
                    time.sleep(wait)
                else:
                    raise
        raise requests.RequestException(f"Failed after {self.max_retries} retries: {url}")

    def get_stats(self) -> Dict:
        """Return statistics about crawled documents."""
        json_files = list(self.output_dir.rglob("*.json"))
        total_docs = len(json_files)
        total_size = sum(f.stat().st_size for f in json_files)
        services = set()
        for f in json_files:
            services.add(f.parent.name)
        return {
            "provider": "aws",
            "total_documents": total_docs,
            "total_size_bytes": total_size,
            "services_crawled": sorted(services),
        }
