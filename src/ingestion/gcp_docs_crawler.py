"""
GCP Documentation Crawler.
Strategy: Web scraping (GCP has no well-organized public GitHub repo).
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.doc_parser import DocParser, Document

logger = logging.getLogger(__name__)


class GCPDocsCrawler:
    """Crawls GCP documentation via web scraping."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir) / "gcp"
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        return s

    def crawl(
        self, services_config: dict, max_services: Optional[int] = None
    ) -> List[Document]:
        """Crawl GCP docs for all configured services."""
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
                f"[green]Crawling GCP docs ({len(all_services)} services)...",
                total=len(all_services),
            )
            for category, svc in all_services:
                svc_name = svc["name"]
                progress.update(task, description=f"[green]GCP: {svc_name}")
                try:
                    docs = self._crawl_service(category, svc)
                    documents.extend(docs)
                    logger.info(
                        "GCP/%s: crawled %d documents", svc_name, len(docs)
                    )
                except Exception as e:
                    logger.error("GCP/%s: crawl failed: %s", svc_name, e)
                progress.advance(task)

        return documents

    def _crawl_service(
        self, category: str, svc: dict, max_depth: int = 3
    ) -> List[Document]:
        """Crawl a single GCP service by following internal links."""
        svc_name = svc["name"]
        base_url = svc["docs_url"]
        svc_dir = self.output_dir / svc_name.lower().replace(" ", "_")
        svc_dir.mkdir(parents=True, exist_ok=True)

        visited: Set[str] = set()
        documents = []
        queue = [(base_url, 0)]
        parsed_base = urlparse(base_url)

        while queue and len(documents) < 80:
            url, depth = queue.pop(0)
            normalized_url = url.split("#")[0].split("?")[0].rstrip("/")

            if normalized_url in visited or depth > max_depth:
                continue
            visited.add(normalized_url)

            time.sleep(self.rate_limit)
            try:
                resp = self._request_with_retry(url)
                if resp.status_code != 200:
                    continue
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                continue

            # Parse document
            doc = self.parser.parse_html(
                resp.text,
                cloud_provider="gcp",
                service_name=svc_name,
                service_category=category,
                url_source=url,
            )
            if doc.word_count >= 20:
                doc_path = svc_dir / f"{doc.doc_id}.json"
                doc_path.write_text(
                    doc.model_dump_json(indent=2), encoding="utf-8"
                )
                documents.append(doc)

            # Follow internal links
            if depth < max_depth:
                soup = BeautifulSoup(resp.text, "lxml")
                for link in soup.find_all("a", href=True):
                    href = urljoin(url, link["href"])
                    href_clean = href.split("#")[0].split("?")[0].rstrip("/")
                    parsed = urlparse(href_clean)

                    # Stay within the same service docs path
                    if (
                        parsed.netloc == parsed_base.netloc
                        and parsed.path.startswith(parsed_base.path)
                        and href_clean not in visited
                        and not any(
                            href_clean.endswith(ext)
                            for ext in (".png", ".jpg", ".gif", ".svg", ".pdf", ".zip")
                        )
                    ):
                        queue.append((href_clean, depth + 1))

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
        raise requests.RequestException(
            f"Failed after {self.max_retries} retries: {url}"
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
            "provider": "gcp",
            "total_documents": total_docs,
            "total_size_bytes": total_size,
            "services_crawled": sorted(services),
        }
