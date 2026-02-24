"""
Document parser for cloud documentation.
Handles Markdown and HTML formats, preserving structure.
"""

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from bs4 import BeautifulSoup, NavigableString
from pydantic import BaseModel, Field


# ============================================================
# Data Models
# ============================================================

class CodeBlock(BaseModel):
    """A code block extracted from documentation."""
    language: str = ""
    code: str
    context: str = ""  # Text before the code block


class Table(BaseModel):
    """A table extracted from documentation."""
    headers: List[str] = []
    rows: List[List[str]] = []
    context: str = ""  # Text before the table


class Section(BaseModel):
    """A section with heading hierarchy."""
    level: int  # 1-6 for H1-H6
    title: str
    content: str
    subsections: List["Section"] = []


class Document(BaseModel):
    """A parsed document with full metadata."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""  # Full clean text

    # Preserved structure
    sections: List[Section] = []
    code_blocks: List[CodeBlock] = []
    tables: List[Table] = []

    # Metadata
    cloud_provider: str = ""  # aws, azure, gcp, k8s, cncf
    service_name: str = ""
    service_category: str = ""  # compute, storage, networking, ai_ml, security
    doc_type: str = ""  # guide, api_reference, tutorial, faq, glossary, concept, task
    url_source: str = ""
    file_path: str = ""
    last_updated: Optional[datetime] = None

    # Statistics
    word_count: int = 0
    char_count: int = 0
    has_code: bool = False
    has_tables: bool = False
    heading_count: int = 0


class Chunk(BaseModel):
    """A chunk produced by any chunking strategy."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    text: str = ""
    token_count: int = 0

    # Document metadata
    cloud_provider: str = ""
    service_name: str = ""
    doc_type: str = ""
    url_source: str = ""

    # Chunk metadata
    section_hierarchy: List[str] = []
    heading_path: str = ""
    chunk_index: int = 0
    total_chunks: int = 0

    # Content type
    has_code: bool = False
    has_table: bool = False
    content_type: str = "narrative"  # narrative, code, table, mixed

    # Normalization (filled by terminology_normalizer)
    normalized_terms: List[str] = []
    detected_siglas: List[str] = []
    cross_cloud_equivalences: Dict[str, List[str]] = {}

    # Chunking metadata
    chunking_strategy: str = ""
    chunk_size_config: int = 0

    # Filled later
    embedding: Optional[List[float]] = None
    bm25_tokens: Optional[List[str]] = None


# ============================================================
# Parser Implementation
# ============================================================

class DocParser:
    """Parses Markdown and HTML documents into structured Document objects."""

    # Hugo shortcode patterns (used in Kubernetes docs)
    HUGO_SHORTCODE_RE = re.compile(
        r'\{\{<\s*(\w+)\s*>\}\}(.*?)\{\{<\s*/\1\s*>\}\}',
        re.DOTALL
    )
    HUGO_SHORTCODE_INLINE_RE = re.compile(
        r'\{\{<\s*(\w+)\s*(?:[^>]*)>\}\}'
    )
    # Front matter pattern
    FRONT_MATTER_RE = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL
    )
    # Markdown heading pattern
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    # Markdown code block pattern
    CODE_BLOCK_RE = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )
    # Markdown table pattern
    TABLE_RE = re.compile(
        r'(\|.+\|)\n(\|[-:\s|]+\|)\n((?:\|.+\|\n?)+)',
        re.MULTILINE
    )

    def parse_file(
        self,
        file_path: Path,
        cloud_provider: str,
        service_name: str,
        service_category: str = "",
        url_source: str = "",
    ) -> Document:
        """Parse a file (.md or .html) into a Document."""
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        suffix = file_path.suffix.lower()

        if suffix in (".md", ".markdown"):
            return self.parse_markdown(
                content,
                cloud_provider=cloud_provider,
                service_name=service_name,
                service_category=service_category,
                url_source=url_source,
                file_path=str(file_path),
            )
        elif suffix in (".html", ".htm"):
            return self.parse_html(
                content,
                cloud_provider=cloud_provider,
                service_name=service_name,
                service_category=service_category,
                url_source=url_source,
                file_path=str(file_path),
            )
        else:
            # Treat as plain text
            return self._build_document(
                title=file_path.stem,
                content=content,
                sections=[],
                code_blocks=[],
                tables=[],
                cloud_provider=cloud_provider,
                service_name=service_name,
                service_category=service_category,
                url_source=url_source,
                file_path=str(file_path),
                front_matter={},
            )

    def parse_markdown(
        self,
        raw_text: str,
        cloud_provider: str = "",
        service_name: str = "",
        service_category: str = "",
        url_source: str = "",
        file_path: str = "",
    ) -> Document:
        """Parse Markdown content into a Document."""
        # 1. Extract front matter
        front_matter, body = self._extract_front_matter(raw_text)

        # 2. Handle Hugo shortcodes (K8s docs)
        body = self._convert_hugo_shortcodes(body)

        # 3. Extract code blocks (replace with placeholders)
        code_blocks, body = self._extract_code_blocks_md(body)

        # 4. Extract tables (replace with placeholders)
        tables, body = self._extract_tables_md(body)

        # 5. Parse sections by headings
        sections = self._parse_sections_md(body)

        # 6. Clean the body text
        clean_content = self._clean_markdown_text(body)

        # 7. Get title from front matter or first heading
        title = front_matter.get("title", "")
        if not title:
            match = self.HEADING_RE.search(raw_text)
            if match:
                title = match.group(2).strip()

        # 8. Infer doc_type
        doc_type = self._infer_doc_type(file_path, front_matter, cloud_provider)

        # 9. Extract last_updated
        last_updated = self._extract_date(front_matter)

        return self._build_document(
            title=title,
            content=clean_content,
            sections=sections,
            code_blocks=code_blocks,
            tables=tables,
            cloud_provider=cloud_provider,
            service_name=service_name,
            service_category=service_category,
            doc_type=doc_type,
            url_source=url_source,
            file_path=file_path,
            front_matter=front_matter,
            last_updated=last_updated,
        )

    def parse_html(
        self,
        raw_html: str,
        cloud_provider: str = "",
        service_name: str = "",
        service_category: str = "",
        url_source: str = "",
        file_path: str = "",
    ) -> Document:
        """Parse HTML content into a Document."""
        soup = BeautifulSoup(raw_html, "lxml")

        # Remove unwanted elements
        for tag in soup.find_all(
            ["nav", "footer", "header", "script", "style", "aside"]
        ):
            tag.decompose()

        # Remove common boilerplate selectors
        for selector in [
            ".devsite-nav", ".devsite-article-meta", ".devsite-banner",
            "#feedback", ".breadcrumb", ".sidebar", ".toc",
            "[role='navigation']", ".was-helpful",
        ]:
            for el in soup.select(selector):
                el.decompose()

        # Find main content area
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("body")
            or soup
        )

        # Extract title
        title_tag = soup.find("title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract code blocks
        code_blocks = []
        for pre in main.find_all("pre"):
            code_tag = pre.find("code")
            lang = ""
            code_text = ""
            if code_tag:
                classes = code_tag.get("class", [])
                for cls in classes:
                    if cls.startswith("language-") or cls.startswith("lang-"):
                        lang = cls.split("-", 1)[1]
                        break
                code_text = code_tag.get_text()
            else:
                code_text = pre.get_text()
            # Get context: previous sibling text
            ctx = ""
            prev = pre.find_previous_sibling(["p", "h1", "h2", "h3", "h4"])
            if prev:
                ctx = prev.get_text(strip=True)
            code_blocks.append(CodeBlock(language=lang, code=code_text, context=ctx))

        # Extract tables
        tables = []
        for table_tag in main.find_all("table"):
            headers = []
            rows = []
            thead = table_tag.find("thead")
            if thead:
                for th in thead.find_all(["th", "td"]):
                    headers.append(th.get_text(strip=True))
            for tr in table_tag.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells and cells != headers:
                    rows.append(cells)
            ctx = ""
            prev = table_tag.find_previous_sibling(["p", "h1", "h2", "h3", "h4"])
            if prev:
                ctx = prev.get_text(strip=True)
            tables.append(Table(headers=headers, rows=rows, context=ctx))

        # Extract sections
        sections = self._parse_sections_html(main)

        # Get clean text
        clean_content = self._get_clean_text_from_soup(main)

        doc_type = self._infer_doc_type(file_path, {}, cloud_provider)

        return self._build_document(
            title=title,
            content=clean_content,
            sections=sections,
            code_blocks=code_blocks,
            tables=tables,
            cloud_provider=cloud_provider,
            service_name=service_name,
            service_category=service_category,
            doc_type=doc_type,
            url_source=url_source,
            file_path=file_path,
            front_matter={},
        )

    # --------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------

    def _extract_front_matter(self, text: str) -> tuple:
        """Extract YAML front matter from markdown."""
        match = self.FRONT_MATTER_RE.match(text)
        if match:
            try:
                fm = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                fm = {}
            body = text[match.end():]
            return fm, body
        return {}, text

    def _convert_hugo_shortcodes(self, text: str) -> str:
        """Convert Hugo shortcodes to bracketed text."""
        def _replace_block(m):
            tag = m.group(1).upper()
            content = m.group(2).strip()
            return f"\n[{tag}: {content}]\n"
        text = self.HUGO_SHORTCODE_RE.sub(_replace_block, text)
        text = self.HUGO_SHORTCODE_INLINE_RE.sub("", text)
        return text

    def _extract_code_blocks_md(self, text: str) -> tuple:
        """Extract fenced code blocks, returning list and text with placeholders."""
        blocks = []
        parts = []
        last_end = 0
        for m in self.CODE_BLOCK_RE.finditer(text):
            before = text[last_end:m.start()]
            # Get context: last paragraph before code block
            ctx_lines = before.strip().rsplit("\n\n", 1)
            ctx = ctx_lines[-1].strip() if ctx_lines else ""
            blocks.append(CodeBlock(
                language=m.group(1),
                code=m.group(2).strip(),
                context=ctx,
            ))
            parts.append(before)
            parts.append(f"\n[CODE_BLOCK_{len(blocks) - 1}]\n")
            last_end = m.end()
        parts.append(text[last_end:])
        return blocks, "".join(parts)

    def _extract_tables_md(self, text: str) -> tuple:
        """Extract Markdown tables."""
        tables = []
        parts = []
        last_end = 0
        for m in self.TABLE_RE.finditer(text):
            before = text[last_end:m.start()]
            ctx_lines = before.strip().rsplit("\n\n", 1)
            ctx = ctx_lines[-1].strip() if ctx_lines else ""

            header_line = m.group(1).strip()
            headers = [c.strip() for c in header_line.strip("|").split("|")]

            rows_text = m.group(3).strip()
            rows = []
            for row_line in rows_text.split("\n"):
                row_line = row_line.strip()
                if row_line:
                    cells = [c.strip() for c in row_line.strip("|").split("|")]
                    rows.append(cells)

            tables.append(Table(headers=headers, rows=rows, context=ctx))
            parts.append(before)
            parts.append(f"\n[TABLE_{len(tables) - 1}]\n")
            last_end = m.end()
        parts.append(text[last_end:])
        return tables, "".join(parts)

    def _parse_sections_md(self, text: str) -> List[Section]:
        """Parse Markdown headings into a section hierarchy."""
        sections = []
        stack: List[Section] = []

        lines = text.split("\n")
        current_content_lines = []

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save content to current section
                if stack:
                    stack[-1].content = "\n".join(current_content_lines).strip()
                elif current_content_lines:
                    # Content before any heading -- ignore or attach to root
                    pass
                current_content_lines = []

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                new_section = Section(level=level, title=title, content="")

                # Find parent
                while stack and stack[-1].level >= level:
                    stack.pop()

                if stack:
                    stack[-1].subsections.append(new_section)
                else:
                    sections.append(new_section)
                stack.append(new_section)
            else:
                current_content_lines.append(line)

        # Flush remaining content
        if stack:
            stack[-1].content = "\n".join(current_content_lines).strip()

        return sections

    def _parse_sections_html(self, soup) -> List[Section]:
        """Parse HTML headings into section hierarchy."""
        sections = []
        heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        stack: List[Section] = []

        for tag in heading_tags:
            level = int(tag.name[1])
            title = tag.get_text(strip=True)
            content_parts = []
            sibling = tag.next_sibling
            while sibling:
                if hasattr(sibling, "name") and sibling.name in [
                    "h1", "h2", "h3", "h4", "h5", "h6"
                ]:
                    break
                if isinstance(sibling, NavigableString):
                    text = str(sibling).strip()
                    if text:
                        content_parts.append(text)
                elif hasattr(sibling, "get_text"):
                    content_parts.append(sibling.get_text(strip=True))
                sibling = sibling.next_sibling

            section = Section(
                level=level,
                title=title,
                content="\n".join(content_parts),
            )
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].subsections.append(section)
            else:
                sections.append(section)
            stack.append(section)

        return sections

    def _clean_markdown_text(self, text: str) -> str:
        """Clean markdown text preserving meaningful content."""
        # Convert links: [text](url) -> text (url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        # Remove image references
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _get_clean_text_from_soup(self, soup) -> str:
        """Extract clean text from BeautifulSoup element."""
        text = soup.get_text(separator="\n")
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _infer_doc_type(
        self, file_path: str, front_matter: dict, provider: str
    ) -> str:
        """Infer document type from path and metadata."""
        fp = file_path.lower()
        fm_type = front_matter.get("content_type", "")

        if fm_type:
            mapping = {
                "concept": "concept",
                "task": "task",
                "tutorial": "tutorial",
                "reference": "api_reference",
            }
            return mapping.get(fm_type, fm_type)

        if "tutorial" in fp:
            return "tutorial"
        if "api" in fp or "reference" in fp:
            return "api_reference"
        if "faq" in fp:
            return "faq"
        if "glossary" in fp or provider == "cncf":
            return "glossary"
        if "concept" in fp:
            return "concept"
        if "task" in fp:
            return "task"
        return "guide"

    def _extract_date(self, front_matter: dict) -> Optional[datetime]:
        """Extract date from front matter."""
        for key in ["ms.date", "date", "last_updated", "last_modified"]:
            val = front_matter.get(key)
            if val:
                if isinstance(val, datetime):
                    return val
                try:
                    return datetime.fromisoformat(str(val))
                except (ValueError, TypeError):
                    pass
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y"]:
                    try:
                        return datetime.strptime(str(val), fmt)
                    except ValueError:
                        continue
        return None

    def _build_document(
        self,
        title: str,
        content: str,
        sections: List[Section],
        code_blocks: List[CodeBlock],
        tables: List[Table],
        cloud_provider: str,
        service_name: str,
        service_category: str = "",
        doc_type: str = "guide",
        url_source: str = "",
        file_path: str = "",
        front_matter: Optional[dict] = None,
        last_updated: Optional[datetime] = None,
    ) -> Document:
        """Build a Document from parsed components."""
        # Compute statistics
        words = content.split()
        heading_count = sum(1 for _ in self._iter_sections(sections))

        return Document(
            title=title or (front_matter or {}).get("title", ""),
            content=content,
            sections=sections,
            code_blocks=code_blocks,
            tables=tables,
            cloud_provider=cloud_provider,
            service_name=service_name,
            service_category=service_category,
            doc_type=doc_type,
            url_source=url_source,
            file_path=file_path,
            last_updated=last_updated,
            word_count=len(words),
            char_count=len(content),
            has_code=len(code_blocks) > 0,
            has_tables=len(tables) > 0,
            heading_count=heading_count,
        )

    def _iter_sections(self, sections: List[Section]):
        """Iterate all sections recursively."""
        for s in sections:
            yield s
            yield from self._iter_sections(s.subsections)
