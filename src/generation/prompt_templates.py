"""
RAG Prompt Templates for different query types.

Templates:
  - SYSTEM_PROMPT: Base system message for all RAG queries
  - RAG_PROMPT: Standard single-topic question
  - CROSS_CLOUD_PROMPT: Multi-provider comparison
  - PROCEDURAL_PROMPT: Step-by-step instructions
"""

# ============================================================
# System prompt (used for all RAG queries)
# ============================================================

SYSTEM_PROMPT = """You are a cloud computing documentation assistant specialized in \
AWS, Azure, GCP, and Kubernetes. Answer questions accurately based ONLY on the \
provided documentation context.

Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain enough info, say: "Based on the available \
documentation, I cannot find sufficient information to fully answer this question."
3. Cite sources: [Source: provider/service/section_path]
4. When comparing providers, clearly label each one
5. Include code examples from context when relevant
6. Be precise with technical terms and configurations"""

# ============================================================
# Standard RAG prompt (conceptual, single-topic)
# ============================================================

RAG_PROMPT = """Context from cloud documentation:
---
{context}
---

Based ONLY on the above context, answer the following question. \
Cite sources using [Source: provider/service/section].

Question: {question}

Answer:"""

# ============================================================
# Cross-cloud comparison prompt
# ============================================================

CROSS_CLOUD_PROMPT = """Context from multiple cloud providers:

{context_by_provider}

Provide a comparative answer based on the above context. \
For each point, indicate which provider(s) it applies to. \
Cite sources using [Source: provider/service/section].

Question: {question}

Comparative Answer:"""

# ============================================================
# Procedural (how-to) prompt
# ============================================================

PROCEDURAL_PROMPT = """Context from cloud documentation:
---
{context}
---

Provide step-by-step instructions based on the above context. \
Include code examples or configurations from the documentation. \
Cite sources for each step.

Question: {question}

Step-by-step Answer:"""

# ============================================================
# Template selector
# ============================================================

TEMPLATE_MAP = {
    "conceptual": RAG_PROMPT,
    "single_provider": RAG_PROMPT,
    "cross_cloud": CROSS_CLOUD_PROMPT,
    "procedural": PROCEDURAL_PROMPT,
    "default": RAG_PROMPT,
}


def get_template(query_type: str) -> str:
    """Return the appropriate prompt template for the query type."""
    return TEMPLATE_MAP.get(query_type, RAG_PROMPT)


def build_context(chunks: list, query_type: str = "default") -> str:
    """Build context string from retrieved chunks.

    For cross_cloud queries, groups context by provider.
    For other queries, concatenates chunks with source labels.
    """
    if query_type == "cross_cloud":
        return _build_cross_cloud_context(chunks)
    return _build_standard_context(chunks)


def _build_standard_context(chunks: list) -> str:
    """Build standard context: numbered chunks with source info."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        provider = chunk.get("cloud_provider", "unknown")
        service = chunk.get("service_name", "unknown")
        heading = chunk.get("heading_path", "")
        text = chunk.get("text", "")
        source = f"[Source: {provider}/{service}"
        if heading:
            source += f"/{heading}"
        source += "]"
        parts.append(f"[{i}] {source}\n{text}")
    return "\n\n".join(parts)


def _build_cross_cloud_context(chunks: list) -> str:
    """Build cross-cloud context: grouped by provider."""
    by_provider = {}
    for chunk in chunks:
        provider = chunk.get("cloud_provider", "unknown")
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(chunk)

    parts = []
    for provider in sorted(by_provider.keys()):
        parts.append(f"### {provider.upper()}")
        for i, chunk in enumerate(by_provider[provider], 1):
            service = chunk.get("service_name", "unknown")
            heading = chunk.get("heading_path", "")
            text = chunk.get("text", "")
            source = f"[Source: {provider}/{service}"
            if heading:
                source += f"/{heading}"
            source += "]"
            parts.append(f"  [{i}] {source}\n  {text}")
        parts.append("")
    return "\n".join(parts)
