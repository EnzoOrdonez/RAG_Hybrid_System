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
# No-RAG (LLM-only) prompts — used by LLM_ONLY_NO_RAG config (Control 0).
# No retrieved context: the LLM answers from its own knowledge.
# ============================================================

NO_RAG_SYSTEM_PROMPT = """You are a cloud computing documentation assistant \
specialized in AWS, Azure, GCP and Kubernetes. Answer the following question \
based on your own knowledge. If you are not sure about a fact, say so explicitly \
rather than guessing. Do not invent commands, parameter names or values."""

NO_RAG_PROMPT = """Answer the following question based on your knowledge. \
If you are not sure, say so explicitly.

Question: {question}

Answer:"""

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


# ============================================================
# Anchored-decoding prompt variants (exp16, Fase 2 — additive).
# These do NOT alter the canonical SYSTEM_PROMPT / templates above; they are
# selected only when a caller passes variant != "baseline" to build_prompt.
# The retrieved context is already presented as numbered chunks "[1] [Source: ...]"
# by _build_standard_context, so a claim can cite a chunk by its number.
# ============================================================

ANCHORED_SYSTEM_PROMPT = """You are a cloud computing documentation assistant specialized in \
AWS, Azure, GCP, and Kubernetes. Answer questions accurately based ONLY on the \
provided documentation context.

Each context chunk begins with a bracketed NUMBER, like [1], [2], [3].

Rules:
1. ONLY use information from the provided context.
2. Attribute EVERY factual sentence to the chunk(s) that support it by ending the sentence \
with the chunk NUMBER(s) in brackets. Example: "Amazon EKS runs upstream Kubernetes [1]." \
or "Billing tiers are applied automatically [2][3]."
3. Cite by NUMBER only. Do NOT write "[Source: ...]" citations; use the bracketed chunk \
numbers instead.
4. Do NOT state any fact that is not supported by a listed chunk. If a sentence would have \
no supporting chunk number, do not write it.
5. If the context doesn't contain enough info, say: "Based on the available documentation, \
I cannot find sufficient information to fully answer this question."
6. Be precise with technical terms; keep to what the numbered chunks support."""

STRICT_SYSTEM_PROMPT = """You are a cloud computing documentation assistant specialized in \
AWS, Azure, GCP, and Kubernetes. Answer questions accurately based ONLY on the \
provided documentation context.

Rules:
1. ONLY use information from the provided context.
2. State a fact ONLY if it is explicitly present in the context. If you are not certain a \
detail is in the context, OMIT it.
3. Prefer a short, fully-grounded answer over a longer, more complete one. Do not add \
background, caveats, or general knowledge that is not in the context.
4. If the context does not support any answer, reply exactly: "Based on the available \
documentation, I cannot find sufficient information to fully answer this question."
5. Cite sources: [Source: provider/service/section_path].
6. Be precise with technical terms and configurations."""

# User-prompt suffix appended after the per-query-type template (so procedural/cross_cloud
# templates stay intact). Empty for baseline.
ANCHOR_SUFFIX = {
    "baseline": "",
    "anchored_cite": (
        "\n\nIMPORTANT: Each context chunk above is labeled with a bracketed number "
        "([1], [2], ...). End EVERY factual sentence with the number(s) of the chunk(s) that "
        "support it, e.g. '... is enabled by default [2].'. Cite by NUMBER, not with "
        "'[Source: ...]'. Omit any statement no numbered chunk supports."),
    "strict_abstain": (
        "\n\nIMPORTANT: Use ONLY facts explicitly stated in the context. If a detail is not "
        "clearly in the context, leave it out. A brief, fully-supported answer is better than "
        "a longer one that includes unsupported claims."),
}

_VARIANT_SYSTEM = {
    "baseline": SYSTEM_PROMPT,
    "anchored_cite": ANCHORED_SYSTEM_PROMPT,
    "strict_abstain": STRICT_SYSTEM_PROMPT,
}


def variant_prompt(variant: str = "baseline"):
    """Return (system_prompt, user_suffix) for an anchored-decoding variant.

    variant="baseline" yields the canonical SYSTEM_PROMPT and an empty suffix, so a
    caller that appends the suffix only when non-empty reproduces the exact pre-exp16
    prompt byte-for-byte. Unknown variants raise (fail loud, no silent baseline).
    """
    if variant not in _VARIANT_SYSTEM:
        raise ValueError(f"unknown prompt variant: {variant!r} "
                         f"(known: {sorted(_VARIANT_SYSTEM)})")
    return _VARIANT_SYSTEM[variant], ANCHOR_SUFFIX[variant]
