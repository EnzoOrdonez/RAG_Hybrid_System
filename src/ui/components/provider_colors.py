"""Consistent colors for providers and systems across the UI."""

PROVIDER_COLORS = {
    "aws": "#FF9900",
    "azure": "#0078D4",
    "gcp": "#4285F4",
    "k8s": "#326CE5",
    "kubernetes": "#326CE5",
    "cncf": "#1B3A5C",
}

SYSTEM_COLORS = {
    "RAG Lexico (BM25)": "#E67E22",
    "RAG Semantico (Dense)": "#3498DB",
    "RAG Hibrido Propuesto": "#27AE60",
}

SYSTEM_LABELS = {
    "lexical": "RAG Lexico (BM25)",
    "semantic": "RAG Semantico (Dense)",
    "hybrid": "RAG Hibrido Propuesto",
}


def get_provider_badge(provider: str) -> str:
    """Return HTML badge colored for a provider."""
    color = PROVIDER_COLORS.get(provider.lower(), "#666")
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:2px 8px;border-radius:4px;font-size:0.8em;">'
        f'{provider.upper()}</span>'
    )


def faithfulness_color(score: float) -> str:
    """Green (>0.8), Yellow (0.5-0.8), Red (<0.5)."""
    if score >= 0.8:
        return "#27AE60"
    elif score >= 0.5:
        return "#F39C12"
    else:
        return "#E74C3C"


def faithfulness_label(score: float) -> str:
    """Human-readable label for faithfulness score."""
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"
