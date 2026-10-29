"""Shared test helpers.

`code_only` lives here because two different source-level guards need it and a second copy is
exactly the duplication pattern that produced this phase's silent defects.
"""

import io
import tokenize

import pytest


@pytest.fixture
def installed_beta(monkeypatch):
    """CLI tests double the local SDK separately from provider commands."""
    from scripts.study_operator import i7_sdk_preflight
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', lambda *a, **kw:
        {'status': 'GCLOUD_BETA_INSTALLED', 'beta_version': 'fixture'})

# Py3.12+ splits f-strings into FSTRING_START/MIDDLE/END, so filtering tokenize.STRING alone
# leaves the literal text of every f-string behind. Match by NAME to stay version-tolerant.
_DROP = {"STRING", "COMMENT", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"}


def code_only(src: str) -> str:
    """Executable tokens only — every string literal and comment removed.

    Prose may legitimately NAME what the code must not reach or hardcode ("not
    bge-reranker-large: that stays the independent oracle"), and a guard that fires on its own
    documentation trains people to weaken it. What matters is what the module can actually do.
    """
    kept = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tokenize.tok_name.get(tok.type) in _DROP:
                continue
            kept.append(tok.string)
    except tokenize.TokenError:  # pragma: no cover - malformed source is its own failure
        return src
    return " ".join(kept)
