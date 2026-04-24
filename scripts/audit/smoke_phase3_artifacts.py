"""
Smoke test for Phase 3 artefacts.

Validates that scripts/generate_phase3_artifacts.py produced the
expected set of _phase3 files with the expected content invariants.

Run with:
    python scripts/generate_phase3_artifacts.py      # must run first
    python scripts/audit/smoke_phase3_artifacts.py   # then this

Assertions (10 mandatory, covering the 10-artefact scope plus the 6
pre-existing baselines):

  1. All 10 *_phase3.{png,tex} exist with size > 0.
  2. All 6 *_pre_phase3.{png,tex} backups exist (one per pre-existing
     baseline). The 4 table_expN_faithfulness_phase3.tex files are new
     and have no pre-Phase-3 counterpart.
  3. md5(fig_retrieval_metrics_phase3.png) !=
     md5(fig_retrieval_metrics_exp8b_phase3.png) — closes Flag 92 from
     the pre-Phase-3 duplicate.
  4. grep '0\\.626' table_retrieval_metrics_phase3.tex → 0 matches
     (d_pooled vestige removed).
  5. grep '0\\.626' table_retrieval_metrics_exp8b_phase3.tex → 0 matches.
  6. table_exp5_faithfulness_phase3.tex contains '190' AND '200'
     (hall_n_effective / hall_n_total for llama3.1 post-rerun).
  7. fig_llm_comparison_phase3.png size > 5 KB (sanity on PNG bytes).
  8. Every _phase3.tex has balanced \\begin{table}/\\end{table}.
  9. table_retrieval_metrics_phase3.tex contains 'd_z' or 'cohens_d_z'
     in the caption / body (evidence of the Phase 2.5 formula change).
 10. No _phase3 artefact contains the literal '"hall_faithfulness_mean": 0'
     (catches placeholder zeros that would have slipped past the
     Phase 1 KeyError fix if a generator still falls back to silent 0).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "paper" / "overleaf_ready" / "figures"

PHASE3_ARTEFACTS = [
    ("fig_llm_comparison", "png", True),
    ("fig_latency_breakdown", "png", True),
    ("fig_retrieval_metrics", "png", True),
    ("fig_retrieval_metrics_exp8b", "png", True),
    ("table_retrieval_metrics", "tex", True),
    ("table_retrieval_metrics_exp8b", "tex", True),
    ("table_exp5_faithfulness", "tex", False),
    ("table_exp6_faithfulness", "tex", False),
    ("table_exp8_faithfulness", "tex", False),
    ("table_exp8b_faithfulness", "tex", False),
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def assert_1_phase3_files_exist() -> None:
    missing = []
    empty = []
    for base, ext, _ in PHASE3_ARTEFACTS:
        p = FIGURES_DIR / f"{base}_phase3.{ext}"
        if not p.exists():
            missing.append(p.name)
        elif p.stat().st_size == 0:
            empty.append(p.name)
    assert not missing, f"missing _phase3 files: {missing}"
    assert not empty, f"zero-byte _phase3 files: {empty}"
    print(f"PASS #1: all {len(PHASE3_ARTEFACTS)} _phase3 files exist with size > 0")


def assert_2_pre_phase3_backups_exist() -> None:
    expected_backups = [
        f"{base}_pre_phase3.{ext}"
        for base, ext, has_baseline in PHASE3_ARTEFACTS
        if has_baseline
    ]
    missing = []
    for name in expected_backups:
        p = FIGURES_DIR / name
        if not p.exists():
            missing.append(name)
    assert not missing, f"missing _pre_phase3 backups: {missing}"
    print(f"PASS #2: all {len(expected_backups)} _pre_phase3 backups exist")


def assert_3_retrieval_figs_distinct() -> None:
    a = FIGURES_DIR / "fig_retrieval_metrics_phase3.png"
    b = FIGURES_DIR / "fig_retrieval_metrics_exp8b_phase3.png"
    h_a = _md5(a)
    h_b = _md5(b)
    assert h_a != h_b, (
        f"fig_retrieval_metrics_phase3.png ({h_a}) is byte-identical to "
        f"fig_retrieval_metrics_exp8b_phase3.png — Flag 92 not closed"
    )
    print(
        f"PASS #3: retrieval figures distinct "
        f"(md5 {h_a[:8]} vs {h_b[:8]})"
    )


def assert_4_no_0626_in_exp8_table() -> None:
    p = FIGURES_DIR / "table_retrieval_metrics_phase3.tex"
    content = p.read_text(encoding="utf-8")
    matches = re.findall(r"0\.626", content)
    assert not matches, (
        f"table_retrieval_metrics_phase3.tex still contains '0.626' "
        f"({len(matches)} match): d_pooled vestige not purged"
    )
    print("PASS #4: table_retrieval_metrics_phase3.tex contains no '0.626'")


def assert_5_no_0626_in_exp8b_table() -> None:
    p = FIGURES_DIR / "table_retrieval_metrics_exp8b_phase3.tex"
    content = p.read_text(encoding="utf-8")
    matches = re.findall(r"0\.626", content)
    assert not matches, (
        f"table_retrieval_metrics_exp8b_phase3.tex still contains '0.626' "
        f"({len(matches)} match)"
    )
    print("PASS #5: table_retrieval_metrics_exp8b_phase3.tex contains no '0.626'")


def assert_6_n_effective_in_exp5_table() -> None:
    p = FIGURES_DIR / "table_exp5_faithfulness_phase3.tex"
    content = p.read_text(encoding="utf-8")
    assert "190" in content, (
        "table_exp5_faithfulness_phase3.tex lacks hall_n_effective=190"
    )
    assert "200" in content, (
        "table_exp5_faithfulness_phase3.tex lacks hall_n_total=200"
    )
    print("PASS #6: table_exp5_faithfulness_phase3.tex contains '190' and '200'")


def assert_7_llm_comparison_png_nonzero() -> None:
    p = FIGURES_DIR / "fig_llm_comparison_phase3.png"
    sz = p.stat().st_size
    assert sz > 5_000, (
        f"fig_llm_comparison_phase3.png size={sz} B (expected > 5000)"
    )
    print(f"PASS #7: fig_llm_comparison_phase3.png size={sz} B (> 5 KB)")


def assert_8_tex_balanced_table_env() -> None:
    tex_artefacts = [
        (base, ext) for base, ext, _ in PHASE3_ARTEFACTS if ext == "tex"
    ]
    for base, ext in tex_artefacts:
        p = FIGURES_DIR / f"{base}_phase3.{ext}"
        content = p.read_text(encoding="utf-8")
        begins = len(re.findall(r"\\begin\{table\}", content))
        ends = len(re.findall(r"\\end\{table\}", content))
        assert begins == ends, (
            f"{p.name}: {begins} \\begin{{table}} vs {ends} \\end{{table}}"
        )
        assert begins >= 1, (
            f"{p.name}: no \\begin{{table}} found"
        )
    print(
        f"PASS #8: all {len(tex_artefacts)} _phase3.tex have balanced "
        "\\begin{{table}} / \\end{{table}}"
    )


def assert_9_d_z_in_retrieval_table() -> None:
    p = FIGURES_DIR / "table_retrieval_metrics_phase3.tex"
    content = p.read_text(encoding="utf-8")
    has_marker = "d_z" in content or "cohens_d_z" in content
    assert has_marker, (
        "table_retrieval_metrics_phase3.tex lacks 'd_z' / 'cohens_d_z' "
        "marker — formula change not evidenced in the caption/body"
    )
    print("PASS #9: table_retrieval_metrics_phase3.tex contains d_z marker")


def assert_10_no_placeholder_zeros() -> None:
    offenders = []
    for base, ext, _ in PHASE3_ARTEFACTS:
        if ext != "tex":
            continue  # binary .png files can't be string-searched
        p = FIGURES_DIR / f"{base}_phase3.{ext}"
        content = p.read_text(encoding="utf-8")
        if '"hall_faithfulness_mean": 0' in content:
            offenders.append(p.name)
    assert not offenders, (
        f"placeholder zero 'hall_faithfulness_mean: 0' in: {offenders}"
    )
    print("PASS #10: no _phase3 artefact contains placeholder "
          '"hall_faithfulness_mean": 0 literal')


def main() -> None:
    assert_1_phase3_files_exist()
    assert_2_pre_phase3_backups_exist()
    assert_3_retrieval_figs_distinct()
    assert_4_no_0626_in_exp8_table()
    assert_5_no_0626_in_exp8b_table()
    assert_6_n_effective_in_exp5_table()
    assert_7_llm_comparison_png_nonzero()
    assert_8_tex_balanced_table_env()
    assert_9_d_z_in_retrieval_table()
    assert_10_no_placeholder_zeros()
    print("\nAll Phase 3 smoke assertions passed.")


if __name__ == "__main__":
    main()
