"""Refuse to overwrite files inside committed experiment evidence.

Two kinds of evidence live under `experiments/results/`, and both are immutable by project
rule, for different reasons:

  SIGNED   `exp3..exp14` (plus `exp8b`) carry the tags `nota3-evidencia-2026-06-11` /
           `nota3-N9-cierre-2026-07-02` and back figures already delivered in the A.3 report
           and the LACCI paper.
  FROZEN   `exp15..exp19a` are the summer phase. No tag covers them, but the rule is the same:
           committed summer evidence is not regenerated, overwritten or reinterpreted without
           fixing the scope first.

In both cases recomputation goes to NEW `_vN` files, never on top of an existing artifact.

Two scripts make that easy to break by accident: `compute_faithfulness_metrics.py`
(`--experiment` defaults to **exp12_matrix**) and `compute_retrieval_metrics.py`
(`--experiment` defaults to **exp8**). Both write in place into the directory they are given,
so running either with no arguments aims straight at signed evidence.

THE REGISTRY IS INVERTED, and that is the point (2026-08-21). It used to be a hand-written
list of what to PROTECT. This phase has already paid twice for hand-written work lists: Pass
N's arm registry silently scored 1 of exp18's 4 arms (ledger entry 21), and
`verify_summer_offline.py`'s hardcoded experiment list left exp18 unverified altogether. A
protect-list goes stale in the dangerous direction -- a new experiment is unprotected by
default and nobody finds out until something is clobbered.

So the default is now PROTECTED, and what gets declared is what is still LIVE. A stale LIVE
entry costs a false refusal: loud, immediate, harmless. A stale protect-list costs overwritten
evidence: silent and permanent.

SCOPE OF THE REFUSAL, unchanged: only OVERWRITING a file that already exists. Creating new
artifacts inside a protected directory stays allowed, because that is how the summer runners
work and how recomputation is sanctioned.

Seccion de Claude Code — 2026-08-21 02:20 (hora local): inversion del registro y motivos por
tipo en el mensaje de error.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "experiments" / "results"

# Experiments a run is still writing. Everything else under experiments/results is protected.
# Remove an id from here the moment its evidence is committed and read as final.
LIVE_EXPERIMENTS = frozenset(["exp19b"])

# The tagged set, kept separate so the refusal can cite the tag ONLY where it exists.
# exp1/exp2 never existed; exp8b is signed alongside exp8.
SIGNED_EXPERIMENTS = frozenset(
    [f"exp{n}" for n in range(3, 15)] + ["exp8b"]
)


def _experiment_id(top: str) -> str:
    """`exp12_matrix` -> `exp12`, `exp8b` -> `exp8b`, `exp19a_selector_probe` -> `exp19a`."""
    return top.split("_")[0]


def _protected_dir_for(path):
    """The protected experiment dir containing `path`, or None.

    Matches on the directory NAME's experiment prefix, so `exp12_matrix`,
    `exp10_retrieval194` and friends are all recognised without listing every suffix.
    """
    try:
        rel = path.resolve().relative_to(RESULTS.resolve())
    except (ValueError, OSError):
        return None
    if not rel.parts:
        return None
    top = rel.parts[0]
    exp_id = _experiment_id(top)
    if not exp_id.startswith("exp"):
        return None                      # not an experiment dir; nothing to protect
    return None if exp_id in LIVE_EXPERIMENTS else top


# Kept so any caller that still reaches for the old name keeps working.
_signed_dir_for = _protected_dir_for


def _reason_for(top: str) -> str:
    """Why this directory is immutable — the true reason, not a boilerplate one."""
    if _experiment_id(top) in SIGNED_EXPERIMENTS:
        return ("carries the nota3-evidencia / nota3-N9 tags and backs figures already "
                "delivered in A.3 and the LACCI paper")
    return ("is committed summer-phase evidence (project rule: exp15+ is not regenerated, "
            "overwritten or reinterpreted without fixing the scope first)")


def guard_write(path, allow_overwrite: bool = False) -> Path:
    """Raise if writing `path` would overwrite an existing file in protected evidence.

    Returns the path so it can be used inline:  ``guard_write(p).write_text(...)``
    """
    path = Path(path)
    protected = _protected_dir_for(path)
    if protected and path.exists() and not allow_overwrite:
        raise SystemExit(
            f"REFUSING to overwrite committed evidence: {path}\n"
            f"  '{protected}' {_reason_for(protected)}.\n"
            f"  Recomputations must go to a NEW _vN file, which this guard allows.\n"
            f"  If the experiment is still being written, add its id to LIVE_EXPERIMENTS in "
            f"src/utils/signed_evidence.py.\n"
            f"  If you genuinely intend to replace it, pass allow_overwrite=True and say so "
            f"in the ledger."
        )
    return path
