"""Refuse to overwrite files inside the signed, immutable experiment evidence.

`experiments/results/exp3..exp14` (plus `exp8b`) are signed with the tags
`nota3-evidencia-2026-06-11` / `nota3-N9-cierre-2026-07-02` and back figures already
delivered in the A.3 report and the LACCI paper. The project rule is that any recomputation
goes to NEW `_vN` files, never on top of an existing artifact.

Two scripts make that rule easy to break by accident: `compute_faithfulness_metrics.py`
(`--experiment` defaults to **exp12_matrix**) and `compute_retrieval_metrics.py`
(`--experiment` defaults to **exp8**). Both write in place into the directory they are
given, so running either with no arguments aims straight at signed evidence. Nothing has
actually been clobbered -- `git diff` against the tag shows additions only -- but the
foot-gun is live and a single distracted invocation is all it takes.

This guard blocks exactly the forbidden action and nothing else: OVERWRITING a file that
already exists inside a signed directory. Creating new `_vN` artifacts there stays allowed,
because that is the sanctioned way to recompute.

The signed set lives here, once. Duplicating it into each caller is the same
copy-of-a-work-list pattern that has already produced several silent defects in this phase.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "experiments" / "results"

# exp1/exp2 never existed; exp8b is signed alongside exp8.
SIGNED_EXPERIMENTS = frozenset(
    [f"exp{n}" for n in range(3, 15)] + ["exp8b"]
)


def _signed_dir_for(path: Path):
    """The signed experiment dir containing `path`, or None.

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
    exp_id = top.split("_")[0]
    return top if exp_id in SIGNED_EXPERIMENTS else None


def guard_write(path, allow_overwrite: bool = False) -> Path:
    """Raise if writing `path` would overwrite an existing file in signed evidence.

    Returns the path so it can be used inline:  ``guard_write(p).write_text(...)``
    """
    path = Path(path)
    signed = _signed_dir_for(path)
    if signed and path.exists() and not allow_overwrite:
        raise SystemExit(
            f"REFUSING to overwrite signed evidence: {path}\n"
            f"  '{signed}' is under experiments/results and carries the "
            f"nota3-evidencia tag; its artifacts are immutable.\n"
            f"  Recomputations must go to a NEW _vN file, which this guard allows.\n"
            f"  If you genuinely intend to replace it, pass allow_overwrite=True and say so "
            f"in the ledger."
        )
    return path
