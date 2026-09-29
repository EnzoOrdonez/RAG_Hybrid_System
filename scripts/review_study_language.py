"""Record a human language annotation without changing a sealed study export."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.ui.components.session_storage import atomic_json, read_json
from src.ui.components.study_protocol import digest

ALLOWED = {"english", "spanish", "mixed", "other", "undetermined"}


def annotate(export_path, language, output_path, reviewer_id, *, now=None):
    export_path, output_path = Path(export_path), Path(output_path)
    if language not in ALLOWED:
        raise ValueError(
            "language must be english, spanish, mixed, other or undetermined"
        )
    if (
        not isinstance(reviewer_id, str)
        or not reviewer_id.strip()
        or len(reviewer_id) > 64
    ):
        raise ValueError("reviewer_id must be a nonempty operator code")
    original_hash = digest(export_path)
    payload = read_json(export_path)
    rows = [
        a for a in payload.get("attempts", []) if a.get("analysis_role") == "free_query"
    ]
    review = dict(
        schema_version=1,
        export=str(export_path.resolve()),
        export_sha256=original_hash,
        reviewer_id=reviewer_id.strip(),
        annotated_at=(now or datetime.now(timezone.utc)).isoformat(),
        annotations=[
            dict(attempt_id=row["attempt_id"], language=language) for row in rows
        ],
    )
    if output_path.exists():
        existing = read_json(output_path)
        comparable = dict(review, annotated_at=existing.get("annotated_at"))
        if existing != comparable:
            raise FileExistsError("Review already exists with different annotation")
    if not output_path.exists():
        atomic_json(output_path, review)
    if digest(export_path) != original_hash:
        raise RuntimeError("Original export changed during review")
    return review


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer-id", required=True)
    args = parser.parse_args(argv)
    annotate(args.export, args.language, args.output, args.reviewer_id)


if __name__ == "__main__":
    main()
