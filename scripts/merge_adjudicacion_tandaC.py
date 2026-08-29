"""Merge the nine Tanda-C adjudications into the stage-A human gold CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import re
import shutil
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output" / "audit"
DEFAULT_CSV = AUDIT / "claim_audit_sample_v4.csv"
EXPECTED_KEYS = {f"J-{idx:02d}" for idx in range(1, 10)}
ALLOWED_VERDICTS = {"correcto", "incorrecto", "dudoso"}


class AdjudicationError(ValueError):
    """Raised when an adjudication export is incomplete or invalid."""


def merge_adjudication(
    *,
    json_path: Path,
    csv_path: Path = DEFAULT_CSV,
    audit_dir: Path = AUDIT,
    date_stamp: str | None = None,
) -> Path:
    """Apply one validated adjudication export and return its backup path."""

    document = json.loads(json_path.read_text(encoding="utf-8"))
    if document.get("formato") != "gold_v4_adjudicacion":
        raise AdjudicationError(
            "El campo formato debe ser exactamente 'gold_v4_adjudicacion'"
        )
    mapping_keys = set(document.get("mapeo", {}))
    judgment_keys = set(document.get("juicios", {}))
    if mapping_keys != EXPECTED_KEYS or judgment_keys != EXPECTED_KEYS:
        raise AdjudicationError(
            "La adjudicación debe contener 9/9 claves J-01..J-09 "
            "tanto en mapeo como en juicios."
        )
    targets = list(document["mapeo"].values())
    if any(not isinstance(target, str) or not re.fullmatch(r"A-\d{3}", target)
           for target in targets):
        raise AdjudicationError("Todo destino del mapeo debe usar el formato A-xxx")
    if len(set(targets)) != len(targets):
        raise AdjudicationError("Los nueve destinos A-xxx deben ser únicos")
    for key in sorted(EXPECTED_KEYS):
        verdict = document["juicios"][key].get("juicio")
        if verdict not in ALLOWED_VERDICTS:
            raise AdjudicationError(
                f"{key}: juicio fuera del vocabulario exacto {sorted(ALLOWED_VERDICTS)}"
            )
        comment = document["juicios"][key].get("comentario", "")
        if not isinstance(comment, str) or not comment.strip():
            raise AdjudicationError(f"{key}: comentario obligatorio no vacío")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rows_by_idx = {int(row["idx"]): row for row in rows}
    missing_targets = [
        target for target in targets
        if int(target.removeprefix("A-")) not in rows_by_idx
    ]
    if missing_targets:
        raise AdjudicationError(
            f"Destinos A-xxx inexistentes en el CSV: {missing_targets}"
        )
    for key, target in document["mapeo"].items():
        row = rows_by_idx[int(target.removeprefix("A-"))]
        adjudication = document["juicios"][key]
        previous = (row.get("comentario") or "").strip()
        comment = f"adjudicado: {adjudication['comentario'].strip()}"
        if previous:
            comment += f" [{previous}]"
        row["juicio_humano"] = adjudication["juicio"]
        row["comentario"] = comment

    stamp = date_stamp or date.today().isoformat()
    backup_dir = audit_dir / f"backups_adjudicacion_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / csv_path.name
    shutil.copy2(csv_path, backup_path)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return backup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json_path is None:
        candidates = sorted(
            AUDIT.glob("gold_v4_adjudicacion_enzo_*.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            print("No existe todavía un JSON de adjudicación gold_v4 para fusionar.")
            return 2
        json_path = candidates[-1]
    else:
        json_path = args.json_path
    backup = merge_adjudication(json_path=json_path)
    print(f"Adjudicación fusionada; backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
