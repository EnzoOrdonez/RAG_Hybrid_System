"""Build a read-only inventory of local and remote-tracking Git branches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import subprocess
from typing import Sequence


TARGET = "summer/taxonomia-759"
PRIMARY = "main"
SNAPSHOT_DATE = date(2026, 8, 23)


@dataclass(frozen=True)
class Branch:
    name: str
    scope: str
    commit: str
    committed_on: date
    object_id: str


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        text=True,
        capture_output=True,
    )


def list_branches(root: Path) -> list[Branch]:
    result = git(
        root,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)%00%(objectname:short)%00%(committerdate:short)%00%(symref)",
        "refs/heads",
        "refs/remotes",
    )
    branches: list[Branch] = []
    for line in result.stdout.splitlines():
        refname, object_id, short_id, date_text, symref = line.split("\0")
        if symref:
            continue
        if refname.startswith("refs/heads/"):
            name = refname.removeprefix("refs/heads/")
            scope = "local"
        else:
            name = refname.removeprefix("refs/remotes/")
            scope = "remota"
        branches.append(
            Branch(
                name=name,
                scope=scope,
                commit=short_id,
                committed_on=date.fromisoformat(date_text),
                object_id=object_id,
            )
        )
    return sorted(branches, key=lambda branch: (branch.scope, branch.name.casefold()))


def is_merged(root: Path, branch: str, base: str) -> bool:
    result = git(root, "merge-base", "--is-ancestor", branch, base, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip())
    return result.returncode == 0


def ahead_behind(root: Path, branch: str, base: str = TARGET) -> tuple[int, int]:
    result = git(root, "rev-list", "--left-right", "--count", f"{base}...{branch}")
    behind_text, ahead_text = result.stdout.strip().split()
    return int(ahead_text), int(behind_text)


def paired_branch(branch: Branch) -> str:
    return branch.name.removeprefix("origin/") if branch.scope == "remota" else f"origin/{branch.name}"


def classify(
    branch: Branch,
    branches_by_name: dict[str, Branch],
    merged_target: bool,
    merged_main: bool,
    ahead: int,
) -> tuple[str, str]:
    logical_name = branch.name.removeprefix("origin/")
    if logical_name == TARGET:
        return "activa", "rama de trabajo actual o su referencia remota"
    if logical_name == PRIMARY:
        return "activa", "rama principal o su referencia remota"

    paired = branches_by_name.get(paired_branch(branch))
    if merged_target and merged_main and branch.scope == "local" and paired:
        if paired.object_id == branch.object_id:
            return (
                "candidata a borrar",
                "copia local totalmente integrada y respaldada por una referencia remota idéntica",
            )
    if merged_target or merged_main:
        destinations = []
        if merged_target:
            destinations.append(TARGET)
        if merged_main:
            destinations.append(PRIMARY)
        return (
            "candidata a archivar",
            f"punta integrada en {', '.join(destinations)}; conservar como hito hasta revisión humana",
        )

    age_days = (SNAPSHOT_DATE - branch.committed_on).days
    if ahead > 0 and age_days <= 60:
        return "activa", f"contiene {ahead} commit(s) propios y tuvo actividad hace {age_days} días"
    return (
        "candidata a archivar",
        f"conserva {ahead} commit(s) no integrados, pero su última actividad fue hace {age_days} días",
    )


def render_inventory(root: Path) -> str:
    branches = list_branches(root)
    branches_by_name = {branch.name: branch for branch in branches}
    rows: list[str] = []
    for branch in branches:
        merged_target = is_merged(root, branch.name, TARGET)
        merged_main = is_merged(root, branch.name, PRIMARY)
        ahead, behind = ahead_behind(root, branch.name)
        classification, reason = classify(
            branch,
            branches_by_name,
            merged_target,
            merged_main,
            ahead,
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    branch.name,
                    branch.scope,
                    branch.commit,
                    branch.committed_on.isoformat(),
                    "sí" if merged_target else "no",
                    "sí" if merged_main else "no",
                    str(ahead),
                    str(behind),
                    classification,
                    reason,
                ]
            )
            + " |"
        )

    return "\n".join(
        [
            "# Inventario de ramas — 2026-08-23",
            "",
            "Inventario generado solo con referencias locales existentes, sin `fetch`, cambios de rama ni operaciones destructivas.",
            f"Los conteos `adelante` y `atrás` son relativos a `{TARGET}`. Una rama figura como mergeada cuando su punta es ancestro de la rama base.",
            "Las clasificaciones son propuestas para decisión humana; este inventario no ejecuta ninguna acción.",
            "",
            "| Rama | Ámbito | Último commit | Fecha | Mergeada a taxonomía | Mergeada a main | Adelante | Atrás | Propuesta | Razón |",
            "|---|---|---:|---:|:---:|:---:|---:|---:|---|---|",
            *rows,
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/BRANCH_INVENTORY_2026-08-23.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text(render_inventory(root), encoding="utf-8", newline="\n")
    print(f"Inventario escrito: {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
