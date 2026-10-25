from pathlib import Path
from io import StringIO
import os
import subprocess
import sys

import pytest

from scripts import scan_secrets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_scan_does_not_follow_directory_links(tmp_path, synthetic_credentials):
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside-root"
    root.mkdir()
    outside.mkdir()
    (outside / "credential.txt").write_text(
        synthetic_credentials["github_personal_token"], encoding="utf-8"
    )
    link = root / "linked-directory"
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True, check=True,
        )
        assert result.returncode == 0
        assert link.is_junction()
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        assert list(scan_secrets.iter_repository_files(root)) == []
        assert scan_secrets.scan_repository(root) == []
    finally:
        # Detach this one fixture link; never recursively remove its target.
        if sys.platform == "win32":
            os.rmdir(link)
        else:
            link.unlink()
    assert (outside / "credential.txt").is_file()


@pytest.fixture
def synthetic_credentials() -> dict[str, str]:
    return {
        "aws_access_key_id": "AKIA" + "A" * 16,
        "github_personal_token": "ghp_" + "b" * 36,
        "sk_token": "sk-" + "c" * 24,
        "private_key": "-----BEGIN " + "PRIVATE" + " KEY-----",
        "password_assignment": "correct-horse-battery-staple",
    }


@pytest.fixture
def synthetic_document(synthetic_credentials: dict[str, str]) -> str:
    return "\n".join(
        [
            synthetic_credentials["aws_access_key_id"],
            synthetic_credentials["github_personal_token"],
            synthetic_credentials["sk_token"],
            synthetic_credentials["private_key"],
            "pass" + "word = \"" + synthetic_credentials["password_assignment"] + "\"",
        ]
    )


def test_scan_text_detects_each_pattern_with_line_numbers(synthetic_document: str) -> None:
    findings = scan_secrets.scan_text(synthetic_document, Path("fixture.txt"))

    assert [finding.pattern for finding in findings] == [
        "aws_access_key_id",
        "github_personal_token",
        "sk_token",
        "private_key",
        "password_assignment",
    ]
    assert [finding.line for finding in findings] == [1, 2, 3, 4, 5]


def test_report_redacts_full_credentials(
    synthetic_document: str,
    synthetic_credentials: dict[str, str],
) -> None:
    findings = scan_secrets.scan_text(synthetic_document, Path("fixture.txt"))
    report = scan_secrets.format_report(findings)

    for secret in synthetic_credentials.values():
        assert secret not in report
    assert "fixture.txt:1" in report
    assert "AKIA..." in report
    assert "corr..." in report


def test_stream_scanner_detects_token_split_across_chunks(
    synthetic_credentials: dict[str, str],
) -> None:
    document = "first line\nxx " + synthetic_credentials["github_personal_token"] + "\n"

    findings = scan_secrets.scan_stream(
        StringIO(document),
        Path("fixture.txt"),
        chunk_size=11,
        overlap_size=64,
    )

    assert len(findings) == 1
    assert findings[0].pattern == "github_personal_token"
    assert findings[0].line == 2


@pytest.mark.parametrize(
    "path",
    [
        Path(".git/config"),
        Path(".venv/Lib/site-packages/example.py"),
        Path(".venv-app/Lib/site-packages/example.py"),
        Path(".pytest_cache/state"),
        Path("data/models/model.bin"),
        Path("experiments/results/example.json.gz"),
        Path(".env"),
        Path(".streamlit/secrets.toml"),
        Path("config/credentials.json"),
        Path("secrets/service.key"),
        Path("secrets/service.pem"),
        Path("output/audit/example_llmjudge_blind.csv"),
    ],
)
def test_protected_and_binary_paths_are_excluded(path: Path) -> None:
    assert scan_secrets.should_exclude(path)


def test_clean_report_and_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(scan_secrets, "scan_repository", lambda root: [])

    assert scan_secrets.main(["--root", "."]) == 0
    assert "ninguno" in capsys.readouterr().out


def test_findings_produce_exit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    finding = scan_secrets.Finding("sk_token", Path("fixture.txt"), 7, "sk-x")
    monkeypatch.setattr(scan_secrets, "scan_repository", lambda root: [finding])

    assert scan_secrets.main(["--root", "."]) == 1


def test_versioned_baseline_declares_public_corpus_and_signed_evidence() -> None:
    exclusions = scan_secrets.load_baseline(PROJECT_ROOT / "secrets_baseline.json")

    assert {exclusion.category for exclusion in exclusions} == {
        "corpus",
        "evidencia_firmada",
    }
    assert {exclusion.reason for exclusion in exclusions} == {
        "corpus público descargado de la documentación oficial",
        "evidencia firmada que cita el corpus",
    }


def test_only_baselined_corpus_findings_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = scan_secrets.Finding(
        "aws_access_key_id",
        Path("data/raw/aws/example.json"),
        12,
        "AKIA",
    )
    monkeypatch.setattr(scan_secrets, "scan_repository", lambda root: [finding])

    exit_code = scan_secrets.main(
        ["--root", str(PROJECT_ROOT), "--baseline", "secrets_baseline.json"]
    )
    report = capsys.readouterr().out

    assert exit_code == 0
    assert "1 excluidos por baseline: corpus" in report
    assert "data/raw" not in report


def test_finding_outside_baseline_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finding = scan_secrets.Finding("sk_token", Path("src/service.py"), 7, "sk-x")
    monkeypatch.setattr(scan_secrets, "scan_repository", lambda root: [finding])

    exit_code = scan_secrets.main(
        ["--root", str(PROJECT_ROOT), "--baseline", "secrets_baseline.json"]
    )
    report = capsys.readouterr().out

    assert exit_code == 1
    assert "src\\service.py:7" in report or "src/service.py:7" in report


# ---------------------------------------------------------------- F-015: enumeracion por git
import shutil  # noqa: E402

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git no disponible")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        capture_output=True, check=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path, synthetic_credentials: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    token = synthetic_credentials["github_personal_token"]
    (repo / ".gitignore").write_text("data/ignored/\nlocal_secret.txt\n", encoding="utf-8")
    (repo / "tracked.txt").write_text(token, encoding="utf-8")
    (repo / "deleted.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "deleted.txt").unlink()  # trackeado pero borrado en el arbol de trabajo
    (repo / "untracked_new.txt").write_text(token, encoding="utf-8")  # nuevo, sin commit, no ignorado
    (repo / "local_secret.txt").write_text(token, encoding="utf-8")  # ignorado
    (repo / "data" / "ignored").mkdir(parents=True)
    (repo / "data" / "ignored" / "big.txt").write_text(token, encoding="utf-8")  # directorio ignorado
    (repo / ".worktrees" / "wt").mkdir(parents=True)
    (repo / ".worktrees" / "wt" / "copy.txt").write_text(token, encoding="utf-8")
    return repo


@needs_git
def test_git_enumeration_reads_tracked_and_new_files_but_not_ignored(git_repo: Path) -> None:
    names = scan_secrets.git_file_names(git_repo)
    assert names is not None
    assert {"tracked.txt", "untracked_new.txt"} <= set(names)
    assert not any(n.startswith("data/ignored") or n == "local_secret.txt" for n in names)

    found = {f.path.as_posix() for f in scan_secrets.scan_repository(git_repo, use_git=True)}
    assert found == {"tracked.txt", "untracked_new.txt"}  # detecta ambos; ignora ignorados y .worktrees
    assert scan_secrets.LAST_ENUMERATION == "git"


@needs_git
def test_git_enumeration_survives_a_tracked_file_deleted_from_disk(git_repo: Path) -> None:
    names = scan_secrets.git_file_names(git_repo)
    assert "deleted.txt" in names  # git la sigue listando
    assert "deleted.txt" not in {p.name for p in scan_secrets.iter_git_files(git_repo, names)}


@needs_git
def test_disk_walk_still_finds_ignored_files_so_the_fallback_is_the_conservative_path(
    git_repo: Path,
) -> None:
    found = {f.path.as_posix() for f in scan_secrets.scan_repository(git_repo)}
    assert {"tracked.txt", "untracked_new.txt", "local_secret.txt", "data/ignored/big.txt"} <= found
    assert scan_secrets.LAST_ENUMERATION == "disco"
    assert not any(p.startswith(".worktrees") for p in found)  # .worktrees se excluye en ambos modos


@needs_git
def test_git_listing_is_refused_for_subdirectories_and_non_repos(
    git_repo: Path, tmp_path: Path, synthetic_credentials: dict[str, str]
) -> None:
    assert scan_secrets.git_file_names(git_repo / "data") is None  # no es la raiz del repo
    plain = tmp_path / "no-git"
    plain.mkdir()
    (plain / "a.txt").write_text(synthetic_credentials["github_personal_token"], encoding="utf-8")
    assert scan_secrets.git_file_names(plain) is None
    # y con use_git el escaneo cae al recorrido en disco sin perder el hallazgo
    assert [f.path.as_posix() for f in scan_secrets.scan_repository(plain, use_git=True)] == ["a.txt"]
    assert scan_secrets.LAST_ENUMERATION == "disco"


@needs_git
def test_cli_without_root_uses_git_and_with_root_walks_the_disk(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    modes = []
    real = scan_secrets.scan_repository
    monkeypatch.setattr(scan_secrets, "scan_repository", lambda root, use_git=False: modes.append(use_git) or [])
    scan_secrets.main([])
    scan_secrets.main(["--root", str(git_repo)])
    assert modes == [True, False]
    monkeypatch.setattr(scan_secrets, "scan_repository", real)
