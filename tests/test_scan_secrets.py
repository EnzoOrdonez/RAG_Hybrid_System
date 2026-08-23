from pathlib import Path
from io import StringIO

import pytest

from scripts import scan_secrets


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
