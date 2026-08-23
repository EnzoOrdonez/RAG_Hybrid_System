"""Static guards for the repository's CPU-only GitHub Actions workflow."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
LOCKFILE = PROJECT_ROOT / "requirements-lock.txt"


def test_ci_targets_main_and_summer_without_services():
    doc = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert doc["on"]["push"]["branches"] == ["main", "summer/*"]
    assert doc["on"]["pull_request"]["branches"] == ["main", "summer/*"]
    assert list(doc["jobs"]) == ["test"]
    assert "services" not in doc["jobs"]["test"]


def test_ci_uses_python_314_and_cpu_only_test_expression():
    source = WORKFLOW.read_text(encoding="utf-8")

    assert 'python-version: "3.14"' in source
    assert "python -m pip install -r requirements.txt" in source
    assert 'python -m pytest tests/ -q -m "not slow and not gpu"' in source
    assert "ollama" not in source.lower()
    assert "cuda" not in source.lower()


def test_ci_runs_redacting_secret_scan_before_tests():
    doc = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = doc["jobs"]["test"]["steps"]
    commands = [step.get("run", "") for step in steps]

    scan_index = commands.index("python scripts/scan_secrets.py")
    test_index = commands.index('python -m pytest tests/ -q -m "not slow and not gpu"')
    assert scan_index < test_index


def test_lockfile_is_pinned_and_alphabetized():
    lines = [
        line
        for line in LOCKFILE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert lines == sorted(lines, key=str.casefold)
    assert all("==" in line or line.startswith("-e ") for line in lines)
    assert any(line.startswith("pytest==") for line in lines)
