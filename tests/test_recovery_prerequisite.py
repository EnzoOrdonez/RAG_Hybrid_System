"""The recovery prerequisite must fail closed without an administrator token."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.measure_interview_gate import read_json

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_gate_recovery.ps1'


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows recovery prerequisite')
def test_non_admin_cannot_attempt_system_registration(tmp_path):
    check = subprocess.check_output(['powershell', '-NoProfile', '-Command',
        '([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)'], text=True).strip()
    if check == 'True':
        pytest.skip('Test requires a non-elevated token; never register real tasks in the suite')
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(SCRIPT), '-Root', str(tmp_path)], capture_output=True)
    assert result.returncode != 0
    # Match the ASCII contract without decoding native-codepage diagnostics.
    assert b'Administrator token required' in result.stderr
    assert read_json(tmp_path / 'invocation.json')['administrator'] is False
    assert {p.name for p in tmp_path.iterdir()} == {'invocation.json'}


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows ETW privilege prerequisite')
def test_memory_probe_refuses_non_admin_before_creating_files(tmp_path):
    check = subprocess.check_output(['powershell', '-NoProfile', '-Command',
        '([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)'], text=True).strip()
    if check == 'True':
        pytest.skip('Never start ETW from the non-admin refusal test')
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', str(SCRIPT.with_name('probe_gate_memory.ps1')), '-Root', str(tmp_path / 'new')],
        capture_output=True)
    assert result.returncode != 0
    assert b'Administrator token required' in result.stderr
    assert not (tmp_path / 'new').exists()
