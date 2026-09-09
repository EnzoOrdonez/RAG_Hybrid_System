"""Exercise process-tree ownership and orchestration without touching real services."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from scripts import measure_interview_gate as gate
from scripts import run_managed_gate as managed


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows job object')
def test_killing_job_owner_terminates_grandchild(tmp_path):
    marker = tmp_path / 'child.json'
    code = '''import json, subprocess, sys, time
from pathlib import Path
from scripts.gate_job import enter
identity = enter()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps(dict(owner=identity, child=child.pid)))
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', code, str(marker)], cwd=gate.PROJECT)
    child = None
    try:
        end = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < end:
            assert parent.poll() is None, 'Job owner exited before spawning child'
            time.sleep(.05)
        identity = json.loads(marker.read_text())
        assert identity['owner']['creation_filetime'] > 0
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        child = kernel.OpenProcess(0x100000, False, identity['child'])
        assert child
        parent.kill()
        parent.wait(timeout=5)
        assert kernel.WaitForSingleObject(child, 5000) == 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child:
            kernel.CloseHandle(child)


def prepare(tmp_path, monkeypatch, *, expired=False):
    gate.write_new(tmp_path / 'window.json', dict(build_id='build', model_digest='d' * 64,
        id='technical', trusted_manifest='original.json'))
    gate.write_new(tmp_path / 'armed.json', dict(deadline_utc=(datetime.now(timezone.utc) +
        timedelta(seconds=-1 if expired else 7200)).isoformat()))
    monkeypatch.setattr(managed.gate_job, 'enter', lambda: dict(pid=1, creation_filetime=1))
    monkeypatch.setattr(gate, 'git', lambda *args: 'build')
    monkeypatch.setattr(gate, 'all_records', lambda *args: [])
    monkeypatch.setattr(gate, 'summarize', lambda *args: dict(complete=True))
    # managed.run updates the process environment; pytest restores every affected variable.
    for name in ('CLOUDRAG_MODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHASHSEED',
                 'PYTHONUTF8', 'CUDA_VISIBLE_DEVICES', 'CLOUDRAG_MEMORY_TRACE', 'CLOUDRAG_MODEL_DIGEST',
                 'CLOUDRAG_BUILD_ID', 'CLOUDRAG_SESSION_DIR', 'CLOUDRAG_ARTIFACT_MANIFEST', 'OLLAMA_HOST',
                 'CLOUDRAG_GATE_DEADLINE', 'CLOUDRAG_GATE_WINDOW_ID'):
        monkeypatch.setenv(name, os.environ.get(name, ''))


def test_managed_payload_stops_before_contrast_when_admission_fails(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        assert kwargs['timeout'] <= 7200
        return subprocess.CompletedProcess(args, 0)

    def reject(root):
        raise RuntimeError('GPU busy')

    monkeypatch.setattr(subprocess, 'run', command)
    monkeypatch.setattr('scripts.observe_interview_gate.admission', reject)
    with pytest.raises(RuntimeError, match='GPU busy'):
        managed.run(tmp_path)
    assert [c[2] for c in commands] == ['verify', 'snapshot', 'verify']
    assert not (tmp_path / 'payload-complete.json').exists()


def test_expired_window_runs_no_commands(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, expired=True)
    with pytest.raises(TimeoutError, match='deadline'):
        managed.run(tmp_path)
    assert not list(tmp_path.glob('*.log'))


def test_bounded_window_uses_one_manifest_and_does_not_claim_cohort_complete(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    from tests.test_warm_gate_protocol import protocol
    cohort = tmp_path / 'shared'
    gate.write_new(cohort / 'source-manifest.json', dict(protocol=dict(protocol(), artifact_manifest_path='fixed.json')))
    window_path = tmp_path / 'window.json'
    window = gate.read_json(window_path)
    window.update(cohort=str(cohort), system='lexical', phase='warm')
    window_path.write_text(json.dumps(window))  # synthetic fixture only
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        assert kwargs['env']['CLOUDRAG_ARTIFACT_MANIFEST'] == 'fixed.json'
        assert kwargs['env']['CLOUDRAG_GATE_DEADLINE']
        assert kwargs['env']['CLOUDRAG_GATE_WINDOW_ID'] == 'technical'
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, 'run', command)
    monkeypatch.setattr('scripts.observe_interview_gate.admission', lambda root: None)
    monkeypatch.setattr(gate, 'pending', lambda *args: [('semantic', 'cold', 0)])
    monkeypatch.setattr(gate, 'summarize', lambda *args, **kwargs: {'passed': False})
    managed.run(tmp_path)
    assert not any('snapshot' in c or 'init' in c for c in commands)
    assert commands[-1][-4:] == ['--system', 'lexical', '--phase', 'warm']
    report = gate.read_json(tmp_path / 'payload-complete.json')
    assert report['condition_complete'] and not report['cohort_complete']


def test_restored_window_cannot_run_again(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    gate.write_new(tmp_path / 'restored.json', {})
    with pytest.raises(RuntimeError, match='not armed'):
        managed.run(tmp_path)


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell Windows guard')
def test_non_admin_manager_refuses_without_files(tmp_path):
    import ctypes
    if ctypes.windll.shell32.IsUserAnAdmin():
        pytest.skip('Explicit non-admin contract')
    result = subprocess.run(['powershell', '-NoProfile', '-File',
        str(Path(gate.PROJECT) / 'scripts/manage_gate_window.ps1'), '-Mode', 'SelfTest',
        '-Root', str(tmp_path / 'window')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Administrator token required' in result.stderr
    assert not (tmp_path / 'window').exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell Windows guard')
@pytest.mark.parametrize('root', [gate.PROJECT, gate.PROJECT.parent.parent])
def test_manager_rejects_checkout_as_evidence_root(root):
    result = subprocess.run(['powershell', '-NoProfile', '-File',
        str(gate.PROJECT / 'scripts/manage_gate_window.ps1'), '-Mode', 'Run', '-Root', str(root)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Use external evidence root' in result.stderr


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell snapshot regression')
def test_interactive_snapshot_preserves_hashtable_paths_and_deduplicates():
    script = gate.PROJECT / 'scripts/manage_gate_window.ps1'
    command = r'''
$ast=[System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$null,[ref]$null)
$node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Interactive-Snapshot'},$true)
if(-not $node){throw 'Missing function'}
. ([scriptblock]::Create($node.Extent.Text))
$rows=@(@{session=1;name='AnyDesk';path='C:/apps/AnyDesk.exe'},@{session=1;name='AnyDesk';path='C:/apps/AnyDesk.exe'},@{session=0;name='AnyDesk';path='C:/service/AnyDesk.exe'},@{session=1;name='NVIDIA Overlay';path='C:/overlay.exe'})
@(Interactive-Snapshot $rows 1) | ConvertTo-Json
'''
    # Pass path in the script body, not as a command string interpreted as shell code.
    command = command.replace('$args[0]', "'" + str(script).replace("'", "''") + "'")
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'path': 'C:/apps/AnyDesk.exe'}
