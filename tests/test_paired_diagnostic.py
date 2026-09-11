"""Prospective paired diagnosis is separate from gate acceptance and service scope."""
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import measure_interview_gate as gate
from scripts import run_lexical_diagnostic as paired
from scripts.lexical_diagnostic import paired_schedule


def rows():
    return [dict(system=s, phase='warm', index=i, status='success', elapsed_s=40,
                 consumes_slot=True) for s, i in paired_schedule()]


def test_full_diagnostic_never_declares_interview_go():
    result = paired.report(rows())
    assert result['complete'] and result['new_interview_verdict'] is None
    assert len(result['cells']) == 2 and all(c['valid'] == 20 for c in result['cells'])


@pytest.mark.parametrize('kind', ['error', 'aborted', 'invalid'])
def test_bad_attempt_consumes_position_but_never_enters_percentiles(kind):
    records = rows()
    records[0].update(elapsed_s=999)
    if kind == 'invalid':
        records[0]['conditions_invalid'] = True
    else:
        records[0]['status'] = kind
    result = paired.report(records)
    assert result['complete'] and result['cells'][0]['valid'] == 19
    assert result['cells'][0]['p95_s'] == 40
    assert ('hybrid', 0) not in paired.remaining(records)


def test_duplicate_or_outside_slots_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        paired.remaining(rows() + [rows()[0]])
    with pytest.raises(ValueError, match='outside'):
        paired.remaining([dict(system='semantic', phase='warm', index=0, status='success')])


def prepare(tmp_path, monkeypatch):
    from tests.test_warm_gate_protocol import protocol
    recipe = dict(protocol(), queries=[dict(question=f'q{i}') for i in range(20)], build_id='test')
    gate.write_new(tmp_path / 'source-manifest.json', dict(protocol=recipe))
    monkeypatch.setattr(paired, 'check_protocol', lambda p: None)
    monkeypatch.setenv('OLLAMA_HOST', 'http://localhost:11434')
    monkeypatch.setenv('CLOUDRAG_GATE_DEADLINE', (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    captured = []
    subject = SimpleNamespace(llm=SimpleNamespace(cache_enabled=False, seed=42, max_retries=1,
        timeout=60, read_timeout=180, default_keep_alive='30m', _ollama_client=object()),
        config=SimpleNamespace(model_dump=lambda **kwargs: {}))
    preparation = SimpleNamespace(prepare=lambda scope: {'id': 'ready'}, ready=lambda scope: True,
        pipeline=lambda system, scope: subject, pipelines={'hybrid': subject, 'lexical': subject},
        last_check={'resident': True})
    def measure(root, metadata, pipeline, **kwargs):
        assert kwargs['before_query']() is pipeline
        captured.append((metadata['system'], metadata['index']))
        record = dict(metadata, status='success', elapsed_s=1)
        gate.write_new(root / 'attempts' / str(len(captured)) / 'result.json', record)
        return record
    monkeypatch.setattr(paired, 'measure_traced_attempt', measure)
    return recipe, preparation, captured


def test_executor_prepares_once_and_runs_exactly_counterbalanced_positions(tmp_path, monkeypatch):
    recipe, preparation, calls = prepare(tmp_path, monkeypatch)
    assert paired.execute(tmp_path, recipe, preparation, observer_factory=lambda *a, **k: None)['complete']
    assert calls == paired_schedule()
    assert len(list((tmp_path / 'warmups').glob('*.json'))) == 1


def test_residency_loss_stops_without_query(tmp_path, monkeypatch):
    recipe, preparation, calls = prepare(tmp_path, monkeypatch)
    preparation.ready = lambda scope: False
    with pytest.raises(RuntimeError, match='residency'):
        paired.execute(tmp_path, recipe, preparation)
    assert not calls


def test_deadline_stops_before_preparation(tmp_path, monkeypatch):
    recipe, preparation, calls = prepare(tmp_path, monkeypatch)
    monkeypatch.setenv('CLOUDRAG_GATE_DEADLINE', datetime.now(timezone.utc).isoformat())
    with pytest.raises(TimeoutError):
        paired.execute(tmp_path, recipe, preparation)
    assert not calls and not list((tmp_path / 'warmups').glob('*.json'))


def test_invalid_conditions_stop_immediately(tmp_path, monkeypatch):
    recipe, preparation, calls = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(paired, 'measure_traced_attempt', lambda *a, **k: dict(status='success', conditions_invalid=True))
    with pytest.raises(RuntimeError, match='failed/invalid'):
        paired.execute(tmp_path, recipe, preparation, observer_factory=lambda *a, **k: None)


def test_synthetic_trace_preserves_hash_result():
    assert paired.synthetic_work(bytes(1024), 1, False) == paired.synthetic_work(bytes(1024), 1, True)


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell policy')
def test_ancestry_requires_valid_overlay_and_non_reused_parent_pid():
    path = str(gate.PROJECT / 'scripts/manage_gate_window.ps1').replace("'", "''")
    command = """
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('PATH',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors | Out-String)}
$node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-NvidiaAncestry'},$true)
. ([scriptblock]::Create($node.Extent.Text))
$rows=@(@{Name='NVIDIA Overlay.exe';ProcessId=3;ParentProcessId=2;CreationDate=3},@{Name='nvcontainer.exe';ProcessId=2;ParentProcessId=1;CreationDate=2},@{Name='nvcontainer.exe';ProcessId=1;ParentProcessId=0;CreationDate=1})
$valid=Test-NvidiaAncestry $rows 3 1
$rows[1].CreationDate=4
$reused=Test-NvidiaAncestry $rows 3 1
@{valid=$valid;reused=$reused} | ConvertTo-Json
""".replace('PATH', path)
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'valid': True, 'reused': False}


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell empty proof')
def test_no_relaunch_proof_serializes_empty_ids_under_strict_mode():
    script = str(gate.PROJECT / 'scripts/manage_gate_window.ps1').replace("'", "''")
    command = """
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest;$rootFull='C:/synthetic'
$ast=[System.Management.Automation.Language.Parser]::ParseFile('SCRIPT',[ref]$null,[ref]$null)
$node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Save-OverlayProof'},$true)
. ([scriptblock]::Create($node.Extent.Text))
function Save-New($Path,$Value) { $Value | ConvertTo-Json -Depth 5 }
Save-OverlayProof ([DateTime]::UtcNow) @{name='NvContainerLocalSystem'} @() @()
""".replace('SCRIPT', script)
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['linked_pids'] == [] and data['processes'] == []


def test_managed_diagnostic_uses_specific_payload_and_never_legacy_cohort(tmp_path, monkeypatch):
    from tests.test_managed_gate import prepare as managed_prepare
    from scripts import run_managed_gate as managed
    from tests.test_warm_gate_protocol import protocol
    managed_prepare(tmp_path, monkeypatch)
    cohort = tmp_path / 'cohort'
    gate.write_new(cohort / 'source-manifest.json', dict(protocol=dict(protocol(), diagnostic_only=True, artifact_manifest_path='fixed')))
    path = tmp_path / 'window.json'
    window = gate.read_json(path)
    window.update(lexical_diagnostic=True, cohort=str(cohort))
    path.write_text(json.dumps(window))
    monkeypatch.setattr('scripts.observe_interview_gate.admission', lambda root: None)
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, 'run', command)
    managed.run(tmp_path)
    assert [c[1:3] for c in commands][-2:] == [['scripts/run_lexical_diagnostic.py', 'contrast'], ['scripts/run_lexical_diagnostic.py', 'run']]
    assert gate.read_json(tmp_path / 'payload-complete.json')['diagnostic_only']


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell restoration')
@pytest.mark.parametrize('intervened', [False, True])
def test_diagnostic_restore_only_touches_recorded_nv_service(tmp_path, intervened):
    script = str(gate.PROJECT / 'scripts/manage_gate_window.ps1').replace("'", "''")
    root = str(tmp_path).replace("'", "''")
    gate.write_new(tmp_path / 'window.json', dict(lexical_diagnostic=True, manage_anydesk=False,
        simulated=False, services=[dict(name='NvContainerLocalSystem')], tasks=[], interactive=[],
        task_name='synthetic-unused', untouched_services=[dict(name='AnyDesk', state='Stopped', start_mode='Auto')]))
    if intervened:
        gate.write_new(tmp_path / 'nvcontainer-intent.json', {})
    command = """
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$rootFull='ROOT'; $global:touched=@()
$ast=[System.Management.Automation.Language.Parser]::ParseFile('SCRIPT',[ref]$null,[ref]$null)
foreach($name in @('Save-New','Read-Json','Restore-Window')) {
 $node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
 . ([scriptblock]::Create($node.Extent.Text))
}
function Event($Kind,$Data) {}
function Restore-Service($Before,$Simulated) {$global:touched += $Before.name}
function Get-CimInstance($ClassName,$Filter) { @{State='Stopped';StartMode='Auto'} }
function Get-ScheduledTask($TaskName) { $null }
function Set-Service { throw 'Forbidden real mutation' }
function Start-Service { throw 'Forbidden real mutation' }
function Stop-Service { throw 'Forbidden real mutation' }
Restore-Window
Restore-Window
@{touched=@($global:touched);restored=(Test-Path -LiteralPath (Join-Path $rootFull 'restored.json'))} | ConvertTo-Json
""".replace('ROOT', root).replace('SCRIPT', script)
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['restored'] and data['touched'] == (['NvContainerLocalSystem'] if intervened else [])
