"""No Ollama, GPU, privileged task or real service mutations in these tests."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import unattended_diagnostic as runner
from scripts import measure_interview_gate as gate


@pytest.mark.parametrize('name', ['brave.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe', 'NVIDIA Other Overlay.exe'])
def test_preflight_rejects_contamination_with_pid(name):
    assert 'PID 37' in ' '.join(runner.process_reasons([dict(pid=37, name=name, path='x')], [], before_cut=True))


def test_only_canonical_target_overlay_allowed_before_cut():
    target = dict(pid=12, name='NVIDIA Overlay.exe', path=runner.OVERLAY)
    assert runner.process_reasons([target], [12], before_cut=True) == []
    assert runner.process_reasons([target], [12], before_cut=False)
    assert runner.process_reasons([dict(target, path=r'C:\Temp\NVIDIA Overlay.exe')], [], before_cut=True)


def test_gpu_unknown_or_spoofed_system_image_rejected(monkeypatch):
    monkeypatch.setenv('SystemRoot', r'C:\Windows')
    assert runner.process_reasons([], [123])
    assert runner.process_reasons([dict(pid=1, name='dwm.exe', path=r'C:\Temp\dwm.exe')], [1])
    assert not runner.process_reasons([dict(pid=1, name='dwm.exe', path=r'C:\Windows\System32\dwm.exe')], [1])


def restored_window(root):
    window = root / 'windows' / 'w1'
    gate.write_new(window / 'window.json', {'simulated': True})
    gate.write_new(window / 'restored.json', {'simulated': True})
    return window


def test_resume_requires_explicit_new_authorization_after_restore(tmp_path):
    window = restored_window(tmp_path)
    with pytest.raises(PermissionError, match='AuthorizeNewWindow'):
        runner.check_resume(tmp_path, False)
    assert runner.check_resume(tmp_path, True) == [window / 'window.json']


def test_unrestored_window_cannot_resume_even_with_authorization(tmp_path):
    gate.write_new(tmp_path / 'windows/w1/window.json', {})
    with pytest.raises(RuntimeError, match='not verified restored'):
        runner.check_resume(tmp_path, True)


def test_energy_abort_is_terminal_and_never_relabels_raw_files(tmp_path):
    window = restored_window(tmp_path)
    row = runner.synthetic_row('hybrid', 0)
    path = tmp_path / 'cohort/attempts/a/result.json'
    gate.write_new(path, row)
    before = path.read_bytes()
    gate.write_new(window / 'energy-event.json', {'event': 'forced_sleep'})
    result = runner.package(tmp_path)
    assert result['energy_aborted'] and not result['complete']
    assert result['groups'][0]['valid'] == 0
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match='terminal'):
        runner.check_resume(tmp_path, True)


@pytest.mark.parametrize('reason', ['prohibited_process', 'external_cpu_load', 'foreign_gpu_or_prohibited_process'])
def test_only_new_protocol_continues_ordinary_invalid_response(reason):
    row = dict(status='success', conditions_invalid=True, control_reasons=[reason])
    assert runner.should_stop(row, unattended=False)
    assert not runner.should_stop(row, unattended=True)


@pytest.mark.parametrize('reason', ['telemetry_persistence_error', 'concurrent_model', 'telemetry_gap', 'power_mode_changed'])
def test_fatal_controls_stop_even_new_protocol(reason):
    assert runner.should_stop(dict(status='success', conditions_invalid=True, control_reasons=[reason]), True)


def test_live_invalidity_sticky_and_durable(tmp_path, monkeypatch):
    from scripts import observe_interview_gate as observe
    monkeypatch.setattr(observe, 'assess', lambda rows: ['prohibited_process'] if rows[0]['bad'] else [])
    runner.live_sample(tmp_path, dict(at='first', bad=True, processes=[]))
    initial = (tmp_path / 'invalid-live.json').read_bytes()
    runner.live_sample(tmp_path, dict(at='later', bad=False, processes=[]))
    assert (tmp_path / 'invalid-live.json').read_bytes() == initial


def test_summary_excludes_invalid_and_uses_real_nli_counts():
    valid = runner.synthetic_row('lexical', 0)
    invalid = runner.synthetic_row('lexical', 1, invalid=True)
    invalid['elapsed_s'] = 900
    result = runner.summarize([valid, invalid])
    lexical = result['groups'][1]
    assert lexical['valid'] == lexical['invalid'] == 1
    assert lexical['metrics']['total_s']['p95'] == .001
    assert lexical['metrics']['nli_pairs']['p95'] == 1
    assert not result['confirmation_ready']


def test_package_tamper_missing_and_escape_rejected(tmp_path):
    restored_window(tmp_path)
    source = tmp_path / 'cohort/attempts/a/result.json'
    gate.write_new(source, runner.synthetic_row('hybrid', 0))
    runner.package(tmp_path)
    runner.verify_package(tmp_path)
    original = source.read_bytes()
    source.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_package(tmp_path)
    source.write_bytes(original)
    extra = tmp_path / 'unexpected.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='inventory'):
        runner.verify_package(tmp_path)
    extra.unlink()
    manifest = gate.read_json(tmp_path / 'manifest.json')
    manifest['files']['../escape'] = {'sha256': 'wrong'}
    runner.replace_view(tmp_path / 'manifest.json', manifest)
    (tmp_path / 'manifest.sha256').write_text(gate.digest(tmp_path / 'manifest.json'))
    with pytest.raises(ValueError, match='missing'):
        runner.verify_package(tmp_path)


def test_summary_repack_keeps_immutable_versions(tmp_path):
    restored_window(tmp_path)
    runner.package(tmp_path)
    old = {p: p.read_bytes() for p in (tmp_path / 'packages').rglob('*.json')}
    runner.package(tmp_path)
    assert all(p.read_bytes() == content for p, content in old.items())
    runner.verify_package(tmp_path)


def test_deadline_has_no_grace_for_inference():
    now = datetime.now(timezone.utc)
    assert not runner.deadline_allows(now, now)
    assert not runner.deadline_allows(now-timedelta(seconds=1), now)
    assert runner.deadline_allows(now+timedelta(seconds=1), now)


def test_expired_synthetic_execution_never_calls_work(tmp_path):
    called = []
    with pytest.raises(TimeoutError):
        runner.synthetic_execute(tmp_path, datetime(2000, 1, 1, tzinfo=timezone.utc),
                                 work=lambda *args: called.append(args))
    assert not called and not gate.local_records(tmp_path)


def test_disk_preflight_refuses_before_intervention(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, 'inventory', lambda: dict(processes=[], gpu_pids=[], anydesk='Stopped'))
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda root: SimpleNamespace(free=1024))
    with pytest.raises(RuntimeError, match='10 GiB'):
        runner.preflight(tmp_path, True)


def test_unattended_observer_flag_cannot_be_omitted(monkeypatch):
    from scripts.run_lexical_diagnostic import check_protocol
    monkeypatch.delenv('CLOUDRAG_UNATTENDED', raising=False)
    with pytest.raises(ValueError, match='observer policy'):
        check_protocol({'unattended_policy': runner.POLICY})


def test_actual_observer_publishes_sticky_invalidation_before_finish(tmp_path, monkeypatch):
    from scripts import observe_interview_gate as observe
    monkeypatch.setenv('CLOUDRAG_UNATTENDED', '1')
    monkeypatch.setattr(observe, 'assess', lambda rows, **kwargs: ['prohibited_process'] if rows[-1]['bad'] else [])
    state = {'bad': False}
    observer = observe.Observer(tmp_path / 'samples.jsonl', interval=100, capture_enabled=False,
                                sampler=lambda: dict(at=gate.now(), bad=state['bad'], processes=[]))
    observer.start()
    try:
        state['bad'] = True
        observer.sample()
        assert (tmp_path / 'samples.control/invalid-live.json').exists()
        state['bad'] = False
    finally:
        result = observer.finish()
    assert result['conditions_invalid'] and 'prohibited_process' in result['control_reasons']


def test_observer_disk_failure_is_fatal(tmp_path, monkeypatch):
    from scripts import observe_interview_gate as observe
    monkeypatch.setenv('CLOUDRAG_UNATTENDED', '1')
    monkeypatch.setattr(runner, 'live_sample', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    observer = observe.Observer(tmp_path / 'samples.jsonl', capture_enabled=False,
                                sampler=lambda: dict(at=gate.now(), processes=[]))
    observer.sample()
    assert observer.persistence_errors
    assert runner.should_stop(dict(status='success', observer_errors=observer.persistence_errors), True)


def test_resume_keeps_all_logical_slots_once():
    from scripts.run_lexical_diagnostic import remaining
    rows = [runner.synthetic_row(*slot) for slot in runner.paired_schedule()[:7]]
    rows += [runner.synthetic_row(*slot) for slot in remaining(rows)]
    assert len(rows) == 40 and remaining(rows) == []
    with pytest.raises(ValueError):
        remaining(rows + [rows[0]])


@pytest.mark.parametrize('protocol,expected', [({'unattended_policy': runner.POLICY}, '1'), ({}, 'absent')])
def test_child_command_receives_registered_observer_policy(protocol, expected):
    from scripts.run_managed_gate import observer_environment
    env = observer_environment(dict(os.environ, CLOUDRAG_UNATTENDED='stale'), protocol)
    child = subprocess.run([sys.executable, '-c',
                            'import os; print(os.environ.get("CLOUDRAG_UNATTENDED", "absent"))'],
                           env=env, capture_output=True, text=True, check=True)
    assert child.stdout.strip() == expected


def test_actual_executor_continues_invalid_slot_only_in_new_protocol(tmp_path, monkeypatch):
    from tests.test_paired_diagnostic import prepare
    from scripts import run_lexical_diagnostic as paired
    recipe, preparation, calls = prepare(tmp_path, monkeypatch)
    recipe['unattended_policy'] = runner.POLICY
    def invalid_first(root, metadata, pipeline, **kwargs):
        assert kwargs['before_query']() is pipeline
        calls.append((metadata['system'], metadata['index']))
        row = dict(metadata, status='success', elapsed_s=1, conditions_invalid=len(calls) == 1,
                   control_reasons=['prohibited_process'] if len(calls) == 1 else [])
        gate.write_new(root / 'attempts' / str(len(calls)) / 'result.json', row)
        return row
    monkeypatch.setattr(paired, 'measure_traced_attempt', invalid_first)
    result = paired.execute(tmp_path, recipe, preparation, observer_factory=lambda *a, **k: None)
    assert result['complete'] and result['cells'][0]['invalid'] == 1
    assert calls == runner.paired_schedule()


@pytest.mark.parametrize('marker', ['trace-identity.json', 'trace-active.json'])
def test_pending_owned_trace_blocks_new_intervention(tmp_path, marker):
    restored_window(tmp_path)
    path = tmp_path / 'cohort/telemetry/a' / marker
    gate.write_new(path, {})
    with pytest.raises(RuntimeError, match='ETW'):
        runner.check_resume(tmp_path, True)
    gate.write_new(path.with_name('trace-recovered.json'), {})
    assert runner.check_resume(tmp_path, True)


def test_fast_dry_run_kills_child_and_packages_without_models(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'external_root', Path)
    monkeypatch.setattr(runner, 'inventory', lambda: pytest.fail('No real system inspection in dry-run'))
    root = tmp_path / 'synthetic'
    runner.dry_run(root)
    result = gate.read_json(root / 'dry-run.json')
    assert all(result['checks'].values())
    assert gate.read_json(root / 'summary.json')['groups'][1]['invalid'] == 1
    runner.verify_package(root)


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell AST/pure energy policy')
def test_powershell_parses_and_power_gap_detects_sleep_or_reboot():
    script = str(gate.PROJECT / 'scripts/manage_gate_window.ps1').replace("'", "''")
    entry = str(gate.PROJECT / 'scripts/run_diagnostic_window.ps1').replace("'", "''")
    code = """
$ErrorActionPreference='Stop'
$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('SCRIPT',[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'manager parse failed'}
[System.Management.Automation.Language.Parser]::ParseFile('ENTRY',[ref]$tokens,[ref]$errors) | Out-Null
if($errors.Count){throw 'entry parse failed'}
$node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-PowerGap'},$true)
. ([scriptblock]::Create($node.Extent.Text))
@{sleep=(Test-PowerGap @{awake_ms=100;elapsed_ms=100} @{awake_ms=200;elapsed_ms=5000});
normal=(Test-PowerGap @{awake_ms=100;elapsed_ms=100} @{awake_ms=200;elapsed_ms=200});
reboot=(Test-PowerGap @{awake_ms=100;elapsed_ms=100} @{awake_ms=1;elapsed_ms=1})} | ConvertTo-Json
""".replace('SCRIPT', script).replace('ENTRY', entry)
    result = subprocess.run(['powershell', '-NoProfile', '-Command', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(sleep=True, normal=False, reboot=True)


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell restoration policy')
def test_watchdog_restores_even_if_event_disk_write_fails():
    script = str(gate.PROJECT / 'scripts/manage_gate_window.ps1').replace("'", "''")
    code = """
$ErrorActionPreference='Stop'
$errors=$null;$tokens=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('SCRIPT',[ref]$tokens,[ref]$errors)
$node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-WatchRestore'},$true)
. ([scriptblock]::Create($node.Extent.Text))
$script:restored=0
function Event { throw 'disk full' }
function Restore-Window { $script:restored++ }
Invoke-WatchRestore @{expired=$true} 3>$null
if($script:restored -ne 1){throw 'Restoration skipped'}
""".replace('SCRIPT', script)
    result = subprocess.run(['powershell', '-NoProfile', '-Command', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
