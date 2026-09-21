"""Human-launched diagnostic: policy, durable packaging and fast synthetic checks.

No real inference or Windows service changes occur in dry-run. Real intervention
belongs exclusively to the reviewed PowerShell supervisor.
"""
import argparse
from datetime import datetime, timezone
import json
import ntpath
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from scripts import measure_interview_gate as gate
from scripts.lexical_diagnostic import paired_schedule

POLICY = 'unattended-paired-v1'
OVERLAY = r'C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe'
DIGEST = '444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852'
BROWSERS = {'brave', 'chrome', 'msedge', 'firefox', 'opera', 'epicgameslauncher', 'steam'}
WINDOWS_GPU = {'dwm.exe', 'explorer.exe', 'shellexperiencehost.exe',
               'startmenuexperiencehost.exe', 'searchhost.exe', 'textinputhost.exe', 'lockapp.exe'}
SOURCES = ('scripts/unattended_diagnostic.py', 'scripts/run_diagnostic_window.ps1',
           'scripts/manage_gate_window.ps1', 'scripts/run_managed_gate.py',
           'scripts/run_lexical_diagnostic.py', 'scripts/lexical_diagnostic.py',
           'scripts/observe_interview_gate.py')


def normalized(path):
    return ntpath.normcase(ntpath.normpath(path or ''))


def process_reasons(processes, gpu_pids, *, before_cut=False):
    """Closed image/path policy; unknown GPU PIDs fail closed, never get killed."""
    reasons = []
    by_pid = {p['pid']: p for p in processes}
    windows = normalized(os.environ.get('SystemRoot', r'C:\Windows')) + '\\'
    ollama = normalized(ntpath.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe'))
    for p in processes:
        name = p['name'].lower().removesuffix('.exe')
        target = name == 'nvidia overlay' and normalized(p.get('path')) == normalized(OVERLAY)
        if name in BROWSERS or name == 'anydesk' or ('overlay' in name and not (before_cut and target)):
            reasons.append(f"close {p['name']} PID {p['pid']}")
    for pid in gpu_pids:
        p = by_pid.get(pid)
        if p is None:
            reasons.append(f'unknown GPU PID {pid}; identify and close before launch')
            continue
        path = normalized(p.get('path'))
        name = p['name'].lower()
        if not name.endswith('.exe'):
            name += '.exe'
        allowed = (name in WINDOWS_GPU and path.startswith(windows))
        allowed |= name == 'ollama.exe' and path == ollama
        allowed |= before_cut and name == 'nvidia overlay.exe' and path == normalized(OVERLAY)
        allowed |= (name in ('windowsterminal.exe', 'openconsole.exe') and
                    path.startswith(r'c:\program files\windowsapps\microsoft.windowsterminal_'))
        if not allowed:
            reasons.append(f"foreign GPU process {p['name']} PID {pid}: {p.get('path')}")
    return sorted(set(reasons))


def inventory():
    command = """
@{processes=@(Get-CimInstance Win32_Process | ForEach-Object {
 @{pid=$_.ProcessId;name=$_.Name;path=$_.ExecutablePath;created=$_.CreationDate}});
 anydesk=(Get-Service AnyDesk).Status.ToString()} | ConvertTo-Json -Depth 5
"""
    data = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True))
    listing = subprocess.check_output(['nvidia-smi'], text=True)
    if 'Processes:' not in listing:
        raise RuntimeError('GPU process listing unavailable; refuse admission')
    table = listing.split('Processes:', 1)[1]
    pids = [int(m.group(1)) for m in re.finditer(r'\|\s*\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+[CG+]+\s', table)]
    if not pids and 'No running processes found' not in table:
        raise RuntimeError('Unrecognized GPU process table; fail closed')
    return dict(at=gate.now(), **data, gpu_pids=pids, gpu_listing=listing)


def preflight(root, before_cut):
    data = inventory()
    reasons = process_reasons(data['processes'], data['gpu_pids'], before_cut=before_cut)
    if data['anydesk'] != 'Stopped':
        reasons.append('AnyDesk must be stopped; this script never changes it')
    if shutil.disk_usage(root).free < 10 * 1024**3:
        reasons.append('need at least 10 GiB free on evidence volume')
    gate.write_new(Path(root) / 'preflight' / f'{uuid.uuid4().hex}.json', dict(data, reasons=reasons))
    if reasons:
        raise RuntimeError('Pre-flight rejected:\n' + '\n'.join(reasons))
    return data


def external_root(root):
    root = Path(root).resolve()
    checkout = (gate.PROJECT / gate.git('rev-parse', '--git-common-dir')).resolve().parent
    if root == checkout or root.is_relative_to(checkout):
        raise ValueError('Evidence must be outside checkout')
    return root


def check_resume(root, authorize):
    if (root / 'energy-aborted.json').exists() or list((root / 'windows').glob('*/energy-event.json')):
        raise RuntimeError('Energy-aborted cohort is terminal; package only, use a NEW cohort')
    windows = list((root / 'windows').glob('*/window.json'))
    if any(not p.with_name('restored.json').exists() for p in windows):
        raise RuntimeError('Previous window not verified restored; restore it before resuming')
    if pending_traces(root):
        raise RuntimeError('Owned ETW trace cleanup unverified; restore/clean previous window first')
    if windows and not authorize:
        raise PermissionError('Resume after restoration requires -AuthorizeNewWindow')
    return windows


def pending_traces(root):
    identities = set(Path(root).rglob('trace-identity.json'))
    identities.update(p.with_name('trace-identity.json') for p in Path(root).rglob('trace-active.json'))
    return [str(p.parent.relative_to(root)) for p in sorted(identities)
            if not p.with_name('trace-stopped.json').exists() and not p.with_name('trace-recovered.json').exists()]


def configure(root, nli_experiment=False):
    os.environ.update(CLOUDRAG_MODE='participant', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONHASHSEED='42', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='',
                      CLOUDRAG_MEMORY_TRACE='1', CLOUDRAG_MODEL_DIGEST=DIGEST,
                      CLOUDRAG_UNATTENDED='1',
                      CLOUDRAG_BUILD_ID=gate.git('rev-parse', 'HEAD'), OLLAMA_HOST='http://localhost:11434',
                      CLOUDRAG_ARTIFACT_MANIFEST=str(root / 'deployment-manifest.json'))
    if nli_experiment:
        os.environ['CLOUDRAG_NLI_EXPERIMENT'] = '1'
    else:
        os.environ.pop('CLOUDRAG_NLI_EXPERIMENT', None)


def prepare(root, resume=False, authorize=False, nli_experiment=False):
    from scripts import run_lexical_diagnostic as diagnostic
    if nli_experiment:
        from scripts import nli_batch_experiment as diagnostic
        if not authorize:
            raise PermissionError('Each NLI window requires -AuthorizeNewWindow, including first launch')
    from src.utils.deployment_artifacts import build_manifest, verify_manifest
    root = external_root(root)
    if gate.git('branch', '--show-current') != 'fix/interview-readiness' or gate.git('status', '--porcelain'):
        raise RuntimeError('Expected clean fix/interview-readiness worktree')
    if resume:
        verify_package(root)
        saved = gate.read_json(root / 'cohort/source-manifest.json')['protocol']
        if bool(saved.get('nli_experiment')) != nli_experiment:
            raise ValueError('Explicit experiment flag must match resumed cohort')
        rows = gate.local_records(root / 'cohort')
        if nli_experiment:
            diagnostic.resume_boundary(rows)
            quality = diagnostic.quality_summary(root / 'cohort', rows)
            if quality['failures']:
                raise RuntimeError('Quality failure is terminal; review evidence before any new experiment')
        if not diagnostic.remaining(rows) and (not nli_experiment or quality['passed']):
            print('COMPLETE', flush=True)
            return None
        check_resume(root, authorize)
        protocol = gate.read_json(root / 'cohort/source-manifest.json')['protocol']
        if protocol.get('unattended_policy') != POLICY:
            raise ValueError('Cannot resume historical or different protocol')
    else:
        if root.exists():
            raise FileExistsError('Use a new evidence path or explicit -Resume')
        root.mkdir(parents=True)
    preflight(root, before_cut=True)
    configure(root, nli_experiment)
    if gate.api('/api/version') != {'version': '0.22.1'}:
        raise RuntimeError('Ollama version mismatch')
    gate.check_model()
    if resume:
        diagnostic.check_protocol(protocol)
    else:
        trusted = 'C:/CloudRAG/operational-20260905T1428Z/deployment-manifest.json'
        verify_manifest(gate.PROJECT, trusted)
        gate.write_new(root / 'deployment-manifest.json', build_manifest(gate.PROJECT))
        protocol = gate.fresh_protocol(systems=list(gate.SYSTEMS), controlled=True)
        protocol.update(diagnostic_only=True, diagnostic_schedule=paired_schedule(),
                        unattended_policy=POLICY, diagnostic_sources={p: gate.digest(gate.PROJECT / p) for p in SOURCES})
        if nli_experiment:
            diagnostic.register(protocol)
        gate.write_new(root / 'cohort/source-manifest.json', dict(created_at=gate.now(), mode='unattended',
                       protocol=protocol, total_attempts=120 if nli_experiment else 40, abort_consumes_slot=True))
    # Real allocated bytes, not a sparse reservation. Restoration releases only this owned file.
    reserve = root / 'runtime/emergency-reserve.bin'
    reserve.parent.mkdir(exist_ok=True)
    if not reserve.exists():
        with reserve.open('xb') as stream:
            for _ in range(16):
                stream.write(bytes(1024 * 1024))
            stream.flush()
            os.fsync(stream.fileno())
    window = root / 'windows' / uuid.uuid4().hex
    gate.write_new(root / 'launches' / f'{window.name}.json', dict(at=gate.now(), resume=resume,
                   authorize_new_window=authorize, window=str(window), build=gate.git('rev-parse', 'HEAD')))
    print(str(window), flush=True)
    return window


def metrics(values):
    values = list(values)
    return dict(n=len(values), p50=gate.percentile(values, .5), p95=gate.percentile(values, .95))


def summarize(rows, energy=False):
    from scripts.run_lexical_diagnostic import remaining
    pending = remaining(rows)  # also rejects duplicates/out-of-protocol positions
    groups = []
    for system in ('hybrid', 'lexical'):
        selected = [r for r in rows if r['system'] == system]
        valid = [r for r in selected if r['status'] == 'success' and not energy
                 and not r.get('conditions_invalid') and not r.get('environment_invalid')]
        fields = {k: [] for k in ('total_s', 'retrieval_s', 'reranking_s', 'generation_s', 'nli_s',
                                  'other_s', 'tokens_output', 'claims', 'nli_calls', 'nli_pairs',
                                  'prompt_chars', 'answer_chars', 'chunks')}
        for row in valid:
            trace, response = row['diagnostic_trace'], row['response']
            stages = {s['name']: s['elapsed_s'] for s in trace['stages']}
            if trace['nli_pairs_attempted'] != sum(c['pairs'] for c in trace['nli']):
                raise ValueError('Measured NLI pair counts disagree')
            values = dict(total_s=row['elapsed_s'], retrieval_s=stages['retrieval'],
                          reranking_s=stages['reranking'], generation_s=stages['generation'],
                          nli_s=stages['hallucination_check'], other_s=row['elapsed_s'] - sum(stages.values()),
                          tokens_output=response['llm_response']['tokens_output'],
                          claims=sum(e['count'] for e in trace['extractions']),
                          nli_calls=trace['nli_predict_calls'], nli_pairs=trace['nli_pairs_attempted'],
                          prompt_chars=sum(g['prompt_chars'] for g in trace['generation']),
                          answer_chars=len(response['answer']), chunks=len(response['retrieved_chunks']))
            for key, value in values.items():
                fields[key].append(value)
        groups.append(dict(system=system, attempts=len(selected), valid=len(valid),
                           energy_aborted_records=len(selected) if energy else 0,
                           errors=sum(r['status'] == 'error' for r in selected),
                           aborted=sum(r['status'] == 'aborted' for r in selected),
                           invalid=sum(bool(r.get('conditions_invalid') or r.get('environment_invalid')) for r in selected),
                           metrics={key: metrics(values) for key, values in fields.items()}))
    return dict(diagnostic_only=True, complete=not pending and not energy,
                confirmation_ready=not energy and all(g['valid'] == 20 for g in groups),
                energy_aborted=energy, pending=pending, groups=groups, verdict='NO-GO vigente; diagnostic only',
                note='Stage percentiles are not additive. Invalid/failed/aborted rows never enter percentiles.')


def replace_view(path, value):
    """Only named current views are replaceable; immutable versions remain in packages/."""
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_package(root):
    root = Path(root).resolve()
    manifest = gate.read_json(root / 'manifest.json')
    expected = (root / 'manifest.sha256').read_text().strip()
    if gate.digest(root / 'manifest.json') != expected:
        raise ValueError('Manifest hash mismatch')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.suffix != '.lock' and 'runtime' not in p.relative_to(root).parts
              and p not in (root / 'manifest.json', root / 'manifest.sha256')}
    if actual != set(manifest['files']):
        raise ValueError('Evidence file inventory changed or missing')
    for name, entry in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or gate.digest(path) != entry['sha256']:
            raise ValueError('Evidence changed or missing: ' + name)
    return manifest


def package(root):
    root = Path(root).resolve()
    with FileLock(str(root / 'package.lock'), timeout=0):
        windows = list((root / 'windows').glob('*/window.json'))
        if any(not p.with_name('restored.json').exists() for p in windows):
            raise RuntimeError('Restoration unverified; refuse final package, keep raw evidence')
        energy = bool(list((root / 'windows').glob('*/energy-event.json')))
        if energy and not (root / 'energy-aborted.json').exists():
            gate.write_new(root / 'energy-aborted.json', dict(at=gate.now(), reason='forced suspend/reboot',
                           classification='all partial cohort data aborted; original files immutable'))
        gate.recover(root / 'cohort')
        rows = gate.local_records(root / 'cohort')
        report = summarize
        protocol_file = root / 'cohort/source-manifest.json'
        experiment = protocol_file.exists() and gate.read_json(protocol_file)['protocol'].get('nli_experiment')
        if experiment:
            from scripts.nli_batch_experiment import EXPERIMENT, summarize as report, quality_summary, bootstrap
            if experiment != EXPERIMENT:
                raise ValueError('Unknown NLI experiment')
        summary = dict(at=gate.now(), **report(rows, energy=energy),
                       cleanup_pending=pending_traces(root),
                       windows=[dict(path=p.parent.relative_to(root).as_posix(),
                                state=gate.read_json(p), restored=gate.read_json(p.with_name('restored.json'))) for p in windows])
        if experiment:
            summary['quality'] = quality_summary(root / 'cohort', rows)
            summary['quality_pass'] = summary['quality']['passed']
            summary['confirmation_ready'] = summary['confirmation_ready'] and summary['quality_pass']
            summary['bootstrap'] = bootstrap(rows) if not energy else []
        version = root / 'packages' / uuid.uuid4().hex
        gate.write_new(version / 'summary.json', summary)
        replace_view(root / 'summary.json', summary)
        files = {}
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix == '.lock' or 'runtime' in path.relative_to(root).parts:
                continue
            if path in (root / 'manifest.json', root / 'manifest.sha256'):
                continue
            before = path.stat()
            digest = gate.digest(path)
            if before.st_size != path.stat().st_size or before.st_mtime_ns != path.stat().st_mtime_ns:
                raise RuntimeError('Evidence still changing; retry packaging after controller exits')
            files[path.relative_to(root).as_posix()] = dict(sha256=digest, bytes=before.st_size)
        manifest = dict(at=gate.now(), files=files, version=str(version.relative_to(root)),
                        exclusions=['manifest.json (self-reference)', 'manifest.sha256', '*.lock', 'runtime/'],
                        summary_sha256=gate.digest(root / 'summary.json'))
        gate.write_new(version / 'manifest.json', manifest)
        # Include the immutable manifest snapshot in the current inventory.
        manifest['files'][str((version / 'manifest.json').relative_to(root)).replace('\\', '/')] = dict(
            sha256=gate.digest(version / 'manifest.json'), bytes=(version / 'manifest.json').stat().st_size)
        replace_view(root / 'manifest.json', manifest)
        (root / 'manifest.sha256').write_text(gate.digest(root / 'manifest.json') + '\n')
        print(json.dumps(dict(package=str(root), summary=str(root / 'summary.json'), complete=summary['complete'])))
        return summary


def live_sample(root, sample):
    """Sticky append-only invalidity marker, called by the actual observer every sample."""
    from scripts.observe_interview_gate import assess
    reasons = assess([sample])
    listing = sample.get('gpu_process_listing', '')
    gpu_pids = [int(m.group(1)) for m in re.finditer(r'\|\s*\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+[CG+]+\s', listing)]
    if process_reasons(sample.get('processes', []), gpu_pids):
        reasons.append('foreign_gpu_or_prohibited_process')
    if reasons and not (Path(root) / 'invalid-live.json').exists():
        gate.write_new(Path(root) / 'invalid-live.json', dict(at=sample['at'], reasons=reasons))
        print('INVALID: ' + ', '.join(reasons), flush=True)
    return reasons


def should_stop(row, unattended=False):
    if row['status'] != 'success' or row.get('environment_invalid') or row.get('observer_errors'):
        return True
    reasons = set(row.get('control_reasons', []))
    ordinary = {'prohibited_process', 'external_cpu_load', 'foreign_gpu_or_prohibited_process'}
    return bool(row.get('conditions_invalid') and (not unattended or not reasons or reasons - ordinary))


def progress(root, row, started, initial_count=0):
    count = len(gate.local_records(Path(root)))
    wall = time.monotonic() - started
    print(f"intento {count}/40 | {row['system']} | elapsed {row['elapsed_s']} s | "
          f"ETA {max(0, 40-count)*wall/max(count-initial_count, 1):.0f} s | "
          f"{row['status']} invalid={row.get('conditions_invalid', False)}", flush=True)


def deadline_allows(deadline, now=None):
    return (deadline - (now or datetime.now(timezone.utc))).total_seconds() > 0


def synthetic_execute(root, deadline, work=None):
    """Same slot journal rules, injected work; the expired path never calls work."""
    from scripts.run_lexical_diagnostic import remaining
    for system, index in remaining(gate.local_records(root)):
        if not deadline_allows(deadline):
            raise TimeoutError('Expired: zero further inference calls')
        row = (work or synthetic_row)(system, index)
        gate.write_new(Path(root) / 'attempts' / f'{system}-{index}' / 'result.json', row)


def synthetic_row(system, index, invalid=False):
    return dict(system=system, phase='warm', index=index, status='success', elapsed_s=.001,
                conditions_invalid=invalid, response=dict(answer='synthetic', retrieved_chunks=[{}],
                llm_response=dict(tokens_output=2)), diagnostic_trace=dict(
                    stages=[dict(name=k, elapsed_s=.0001) for k in
                            ('retrieval', 'reranking', 'generation', 'hallucination_check')],
                    nli=[dict(pairs=1)], nli_pairs_attempted=1, nli_predict_calls=1,
                    extractions=[dict(count=1)], generation=[dict(prompt_chars=9)]))


def synthetic_watch(root, pid, deadline):
    """Independent child used only by dry-run/tests; simulates service restoration."""
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if handle:
            try:
                kernel.WaitForSingleObject(handle, max(0, int((deadline-time.time())*1000)))
            finally:
                kernel.CloseHandle(handle)
    else:
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.02)
    gate.write_new(Path(root) / 'restored.json', dict(at=gate.now(), simulated=True, restorer_pid=os.getpid()))


def dry_run(root):
    root = external_root(root)
    root.mkdir(parents=True, exist_ok=False)
    checks = {}
    checks['preflight_rejected'] = bool(process_reasons([dict(pid=1, name='brave.exe', path='x')], []))
    window = root / 'windows/synthetic'
    gate.write_new(window / 'window.json', dict(simulated=True, at=gate.now()))
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    watch = subprocess.Popen([sys.executable, __file__, 'synthetic-watch', '--root', str(window),
                              '--pid', str(child.pid), '--deadline', str(time.time()+5)])
    try:
        child.kill()
        child.wait(timeout=5)
        watch.wait(timeout=8)
        checks['restored_after_kill'] = (window / 'restored.json').exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if watch.poll() is None:
            watch.kill()
            watch.wait()
    try:
        check_resume(root, False)
        checks['authorization_required'] = False
    except PermissionError:
        checks['authorization_required'] = True
    check_resume(root, True)
    called = []
    try:
        synthetic_execute(root / 'expired', datetime(2000, 1, 1, tzinfo=timezone.utc),
                          work=lambda *args: called.append(args))
    except TimeoutError:
        checks['expired_no_inference'] = not called
    for system, index in paired_schedule()[:7]:
        gate.write_new(root / 'cohort/attempts' / f'{system}-{index}' / 'result.json', synthetic_row(system, index))
    synthetic_execute(root / 'cohort', datetime.fromtimestamp(time.time()+30, timezone.utc),
                      work=lambda s, i: synthetic_row(s, i, invalid=(s == 'lexical' and i == 8)))
    summary = package(root)
    checks['resume_without_duplicates'] = sum(g['attempts'] for g in summary['groups']) == 40
    checks['invalid_retained'] = sum(g['invalid'] for g in summary['groups']) == 1
    checks['integrity'] = bool(verify_package(root))
    energy = root / 'scenarios/energy'
    gate.write_new(energy / 'windows/w/window.json', dict(simulated=True))
    gate.write_new(energy / 'windows/w/restored.json', dict(simulated=True))
    gate.write_new(energy / 'windows/w/energy-event.json', dict(simulated=True, kind='forced_suspend'))
    gate.write_new(energy / 'cohort/attempts/a/result.json', synthetic_row('hybrid', 0))
    energy_summary = package(energy)
    checks['energy_partials_aborted'] = energy_summary['energy_aborted'] and not energy_summary['groups'][0]['valid']
    try:
        check_resume(energy, True)
        checks['energy_resume_refused'] = False
    except RuntimeError:
        checks['energy_resume_refused'] = True
    gate.write_new(root / 'dry-run.json', dict(at=gate.now(), simulated=True, checks=checks))
    package(root)
    if not all(checks.values()):
        raise RuntimeError('Synthetic scenario failed')
    print(json.dumps(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'package', 'verify', 'dry-run', 'synthetic-watch'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--authorize-new-window', action='store_true')
    parser.add_argument('--nli-experiment', action='store_true')
    parser.add_argument('--before-cut', action='store_true')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root, args.resume, args.authorize_new_window, args.nli_experiment)
    elif args.command == 'preflight':
        preflight(args.root, args.before_cut)
    elif args.command == 'package':
        package(args.root)
    elif args.command == 'verify':
        verify_package(args.root)
    elif args.command == 'dry-run':
        if args.nli_experiment:
            from scripts.nli_batch_experiment import dry_run as dry_experiment
            dry_experiment(args.root)
        else:
            dry_run(args.root)
    else:
        synthetic_watch(args.root, args.pid, args.deadline)


if __name__ == '__main__':
    main()
