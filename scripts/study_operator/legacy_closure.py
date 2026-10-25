"""Complete an unsealed inherited closure without changing its original deadlines.

Recovery is explicitly late, under the incoming iteration. No measured results
are synthesized, and failed historical receipts are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.study_operator.audit_package import census
from scripts.study_operator.run_control import Recorder, require_limited, utc


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def closure_guard(global_close, audit, controls, followup):
    expected = [(global_close, 'SAFE_RESOURCES_VERIFIED'),
                (audit, 'SAFE_RESOURCE_CLOSURE_AUDITED_NOT_PARTICIPANT_ACCEPTANCE'),
                (controls, 'PASS_FINAL_FUNCTIONAL_CONTROLS_STATIC_FREEZE_ONLY'),
                (followup, 'ABSENT_OWNED_IP_RECONCILED_ONCE_AFTER_SAFE_CLOSURE')]
    if any(row.get('status') != status for row, status in expected) or global_close.get('errors'):
        raise ValueError('Incomplete inherited resource or control closure; preserve failure')


def removable_tasks(rows, allowed, now):
    names = [row['name'] for row in rows]
    if len(set(names)) != len(names) or not set(names).issubset(allowed):
        raise ValueError('Unknown or duplicate task ownership')
    for row in rows:
        future = row.get('next_run_utc')
        if row['state'] == 'Running' or (future and datetime.fromisoformat(future) > now):
            raise ValueError('Running or pending task; respect inherited closure')
    return names


def verify_claims(root):
    text = (root / 'CLAIMS_LEDGER.md').read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        if not re.match(r'^V\d+ \|', line):
            continue
        match = re.search(r'^(V\d+) \|.*?\| `([^`]+)` \| SHA-256 `([a-f0-9]{64})`', line)
        if match is None:
            raise ValueError('Unparsed claim: ' + line.split(' |', 1)[0])
        identifier, filename, expected = match.groups()
        path = Path(filename)
        actual = digest(path)
        rows.append(dict(claim=identifier, path=str(path), expected=expected,
                         actual=actual, matched=actual == expected))
    if not rows or any(not row['matched'] for row in rows):
        raise ValueError('Inherited claim digest mismatch; no repair permitted')
    return dict(status='PASS_CONTENT_HASHES_ONLY', at=utc(), checked=len(rows), rows=rows,
                ledger_snapshot_sha256=digest(root / 'CLAIMS_LEDGER.md'),
                historical_execution_not_repeated=True)


def permitted_task_preservation(receipt, stderr):
    """A documented Limited-token denial never authorizes elevation or cancellation."""
    if receipt.get('exit_code') == 0 or b'0x80070005' not in stderr or b'Unregister-ScheduledTask' not in stderr:
        raise ValueError('No proven task-removal permission denial')
    return dict(status='BLOQUEADO-HUMANO', action='Retire completed I4 task definitions with a separately authorized administrator',
                no_elevation_or_acl_change=True)


def recover(root, package, app, sdk, *, attempt='01', preserve_unremovable=False):
    root, package, app = map(Path, (root, package, app))
    if (root / 'MANIFEST_SHA256.jsonl').exists():
        raise ValueError('Sealed inherited package is read-only; inspect external seal')
    if (root / 'RECOVERY_ITERATION5.json').exists():
        return finalize_report(root, package)
    state = read(root / 'STATE.json')
    original_deadline = state['deadline_utc']
    if datetime.now(timezone.utc) <= datetime.fromisoformat(original_deadline):
        raise ValueError('This recovery is only for a late failed closure')
    audit, controls = read(root / 'closure-audit01.json'), read(root / 'closure-controls01.json')
    closure_guard(read(root / 'global-closure-receipt.json'), audit, controls,
                  read(root / 'postclosure-ip-reconcile02-result.json'))
    verifier = Path('C:/CloudRAG/verify_iteration4_package_readonly.py')
    proof = read(root / 'external-verifier-controls02.json')
    if proof['status'] != 'PASS_SEVEN_EXTERNAL_VERIFIER_FIXTURES' or digest(verifier) != proof['source_sha256']:
        raise ValueError('Inherited independent verifier identity changed')
    recorder = Recorder(package, app, model=read(package / 'STATE.json')['model'], phase=0)
    claims = verify_claims(root)
    with (package / f'inherited-claims-verification-{attempt}.json').open('x', encoding='utf-8') as stream:
        json.dump(claims, stream, indent=2)
    rows = [('base:' + str(i), row) for i, row in enumerate(read(root / 'phase-command-supervisor-census12.json')['rows'])]
    for i, line in enumerate((root / 'COMMANDS.log').read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('exit_code') is not None and row.get('started_utc') and row.get('ended_utc'):
            rows.append(('COMMANDS.log:' + str(i), row))
    times = census(rows)
    time_name = f'phase-command-supervisor-recovery-i5-{attempt}.json'
    with (root / time_name).open('x', encoding='utf-8') as stream:
        json.dump(times, stream, indent=2)
    commands = [
        ('VMs', ['compute', 'instances', 'list', '--format=json(name,id,status,deletionProtection,disks,metadata.items.key)']),
        ('disks', ['compute', 'disks', 'list', '--format=json(name,id,zone,sizeGb,type,users)']),
        ('snapshots', ['compute', 'snapshots', 'list', '--format=json(name,id,storageBytes,sourceDiskId,status,storageLocations)']),
        ('addresses', ['compute', 'addresses', 'list', '--format=json(name,id,region,status,users)']),
        ('firewalls', ['compute', 'firewall-rules', 'list', '--format=json(name,id,disabled,network,allowed,sourceRanges,targetTags)'])]
    observed = {}
    for name, args in commands:
        label = 'legacy-live-' + name.lower() + '-' + attempt
        recorder.run(dict(name=label, argv=[sdk, *args, '--project=pure-loop-474323-a8', '--quiet'], timeout=120))
        observed[name] = read(package / (label + '.stdout'))
    by_id = {str(row['id']): row for row in observed['VMs']}
    if set(by_id) != {row['id'] for row in audit['VMs']}:
        raise ValueError('VM inventory changed; reconcile before completing closure')
    for row in by_id.values():
        keys = [item.get('key') for item in row.get('metadata', {}).get('items', [])]
        if (row['status'] != 'TERMINATED' or not row.get('deletionProtection')
                or any(d.get('autoDelete') for d in row['disks']) or 'startup-script' in keys):
            raise ValueError('Unsafe live VM state')
    if {str(row['id']) for row in observed['disks']} != set(audit['disk_ids']):
        raise ValueError('Retained disk inventory changed')
    if {str(row['id']) for row in observed['snapshots']} != set(audit['snapshot_ids']):
        raise ValueError('Retained snapshot inventory changed')
    if observed['addresses'] or any(row['name'] == 'cloudrag-i4-iap-20261004' for row in observed['firewalls']):
        raise ValueError('Inherited temporary address or IAP remains')
    disabled = {row['name']: row.get('disabled', False) for row in observed['firewalls']}
    if not all(disabled.get(name) for name in ['default-allow-ssh', 'default-allow-rdp']):
        raise ValueError('Inherited public management firewall rules enabled')

    ps = """$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new();
$rows=@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'CloudRAG-I4-*'} | ForEach-Object {
$i=Get-ScheduledTaskInfo -TaskName $_.TaskName;
[pscustomobject]@{name=$_.TaskName;state=$_.State.ToString();run_level=$_.Principal.RunLevel.ToString();
next_run_utc=$(if($i.NextRunTime){$i.NextRunTime.ToUniversalTime().ToString('o')}else{$null});last_result=$i.LastTaskResult}});
ConvertTo-Json -InputObject $rows -Compress"""
    task_label = 'legacy-tasks-before-' + attempt
    recorder.run(dict(name=task_label, argv=['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', ps], timeout=60))
    allowed = {row['task'] for row in state['local_system_changes'] if row.get('task', '').startswith('CloudRAG-I4-')}
    allowed.update(['CloudRAG-I4-GlobalClosure-20261004', 'CloudRAG-I4-PostClosureIP2-20261006',
                    'CloudRAG-I4-ClosureAudit-20261006', 'CloudRAG-I4-CloseReportHandoff2-20261006'])
    for path in root.glob('*-task-receipt.json'):
        row = json.loads(path.read_text(encoding='utf-8-sig'))
        if row.get('task', '').startswith('CloudRAG-I4-'):
            allowed.add(row['task'])
    task_rows = read(package / (task_label + '.stdout'))
    names = removable_tasks(task_rows, allowed, datetime.now(timezone.utc))
    if any(not re.fullmatch('CloudRAG-I4-[A-Za-z0-9_.-]+', name) for name in names):
        raise ValueError('Unsafe task name')
    xml = root / ('recovered-task-xml-i5-' + attempt)
    xml.mkdir()
    with (root / 'SYSTEM_CHANGES.md').open('a', encoding='utf-8') as stream:
        stream.write(f'BEFORE late I5 recovery: retire {len(names)} known completed I4 task definitions; export exact XML. No original schedule advanced/cancelled before execution; no user application affected. Original deadline {original_deadline} unchanged.\n')
    cleanup_plan = root / ('recovery-task-retirement-i5-' + attempt + '.json')
    with cleanup_plan.open('x', encoding='utf-8') as stream:
        json.dump(dict(names=names, xml_root=str(xml)), stream, indent=2)
    task_block = None
    if preserve_unremovable:
        task_block = permitted_task_preservation(read(package / 'legacy-tasks-retire-receipt.json'),
                                                (package / 'legacy-tasks-retire.stderr').read_bytes())
        # Copy definitions without retrying removal or changing their DACL/schedule.
        recorder.run(dict(name='legacy-tasks-preserve-' + attempt,
                         argv=['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                               str(app / 'scripts/study_operator/preserve_legacy_tasks.ps1'),
                               '-Plan', str(cleanup_plan)], timeout=120))
    else:
        recorder.run(dict(name='legacy-tasks-retire-' + attempt, argv=['powershell.exe', '-NoProfile', '-NonInteractive',
                          '-File', str(app / 'scripts/study_operator/retire_legacy_tasks.ps1'),
                          '-Plan', str(cleanup_plan)], timeout=180))

    recovery = dict(status='LATE_REPORT_RECOVERY_RESOURCES_VERIFIED_SEAL_PENDING', at=utc(),
                    incoming_iteration=5, original_deadline_utc=original_deadline, original_deadline_met=False,
                    original_failure_preserved='close-report-handoff01-wrapper.stderr',
                    inherited_receipts={name: digest(root / name) for name in ['global-closure-receipt.json',
                       'closure-audit01.json', 'closure-controls01.json', 'postclosure-ip-reconcile02-result.json']},
                    current_live_verification=observed, inherited_claims_verified=claims['checked'],
                    old_tasks_retired=names if task_block is None else [],
                    terminal_task_definitions_preserved=names if task_block else [], task_retirement_block=task_block,
                    old_tasks_quiescent_no_future_trigger=True,
                    new_GPU_measurements=False, stimulus_smoke_gate_not_measured=True,
                    identity_live_final_not_verified=True, seal_not_yet_verified=True)
    with (root / 'RECOVERY_ITERATION5.json').open('x', encoding='utf-8') as stream:
        json.dump(recovery, stream, indent=2)
    return finalize_report(root, package, time_name=time_name)


def report_sections(working):
    sections = re.findall(r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', working, re.M | re.S)
    if len(sections) != 20 or len({title for title, _ in sections}) != 20:
        raise ValueError('Expected inherited 20 distinct report sections')
    return sections


def finalize_report(root, package, *, time_name=None):
    """Resume only document finalization from immutable actual resource proof."""
    recovery = read(root / 'RECOVERY_ITERATION5.json')
    for name, expected in recovery['inherited_receipts'].items():
        if digest(root / name) != expected:
            raise ValueError('Inherited closure proof changed before finalization')
    if not recovery['old_tasks_quiescent_no_future_trigger']:
        raise ValueError('No actual quiescence proof')
    state, audit, claims = read(root / 'STATE.json'), read(root / 'closure-audit01.json'), verify_claims(root)
    original_deadline = recovery['original_deadline_utc']
    task_block = recovery['task_retirement_block']
    working = (root / 'REPORT_WORKING05.md').read_text(encoding='utf-8')
    sections = report_sections(working)  # Validate structure before any report/ledger write.
    if state.get('closure_report_complete'):
        report_sections((root / 'REPORT_FINAL.md').read_text(encoding='utf-8'))
        if not (root / 'DATOS_B4_V6.md').exists():
            raise ValueError('Completed state has missing B4 handover')
        return recovery
    if time_name is None:
        candidates = sorted(root.glob('phase-command-supervisor-recovery-i5-*.json'))
        if not candidates:
            raise ValueError('Missing time census for partial recovery')
        time_name = candidates[-1].name
    ledger = root / 'CLAIMS_LEDGER.md'
    number = max(map(int, re.findall(r'^V(\d+) \|', ledger.read_text(encoding='utf-8'), re.M)))
    def claim(name, assertion):
        nonlocal number
        path = root / name
        existing = [row['claim'] for row in claims['rows']
                    if Path(row['path']) == path and row['actual'] == digest(path)]
        if existing:
            return existing[-1]
        number += 1
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(f'\nV{number:03} | {assertion} | `{path.as_posix()}` | SHA-256 `{digest(path)}` | `Get-FileHash -Algorithm SHA256 -LiteralPath "{path.as_posix()}"`\n')
        return f'V{number:03}'
    recovery_ref = claim('RECOVERY_ITERATION5.json', 'VERIFICADO: recuperación tardía I5 y lecturas propias de recursos; no cumplimiento del deadline original ni aceptación del despliegue')
    time_ref = claim(time_name, 'VERIFICADO: unión de intervalos válidos; timestamps inválidos originales conservados y excluidos, sin inventar su ubicación')
    header = (f'# Iteración 4: cierre recuperado durante la iteración 5\n\nPrueba de recursos: {recovery["at"]}\n\n'
              f'El reporte programado falló y el sello no se lanzó dentro del plazo original ({original_deadline}). '
              'La recuperación es tardía y no corrige ese incumplimiento. El checkpoint histórico se conserva debajo de cada actualización. '
              'No se midieron la aceptación final del estímulo, el smoke ni la compuerta nueva. '
              f'Recursos actuales verificados: {recovery_ref}. Sello confirmado sólo por el recibo externo posterior. '
              'Retirada de definiciones de tareas: '+('BLOQUEADO-HUMANO por acceso denegado con token Limited; XML preservado y cero escritores/futuros disparadores comprobados.' if task_block else 'completada.')+'\n\n')
    updates = {
        'Resumen ejecutivo': f'Estímulo: aceptación final pendiente.\nPrivacidad y borrado: integración pendiente.\nSeguridad: cierre de recursos verificado.\nTLS/capacidad: VM detenidas; IP/IAP propios ausentes.\nCompuerta nueva: no medida.\nCosto: estimado heredado USD {audit["cost_estimate_usd"]}; márgenes separados, no factura. {recovery_ref}',
        'Auditoría': f'Recibos originales y {recovery["inherited_claims_verified"]} hashes del ledger heredado re-verificados por lectura antes de añadir el cierre; suites originales no re-ejecutadas en esta recuperación. Sello pendiente de recibo externo. {recovery_ref}',
        'Plazos y tiempo': f'Deadline original INCUMPLIDO para reporte y sello; unión de duraciones verificables en {time_ref}. Los intervalos invertidos no se reconstruyen ni suman dos veces. La recuperación se contabiliza en I5.',
        'TLS, IP estática y capacidad': f'Lectura API nueva: ambas VM TERMINATED/protegidas, discos e instantáneas retenidos, IP y regla IAP propias ausentes. No hay URL operativa ni tres arranques calificados. {recovery_ref}',
        'Costos': f'Estimado heredado a {audit["at"]}: USD {audit["cost_estimate_usd"]}; egress superior separado {audit["image_egress_upper_separate_usd"]}; margen separado {audit["initial_margin_separate_usd"]}; retención sin IP {audit["current_no_IP_idle_upper_usd_day"]} USD/día. Recuperación sin nuevas VM ni recursos pagados. No factura final.',
        'Veredicto': f'Recursos seguros; reporte recuperado tardíamente; sello depende del recibo externo. Aceptación final de participantes no demostrada. {recovery_ref}',
        'Falta': f'Reporte y conciliación del cierre ya recuperados; inventario/sello se comprobarán externamente. Siguen pendientes identidad y contextos vivos finales, aceptación 12/12, arranques/continuidad, smoke público, piloto y compuerta, privacidad integrada y factura final. {recovery_ref}'
    }
    text = header
    for title, body in sections:
        update = updates.get(title, 'Resultados de checkpoint heredados: no se ejecutó una medición adicional al recuperar el cierre.')
        text += f'## {title}\n\n{update}\n\n**Checkpoint histórico, anterior al cierre:**\n\n{body.strip()}\n\n'
    report = root / 'REPORT_FINAL.md'
    if report.exists():
        if report.read_text(encoding='utf-8') != text:
            raise ValueError('Partial report differs; preserve it without overwrite')
    else:
        with report.open('x', encoding='utf-8') as stream:
            stream.write(text)
    b4 = (root / 'DATOS_B4_V6_WORKING04.md').read_text(encoding='utf-8')
    b4_path = root / 'DATOS_B4_V6.md'
    if not b4_path.exists():
        with b4_path.open('x', encoding='utf-8') as stream:
            stream.write(b4 + f'\n\nRecuperación tardía I5: {recovery_ref}. Recursos detenidos y URL invalidada; aceptación integrada pendiente. No fija plazo legal ni edita documentos del investigador.\n')
    claim('REPORT_FINAL.md', 'VERIFICADO: reporte de 20 secciones con incumplimiento de plazo y ensayos no medidos explícitos; no dictamen nuevo de aptitud')
    claim('DATOS_B4_V6.md', 'VERIFICADO: entrega técnica heredada y actualización del cierre, sin certificar cumplimiento legal ni privacidad integrada')
    state.update(status='CLOSED_LATE_EXTERNAL_SEAL_PENDING', iteration_open=False, phase=7,
                 updated_utc=utc(), closure_report_complete=True, final_package_sealed=False,
                 recovery_iteration=5, original_deadline_met=False,
                 task_retirement_block=task_block,
                 final_seal_receipt=str(package / 'iteration4-recovered-seal.json'))
    # Pending historical scheduled flags are superseded by this provenance, not repaired in place.
    with (root / 'STATE.json').open('w', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    with (root / 'RUN_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write(utc()+' | Codex | GPT-6 runtime family | I5 late closure recovery: original failed receipt and deadline preserved; report20/B4 and quiescence complete; external seal still pending.\n')
    with (root / 'DESTRUCTION_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write('BEFORE recovery seal: external staging in I5 package declared disposable, preserved on failure. No evidence files deleted; manifest self-excluded only. No inherited root writer allowed after manifest.\n')
    return recovery


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ['root', 'package', 'app', 'sdk']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--attempt', default='01')
    parser.add_argument('--preserve-unremovable-tasks', action='store_true')
    args = parser.parse_args(argv)
    require_limited()
    if not re.fullmatch(r'\d{2}', args.attempt):
        raise ValueError('Two-digit immutable recovery attempt required')
    result = recover(args.root, args.package, args.app, args.sdk, attempt=args.attempt,
                     preserve_unremovable=args.preserve_unremovable_tasks)
    print(json.dumps(dict(status=result['status'],claims=result['inherited_claims_verified'])))


if __name__ == '__main__':
    main()
