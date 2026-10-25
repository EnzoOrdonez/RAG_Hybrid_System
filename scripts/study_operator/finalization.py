"""Reserve-triggered safety/report/seal; all post-seal output stays external."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock

from scripts.study_operator import package_verify
from scripts.study_operator.audit_package import MANIFEST, census, seal, verify_existing
from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.cloud_safety import close
from scripts.study_operator.evidence import add, verify
from scripts.study_operator.final_report import render
from scripts.study_operator.run_control import require_limited, utc
from src.ui.components.session_storage import atomic_json


def external_paths(plan):
    root, external = Path(plan['package']).resolve(), Path(plan['external']).resolve()
    if (external.parent != root.parent or external == root
            or not external.name.startswith('iteration5-finalization-')
            or not root.name.startswith('iteration5-run-')):
        raise ValueError('Own sibling finalization output required')
    return root, external


def admitted(state, now):
    reserve = datetime.fromisoformat(state['closure_reserved_utc'])
    if now.tzinfo is None or reserve.tzinfo is None or now < reserve:
        raise ValueError('Finalizer cannot close before reserved time')
    if state['iteration'] != 5 or not state.get('agent') or not state.get('model'):
        raise ValueError('Own iteration5 actor required')


def retire(plan, plan_path, *, invoke=subprocess.run):
    root, external = external_paths(plan)
    value = dict(plan, coordinator_pid=os.getpid(), retirement_receipt=str(root/'finalization-task-retirement.json'))
    # The retirement script reads this exact external runtime plan. Scheduled
    # actions remain bound to the immutable entry plan passed separately.
    value['entry_plan'] = str(plan_path)
    runtime = external/('retirement-runtime-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(runtime, value)
    argv = ['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
            str(Path(plan['app'])/'scripts/study_operator/retire_own_tasks.ps1'), '-Plan', str(runtime)]
    result = invoke(argv, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Own writer retirement failed; cloud safety still required')
    receipt = json.loads((root/'finalization-task-retirement.json').read_text(encoding='utf-8-sig'))
    if (receipt['status'] != 'OWN_WRITERS_QUIESCENT_BACKUPS_PRESERVED' or receipt['remaining_python'] != 0
            or not isinstance(receipt['remaining_safety_tasks'], list)):
        raise ValueError('Quiescence not proven')
    return receipt


def retire_safety(plan, plan_path, *, invoke=subprocess.run):
    """Only after external verification; retries never modify sealed evidence."""
    root, external = external_paths(plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = external/('safety-retirement-'+stamp+'.json')
    runtime = external/('safety-retirement-runtime-'+stamp+'.json')
    atomic_json(runtime, dict(plan, coordinator_pid=os.getpid(), retirement_receipt=str(output),
                             entry_plan=str(plan_path)))
    result = invoke(['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
        str(Path(plan['app'])/'scripts/study_operator/retire_own_tasks.ps1'), '-Plan', str(runtime),
        '-SealedCleanup'], capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('External safety retirement pending; sealed package stays read-only')
    receipt = json.loads(output.read_text(encoding='utf-8-sig'))
    if (receipt['status'] != 'OWN_SAFETY_TASKS_RETIRED_AFTER_VERIFIED_SEAL'
            or receipt['sealed_package_not_modified'] is not True):
        raise ValueError('External task cleanup not verified')
    return dict(status=receipt['status'], receipt=str(output))


def finalize(plan, plan_path, cloud, *, quiesce=retire, sealer=seal, cleanup=retire_safety, now=None):
    root, external = external_paths(plan)
    external.mkdir(exist_ok=True)
    seal_receipt = external/'seal.json'
    verifier = external/'package_verify.py'
    # A backup must not even create a lock inside the inventoried package.
    if (root/MANIFEST).exists():
        if not seal_receipt.exists():
            raise ValueError('Existing manifest lacks external pin; no repin or repair')
        retry = external/('read-only-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        result = verify_existing(root, retry, verifier, seal_receipt)
        if result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED':
            result['external_task_cleanup'] = cleanup(plan, plan_path)
        return result
    state_path = root/'STATE.json'
    state = json.loads(state_path.read_bytes())
    admitted(state, now or datetime.now(timezone.utc))
    state.update(status='CLOSING', phase=6, next_action='Independent finalization; paid admission closed')
    atomic_json(state_path, state)
    started, begin = utc(), time.monotonic()
    retire_error = None
    try:
        retirement = quiesce(plan, plan_path)
    except Exception as exc:
        retire_error = type(exc).__name__
    safety = close(root, cloud)
    if retire_error or safety['status'] != 'OWN_VMS_TERMINATED_VERIFIED':
        raise RuntimeError('Closure incomplete; no manifest: '+(retire_error or 'STOP_PENDING'))
    verify(root)
    before = json.loads((root/'claims.json').read_bytes())
    command = [sys.executable, '-B', '-m', 'scripts.study_operator.finalization', '--plan', str(plan_path)]
    receipt = dict(command=command, exit_code=0, phase=6, agent=state['agent'], model=state['model'],
        started_utc=started, ended_utc=utc(), duration_s=time.monotonic()-begin,
        status='PRE_SEAL_SAFETY_AND_QUIESCENCE_VERIFIED', sealing_not_yet_done=True,
        safety_receipt=Path(json.loads(state_path.read_bytes())['safety_receipt']).name)
    atomic_json(root/'finalization-pre-seal-receipt.json', receipt)
    add(root, [dict(key='independent_finalization_safety', certainty='VERIFICADO',
        statement='El cierre independiente verificó VM propias TERMINATED y quiescencia de escritores, y retiró trabajos ordinarios sin cerrar aplicaciones de Enzo. Conservó tareas de respaldo hasta verificar el sello; su retiro final se documenta fuera del paquete. Esto no demuestra aceptación del estímulo, smoke, compuerta ni aptitud para participantes.',
        evidence=['finalization-task-retirement.json', receipt['safety_receipt'], 'finalization-pre-seal-receipt.json'],
        command_receipts=['finalization-pre-seal-receipt.json'])])
    claims = json.loads((root/'claims.json').read_bytes())
    added = next(r['id'] for r in claims if r['key'] == 'independent_finalization_safety')
    report_plan = json.loads(Path(plan['report_plan']).read_bytes())
    if added not in report_plan['sections']['Veredicto']:
        report_plan['sections']['Veredicto'].append(added)
    report = render(root, report_plan)
    (root/'REPORT_FINAL.md').write_text(report, encoding='utf-8', newline='\n')
    rows = []
    for line in (root/'COMMANDS.log').read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        if row.get('event') != 'INTENT':
            rows.append(('COMMANDS.log', row))
    rows.append(('finalization-pre-seal-receipt.json', receipt))
    atomic_json(root/'finalization-time-census.json', census(rows))
    current = json.loads(state_path.read_bytes())
    current.update(status='CLOSED_AWAITING_EXTERNAL_SEAL', phase=6, updated_utc=utc(),
        scheduled_tasks=retirement['remaining_safety_tasks'],
        scheduled_tasks_pending_external_cleanup=True,
        next_action='Read external seal and task cleanup receipts; independent audit and ethical approval pending',
        finalization=dict(preseal_receipt='finalization-pre-seal-receipt.json', prior_claims=len(before),
            aptitude_not_decided=True, report='REPORT_FINAL.md'))
    atomic_json(state_path, current)
    (root/'HANDOVER.md').write_text('# Cierre\n\n'+json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (root/'RUN_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write(utc()+' | '+state['agent']+' | '+state['model']+' | Pre-sello: trabajos ordinarios retirados; recursos detenidos; respaldos conservados hasta verificar sello. Consultar recibos externos.\n')
    # No package write after this line, including errors or second invocation.
    verifier.write_bytes(Path(package_verify.__file__).read_bytes())
    result = sealer(root, seal_receipt, verifier)
    if result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED':
        result['external_task_cleanup'] = cleanup(plan, plan_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args(argv)
    require_limited()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_bytes())
    root, external = external_paths(plan)
    if plan_path.parent != external:
        raise ValueError('Entry plan must live outside the sealed package')
    external.mkdir(exist_ok=True)
    result_path = external/('attempt-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    try:
        with FileLock(str(external/'coordinator.lock'), timeout=0):
            cloud = None
            if not (root/MANIFEST).exists():
                admitted(json.loads((root/'STATE.json').read_bytes()), datetime.now(timezone.utc))
                cloud = Cloud(plan['sdk'], 'pure-loop-474323-a8', root/'finalization-api')
            result = finalize(plan, plan_path, cloud)
    except Exception as exc:
        result = dict(status='FINALIZATION_FAILURE_PRESERVED', error_type=type(exc).__name__,
            at=utc(), acceptance_not_inferred=True)
    atomic_json(result_path, result)
    print(json.dumps(dict(status=result['status'], external_receipt=str(result_path))))
    return int(result['status'] != 'SEALED_AND_EXTERNALLY_VERIFIED')


if __name__ == '__main__':
    raise SystemExit(main())
