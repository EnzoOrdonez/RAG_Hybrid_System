"""One owner entrypoint per November step; --check is local and never pays."""
import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

STEPS = ('cpu-build','gpu-restore','ready','tls-three','smoke','restore','privacy',
         'withdraw-purge','stimulus','gate','close')


def profile(kind):
    if kind not in ('gate','stimulus','smoke'):
        raise ValueError('FIXED_SYNTHETIC_PROFILE_REQUIRED')
    return {'i7_profile':'gate' if kind == 'gate' else 'standard',
            'native_stop_s':10800}


def validate_steps():
    for module in ('i7','i7_gate_host','i7_restore','i7_privacy','cpu_build','build_context',
                   'i7_service_disk','i7_cpu_pipeline','i7_smoke'):
        if importlib.util.find_spec('scripts.study_operator.'+module) is None:
            raise ValueError('RUNBOOK_MODULE_MISSING')
    return {'status':'LOCAL_RUNBOOK_PARTIAL','steps':len(STEPS),'cloud_measured':False,
            'blockers':['VALIDATED_BROWSER_AUTOMATOR_NOT_INTEGRATED',
                        'PUBLISHED_ENVIRONMENT_ANNEX_ADMISSION_NOT_INTEGRATED']}


def switch_profile(operator, kind):
    desired = profile(kind)
    operator.stop()
    operator.admit_paid(0)
    operator.cloud.command(['compute','instances','set-scheduling',operator.selected()['name'],
        '--zone='+operator.selected()['zone'],'--max-run-duration='+str(desired['native_stop_s'])+'s',
        '--instance-termination-action=STOP'])
    operator.config.update(desired)
    operator.persist()


def gate(operator):
    # No validated HTTPS automator is integrated yet. A manual complete export
    # does not meet the published amendment. Never silently substitute it.
    raise ValueError('BLOCKED_VALIDATED_BROWSER_AUTOMATOR_NOT_INTEGRATED')


def _gate_after_prerequisites(operator):
    """Not exposed by CLI until the missing published prerequisites are wired."""
    acceptance = json.loads((operator.root/'stimulus-acceptance.json').read_bytes())
    if acceptance['status'] != 'ACCEPTED':
        raise ValueError('I7_STIMULUS_ACCEPTED_REQUIRED')
    from scripts.study_operator.stimulus_evidence import accept
    from src.ui.components.study_protocol import verify_draw
    proofs=[json.loads(p.read_bytes()) for p in
            (operator.root/'private'/'stimulus'/'coded-P999').glob('*.json')]
    if (any(p['inventory']['image']['image_id'] != operator.config['image_id'] for p in proofs)
            or accept(proofs,verify_draw(operator.config['local_reviewed_config'])['config'])['status'] != 'ACCEPTED'):
        raise ValueError('CURRENT_IMAGE_STIMULUS_REQUIRED')
    attempt=operator.root/'gate-cohort-intent.json'
    with attempt.open('x',encoding='utf-8') as stream:
        json.dump({'started_utc':operator.now().isoformat(),'replay_allowed':False},stream)
    if not operator.state.get('reservation') or operator.state['reservation'].get('deleted_verified'):
        operator.reserve(5)
    from scripts.study_operator.i7_workflows import wait_ready
    try:
        # The preregistered full smoke follows the twelve complete cold trials.
        from scripts.study_operator.i7_smoke import run
        switch_profile(operator,'smoke')
        run(operator)
        restore_latest(operator)
        withdraw_purge(operator)
        switch_profile(operator,'gate')
        operator.start()
        wait_ready(operator)
        result = operator.bridge({'operation':'gate-start'},private=True)
        if result.get('status') != 'GATE_STARTED_NOT_GO':
            raise ValueError('GATE_START_UNCONFIRMED')
        deadline = time.monotonic()+9000
        while time.monotonic() < deadline:
            value = operator.bridge({'operation':'gate-status'},private=True)
            if value['status'] == 'GATE_COMPLETE_UNANALYZED':
                receipt=operator.collect('gate')
                from scripts.study_operator.i7_technical import cleanup
                cleanup(operator)
                operator.bridge({'operation':'gate-finalized','proof_sha256':receipt['proof_sha256']},private=True)
                return receipt
            if value['status'] == 'INCOMPLETE_TERMINAL':
                from scripts.study_operator.i7_technical import cleanup,collect_terminal
                receipt=collect_terminal(operator)
                cleanup(operator)
                operator.bridge({'operation':'gate-finalized','proof_sha256':receipt['proof_sha256']},private=True)
                raise ValueError('GATE_TERMINAL_NO_REPLACEMENT')
            operator.sleep(2)
        raise TimeoutError('GATE_OWNER_DEADLINE')
    finally:
        operator.stop()
        operator.guard()


def tls(operator):
    from scripts.study_operator.i7_workflows import check_tls_sequence, wait_ready
    attempts = []
    for _ in range(3):
        record = {'start_requested_utc':operator.now().isoformat(),'status':'FAILED'}
        try:
            operator.start()
            wait_ready(operator)
            record.update(operator.state['ready'],status='READY',source_commit=operator.config['commit'])
            attempts.append(record)
        finally:
            operator.receipt('boot-attempt',record)
            operator.stop()
    return operator.receipt('tls',check_tls_sequence(attempts))


def withdraw_purge(operator):
    if operator.state.get('purpose') != 'smoke':
        raise ValueError('SYNTHETIC_SMOKE_PERIOD_REQUIRED')
    records = []
    for operation, code in (('withdraw','P999'),('purge-study',None)):
        operator.delete_sessions(operation,code=code,dry_run=True)
        result = operator.delete_sessions(operation,code=code,dry_run=False)
        if not all(result[k] for k in ('remote_versions_empty','zero_retention_policy_verified',
                                      'normal_objects_empty','disk_empty')):
            raise ValueError('FOUR_DELETION_CONTROLS_REQUIRED')
        records.append(result)
    return operator.receipt('withdraw-purge',{'status':'FOUR_CONTROLS_VERIFIED','results':records})


def restore_latest(operator):
    rows=sorted((operator.root/'receipts'/'smoke').glob('*.json'))
    if not rows:
        raise ValueError('GENERATION_VERIFIED_SMOKE_BACKUP_REQUIRED')
    smoke=json.loads(rows[-1].read_bytes())
    objects=smoke['objects']
    result=operator.restore('P999',smoke['session_id'],
        full_generation=objects['full_session.json']['generation'],
        manifest_generation=objects['export_manifest.json']['generation'])
    if (result.get('status')!='RESTORED_VERIFIED' or result.get('export_sha256')!=smoke['export_sha256']
            or not result.get('new_app_instance') or not result.get('original_store_unchanged')
            or not result.get('invitation_revoked')):
        raise ValueError('DEPLOYED_RESTORE_FAILED')
    return operator.receipt('restore',result)


def dispatch(operator, step, spec):
    if step == 'cpu-build':
        from scripts.study_operator.i7_cpu_pipeline import build
        return build(operator,spec)
    if step == 'gpu-restore':
        from scripts.study_operator.i7_cpu_pipeline import gpu_restore
        return gpu_restore(operator)
    if step == 'ready':
        from scripts.study_operator.i7_workflows import wait_ready
        return wait_ready(operator)
    if step == 'tls-three':
        return tls(operator)
    if step == 'gate':
        return gate(operator)
    if step == 'withdraw-purge':
        return withdraw_purge(operator)
    if step == 'restore':
        return restore_latest(operator)
    if step == 'privacy':
        from scripts.study_operator.i7_workflows import wait_ready, privacy
        operator.stop()
        operator.start('smoke')
        wait_ready(operator)
        return privacy(operator)
    if step == 'smoke':
        from scripts.study_operator.i7_smoke import run
        return run(operator)
    if step == 'stimulus':
        from scripts.study_operator.i7_workflows import series
        switch_profile(operator,'stimulus')
        return series(operator)
    from scripts.study_operator.i7_workflows import dispatch as workflow
    return workflow(operator,step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--sdk')
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('step',nargs='?',choices=STEPS)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(validate_steps()))
        return 0
    if (not args.live or not args.root or not args.sdk or not args.step
            or datetime.now(timezone.utc) < datetime(2026,11,1,tzinfo=timezone.utc)):
        parser.error('Explicit --live, root, SDK and November authorization required')
    from filelock import FileLock

    from scripts.study_operator.cloud_client import Cloud
    from scripts.study_operator.i7 import Operator
    with FileLock(str(args.root/'i7.lock'),timeout=1):
        operator = Operator(args.root,Cloud(args.sdk,'pure-loop-474323-a8',args.root/'runs'))
        from scripts.study_operator.i7_workflows import verify_tool_source
        verify_tool_source(operator)
        spec = json.loads(args.spec.read_bytes()) if args.spec else {}
        result = dispatch(operator,args.step,spec)
        print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__ == '__main__':
    main()
