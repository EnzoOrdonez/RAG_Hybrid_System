"""One owner entrypoint per November step; --check is local and never pays."""
import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

STEPS = ('cpu-build','gpu-restore','ready','tls-three','contexts','smoke','restore','privacy',
         'withdraw-purge','stimulus','gate','technical-cycle','reconcile','close')


def profile(kind):
    if kind not in ('gate','stimulus','smoke'):
        raise ValueError('FIXED_SYNTHETIC_PROFILE_REQUIRED')
    return {'i7_profile':'gate' if kind == 'gate' else 'standard',
            'native_stop_s':10800}


def validate_steps(evidence=None, *, root=None,november_authorized=False):
    for module in ('i7','i7_gate_host','i7_restore','i7_privacy','cpu_build','build_context',
                   'i7_service_disk','i7_cpu_pipeline','i7_smoke','i7_admission','i7_technical','i7_reconcile','i7_costs'):
        if importlib.util.find_spec('scripts.study_operator.'+module) is None:
            raise ValueError('RUNBOOK_MODULE_MISSING')
    from scripts.study_operator.i7_readiness import evaluate
    result=dict(evaluate(evidence,root=root,november_authorized=november_authorized),steps=len(STEPS))
    from scripts.study_operator.i7_browser_preflight import verify
    try:
        result['browser']=verify()
    except ValueError:
        result['status']='LOCAL_RUNBOOK_PARTIAL'
        result['blockers'].append({'owner':'code','code':'PINNED_BROWSER_RUNTIME_REQUIRED'})
    return result


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
    from scripts.study_operator.i7_admission import check_files
    from scripts.study_operator.i7_smoke import validate_receipt
    try:
        publication=check_files(operator.root)
    except (OSError,KeyError,ValueError) as error:
        raise ValueError('ENZO_PUBLISHED_EFFECTIVE_ANNEX_REQUIRED') from error
    validate_receipt(operator.root/'browser-validation.json')
    return _gate_after_prerequisites(operator,publication)


def _gate_after_prerequisites(operator,publication):
    acceptance = json.loads((operator.root/'stimulus-acceptance.json').read_bytes())
    if acceptance['status'] != 'ACCEPTED':
        raise ValueError('I7_STIMULUS_ACCEPTED_REQUIRED')
    from scripts.study_operator.stimulus_evidence import accept
    from src.ui.components.study_protocol import verify_draw
    proofs=[json.loads(p.read_bytes()) for p in
            (operator.root/'private'/'stimulus'/'coded-P999').glob('*.json')]
    verified=accept(proofs,verify_draw(operator.config['local_reviewed_config'])['config'])
    if (any(p['inventory']['image']['image_id'] != operator.config['image_id'] for p in proofs)
            or verified['status'] != 'ACCEPTED' or verified['software_sha256']!=publication['software_sha256']):
        raise ValueError('CURRENT_IMAGE_STIMULUS_REQUIRED')
    from scripts.study_operator.i7_admission import compare_contexts
    from scripts.study_operator.stimulus_host_evidence import sha
    contexts={}
    for proof in proofs:
        for row in proof['rows']:
            if row['slot']['role']!='target':
                continue
            key=row['slot']['query_id']+'|'+row['slot']['condition']
            if key in contexts and contexts[key]!=row['contexts']:
                raise ValueError('STIMULUS_CONTEXT_VARIANTS_FORBIDDEN')
            contexts[key]=row['contexts']
    reference=json.loads(Path(operator.config['i3_context_reference_path']).read_bytes())['rag']['contexts']
    compare_contexts(contexts,reference)
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
        smoke=run(operator)
        restore_latest(operator)
        withdraw_purge(operator)
        switch_profile(operator,'gate')
        operator.start()
        wait_ready(operator)
        identity={'boot_id':operator.state['ready']['boot_id'],'image_id':operator.config['image_id'],
                  'source_commit':operator.config['commit']}
        admission={'measurement_identity':identity,'publication':publication,'smoke':smoke,
            'stimulus':{'status':'ACCEPTED','synthetic':False,'boots':len(proofs),'targets':120,
                'calls':sum(len(p['rows']) for p in proofs),'software_sha256':verified['software_sha256'],
                'proofs_sha256':sha(proofs)}}
        from scripts.study_operator.i7_admission import validate_gate_receipt
        from scripts.study_operator.i7_browser import PIN
        digest=validate_gate_receipt(admission,identity,publication['software_sha256'],
            source_sha=hashlib.sha256(Path(__file__).with_name('i7_browser.py').read_bytes()).hexdigest(),
            pin_sha=hashlib.sha256(PIN.read_bytes()).hexdigest())
        operator.state['gate_admission_sha256']=digest
        operator.persist()
        result = operator.bridge({'operation':'gate-start','admission':admission},private=True)
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
    if step=='contexts':
        from scripts.study_operator.i7_workflows import wait_ready
        operator.stop()
        operator.start()
        try:
            wait_ready(operator)
            operator.maintenance()
            proof=operator.bridge({'operation':'i7-contexts'},private=True,timeout=900)
            if proof['mode']!='LIVE_NO_GENERATION' or proof['inventory']['image']['image_id']!=operator.config['image_id']:
                raise ValueError('LIVE_CURRENT_CONTEXT_PROOF_REQUIRED')
            with (operator.root/'live-contexts.json').open('x',encoding='utf-8') as stream:
                json.dump(proof,stream,ensure_ascii=False,sort_keys=True)
            return operator.receipt('contexts',{'status':'TWELVE_CONTEXTS_COLLECTED_NOT_PUBLISHED'})
        finally:
            operator.stop()
    if step=='technical-cycle':
        from scripts.study_operator.i7_technical import cleanup
        return cleanup(operator)
    if step=='reconcile':
        from scripts.study_operator.i7_reconcile import reconcile
        return reconcile(operator)
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
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('step',nargs='?',choices=STEPS)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(validate_steps(args.evidence,root=args.root,
            november_authorized=args.live and datetime.now(timezone.utc)>=datetime(2026,11,1,tzinfo=timezone.utc))))
        return 0
    if (not args.live or not args.root or not args.sdk or not args.step
            or datetime.now(timezone.utc) < datetime(2026,11,1,tzinfo=timezone.utc)):
        parser.error('Explicit --live, root, SDK and November authorization required')
    from scripts.study_operator.i7_browser_preflight import verify
    try:
        verify()
    except ValueError as error:
        parser.error(str(error))
    if args.step not in ('close','reconcile','technical-cycle'):
        from scripts.study_operator.i7_readiness import evaluate
        if any(row['owner']=='code' for row in evaluate(args.evidence,november_authorized=True)['blockers']):
            parser.error('Current source-bound Windows/Linux/I7 verification receipt required')
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
