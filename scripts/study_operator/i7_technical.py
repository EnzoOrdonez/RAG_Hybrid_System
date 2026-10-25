"""Close, restore and purge synthetic gate sessions, retaining private evidence."""
import hashlib
import json

CONTROLS=('remote_versions_empty','zero_retention_policy_verified','normal_objects_empty','disk_empty')


def close_sessions(store,backup):
    from src.ui.components.study_sessions import StudySession
    if store.purpose!='technical':
        raise ValueError('TECHNICAL_SYNTHETIC_ONLY')
    active=store._read().get('active') if store.path.exists() else None
    if active:
        store.abandon(active)
    rows=[]
    for checkpoint in sorted(store.root.glob('*/study_checkpoint.json')):
        session=StudySession.load(store,checkpoint.parent.name)
        code=session.data['assignment']['participant_id']
        if not code.startswith('P') or not code[1:].isdigit() or int(code[1:])<900:
            raise ValueError('SYNTHETIC_GATE_CODE_REQUIRED')
        raw=session.export().read_bytes()
        receipt=backup(checkpoint.parent)
        digest=hashlib.sha256(raw).hexdigest()
        if receipt['status']!='complete' or receipt['sha256']!=digest:
            raise ValueError('TECHNICAL_BACKUP_NOT_VERIFIED')
        rows.append({'code':code,'session_id':session.session_id,'export_sha256':digest,
                     'objects':receipt['objects']})
    return {'status':'TECHNICAL_SESSIONS_CLOSED_BACKED','sessions':rows}


def cleanup(operator):
    if operator.state.get('purpose')!='technical':
        raise ValueError('TECHNICAL_PERIOD_REQUIRED')
    maintenance=operator.bridge({'operation':'maintenance'},private=True)
    if maintenance.get('status')!='MAINTENANCE' or maintenance.get('admission_closed') is not True:
        raise ValueError('TECHNICAL_MAINTENANCE_REQUIRED')
    inventory=operator.bridge({'operation':'i7-technical-close'},private=True)
    if inventory['status']!='TECHNICAL_SESSIONS_CLOSED_BACKED':
        raise ValueError('TECHNICAL_BACKUPS_REQUIRED')
    operator.receipt('technical-inventory',inventory)
    rows=inventory['sessions']
    if rows:
        first=rows[0]
        objects=first['objects']
        restored=operator.restore(first['code'],first['session_id'],
            full_generation=objects['full_session.json']['generation'],
            manifest_generation=objects['export_manifest.json']['generation'])
        if (restored.get('status')!='RESTORED_VERIFIED' or restored.get('export_sha256')!=first['export_sha256']
                or any(restored.get(k) is not True for k in
                       ('invitation_revoked','new_app_instance','original_store_unchanged'))):
            raise ValueError('TECHNICAL_RESTORE_FAILED')
        operator.receipt('technical-restore',restored)
        operator.delete_sessions('withdraw',code=first['code'],dry_run=True)
        withdrawn=operator.delete_sessions('withdraw',code=first['code'],dry_run=False)
        if not all(withdrawn.get(k) is True for k in CONTROLS):
            raise ValueError('TECHNICAL_WITHDRAWAL_INCOMPLETE')
    operator.delete_sessions('purge-study',dry_run=True)
    result=operator.delete_sessions('purge-study',dry_run=False)
    if not all(result.get(k) is True for k in CONTROLS):
        raise ValueError('TECHNICAL_PURGE_INCOMPLETE')
    return operator.receipt('technical-cycle',{'status':'TECHNICAL_CYCLE_VERIFIED',
        'sessions':len(rows),'restored_representative':bool(rows),'purge':result,'terminal_evidence_retained':True})


def collect_terminal(operator):
    from scripts.study_operator.stimulus_host_evidence import sha
    response=operator.bridge({'operation':'gate-terminal-evidence'},private=True)
    proof=response['proof']
    host=proof['host_evidence']
    if (sha(proof)!=response['proof_sha256'] or host['status']!='INCOMPLETE_TERMINAL'
            or host['boot_id']!=operator.state['ready']['boot_id']
            or host['app_image_id']!=operator.config['image_id']):
        raise ValueError('TERMINAL_HOST_IDENTITY_DIFFERS')
    folder=operator.root/'private'/'gate-terminal'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(host['boot_id']+'.json')
    if path.exists():
        if sha(json.loads(path.read_bytes()))!=sha(proof):
            raise ValueError('TERMINAL_EVIDENCE_CHANGED')
    else:
        with path.open('x',encoding='utf-8') as stream:
            json.dump(proof,stream,ensure_ascii=False,sort_keys=True)
    return operator.receipt('gate-terminal',{'status':'TERMINAL_DOWNLOADED_NOT_GO',
        'proof_sha256':sha(proof),'attempts':len(proof['rows'])})
