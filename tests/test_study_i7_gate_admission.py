import pytest


def receipt():
    identity={'boot_id':'gate','image_id':'sha256:'+'a'*64,'source_commit':'b'*40}
    value={'measurement_identity':identity,'publication':{'status':'PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED',
        'software_sha256':'c'*64,'contexts_sha256':'d'*64,'anchor_sha256':'e'*64,'combinations':12},
        'smoke':{'status':'BROWSER_EXPORT_VERIFIED','local_validation':False,
                 'identity':dict(identity,boot_id='smoke'),'attempts':8,'instruments':2,
                 'export_sha256':'f'*64,'automator':{'source_sha256':'1'*64,'pin_sha256':'2'*64}},
        'stimulus':{'status':'ACCEPTED','synthetic':False,'boots':12,'targets':120,'calls':144,
                    'software_sha256':'c'*64,'proofs_sha256':'3'*64}}
    return value,identity


def test_gate_receipt_binds_publication_actual_smoke_stimulus_and_current_boot():
    from scripts.study_operator.i7_admission import validate_gate_receipt
    value,identity=receipt()
    assert validate_gate_receipt(value,identity,'c'*64,source_sha='1'*64,pin_sha='2'*64)


@pytest.mark.parametrize('fault',['local_smoke','boot','software','source','stimulus','missing'])
def test_gate_receipt_rejects_partial_synthetic_or_mixed_prerequisites(fault):
    from scripts.study_operator.i7_admission import validate_gate_receipt
    value,identity=receipt()
    if fault=='local_smoke':
        value['smoke']['local_validation']=True
    if fault=='boot':
        value['measurement_identity']['boot_id']='other'
        identity=dict(identity,boot_id='gate')
    if fault=='software':
        value['publication']['software_sha256']='4'*64
    if fault=='source':
        value['smoke']['automator']['source_sha256']='4'*64
    if fault=='stimulus':
        value['stimulus']['boots']=11
    if fault=='missing':
        value.pop('publication')
    with pytest.raises(ValueError):
        validate_gate_receipt(value,identity,'c'*64,source_sha='1'*64,pin_sha='2'*64)


def test_guest_launches_only_bound_admission_on_empty_cold_technical_boot(tmp_path):
    import hashlib
    import json
    from datetime import datetime, timedelta, timezone
    from test_study_i7_gate import full_host_proof
    from scripts.study_operator.i7_gate_guest import dispatch_gate
    from scripts.study_operator.stimulus_evidence import software_projection
    from scripts.study_operator.stimulus_host_evidence import sha
    from scripts.study_operator.i7_browser import PIN
    inventory=full_host_proof()['inventory']
    boot='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    inventory['observed']['boot_id']=boot
    value,identity=receipt()
    identity.update(boot_id=boot,image_id=inventory['image']['image_id'],source_commit=inventory['source']['commit'])
    value['measurement_identity']=identity
    value['smoke']['identity']=dict(identity,boot_id='other-smoke-boot')
    software=sha(software_projection(inventory))
    value['publication']['software_sha256']=value['stimulus']['software_sha256']=software
    code=PIN.parent.parent
    value['smoke']['automator']={'source_sha256':hashlib.sha256((code/'scripts/study_operator/i7_browser.py').read_bytes()).hexdigest(),
                                'pin_sha256':hashlib.sha256(PIN.read_bytes()).hexdigest()}
    root=tmp_path/'boot'
    (root/'meta').mkdir(parents=True)
    (root/'sockets').mkdir()
    (root/'meta'/'environment_identity.json').write_text(json.dumps(inventory))
    (root/'sockets'/'service-state.json').write_text(json.dumps(dict(schema_version=1,mode='fresh_runner',phase='STARTING',
        sequence=1,boot_id=boot,request_id=None,deadline_monotonic_s=None)))
    sessions=tmp_path/'period'/'sessions'
    sessions.mkdir(parents=True)
    (sessions/'_i4_root.json').write_text(json.dumps(dict(schema_version=1,purpose='technical')))
    now=datetime(2026,11,1,tzinfo=timezone.utc)
    active=dict(boot_root=str(root),boot_id=boot,session_root=str(sessions),
        config=dict(purpose='technical',host_code=str(code),image_id=identity['image_id'],commit=identity['source_commit']),
        guest_deadline_utc=(now+timedelta(minutes=165)).isoformat(),native_deadline_utc=(now+timedelta(hours=3)).isoformat())
    effects=[]
    result=dispatch_gate(active,dict(operation='gate-start',admission=value),
        execute=lambda command,**kw:effects.append(command),controller=lambda *a:dict(active_session=None,session_count=0,invitation_count=0),
        maintenance=lambda:effects.append('maintenance'),now=lambda:now)
    assert result['status']=='GATE_STARTED_NOT_GO'
    assert effects[0]=='maintenance' and effects[1][0]=='systemd-run'
    assert json.loads((root/'i7-gate'/'prerequisites.json').read_bytes())==value
    with pytest.raises(ValueError,match='ALREADY_REGISTERED'):
        dispatch_gate(active,dict(operation='gate-start',admission=value),execute=lambda *a:pytest.fail('No replay'),
            controller=lambda *a:pytest.fail('No replay'),maintenance=lambda:None,now=lambda:now)
