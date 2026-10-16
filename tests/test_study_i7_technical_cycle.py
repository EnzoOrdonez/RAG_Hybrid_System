import json

import pytest


def test_technical_cycle_restores_then_withdraws_and_purges(tmp_path):
    from scripts.study_operator.i7_technical import cleanup
    calls=[]
    class Operator:
        state={'purpose':'technical'}
        root=tmp_path
        def bridge(self,request,**kwargs):
            calls.append(request['operation'])
            if request['operation']=='maintenance':
                return {'status':'MAINTENANCE','admission_closed':True}
            return {'status':'TECHNICAL_SESSIONS_CLOSED_BACKED','sessions':[
                {'code':'P900001','session_id':'a'*32,'export_sha256':'b'*64,
                 'objects':{name:{'generation':'1'} for name in ('full_session.json','export_manifest.json')}}]}
        def restore(self,code,sid,**kwargs):
            calls.append('restore')
            return {'status':'RESTORED_VERIFIED','export_sha256':'b'*64,'invitation_revoked':True,
                    'new_app_instance':True,'original_store_unchanged':True}
        def delete_sessions(self,operation,**kwargs):
            calls.append(operation+('-dry' if kwargs['dry_run'] else ''))
            return dict.fromkeys(('remote_versions_empty','zero_retention_policy_verified','normal_objects_empty','disk_empty'),True)
        def receipt(self,kind,result):
            return result
    terminal=tmp_path/'terminal.json'
    terminal.write_text('preserved')
    result=cleanup(Operator())
    assert result['status']=='TECHNICAL_CYCLE_VERIFIED'
    assert calls==['maintenance','i7-technical-close','restore','withdraw-dry','withdraw','purge-study-dry','purge-study']
    assert terminal.read_text()=='preserved'


@pytest.mark.parametrize('fault',['api','identity','absent','source'])
def test_unknown_effect_reconciliation_never_invents_success(tmp_path,fault):
    from scripts.study_operator.i7_reconcile import reconcile
    class Cloud:
        def command(self,args,**kwargs):
            if fault=='api':
                raise RuntimeError('API unavailable')
            return [] if fault=='absent' else [{'id':'1','name':'cloudrag-i7-disk','zone':'zones/us-central1-a',
                'description':'wrong' if fault=='identity' else 'owned',
                'sourceSnapshotId':'wrong' if fault=='source' else '3'}]
    class Operator:
        root=tmp_path
        config={'resource_marker':'owned'}
        state={'creation_intents':[{'kind':'disks','name':'cloudrag-i7-disk','zone':'us-central1-a',
                                   'marker':'owned','source_snapshot_id':'3','status':'UNKNOWN'}]}
        cloud=Cloud()
        def persist(self):
            pass
        def receipt(self,kind,value):
            return value
    result=reconcile(Operator())
    assert result['status']=='BLOCKED_UNKNOWN_RESOURCE'
    assert result['resources'][0]['next_command'][:3]==['gcloud','compute','disks']


def test_close_active_technical_session_exports_and_backs_up(tmp_path):
    from scripts.study_operator.i7_technical import close_sessions
    from src.ui.components.study_sessions import StudyStore
    from tests.study_helpers import configured
    _,_,protocol=configured(tmp_path)
    store=StudyStore(tmp_path/'sessions',protocol,purpose='technical')
    store.freeze()
    session=store.admit(store.issue('P900001',cell=1,profile='without_experience'))
    session.familiarization_done()
    session.begin()
    def backup(folder):
        import hashlib
        raw=(folder/'full_session.json').read_bytes()
        return {'status':'complete','sha256':hashlib.sha256(raw).hexdigest(),
                'objects':{name:{'generation':'1','sha256':'a'*64} for name in ('full_session.json','export_manifest.json')}}
    result=close_sessions(store,backup)
    full=json.loads((store.root/session.session_id/'full_session.json').read_bytes())
    assert full['stage']=='abandoned' and full['attempts'][0]['error']=='interrupted'
    assert len(result['sessions'])==1


@pytest.mark.parametrize('stage',['close','restore','purge'])
def test_cycle_api_error_retains_terminal_and_never_reports_success(tmp_path,stage):
    from scripts.study_operator.i7_technical import cleanup
    terminal=tmp_path/'terminal.json'
    terminal.write_bytes(b'original terminal evidence')
    class Operator:
        state={'purpose':'technical'}
        root=tmp_path
        def bridge(self,request,**kw):
            if request['operation']=='maintenance':
                return {'status':'MAINTENANCE','admission_closed':True}
            if stage=='close':
                raise RuntimeError('API unavailable')
            return {'status':'TECHNICAL_SESSIONS_CLOSED_BACKED','sessions':[{
                'code':'P900001','session_id':'a'*32,'export_sha256':'b'*64,
                'objects':{name:{'generation':'1'} for name in ('full_session.json','export_manifest.json')}}]}
        def restore(self,*args,**kwargs):
            if stage=='restore':
                raise RuntimeError('API unavailable')
            return dict(status='RESTORED_VERIFIED',export_sha256='b'*64,new_app_instance=True,
                        invitation_revoked=True,original_store_unchanged=True)
        def delete_sessions(self,operation,**kwargs):
            if operation=='purge-study':
                raise RuntimeError('API unavailable')
            return dict.fromkeys(('remote_versions_empty','zero_retention_policy_verified','normal_objects_empty','disk_empty'),True)
        def receipt(self,kind,value):
            return value
    with pytest.raises(RuntimeError):
        cleanup(Operator())
    assert terminal.read_bytes()==b'original terminal evidence'
