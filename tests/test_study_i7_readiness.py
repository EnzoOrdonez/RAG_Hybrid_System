import hashlib
import json


def test_ready_requires_matching_source_all_suites_and_real_preconditions(tmp_path,monkeypatch):
    from scripts.study_operator import i7_admission,i7_readiness
    log=tmp_path/'suite.txt'
    log.write_bytes(b'3 passed')
    value={'source_sha256':'a'*64,'freeze':{'modules':19,'documents':3},'four_posix_tests_passed':True,
        'tests':{kind:{'exit_code':0,'log_path':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
                 for kind in ('windows','linux','i7')}}
    evidence=tmp_path/'local-readiness.json'
    evidence.write_text(json.dumps(value))
    monkeypatch.setattr(i7_readiness,'source_sha256',lambda:'a'*64)
    partial=i7_readiness.evaluate(evidence)
    assert {r['owner'] for r in partial['blockers']}=={'Enzo'}
    (tmp_path/'published-annex.json').write_text('{}')
    monkeypatch.setattr(i7_admission,'check_files',lambda root:{'status':'PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED'})
    assert i7_readiness.evaluate(evidence,root=tmp_path,november_authorized=True)['status']=='LOCAL_RUNBOOK_READY'
    log.write_bytes(b'changed log')
    assert any(r['owner']=='code' for r in i7_readiness.evaluate(evidence)['blockers'])
