from scripts.study_operator.linux_build import check_commands,Guest,host_check_code

import subprocess
import time

import pytest


def test_linux_checks_preserve_pins_offline_models_and_persistent_cut_test(tmp_path):
    commands = check_commands('sha256:fixture',tmp_path,'build01')
    assert [name for name,_,_ in commands] == ['pip-check','linux-full','linux-filtered','posix-durability','packages']
    for _,argv,limit in commands:
        assert '--rm=false' in argv and argv[argv.index('--network')+1] == 'none'
        assert '--gpus' not in argv  # CPU validation cannot stand in for L4 acceptance.
        assert 'HF_HUB_OFFLINE=1' in argv and 'TRANSFORMERS_OFFLINE=1' in argv
        mounts = [argv[i+1] for i,v in enumerate(argv[:-1]) if v == '--mount']
        assert any('persistent-test-tmp' in v and 'target=/test-tmp' in v for v in mounts)
        assert all('readonly' in v for v in mounts if 'data/models' in v or 'data/indices' in v)
        assert limit == 900
    assert 'tests/test_session_durability.py' in commands[3][1]
    assert '-m' in commands[2][1] and 'not slow and not gpu' in commands[2][1]


@pytest.mark.parametrize('key,value',[('purpose','study'),('operation','OTHER'),('label','../escape'),
                                    ('bucket','other'),('prefix','sessions/participant')])
def test_guest_rejects_scope_before_any_file_or_process(key,value):
    job = dict(purpose='technical',operation='I5_CPU_BUILD',label='build01',
               bucket='cloudrag-study-103950017681-20261002',prefix='iteration4/iteration5/run/build01')
    job[key] = value
    with pytest.raises(ValueError,match='scope'):
        Guest(job)


def test_command_timeout_kills_owned_group_and_records_exit124(tmp_path,monkeypatch):
    import scripts.study_operator.linux_build as module
    receipts,kills = [],[]
    monkeypatch.setattr(module,'persist',lambda path,value:receipts.append(value))
    monkeypatch.setattr(module.os,'killpg',lambda pid,sig:kills.append((pid,sig)),raising=False)
    class Child:
        pid = 123
        returncode = -15
        def __init__(self):
            self.calls = 0
        def wait(self,timeout):
            self.calls += 1
            if self.calls == 1:
                assert 0 < timeout <= 20
                raise subprocess.TimeoutExpired(['fixture'],timeout)
        def poll(self):
            return None if self.calls == 1 else self.returncode
    guest = Guest.__new__(Guest)
    guest.logs,guest.deadline,guest.active = tmp_path,time.monotonic()+20,None
    guest.invoke = lambda *args,**kwargs:Child()
    with pytest.raises(ValueError,match='STEP_FAILED_fixture'):
        guest.step('fixture',['fixture'],timeout=100)
    assert kills and kills[0][0] == 123
    assert receipts[0]['exit_code'] == 124 and receipts[0]['timed_out']
    assert guest.active is None


def test_host_validation_uses_the_actual_stdlib_and_unit_without_starting_any_service(tmp_path):
    import ast
    code = host_check_code('/srv/cloudrag/iteration5/code-final03',tmp_path/'retained.service')
    ast.parse(code)
    assert 'stimulus_host.unit_text(active,1)' in code
    assert '"torch","filelock","pydantic"' in code
    assert 'systemd-run' not in code and 'Popen' not in code and 'query(' not in code


def test_new_build_collector_requires_native_host_checks_before_false_pass(tmp_path,monkeypatch):
    from test_study_cpu_build import fixture, objects

    build,_,_,_,_,_=fixture(tmp_path,monkeypatch)
    objects(tmp_path,monkeypatch,build)
    import hashlib
    import json
    path = tmp_path/'cpu-build-final01-job-input.json'
    job = json.loads(path.read_bytes())
    proof_sha = hashlib.sha256((tmp_path/'cpu-restoration-bootstrap01-proof.json').read_bytes()).hexdigest()
    job.update(cpu_label='bootstrap01',cpu_restoration_proof_sha256=proof_sha)
    path.write_text(json.dumps(job))
    build.state.update(lambda state:state['build_jobs']['final01'].update(cpu_label='bootstrap01',
                       cpu_restoration_proof_sha256=proof_sha))
    with pytest.raises(ValueError,match='complete zero-exit'):
        build.collect('final01')
    assert not (tmp_path/'linux-build-final01-download-proof.json').exists()
