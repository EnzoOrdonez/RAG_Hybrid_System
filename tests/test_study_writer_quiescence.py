import json

import pytest

from scripts.study_operator.writer_quiescence import binding, coordinator_exclusions, normalized, writer_pids


def row(pid,exe,command):
    return dict(ProcessId=pid,ExecutablePath=exe,CommandLine=command)


def test_native_redirected_venv_and_sdk_writers_are_detected_without_private_output():
    paths=['C:/app/.venv/Scripts/python.exe','C:/native/python.exe','C:/sdk/python.exe']
    rows=[row(1,paths[0],'python --plan C:/own/run/plan.json'),
        row(2,paths[1],'python --plan c:\\own\\run\\plan.json PRIVATE_CLIENT_CANARY'),
        row(3,paths[2],'gcloud --project=pure-loop-474323-a8'),
        row(4,paths[1],'python C:/unrelated/userapp.py'),
        row(5,paths[1],'python --plan C:/own/run/finalizer.json')]
    result=writer_pids(rows,paths,['C:/own/run','--project=pure-loop-474323-a8'],5)
    assert result['writer_pids']==[1,2,3]
    assert result['processes_not_stopped'] and result['command_lines_not_persisted']
    assert 'PRIVATE_CLIENT_CANARY' not in json.dumps(result)


def test_native_binding_observes_current_runtime_and_normalization_is_windows_portable():
    result=binding()
    assert result['native_executable'] and result['launcher']
    assert normalized('C:\\NATIVE\\python.exe')==normalized('c:/native/python.exe')


@pytest.mark.parametrize('pid',[True,0,-1])
def test_invalid_coordinator_cannot_exclude_other_process(pid):
    with pytest.raises(ValueError):
        writer_pids([],['C:/native/python.exe'],['C:/own/run'],pid)


def test_duplicate_invalid_or_unknown_process_is_not_silently_treated_as_quiescence():
    with pytest.raises(ValueError):
        writer_pids([row(2,'C:/native/python.exe','C:/own/run')]*2,
            ['C:/native/python.exe'],['C:/own/run'],1)
    with pytest.raises(ValueError):
        writer_pids([row(True,'C:/native/python.exe','C:/own/run')],
            ['C:/native/python.exe'],['C:/own/run'],1)


def fixture_coordinator():
    args=['-B','-m','scripts.study_operator.finalization','--plan','C:/own/finalizer.json']
    rows=[dict(row(10,'C:/native/python.exe','coordinator'),ParentProcessId=20),
          dict(row(20,'C:/app/venv/python.exe','redirector'),ParentProcessId=30),
          row(30,'C:/native/python.exe','C:/own/unrelated-job')]
    commands={'coordinator':['native',*args],'redirector':['launcher',*args]}
    return rows,dict(native='C:/native/python.exe',launcher='C:/app/venv/python.exe',
                     plan='c:\\own\\finalizer.json'),commands


def test_only_direct_bound_redirector_is_exempt_never_other_native_writer():
    rows,bindings,commands=fixture_coordinator()
    result=writer_pids(rows,[bindings['native'],bindings['launcher']],['C:/own'],10,
                      coordinator_binding=bindings,parse=commands.__getitem__)
    assert result['writer_pids']==[30]
    assert 'coordinator' not in json.dumps(result)


@pytest.mark.parametrize('defect',['native','plan','parent_args','missing','duplicate'])
def test_coordinator_or_redirector_binding_mismatch_never_proves_quiescence(defect):
    rows,bindings,commands=fixture_coordinator()
    if defect=='native':
        rows[0]['ExecutablePath']='C:/unrelated/python.exe'
    elif defect=='plan':
        commands['coordinator'][-1]='C:/other/finalizer.json'
    elif defect=='parent_args':
        commands['redirector'][-1]='C:/other/finalizer.json'
    elif defect=='missing':
        rows.pop(0)
    else:
        rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        writer_pids(rows,[bindings['native'],bindings['launcher']],['C:/own'],10,
                    coordinator_binding=bindings,parse=commands.__getitem__)


def test_non_launcher_parent_remains_subject_to_ordinary_writer_detection():
    rows,bindings,commands=fixture_coordinator()
    rows[1]['ExecutablePath']=bindings['native']
    assert coordinator_exclusions(rows,10,**bindings,parse=commands.__getitem__)=={10}
