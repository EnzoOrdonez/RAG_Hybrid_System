"""cpu-build-04 (ADR-0023): a boot-disk clone attached at boot can become / or /boot."""
import os
from types import SimpleNamespace

import pytest

PERIOD = '5e9d9dac3ca74bfb95468333b0b2820f'
MARKER = 'cloudrag-i7-'+PERIOD
TARGET = 'cloudrag-i7-service-'+PERIOD[:12]
STAGE = 'cloudrag-i7-stage-'+PERIOD[:12]


class StopAtFirstSsh(Exception):
    pass


def boot_only_then_hot_attach(commands):
    """Return True only for start -> attach-disk(clone) -> first ssh."""
    names = [tuple(c[1:3]) for c in commands]
    start = names.index(('instances','start'))
    attach = next(i for i,c in enumerate(commands)
                  if tuple(c[1:3]) == ('instances','attach-disk') and '--disk='+TARGET in c)
    ssh = next(i for i,c in enumerate(commands) if tuple(c[:2]) == ('compute','ssh'))
    assert start < attach < ssh, 'clone must be hot-attached after boot-only start, before mount'
    return True


def test_clean_snapshot_starts_boot_only_then_hot_attaches_clone(tmp_path):
    from scripts.study_operator.i7_cpu_pipeline import clean_snapshot
    commands = []
    class Cloud:
        def command(self,args,**kwargs):
            commands.append(list(args))
            if args[:2] == ['compute','ssh']:
                raise StopAtFirstSsh()
            if args[1:3] == ['snapshots','list']:
                return []
            if args[1:3] == ['snapshots','describe']:
                return {'id':'51','name':STAGE,'description':MARKER,'storageLocations':['us-central1']}
            if args[1:3] == ['disks','describe']:
                return {'id':'61','name':TARGET,'description':MARKER,'sourceSnapshotId':'51'}
            return {}
    class Operator:
        root = tmp_path
        config = {'resource_marker':MARKER,'period_id':PERIOD,'host_code':'/srv/cloudrag/iteration5/code-finali7'}
        state = {}
        def persist(self):
            pass
    resources = {'vm':{'name':'cloudrag-i5-restore-i7cpu-5e9d9dac','id':'84'},
                 'disk':{'name':'cloudrag-i5-restore-i7cpu-5e9d9dac-boot','id':'20'}}
    with pytest.raises(StopAtFirstSsh):
        clean_snapshot(Operator(),Cloud(),resources,{'status':'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED'})
    assert boot_only_then_hot_attach(commands)
    assert commands[-1][1:3] == ['instances','stop']  # finally still stops the CPU VM


def test_order_checker_rejects_inverted_mutant():
    mutant = [['compute','instances','attach-disk','vm','--disk='+TARGET],
              ['compute','instances','start','vm'],['compute','ssh','vm']]
    with pytest.raises(AssertionError):
        boot_only_then_hot_attach(mutant)


def test_root_guard_rejects_clone_backing_system_paths():
    from scripts.study_operator.i7_service_disk import refuse_root_on_clone
    job = {'disk_name':TARGET}
    clone = lambda job:{11,12,13}
    def stat_for(devices):
        def stat(path):
            if path not in devices:
                raise FileNotFoundError(path)
            return SimpleNamespace(st_dev=devices[path])
        return stat
    for path in ('/','/boot','/boot/efi'):
        devices = {'/':1,'/boot':2,'/boot/efi':3}
        devices[path] = 12
        with pytest.raises(ValueError,match='ROOT_ON_SERVICE_CLONE'):
            refuse_root_on_clone(job,devices=clone,stat=stat_for(devices))
    refuse_root_on_clone(job,devices=clone,stat=stat_for({'/':1,'/boot':2}))
    with pytest.raises(ValueError,match='SERVICE_DISK_BY_ID_MISSING'):
        refuse_root_on_clone(job,devices=lambda job:set(),stat=stat_for({'/':1}))


@pytest.mark.skipif(os.name != 'posix', reason='POSIX device numbers')
def test_clone_devices_reads_real_by_id_links_and_filters_by_exact_name(tmp_path,monkeypatch):
    from scripts.study_operator import i7_service_disk
    (tmp_path/('google-'+TARGET+'-part1')).symlink_to('/dev/null')
    (tmp_path/'google-otro-part1').symlink_to('/dev/zero')
    # Same prefix, other disk: inside the glob, so only the exact-name filter excludes it.
    (tmp_path/('google-'+TARGET+'9-part1')).symlink_to('/dev/zero')
    monkeypatch.setattr(i7_service_disk,'BY_ID',tmp_path)
    assert i7_service_disk.clone_devices({'disk_name':TARGET}) == {os.stat('/dev/null').st_rdev}


def test_verify_mount_runs_root_guard_first(monkeypatch):
    from scripts.study_operator import i7_service_disk
    class Guarded(Exception):
        pass
    def guard(job):
        raise Guarded()
    monkeypatch.setattr(i7_service_disk,'refuse_root_on_clone',guard)
    with pytest.raises(Guarded):
        i7_service_disk.verify_mount(i7_service_disk.MOUNT,{'disk_name':TARGET,'cpu_id':'84'})


def test_mount_operation_waits_guards_mounts_then_verifies(tmp_path):
    from scripts.study_operator.i7_service_disk import mount_service
    calls = []
    job = {'disk_name':TARGET,'cpu_id':'84'}
    def run(argv,**kwargs):
        calls.append('mount' if argv[0] == 'mount' else argv[0])
    hooks = dict(wait=lambda job:calls.append('wait'),guard=lambda job:calls.append('guard'),run=run,
                 verify=lambda root,job:calls.append('verify'))
    assert mount_service(job,mount_point=tmp_path/'service',**hooks) == {'status':'OWN_SERVICE_DISK_MOUNTED'}
    assert calls == ['wait','guard','mount','verify']
    calls.clear()
    def refuse(job):
        calls.append('guard')
        raise ValueError('ROOT_ON_SERVICE_CLONE')
    with pytest.raises(ValueError,match='ROOT_ON_SERVICE_CLONE'):
        mount_service(job,mount_point=tmp_path/'other',**dict(hooks,guard=refuse))
    assert calls == ['wait','guard']


def test_by_id_wait_is_bounded():
    from scripts.study_operator.i7_service_disk import wait_target
    runs, ticks = [], iter(range(100))
    with pytest.raises(ValueError,match='SERVICE_DISK_BY_ID_TIMEOUT'):
        wait_target({'disk_name':TARGET},run=lambda argv,**kw:runs.append(argv),exists=lambda path:False,
                    sleep=lambda s:None,clock=lambda:next(ticks),timeout=3)
    assert runs[0][:2] == ['udevadm','settle']
    found = wait_target({'disk_name':TARGET},run=lambda argv,**kw:None,exists=lambda path:True,
                        sleep=lambda s:None,clock=lambda:0,timeout=3)
    assert found.name == 'google-'+TARGET+'-part1'
