"""cpu-build-04/06 (ADR-0023, ADR-0024): a boot-disk clone present at boot can become / or /boot."""
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

PERIOD = '5e9d9dac3ca74bfb95468333b0b2820f'
MARKER = 'cloudrag-i7-'+PERIOD
TARGET = 'cloudrag-i7-service-'+PERIOD[:12]
STAGE = 'cloudrag-i7-stage-'+PERIOD[:12]
BOOT_WAIT = ['compute','ssh','rpc:boot-status']


class StopAtMountSsh(Exception):
    pass


def boots_then_hot_attach(commands):
    """Return True only for start -> guest-boot wait -> attach-disk(clone) -> mount rpc."""
    names = [tuple(c[1:3]) for c in commands]
    start = names.index(('instances','start'))
    wait = commands.index(BOOT_WAIT)
    attach = next(i for i,c in enumerate(commands)
                  if tuple(c[1:3]) == ('instances','attach-disk') and '--disk='+TARGET in c)
    mount = commands.index(['compute','ssh','rpc:mount'])
    assert start < wait < attach < mount, 'clone must be hot-attached only after the guest finished booting'
    return True


def clean_run(tmp_path, boot_state):
    from scripts.study_operator.i7_cpu_pipeline import clean_snapshot
    commands = []
    class Cloud:
        def command(self,args,**kwargs):
            if args[:2] == ['compute','ssh']:
                # Both waits go through the pinned private RPC (input_data present).
                request = json.loads(kwargs['input_data'])
                commands.append(['compute','ssh','rpc:'+request['operation']])
                if request['operation'] == 'boot-status':
                    assert 1 <= request['wait_s'] <= 300 and kwargs['timeout'] <= 420
                    return json.dumps({'state':boot_state}).encode()
                raise StopAtMountSsh()
            commands.append(list(args))
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
    return commands,lambda:clean_snapshot(Operator(),Cloud(),resources,
                                          {'status':'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED'})


def test_clean_snapshot_waits_for_guest_boot_then_hot_attaches_clone(tmp_path):
    commands,run = clean_run(tmp_path,'running')
    with pytest.raises(StopAtMountSsh):
        run()
    assert boots_then_hot_attach(commands)
    assert commands[-1][1:3] == ['instances','stop']  # finally still stops the CPU VM


def test_no_attach_when_guest_boot_wait_expires(tmp_path):
    commands,run = clean_run(tmp_path,'starting')
    with pytest.raises(ValueError,match='GUEST_BOOT_WAIT_TIMEOUT'):
        run()
    assert BOOT_WAIT in commands
    assert not any(c[1:3] == ['instances','attach-disk'] for c in commands)
    assert commands[-1][1:3] == ['instances','stop']


def test_order_checker_rejects_attach_before_boot_wait_mutant():
    start, attach = ['compute','instances','start','vm'], ['compute','instances','attach-disk','vm','--disk='+TARGET]
    mount = ['compute','ssh','rpc:mount']
    for mutant in ([attach,start,BOOT_WAIT,mount],[start,attach,BOOT_WAIT,mount]):
        with pytest.raises(AssertionError):
            boots_then_hot_attach(mutant)


class Sentinel(Exception):
    pass


def test_guest_boot_wait_is_bounded():
    from scripts.study_operator.i7_cpu_pipeline import wait_guest_booted
    from scripts.study_operator.policy import ReadyPending
    calls, ticks, sleeps = [], iter(range(0,100000,10)), []
    def pending(request,timeout):
        calls.append((request,timeout))
        raise ReadyPending('keys pending')
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 50:
            raise Sentinel()  # an unbounded wait would end here instead of timing out
    with pytest.raises(ValueError,match='GUEST_BOOT_WAIT_TIMEOUT'):
        wait_guest_booted(pending,clock=lambda:next(ticks),sleep=sleep,timeout=300)
    assert calls and all(r['operation'] == 'boot-status' and 1 <= r['wait_s'] <= 300 and t <= 420
                         for r,t in calls)
    for state in ('running','degraded'):
        wait_guest_booted(lambda r,timeout,s=state:{'state':s},clock=lambda:0,sleep=sleep)
    for state in ('starting','initializing','timeout',''):
        with pytest.raises(ValueError,match='GUEST_BOOT_WAIT_TIMEOUT'):
            wait_guest_booted(lambda r,timeout,s=state:{'state':s},clock=lambda:0,sleep=sleep)


def test_guest_boot_status_is_bounded_and_reads_systemd():
    from scripts.study_operator.i7_service_disk import boot_status
    seen = []
    def run(argv,**kwargs):
        seen.append((argv,kwargs['timeout']))
        return SimpleNamespace(stdout=b'degraded\n',returncode=1)  # degraded still means booted
    assert boot_status({'wait_s':120},run=run) == {'state':'degraded'}
    assert seen == [(['systemctl','is-system-running','--wait'],120)]
    def slow(argv,**kwargs):
        raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
    assert boot_status({'wait_s':5},run=slow) == {'state':'timeout'}
    for wait in (0,301,None,'60',True):
        with pytest.raises(ValueError,match='BOOT_WAIT_BOUND_REQUIRED'):
            boot_status({'wait_s':wait},run=run)


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
