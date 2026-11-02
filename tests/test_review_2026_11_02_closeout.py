"""Tests del cierre por inventario (docs/review/2026-11-02/closeout/closeout.py) con un gcloud falso.

Ninguna llamada sale del proceso: `runner` sustituye a subprocess. Cubre lectura sin efectos,
confirmación por recurso, re-describe con comparación de id, orden de borrado, protección contra
borrado solo en la VM que se borra, reanudación, regla permanente y recursos sin marcador.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'docs'/'review'/'2026-11-02'/'closeout'/'closeout.py'
spec = importlib.util.spec_from_file_location('review_closeout', PATH)
closeout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closeout)

TECH = '0123456789abcdef0123456789abcdef'
SMOKE = 'fedcba9876543210fedcba9876543210'
MARKER = 'cloudrag-i7-'+TECH
LABEL = 'i7cpu-'+TECH[:8]
CPU_MARKER = 'CloudRAG-I5-restore-cpu-'+LABEL
Z = 'https://www.googleapis.com/compute/v1/projects/p/zones/us-central1-a'
MUTATING = ('delete', 'update', 'create', 'stop', 'start', 'add-metadata', 'resize')


def world():
    """Restos de un intento: CPU (F-01/B-3), reserva sin VM (B-4), retenidos, ajenos y la regla permanente."""
    return {
        'instances': {
            'cloudrag-i5-restore-'+LABEL: dict(id='101', zone=Z, status='TERMINATED', deletionProtection=True,
                                               description=CPU_MARKER),
            'cloudrag-i7-vm-'+TECH[:12]: dict(id='102', zone=Z, status='TERMINATED', deletionProtection=True,
                                              description=MARKER),
            'cloudrag-i6-cowork': dict(id='103', zone=Z, status='TERMINATED', deletionProtection=True,
                                       description='CloudRAG-I6'),
        },
        'reservations': {
            'cloudrag-i7-1101172506-'+TECH[:8]: dict(id='201', zone=Z, status='READY', description=MARKER),
        },
        'disks': {
            'cloudrag-i5-restore-'+LABEL+'-boot': dict(id='301', zone=Z, sizeGb='180', status='READY', description=CPU_MARKER),
            'cloudrag-i7-service-'+TECH[:12]: dict(id='302', zone=Z, sizeGb='180', status='READY', description=MARKER),
            'cloudrag-i7-vm-'+TECH[:12]+'-boot': dict(id='303', zone=Z, sizeGb='180', status='READY', description=MARKER),
            'cloudrag-i7-unmarked': dict(id='304', zone=Z, sizeGb='10', status='READY', description=''),
        },
        'snapshots': {
            'cloudrag-i7-stage-'+TECH[:12]: dict(id='401', storageBytes='9', status='READY', description=MARKER),
            'cloudrag-i7-clean-'+TECH[:12]: dict(id='402', storageBytes='9', status='READY', description=MARKER),
            'cloudrag-i6-cierre-disco-20261016': dict(id='403', storageBytes='9', status='READY', description='i6'),
        },
        'firewall-rules': {
            'cloudrag-i5-restore-'+LABEL+'-iap': dict(id='501', description=CPU_MARKER),
            'cloudrag-i5-i6cowork-iap': dict(id='502', description=CPU_MARKER),
        },
    }


class FakeGcloud:
    def __init__(self, resources):
        self.resources = resources
        self.calls = []
        self.fail = {}          # (kind, verb, name) -> (code, stderr), consumido una vez

    def __call__(self, argv):
        assert argv[0] == 'gcloud' and argv[1] == 'compute' and '--project=pure-test-project' in argv
        self.calls.append(argv)
        kind, verb = argv[2], argv[3]
        name = argv[4] if len(argv) > 4 and not argv[4].startswith('--') else None
        if (kind, verb, name) in self.fail:
            return self.fail.pop((kind, verb, name))
        table = self.resources[kind]
        if verb == 'list':
            return 0, json.dumps([dict(v, name=k) for k, v in table.items()]), ''
        if name not in table:
            return 1, '', 'ERROR: (gcloud.compute.%s.%s) Could not fetch resource:\n - The resource was not found' % (kind, verb)
        if verb == 'describe':
            return 0, json.dumps(dict(table[name], name=name)), ''
        if verb == 'update':
            assert '--no-deletion-protection' in argv
            table[name]['deletionProtection'] = False
            return 0, '', ''
        if verb == 'delete':
            if kind == 'instances':
                assert '--keep-disks=all' in argv
                assert table[name]['deletionProtection'] is False
            del table[name]
            return 0, '', ''
        raise AssertionError(argv)

    def mutations(self):
        return [(c[2], c[3], c[4]) for c in self.calls if c[3] in MUTATING]


def root(tmp_path):
    folder = tmp_path/'root'
    folder.mkdir(exist_ok=True)
    (folder/'installation.json').write_text(json.dumps({
        'resource_marker': MARKER, 'period_ids': {'technical': TECH, 'smoke': SMOKE},
        'period_id': SMOKE, 'purpose': 'smoke'}))     # estado F-01: periodo del smoke persistido
    return folder


def read(tmp_path, cloud, *extra):
    out = io.StringIO()
    code = closeout.main(['--sdk', 'gcloud', '--project', 'pure-test-project', '--out', str(tmp_path/'out'), *extra],
                         runner=cloud, out=out)
    assert code == 0
    return json.loads((tmp_path/'out'/'inventory.json').read_text()), out.getvalue()


def remove(tmp_path, cloud, answers, *extra):
    asked = []
    def ask(prompt):
        asked.append(prompt)
        word = prompt.rsplit('Escribe ', 1)[1].split(' ', 1)[0]
        return answers(word, prompt)
    code = closeout.main(['--sdk', 'gcloud', '--project', 'pure-test-project', '--out', str(tmp_path/'out'),
                          '--delete', '--inventory', str(tmp_path/'out'/'inventory.json'), *extra],
                         runner=cloud, ask=ask, out=io.StringIO())
    log = [json.loads(line) for line in (tmp_path/'out'/'closeout-log.jsonl').read_text().splitlines()]
    return code, asked, log


def yes(word, prompt):
    return word


def names(rows):
    return {r['name'] for r in rows}


def test_default_mode_only_lists_and_never_mutates(tmp_path):
    cloud = FakeGcloud(world())
    saved, printed = read(tmp_path, cloud, '--root', str(root(tmp_path)))
    assert cloud.mutations() == []
    assert {c[3] for c in cloud.calls} == {'list'}
    assert names(saved['resources']) == {
        'cloudrag-i5-restore-'+LABEL, 'cloudrag-i7-vm-'+TECH[:12], 'cloudrag-i7-1101172506-'+TECH[:8],
        'cloudrag-i5-restore-'+LABEL+'-boot', 'cloudrag-i7-service-'+TECH[:12], 'cloudrag-i7-vm-'+TECH[:12]+'-boot',
        'cloudrag-i7-stage-'+TECH[:12], 'cloudrag-i7-clean-'+TECH[:12], 'cloudrag-i5-restore-'+LABEL+'-iap'}
    assert 'cloudrag-i5-i6cowork-iap' not in printed and 'cloudrag-i7-unmarked' not in printed
    row = next(r for r in saved['resources'] if r['name'] == 'cloudrag-i7-service-'+TECH[:12])
    assert (row['id'], row['status'], row['size'], row['class']) == ('302', 'READY', '180', 'temporary')
    assert len({r['confirm'] for r in saved['resources']}) == len(saved['resources'])
    assert 'nada borrado' in printed


def test_root_scope_uses_the_technical_period_even_after_smoke(tmp_path):
    """La raíz quedó con period_id del smoke (F-01): el alcance sigue reconociendo la CPU técnica."""
    assert closeout.root_scope(root(tmp_path)) == {'marker': MARKER, 'cpu_label': LABEL}


def test_another_attempts_marker_is_excluded_with_root_scope(tmp_path):
    resources = world()
    other = 'cloudrag-i7-'+'9'*32
    resources['disks']['cloudrag-i7-service-999999999999'] = dict(id='399', zone=Z, sizeGb='180', description=other)
    saved, _ = read(tmp_path, FakeGcloud(resources), '--root', str(root(tmp_path)))
    assert 'cloudrag-i7-service-999999999999' not in names(saved['resources'])
    # Sin --root se listan todos los intentos I7 (por ejemplo, restos del intento 6).
    tmp2 = tmp_path/'all'
    tmp2.mkdir()
    saved, _ = read(tmp2, FakeGcloud(copy.deepcopy(resources)))
    assert 'cloudrag-i7-service-999999999999' in names(saved['resources'])


def test_delete_requires_inventory_and_matching_project(tmp_path):
    cloud = FakeGcloud(world())
    with pytest.raises(SystemExit):
        closeout.main(['--sdk', 'gcloud', '--project', 'pure-test-project', '--out', str(tmp_path/'out'), '--delete'],
                      runner=cloud, ask=lambda p: '', out=io.StringIO())
    read(tmp_path, cloud)
    with pytest.raises(SystemExit):
        closeout.main(['--sdk', 'gcloud', '--project', 'other-test-project', '--out', str(tmp_path/'out'), '--delete',
                       '--inventory', str(tmp_path/'out'/'inventory.json')], runner=FakeGcloud(world()),
                      ask=lambda p: '', out=io.StringIO())
    assert cloud.mutations() == []


def test_each_resource_needs_its_own_word(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    words = []
    def reuse_first(word, prompt):
        words.append(word)
        return words[0]                 # repetir la primera palabra no confirma los demás
    code, asked, log = remove(tmp_path, cloud, reuse_first, '--root', str(root(tmp_path)))
    deleted = [m for m in cloud.mutations() if m[1] == 'delete']
    assert deleted == [('instances', 'delete', 'cloudrag-i5-restore-'+LABEL)]
    assert len(set(words)) == len(words)
    assert sum(e.get('result') == 'NOT_CONFIRMED_SKIPPED' for e in log) == len(asked) - 1


def test_full_closeout_order_protection_and_retained_resources(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    code, asked, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))
    assert code == 0
    assert cloud.mutations() == [
        ('instances', 'update', 'cloudrag-i5-restore-'+LABEL),
        ('instances', 'delete', 'cloudrag-i5-restore-'+LABEL),
        ('reservations', 'delete', 'cloudrag-i7-1101172506-'+TECH[:8]),
        ('disks', 'delete', 'cloudrag-i5-restore-'+LABEL+'-boot'),
        ('disks', 'delete', 'cloudrag-i7-service-'+TECH[:12]),
        ('snapshots', 'delete', 'cloudrag-i7-stage-'+TECH[:12]),
        ('firewall-rules', 'delete', 'cloudrag-i5-restore-'+LABEL+'-iap')]
    # Protección quitada solo en la VM borrada; la GPU retenida conserva la suya.
    assert cloud.resources['instances']['cloudrag-i7-vm-'+TECH[:12]]['deletionProtection'] is True
    retained = {e['resource'] for e in log if e.get('result') == 'RETAINED_SKIPPED'}
    assert retained == {'cloudrag-i7-vm-'+TECH[:12], 'cloudrag-i7-vm-'+TECH[:12]+'-boot', 'cloudrag-i7-clean-'+TECH[:12]}
    assert 'cloudrag-i5-i6cowork-iap' in cloud.resources['firewall-rules']
    assert 'cloudrag-i7-unmarked' in cloud.resources['disks']
    assert 'cloudrag-i6-cowork' in cloud.resources['instances']
    assert sum(e.get('result') == 'DELETED_VERIFIED' for e in log) == 6


def test_include_retained_deletes_gpu_vm_disk_and_clean_snapshot(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)), '--include-retained')
    assert not cloud.resources['reservations'] and not cloud.resources['snapshots'].get('cloudrag-i7-clean-'+TECH[:12])
    assert set(cloud.resources['instances']) == {'cloudrag-i6-cowork'}
    assert set(cloud.resources['disks']) == {'cloudrag-i7-unmarked'}
    assert set(cloud.resources['firewall-rules']) == {'cloudrag-i5-i6cowork-iap'}


def test_identity_change_between_inventory_and_delete_aborts_only_that_resource(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    cloud.resources['disks']['cloudrag-i7-service-'+TECH[:12]]['id'] = '999'      # recreado con otro id
    cloud.resources['snapshots']['cloudrag-i7-stage-'+TECH[:12]]['description'] = 'otro dueño'
    code, _, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))
    assert code == 1
    aborted = {e['resource'] for e in log if e.get('result') == 'ABORTED_IDENTITY_CHANGED'}
    assert aborted == {'cloudrag-i7-service-'+TECH[:12], 'cloudrag-i7-stage-'+TECH[:12]}
    assert 'cloudrag-i7-service-'+TECH[:12] in cloud.resources['disks']
    assert 'cloudrag-i5-restore-'+LABEL+'-boot' not in cloud.resources['disks']


def test_tampered_inventory_cannot_reach_permanent_or_unmarked_resources(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud)
    path = tmp_path/'out'/'inventory.json'
    saved = json.loads(path.read_text())
    for kind, name, rid in (('firewall-rules', 'cloudrag-i5-i6cowork-iap', '502'), ('disks', 'cloudrag-i7-unmarked', '304'),
                            ('instances', 'cloudrag-i6-cowork', '103')):
        saved['resources'].append({'kind': kind, 'name': name, 'id': rid, 'zone': 'us-central1-a', 'class': 'temporary'})
    path.write_text(json.dumps(saved))
    _, _, log = remove(tmp_path, cloud, yes)
    assert 'cloudrag-i5-i6cowork-iap' in cloud.resources['firewall-rules']
    assert 'cloudrag-i7-unmarked' in cloud.resources['disks'] and 'cloudrag-i6-cowork' in cloud.resources['instances']
    results = {e['resource']: e['result'] for e in log if 'resource' in e}
    assert results['cloudrag-i5-i6cowork-iap'] == 'PERMANENT_NEVER_TOUCHED'
    assert results['cloudrag-i7-unmarked'] == results['cloudrag-i6-cowork'] == 'ABORTED_IDENTITY_CHANGED'


def test_resumes_after_partial_failure_and_logs_absent_resources(tmp_path):
    """Caso B-3: la VM ya se borró y falló un disco; el segundo paso registra la ausencia y termina."""
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    cloud.fail[('disks', 'delete', 'cloudrag-i7-service-'+TECH[:12])] = (1, '', 'ERROR: Operation timed out')
    code, _, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))
    assert code == 1 and any(e.get('result') == 'DELETE_ERROR_RETRY_LATER' for e in log)
    assert 'cloudrag-i7-service-'+TECH[:12] in cloud.resources['disks']
    code, _, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))   # mismo inventario
    assert code == 0
    second = [e for e in log if e.get('mode') == 'DELETE'][-1]
    tail = log[log.index(second)+1:]
    assert {e['resource'] for e in tail if e.get('result') == 'ALREADY_ABSENT'} >= {
        'cloudrag-i5-restore-'+LABEL, 'cloudrag-i7-1101172506-'+TECH[:8], 'cloudrag-i5-restore-'+LABEL+'-iap'}
    assert 'cloudrag-i7-service-'+TECH[:12] not in cloud.resources['disks']


def test_describe_error_other_than_not_found_is_not_treated_as_absent(tmp_path):
    cloud = FakeGcloud(world())
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    cloud.fail[('disks', 'describe', 'cloudrag-i7-service-'+TECH[:12])] = (1, '', 'ERROR: PERMISSION_DENIED')
    code, _, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))
    assert code == 1
    assert {e['resource']: e['result'] for e in log if 'resource' in e}['cloudrag-i7-service-'+TECH[:12]] == 'DESCRIBE_ERROR_SKIPPED'
    assert 'cloudrag-i7-service-'+TECH[:12] in cloud.resources['disks']


def test_running_vm_and_consumed_reservation_are_not_deleted(tmp_path):
    resources = world()
    resources['instances']['cloudrag-i5-restore-'+LABEL]['status'] = 'RUNNING'
    resources['instances']['cloudrag-i7-vm-'+TECH[:12]]['reservationAffinity'] = {
        'consumeReservationType': 'SPECIFIC_RESERVATION', 'values': ['cloudrag-i7-1101172506-'+TECH[:8]]}
    cloud = FakeGcloud(resources)
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    code, _, log = remove(tmp_path, cloud, yes, '--root', str(root(tmp_path)))
    results = {e['resource']: e['result'] for e in log if 'resource' in e}
    assert code == 1
    assert results['cloudrag-i5-restore-'+LABEL] == 'ABORTED_VM_NOT_TERMINATED'
    assert results['cloudrag-i7-1101172506-'+TECH[:8]] == 'ABORTED_RESERVATION_HAS_CONSUMER'
    assert cloud.resources['instances']['cloudrag-i5-restore-'+LABEL]['deletionProtection'] is True


def test_b4_case_reservation_without_any_vm_is_deleted(tmp_path):
    """Caso B-4: gpu_restore falló tras reserve(13.5); no existe VM GPU ni disco."""
    resources = world()
    del resources['instances']['cloudrag-i7-vm-'+TECH[:12]]
    del resources['disks']['cloudrag-i7-vm-'+TECH[:12]+'-boot']
    cloud = FakeGcloud(resources)
    read(tmp_path, cloud, '--root', str(root(tmp_path)))
    remove(tmp_path, cloud, lambda word, prompt: word if 'reservations' in prompt else 'no', '--root', str(root(tmp_path)))
    assert cloud.mutations() == [('reservations', 'delete', 'cloudrag-i7-1101172506-'+TECH[:8])]
    assert not cloud.resources['reservations']
