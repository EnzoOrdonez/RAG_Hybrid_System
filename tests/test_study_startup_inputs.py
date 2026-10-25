import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.study_operator.startup import write_startup
from scripts.study_operator.startup_inputs import FILES, assemble, guest_source, staging_program, validate


def inputs(tmp_path):
    reviewed = tmp_path/'reviewed'
    reviewed.mkdir()
    for name in FILES:
        if name.startswith('reviewed/'):
            (reviewed/name.split('/')[-1]).write_bytes(('synthetic '+name).encode())
    manifest, prereg = tmp_path/'manifest.json', tmp_path/'prereg.md'
    manifest.write_bytes(b'synthetic artifact inventory')
    prereg.write_bytes(b'synthetic preregistration')
    return assemble(reviewed, manifest, prereg)


def config(value):
    root = value['root']
    return dict(bootstrap_inputs=value, reviewed_config=root+'/reviewed',
                artifact_manifest=root+'/deployment-artifacts.json', preregistration_file=root+'/service-preregistration.md')


def run_guest(value, tmp_path, monkeypatch):
    namespace = {'Path': lambda path: tmp_path/'guest-inputs' if path == value['root'] else Path(path)}
    exec(compile(guest_source(), 'guest-public-inputs', 'exec'), namespace)
    # The production guest is POSIX; emulate its directory fsync primitive in
    # this Windows functional test, not a claimed POSIX durability measurement.
    monkeypatch.setattr(namespace['os'], 'fsync', lambda fd: None)
    original = namespace['os'].open
    monkeypatch.setattr(namespace['os'], 'open', lambda path, flags: original(tmp_path/'fd', flags))
    (tmp_path/'fd').write_bytes(b'fixture directory fd')
    namespace['materialize'](value)
    return namespace['materialize'], value


def test_guest_stages_every_input_then_accepts_same_inventory(tmp_path, monkeypatch):
    value = inputs(tmp_path)
    apply, guest = run_guest(value, tmp_path, monkeypatch)
    expected = {r['name']: base64.b64decode(r['base64']) for r in value['files']}
    for name, data in expected.items():
        assert (tmp_path/'guest-inputs'/name).read_bytes() == data
    apply(guest)
    target = tmp_path/'guest-inputs'/'reviewed/study.json'
    target.write_bytes(b'synthetic prior content must survive')
    with pytest.raises(ValueError, match='ALREADY_DIFFERS'):
        apply(guest)
    assert target.read_bytes() == b'synthetic prior content must survive'


def test_guest_validates_whole_inventory_before_any_write(tmp_path, monkeypatch):
    value = inputs(tmp_path)
    value['files'][-1]['sha256'] = '0'*64
    value['root'] = '/srv/cloudrag/iteration5/inputs/'+hashlib.sha256(
        json.dumps(value['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='HASH_CHANGED'):
        run_guest(value, tmp_path, monkeypatch)
    assert not (tmp_path/'guest-inputs').exists()


@pytest.mark.parametrize('alteration', ['root', 'extra', 'duplicate', 'payload', 'digest'])
def test_inventory_tamper_rejected_before_startup_write(tmp_path, alteration):
    value = inputs(tmp_path)
    if alteration == 'root':
        value['root'] = '/srv/cloudrag/iteration4/inputs/'+'a'*64
    elif alteration == 'extra':
        value['files'][0]['name'] = '../session.json'
    elif alteration == 'duplicate':
        value['files'].append(copy.deepcopy(value['files'][0]))
    elif alteration == 'payload':
        value['files'][0]['base64'] = base64.b64encode(b'altered').decode()
    else:
        value['root'] = '/srv/cloudrag/iteration5/inputs/'+'a'*64
    with pytest.raises(ValueError):
        write_startup(tmp_path/'startup.sh', config(value))
    assert not (tmp_path/'startup.sh').exists()


def test_startup_stages_inputs_before_host_launch_without_new_host_module(tmp_path):
    value = inputs(tmp_path)
    validate(value)
    source = write_startup(tmp_path/'startup.sh', config(value)).read_text()
    ast.parse(source.split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    assert source.index("materialize(c['bootstrap_inputs'])") < source.index('subprocess.Popen')
    assert 'scripts.study_operator.startup_inputs' not in source
    assert staging_program({}) == ''
    changed = dict(config(value), reviewed_config='/srv/cloudrag/iteration4/old')
    with pytest.raises(ValueError, match='paths differ'):
        staging_program(changed)
