import copy
import getpass
import sys
from types import SimpleNamespace
import pytest

from scripts.study_operator.deployment import app_command, assert_isolation, caddyfile, checked_config
from scripts.study_operator.policy import OperatorError


def config():
    return dict(schema_version=1, project='pure-loop-474323-a8', zone='us-central1-a',
        machine_type='g2-standard-4', sessions_bucket='cloudrag-study-i4-103950017681-20261004',
        technical_bucket='cloudrag-study-103950017681-20261002', purpose='smoke', period_id='a'*32,
        image_id='sha256:'+'b'*64, commit='c'*40, model_digest='d'*64, static_ip='203.0.113.8',
        hostname='203.0.113.8.sslip.io', ollama_image='ollama/ollama@sha256:'+'e'*64,
        caddy_image='caddy@sha256:'+'f'*64, asset_root='/srv/cloudrag/assets',
        ollama_models='/srv/cloudrag/models', host_code='/srv/cloudrag/iteration4/code')


@pytest.mark.parametrize('key,value', [('zone','europe-west1-b'), ('project','other'),
    ('hostname','other.sslip.io'), ('ollama_image','ollama:latest'), ('host_code','/etc'),
    ('period_id','../anything')])
def test_installation_scope_is_fail_closed(key, value):
    value_config = config()
    value_config[key] = value
    with pytest.raises((OperatorError, ValueError)):
        checked_config(value_config)


@pytest.mark.parametrize('zone', ['us-east1-b','us-west4-c','us-west1-a'])
def test_authorized_us_regions_keep_frozen_generation_and_isolation(zone):
    value = config()
    value['zone'] = zone
    assert checked_config(value)['zone'] == zone
    command = app_command(value,'/srv/cloudrag/iteration5/boot','/srv/cloudrag/iteration5/sessions','app')
    assert command[command.index('--network')+1] == 'none'
    assert 'USER=cloudrag' in command


def test_app_and_freeze_share_service_and_generation_environment():
    serving = app_command(config(), '/srv/cloudrag/iteration4/boot', '/srv/cloudrag/iteration4/sessions', 'app')
    freezing = app_command(config(), '/srv/cloudrag/iteration4/boot', '/srv/cloudrag/iteration4/sessions', 'freeze',
        operation='freeze', output='/deployment/deployment.json')
    for command in (serving, freezing):
        assert command[command.index('--network')+1] == 'none'
        assert command[command.index('--user')+1] == '10001:10001'
        assert '--read-only' in command and '--rm=false' in command
        assert 'CLOUDRAG_ISOLATED_SERVICE=1' in command
        assert 'CLOUDRAG_DEMO_GPU=1' in command and 'CUDA_VISIBLE_DEVICES=0' in command
        assert not any('/opt/cloudrag/repository/data/llm_cache:' in argument for argument in command)
        assert 'type=bind,source=/srv/cloudrag/assets/data/models,target=/opt/cloudrag/repository/data/models,readonly' in command
        assert 'type=bind,source=/srv/cloudrag/iteration4/boot/embeddings-initialization,target=/opt/cloudrag/repository/data/embeddings,readonly' in command
        assert '/service/generation.sock' in command
    assert 'type=bind,source=/srv/cloudrag/iteration4/boot/meta,target=/deployment,readonly' in serving
    assert 'type=bind,source=/srv/cloudrag/iteration4/boot/meta,target=/deployment' in freezing


def test_numeric_uid_without_passwd_uses_technical_runtime_user(monkeypatch):
    """Torch's default cache path must work without a named /etc/passwd entry."""
    def unknown_uid(uid):
        raise KeyError(uid)
    monkeypatch.setitem(sys.modules, 'pwd', SimpleNamespace(getpwuid=unknown_uid))
    monkeypatch.setattr(getpass.os, 'getuid', lambda: 10001, raising=False)
    monkeypatch.setattr(getpass.os, 'environ', {})
    with pytest.raises((KeyError, OSError)):
        getpass.getuser()
    command = app_command(config(), '/srv/cloudrag/iteration4/boot',
                          '/srv/cloudrag/iteration4/sessions', 'app')
    environment = dict(argument.split('=', 1) for index, argument in enumerate(command)
                       if index and command[index-1] == '-e')
    monkeypatch.setattr(getpass.os, 'environ', environment)
    assert getpass.getuser() == 'cloudrag'
    assert environment['HOME'] == '/tmp'
    assert command[command.index('--user')+1] == '10001:10001'


def isolated():
    return dict(Image='sha256:'+'b'*64, Config={'User':'10001:10001', 'Env':['USER=cloudrag']},
        HostConfig=dict(NetworkMode='none', PidMode='', ReadonlyRootfs=True, CapDrop=['ALL'],
            SecurityOpt=['no-new-privileges'], LogConfig={'Type':'none'},
            Tmpfs={'/tmp':'rw,size=512m'}),
        Mounts=[dict(Destination=p, RW=False) for p in
            ['/service','/deployment','/reviewed','/opt/cloudrag/repository/data/models',
             '/opt/cloudrag/repository/data/indices','/opt/cloudrag/repository/data/embeddings']])


@pytest.mark.parametrize('mutation', ['missing','writable'])
def test_embedding_initialization_cannot_write_query_caches(mutation):
    value = isolated()
    mount = next(row for row in value['Mounts']
                 if row['Destination'] == '/opt/cloudrag/repository/data/embeddings')
    if mutation == 'missing':
        value['Mounts'].remove(mount)
    else:
        mount['RW'] = True
    with pytest.raises(OperatorError, match='solo lectura'):
        assert_isolation(value, 'sha256:'+'b'*64)


@pytest.mark.parametrize('mutation', ['network','pid','socket','logs','mount','cache','cache_file','source_parent','runtime_user'])
def test_actual_docker_isolation_rejects_escape_paths(mutation):
    value = copy.deepcopy(isolated())
    if mutation == 'runtime_user':
        value['Config']['Env'] = ['USER=root']
    elif mutation in ('network','pid'):
        value['HostConfig']['NetworkMode' if mutation == 'network' else 'PidMode'] = 'host'
    elif mutation == 'socket':
        value['Mounts'].append(dict(Destination='/var/run/docker.sock',RW=False))
    elif mutation == 'logs':
        value['HostConfig']['LogConfig']['Type'] = 'json-file'
    elif mutation == 'cache':
        value['HostConfig']['Tmpfs']['/opt/cloudrag/repository/data/llm_cache'] = 'rw,size=8m'
    elif mutation == 'cache_file':
        value['Mounts'].append(dict(Destination='/opt/cloudrag/repository/data/llm_cache/granite4.1_8b_cache.json',RW=True))
    elif mutation == 'source_parent':
        value['Mounts'].append(dict(Destination='/opt/cloudrag/repository/data',RW=False))
    else:
        value['Mounts'][0]['RW'] = True
    with pytest.raises(OperatorError):
        assert_isolation(value, 'sha256:'+'b'*64)


def test_isolation_and_caddy_fixed_privacy_configuration():
    assert assert_isolation(isolated(), 'sha256:'+'b'*64)['container_isolation_verified']
    text = caddyfile(config()['hostname'])
    assert text.count('output discard') == 2
    assert 'admin off' in text and 'persist_config off' in text
    assert 'disable_http_challenge' in text and 'ZeroSSL' not in text
    assert 'header_up -User-Agent' in text and 'unix//web/streamlit.sock' in text
