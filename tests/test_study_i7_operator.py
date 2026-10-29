import copy
import json
from datetime import datetime, timezone

import pytest

from scripts.study_operator.i7 import Operator, reservation_charge
from tests.test_study_operator_deployment import config as deployment_config


def setup(tmp_path):
    from scripts.study_operator.i7_costs import KEYS
    cfg = dict(deployment_config(),iteration=7,service_account='sa',network='default',subnet='default',
        primary_vm={'name':'cloudrag-i7-vm','id':'10','zone':'us-central1-a'},
        native_stop_s=10800,cuda_cache='/srv/cloudrag/i7/jit',cuda_cache_seed={'k':'a'*64},
        cuda_cache_driver='580.178.04',budget_usd=20,closure_reserve_usd=3,
        rates={'compute_usd_h':.706832255,'price_receipt_sha256':'a'*64,'verified_utc':'2026-11-01T00:00:00+00:00'},
        resource_marker='cloudrag-i7-test')
    cfg['cost_plan']={'status':'PRICED_LIST','verified_utc':'2026-11-01T00:00:00+00:00',
        'rates':{key:{'usd':.000001,'sku_id':key} for key in KEYS}}
    (tmp_path/'installation.json').write_text(json.dumps(cfg))
    (tmp_path/'I7.json').write_text(json.dumps({'iteration':7}))
    reservation = {'name': 'cloudrag-i7-res','id': '20','zone': 'zones/us-central1-a','description': 'cloudrag-i7-test',
        'specificReservation': {'count': '1','instanceProperties': {'machineType': 'g2-standard-4',
                'guestAccelerators': [{'acceleratorType': 'nvidia-l4','acceleratorCount': 1}]}}}
    class Cloud:
        def require_beta(self):
            return {'status': 'GCLOUD_BETA_INSTALLED', 'beta_version': 'fixture'}

        def __init__(self):
            self.calls=[]
            self.error=None
            self.exists=True
        def command(self,argv,**kwargs):
            self.calls.append(argv)
            if argv[:3] == ['compute','reservations','list']:
                if self.error == 'list':
                    raise RuntimeError('API_FAILED')
                return [copy.deepcopy(reservation)] if self.exists else []
            if argv[:3] == ['compute','reservations','describe']:
                return copy.deepcopy(reservation)
            if argv[:3] == ['compute','reservations','delete']:
                if self.error == 'delete':
                    raise RuntimeError('API_FAILED')
                self.exists=False
                return None
            if argv[:3] == ['compute','instances','describe']:
                return {'name': cfg['primary_vm']['name'],'id': '10','zone': 'zones/us-central1-a','status': 'TERMINATED'}
            raise AssertionError(argv)
    cloud = Cloud()
    op = Operator(tmp_path,cloud,now=lambda:datetime(2026,11,1,tzinfo=timezone.utc))
    op.state['reservation'] = dict(reservation,created_utc='2026-11-01T00:00:00+00:00',
        cost_basis={'hours':12,'compute_usd_h':cfg['rates']['compute_usd_h'],
                    'exposure_usd':12*cfg['rates']['compute_usd_h']+.25,
                    'price_receipt_sha256':cfg['rates']['price_receipt_sha256']})
    return op,cloud,reservation


def test_guard_requires_delete_success_identity_and_absence(tmp_path):
    op,cloud,_ = setup(tmp_path)
    assert op.guard()['status'] == 'RESERVATION_DELETED_VERIFIED'
    assert cloud.calls[-1][:3] == ['compute','reservations','list']
    assert op.state['reservation']['deleted_verified']


@pytest.mark.parametrize('failure',['delete','list','identity'])
def test_guard_never_reports_deleted_on_failed_api_or_replacement(tmp_path,failure):
    op,cloud,_ = setup(tmp_path)
    if failure == 'identity':
        op.state['reservation']['id']='wrong'
    else:
        cloud.error=failure
    with pytest.raises((ValueError,RuntimeError)):
        op.guard()
    assert not op.state['reservation'].get('deleted_verified')


def test_no_paid_effect_before_floor_and_budget_reserves_closure(tmp_path):
    op,cloud,_ = setup(tmp_path)
    op.now=lambda:datetime(2026,10,16,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='^NO_PAID_EFFECT_BEFORE_2026_10_28$'):
        op.admit_paid(1)
    assert not cloud.calls
    op.now=lambda:datetime(2026,11,1,tzinfo=timezone.utc)
    op.state['spent_usd']=16.9
    with pytest.raises(ValueError,match='BUDGET'):
        op.admit_paid(.2)
    op.admit_paid(.1)


def test_reservation_cost_includes_idle_and_does_not_double_bill_consumed_vm():
    assert reservation_charge([('2026-11-01T00:00:00Z','2026-11-01T02:00:00Z')],
                              [('2026-11-01T01:00:00Z','2026-11-01T03:00:00Z')],1) == 3


def test_three_ready_rows_cannot_hide_an_intervening_failed_boot():
    from scripts.study_operator.i7_workflows import check_tls_sequence
    rows = [{'start_requested_utc': str(i),'status': 'READY','boot_id': str(i),'start_to_ready_s': 1,
                 'url': 'https://same','source_commit': 'a','tls': {'certificate_sha256':'b'}} for i in range(4)]
    rows[2]={'start_requested_utc':'2','status':'FAILED'}
    with pytest.raises(ValueError):
        check_tls_sequence(rows)


def test_version_archive_is_immutable(tmp_path):
    from scripts.study_operator.i7_workflows import archive_tools
    source = tmp_path/'source'
    (source/'scripts').mkdir(parents=True)
    (source/'scripts'/'tool.py').write_bytes(b'original')
    archive_tools(source,tmp_path/'versions','a'*40)
    (source/'scripts'/'tool.py').write_bytes(b'changed')
    with pytest.raises(ValueError):
        archive_tools(source,tmp_path/'versions','a'*40)


def test_unverified_download_never_acknowledges_shutdown(tmp_path,monkeypatch):
    op,_,_ = setup(tmp_path)
    calls=[]
    def bridge(request,**kwargs):
        calls.append(request)
        return {'proof':{},'proof_sha256':'0'*64}
    monkeypatch.setattr(op,'bridge',bridge)
    with pytest.raises(ValueError,match='HASH'):
        op.collect('stimulus')
    assert calls == [{'operation':'stimulus-evidence'}]


def test_scheduler_registers_without_administrator_and_pins_version(tmp_path,monkeypatch):
    import subprocess
    import sys

    from scripts.study_operator.i7_workflows import schedule
    op,cloud,_ = setup(tmp_path)
    cloud.sdk='gcloud.cmd'
    monkeypatch.setattr(sys,'platform','win32')
    captured=[]
    def invoke(argv,**kwargs):
        captured.append(argv)
        return subprocess.CompletedProcess(argv,0)
    result=schedule(op,invoke=invoke)
    text=(tmp_path/'scheduled-series.ps1').read_text(encoding='utf8')
    assert 'RunLevel Limited' in text and 'Highest' not in text
    assert result['tool_version']['files'] > 0
    assert len(captured)==1


def test_unknown_reservation_intent_blocks_another_create(tmp_path):
    op,cloud,_=setup(tmp_path)
    op.state.pop('reservation')
    op.state['reservation_intent']={'status':'UNKNOWN'}
    with pytest.raises(ValueError,match='UNKNOWN'):
        op.reserve(1)
    assert not cloud.calls


def test_successful_delete_with_failed_readback_can_be_reconciled_by_read_only(tmp_path):
    op,cloud,_=setup(tmp_path)
    cloud.error='list'
    with pytest.raises(RuntimeError):
        op.guard()
    cloud.error=None
    before=len(cloud.calls)
    result=op.reconcile()
    assert result['status']=='RESERVATION_ABSENT_RECONCILED'
    assert op.state['reservation']['deleted_verified']
    assert all(c[:3]==['compute','reservations','list'] for c in cloud.calls[before:])


def test_paid_floor_is_owner_dated_2026_10_28_adr_0016(tmp_path):
    from scripts.study_operator.i7 import PAID_NOT_BEFORE_UTC
    assert PAID_NOT_BEFORE_UTC == datetime(2026,10,28,tzinfo=timezone.utc)
    op,cloud,_ = setup(tmp_path)
    op.config['rates']['verified_utc'] = op.config['cost_plan']['verified_utc'] = '2026-10-27T00:00:00+00:00'
    op.now=lambda:datetime(2026,10,27,23,59,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='^NO_PAID_EFFECT_BEFORE_2026_10_28$'):
        op.admit_paid(.1)
    op.now=lambda:PAID_NOT_BEFORE_UTC
    op.admit_paid(.1)
    assert not cloud.calls
