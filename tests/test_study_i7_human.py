"""Human phase on the I7 deployment (F-020, ADR-0018): GO-gated, separate USD 70 ledger."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from tests.test_study_i7_operator import setup


def gate_receipt(root, decision, name='a', *, image=None, write_proof=True):
    from scripts.study_operator.stimulus_host_evidence import sha
    config = json.loads((root/'installation.json').read_bytes())
    boot = name.ljust(8,'0')+'-0000-0000-0000-000000000000'
    proof = {'inventory':{'image':{'image_id':image or config['image_id']},'source':{'commit':config['commit']}}}
    private = root/'private'/'gate'/'coded-P999'
    private.mkdir(parents=True,exist_ok=True)
    path = private/(boot+'.json')
    if write_proof:
        path.write_text(json.dumps(proof,sort_keys=True))
    folder = root/'receipts'/'gate'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps({'status':'DOWNLOAD_VERIFIED','verified':{'decision':decision},
        'boot_id':boot,'proof_sha256':sha(proof),
        'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if write_proof else 'c'*64}))


def closed_phase_b(tmp_path, decision='GO'):
    op,cloud,_ = setup(tmp_path)
    op.state['reservation']['deleted_verified'] = True
    op.state['spent_usd'] = 15.0
    if decision:
        gate_receipt(tmp_path,decision)
    return op,cloud


@pytest.mark.parametrize('decisions',[[],['NO-GO'],['GO','NO-GO'],['INCOMPLETE']])
def test_human_phase_requires_only_verified_go(tmp_path,decisions):
    op,cloud = closed_phase_b(tmp_path,None)
    for index,decision in enumerate(decisions):
        gate_receipt(tmp_path,decision,str(index))
    with pytest.raises(ValueError,match='VERIFIED_GATE_GO_REQUIRED'):
        op.human_phase_open()
    assert 'human_phase' not in op.state and not cloud.calls


def test_human_phase_requires_closed_phase_b_reservation(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.state['reservation'].pop('deleted_verified')
    with pytest.raises(ValueError,match='PHASE_B_RESERVATION_CLOSE_REQUIRED'):
        op.human_phase_open()


def test_human_phase_opens_once_and_keeps_phase_b_spend_apart(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    receipt = op.human_phase_open()
    assert receipt['status'] == 'HUMAN_PHASE_OPEN' and receipt['budget_usd'] == 70
    assert op.state['human_phase']['phase_b_spent_usd'] == 15.0
    assert len(op.state['human_phase']['gate_receipt_sha256']) == 64
    with pytest.raises(ValueError,match='HUMAN_PHASE_ALREADY_OPEN'):
        op.human_phase_open()


def test_human_start_requires_open_phase(tmp_path):
    op,cloud = closed_phase_b(tmp_path)
    with pytest.raises(ValueError,match='I7_HUMAN_PHASE_NOT_OPEN'):
        op.start('rehearsal')
    assert not cloud.calls


def test_human_purpose_uses_standard_profile_and_study_needs_ethics(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.config['i7_profile'] = 'gate'
    op.human_phase_open()
    assert op.human_start_profile('rehearsal') == {'i7_profile':'standard'}
    with pytest.raises(Exception,match='tica'):
        op.human_start_profile('study')
    with pytest.raises(ValueError,match='I7_PHASE_B_SYNTHETIC_TECHNICAL_ONLY'):
        op.human_start_profile('participants')


def test_human_ledger_replaces_phase_b_window_and_cap(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.human_phase_open()
    op.state['budget_started_utc'] = '2026-10-20T00:00:00+00:00'  # Phase B 24 h window long gone
    op.admit_paid(66.9)  # USD 3 closure reserve and one projected retained day stay free
    with pytest.raises(ValueError,match='I7_HUMAN_BUDGET_STOP'):
        op.admit_paid(67.1)


def test_human_ledger_counts_retained_disk_snapshot_and_ip(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.config['cost_plan']['rates']['pd_gib_h']['usd'] = .001
    op.human_phase_open()
    start = op.now()
    op.now = lambda: start+timedelta(days=5)
    op.config['rates']['verified_utc'] = op.now().isoformat()
    with pytest.raises(ValueError,match='I7_HUMAN_BUDGET_STOP'):
        op.admit_paid(10)  # 540 GiB * 0.001 USD/GiB-h * 120 h = 64.8 USD retained already


def test_human_ledger_still_requires_fresh_prices(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.human_phase_open()
    op.config['rates']['verified_utc'] = (op.now()-timedelta(days=8)).isoformat()
    with pytest.raises(ValueError,match='FRESH_USD_PRICE_RECEIPT_REQUIRED'):
        op.admit_paid(1)


def test_human_reservation_is_bounded_to_four_hours(tmp_path):
    op,cloud = closed_phase_b(tmp_path)
    op.human_phase_open()
    with pytest.raises(ValueError,match='HUMAN_RESERVATION_TTL_REQUIRED'):
        op.reserve(4.5)
    assert not cloud.calls


def test_refresh_prices_updates_rates_and_plan_with_receipt(tmp_path,monkeypatch):
    from scripts.study_operator import i7_costs, pricing
    op,_ = closed_phase_b(tmp_path)
    older = (op.now()-timedelta(days=6)).isoformat()
    op.config['rates']['verified_utc'] = op.config['cost_plan']['verified_utc'] = older
    monkeypatch.setattr(pricing,'quote_archive',lambda folder,machine,region:
        {'usd_per_hour':'0.8','catalog_receipt_sha256':'b'*64})
    plan = dict(op.config['cost_plan'],verified_utc=op.now().isoformat())
    monkeypatch.setattr(i7_costs,'list_prices',lambda compute,storage: plan)
    (tmp_path/'compute').mkdir()
    (tmp_path/'compute'/'receipt.json').write_text(json.dumps({'at':op.now().isoformat()}))
    receipt = op.refresh_prices(tmp_path/'compute',tmp_path/'storage')
    assert receipt['status'] == 'LIST_PRICES_REFRESHED'
    assert op.config['rates'] == {'compute_usd_h':.8,'verified_utc':op.now().isoformat(),'price_receipt_sha256':'b'*64}
    assert op.config['cost_plan'] == plan
    plan['status'] = 'UNPRICED'
    with pytest.raises(ValueError,match='PRICED_LIST_REQUIRED'):
        op.refresh_prices(tmp_path/'compute',tmp_path/'storage')


def test_cli_exposes_human_operations_behind_live_and_date_guards(tmp_path,capsys):
    from scripts.study_operator import i7
    for operation in ('human-phase-open','refresh-prices','invite','revoke'):
        with pytest.raises(SystemExit):
            i7.main(['--root',str(tmp_path),operation,'--code','P998'])
        assert 'November authorization' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        i7.main(['--root',str(tmp_path),'start','--purpose','participants'])


@pytest.mark.parametrize('fault',['missing_proof','other_image','not_a_dict'])
def test_gate_go_is_bound_to_downloaded_proof_and_current_image(tmp_path,fault):
    op,_ = closed_phase_b(tmp_path,None)
    if fault == 'not_a_dict':
        (tmp_path/'receipts'/'gate').mkdir(parents=True)
        (tmp_path/'receipts'/'gate'/'x.json').write_text('[]')
    else:
        gate_receipt(tmp_path,'GO',image='sha256:'+'d'*64 if fault=='other_image' else None,
                     write_proof=fault!='missing_proof')
    with pytest.raises(ValueError,match='VERIFIED_GATE_GO_REQUIRED'):
        op.human_phase_open()


def test_human_start_rejects_gate_evidence_changed_after_opening(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.human_phase_open()
    gate_receipt(tmp_path,'GO','b')
    with pytest.raises(ValueError,match='GATE_EVIDENCE_CHANGED_SINCE_OPEN'):
        op.human_start_profile('rehearsal')


@pytest.mark.parametrize('fault',['intent','exposure'])
def test_human_phase_requires_no_open_phase_b_exposure(tmp_path,fault):
    op,_ = closed_phase_b(tmp_path)
    if fault == 'intent':
        op.state['reservation_intent'] = {'status':'UNKNOWN'}
    else:
        op.state['open_exposure_usd'] = 1.0
    with pytest.raises(ValueError,match='PHASE_B_OPEN_EXPOSURE_REQUIRES_RECONCILE'):
        op.human_phase_open()


@pytest.mark.parametrize('rate',[math.nan,-1.0,None])
def test_human_ledger_rejects_invalid_or_missing_rates(tmp_path,rate):
    op,_ = closed_phase_b(tmp_path)
    op.human_phase_open()
    if rate is None:
        op.config['cost_plan']['rates'].pop('ip_unused_h')
    else:
        op.config['cost_plan']['rates']['pd_gib_h']['usd'] = rate
    with pytest.raises(ValueError,match='I7_HUMAN_ALL_SKU_PRICES_REQUIRED'):
        op.admit_paid(0)


def test_human_ledger_requires_fresh_storage_plan_too(tmp_path):
    op,_ = closed_phase_b(tmp_path)
    op.human_phase_open()
    op.config['cost_plan']['verified_utc'] = (op.now()-timedelta(days=8)).isoformat()
    with pytest.raises(ValueError,match='FRESH_ALL_SKU_LIST_PRICES_REQUIRED'):
        op.admit_paid(0)


def test_refresh_prices_rejects_older_archives(tmp_path,monkeypatch):
    from scripts.study_operator import i7_costs, pricing
    op,_ = closed_phase_b(tmp_path)
    monkeypatch.setattr(pricing,'quote_archive',lambda folder,machine,region:
        {'usd_per_hour':'0.8','catalog_receipt_sha256':'b'*64})
    old = (op.now()-timedelta(days=1)).isoformat()
    monkeypatch.setattr(i7_costs,'list_prices',lambda compute,storage:
        dict(op.config['cost_plan'],verified_utc=old))
    (tmp_path/'compute').mkdir()
    (tmp_path/'compute'/'receipt.json').write_text(json.dumps({'at':old}))
    with pytest.raises(ValueError,match='OLDER_PRICE_ARCHIVE_REJECTED'):
        op.refresh_prices(tmp_path/'compute',tmp_path/'storage')


def test_reconcile_charges_expired_reservation_only_until_native_deletion(tmp_path):
    op,cloud = closed_phase_b(tmp_path)
    reservation = op.state['reservation']
    reservation.pop('deleted_verified')
    reservation.update(deleteAtTime='2026-11-01T04:00:00+00:00',delete_api_success=False)
    cloud.exists = False
    op.state['spent_usd'] = 0
    op.now = lambda: datetime(2026,11,4,tzinfo=timezone.utc)
    op.reconcile()
    assert op.state['spent_usd'] == pytest.approx(4*op.config['rates']['compute_usd_h'])


def test_revoke_validates_code_before_any_cloud_call(tmp_path):
    op,cloud = closed_phase_b(tmp_path)
    with pytest.raises(Exception,match='digitos|dígitos'):
        op.revoke('Ana')
    assert not cloud.calls


@pytest.mark.parametrize('purpose', ['rehearsal', 'pilot'])
@pytest.mark.parametrize('number', range(1, 21))
def test_human_rehearsal_and_pilot_reject_primary_codes(tmp_path, purpose, number):
    from src.ui.components.study_sessions import StudyStore
    from tests.study_helpers import configured
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, purpose=purpose)
    store.freeze()
    with pytest.raises(ValueError, match='fictitious P900'):
        store.issue(f'P{number:02}', cell=1, profile='without_experience')


@pytest.mark.parametrize('purpose,code', [('rehearsal', 'P998'), ('pilot', 'P900')])
def test_human_synthetic_codes_remain_admissible(tmp_path, purpose, code):
    from src.ui.components.study_sessions import StudyStore
    from tests.study_helpers import configured
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, purpose=purpose)
    store.freeze()
    session = store.admit(store.issue(code, cell=1, profile='without_experience'))
    assert session.data['assignment']['participant_id'] == code
    assert session.data['purpose'] == purpose
