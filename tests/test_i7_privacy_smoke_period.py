"""privacy-08: the Phase B privacy step boots in the synthetic smoke period (i7_phase_b dispatch
calls start('smoke')), so the guest must accept that period. Human periods stay rejected."""
import pytest

from scripts.study_operator import guest_bridge, i7_privacy

REQUEST = {'operation': 'i7-privacy', 'ip': '192.0.2.10', 'user_agent': 'CloudRAG-I7-synthetic'}


def active(purpose, iteration=7):
    return {'config': {'iteration': iteration, 'purpose': purpose}}


@pytest.fixture
def scanned(monkeypatch):
    calls = []
    monkeypatch.setattr(i7_privacy, 'host_sources', lambda value: {'marker': value['config']['purpose']})
    monkeypatch.setattr(i7_privacy, 'scan', lambda sources, needles: calls.append((sources, needles))
                        or {'domains': {}, 'status': 'SCANNED'})
    return calls


@pytest.mark.parametrize('purpose', ['smoke', 'technical'])
def test_privacy_scan_runs_in_synthetic_periods(scanned, purpose):
    result = guest_bridge.disk_dispatch(active(purpose), dict(REQUEST))
    assert result['status'] == 'SCANNED'
    assert scanned == [({'marker': purpose}, (REQUEST['ip'].encode(), REQUEST['user_agent'].encode()))]


@pytest.mark.parametrize('purpose', ['study', 'pilot', 'rehearsal'])
def test_privacy_scan_is_refused_in_human_periods(scanned, purpose):
    with pytest.raises(ValueError, match='^I7_SYNTHETIC_PRIVACY_ONLY$'):
        guest_bridge.disk_dispatch(active(purpose), dict(REQUEST))
    assert scanned == []


def test_privacy_scan_is_refused_outside_iteration_7(scanned):
    with pytest.raises(ValueError, match='^I7_SYNTHETIC_PRIVACY_ONLY$'):
        guest_bridge.disk_dispatch(active('smoke', iteration=6), dict(REQUEST))
    assert scanned == []
