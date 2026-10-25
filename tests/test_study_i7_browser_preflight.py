import json
from datetime import datetime, timezone

import pytest


def test_missing_pinned_browser_is_named_in_phase_b_check_before_cloud(tmp_path,monkeypatch,capsys):
    from scripts.study_operator import cloud_client, i7_phase_b
    monkeypatch.setenv('PLAYWRIGHT_BROWSERS_PATH',str(tmp_path/'empty-browsers'))
    monkeypatch.setattr(cloud_client,'Cloud',lambda *a,**kw:pytest.fail('No cloud client during local preflight'))
    assert i7_phase_b.main(['--check'])==0
    result=json.loads(capsys.readouterr().out)
    assert any(row['code']=='PINNED_BROWSER_RUNTIME_REQUIRED' for row in result['blockers'])


def test_live_step_rejects_missing_browser_before_cloud_client(tmp_path,monkeypatch):
    from scripts.study_operator import cloud_client, i7_phase_b, i7_readiness
    class November(datetime):
        @classmethod
        def now(cls,tz=None):
            return cls(2026,11,1,tzinfo=timezone.utc)
    monkeypatch.setattr(i7_phase_b,'datetime',November)
    monkeypatch.setattr(i7_readiness,'evaluate',lambda *a,**kw:{'blockers':[]})
    monkeypatch.setenv('PLAYWRIGHT_BROWSERS_PATH',str(tmp_path/'empty-browsers'))
    monkeypatch.setattr(cloud_client,'Cloud',lambda *a,**kw:pytest.fail('Cloud was constructed before browser check'))
    with pytest.raises(SystemExit):
        i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused','cpu-build'])


@pytest.mark.parametrize('fault',['tampered','missing','outside'])
def test_pinned_browser_binary_cannot_be_replaced(tmp_path,monkeypatch,fault):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from tests.i7_browser_guard import require_i7_browser
    require_i7_browser(need_binary=False)
    import playwright.sync_api
    from scripts.study_operator.i7_browser_preflight import verify
    root=tmp_path/'browsers'
    executable=(tmp_path/'elsewhere/chrome') if fault=='outside' else root/'chromium-1208/chrome'
    root.mkdir()
    executable.parent.mkdir(parents=True,exist_ok=True)
    if fault!='missing':
        executable.write_bytes(b'replacement browser')
    monkeypatch.setenv('PLAYWRIGHT_BROWSERS_PATH',str(root))
    @contextmanager
    def driver():
        yield SimpleNamespace(chromium=SimpleNamespace(executable_path=str(executable)))
    monkeypatch.setattr(playwright.sync_api,'sync_playwright',driver)
    with pytest.raises(ValueError,match='PINNED_BROWSER_RUNTIME_REQUIRED'):
        verify()


def test_real_installed_browser_matches_revision_path_and_hash():
    from tests.i7_browser_guard import require_i7_browser
    require_i7_browser()
    from scripts.study_operator.i7_browser_preflight import verify
    result=verify()
    assert result['status']=='PINNED_BROWSER_READY' and result['chromium_revision']=='1208'
    assert 'chromium-1208' in result['executable_path']
    assert len(result['browser_sha256'])==64
