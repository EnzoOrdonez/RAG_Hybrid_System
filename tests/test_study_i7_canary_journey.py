
from scripts.study_operator.i7_privacy import DOMAINS,scan,tree_sources
from tests.test_study_i7_browser import local_app


def test_canaries_cross_real_browser_socket_generation_double_and_backup(tmp_path):
    from scripts.study_operator.i7_browser import run
    ip,ua='203.0.113.10','CloudRAG-Synthetic-Canary'
    with local_app(tmp_path,canary_probe=True) as (url,token,protocol,identity,observe):
        result=run(url,token,protocol,identity,observe,local_test=True,ip=ip,user_agent=ua)
    assert all(result['canary_transport'].values())
    for domain in ('app','runner','backups','bucket'):
        data=(tmp_path/'canary-domains'/domain/'trace.json').read_bytes()
        assert ip.encode() in data and ua.encode() in data
    # Three unavailable OS/proxy domains are deliberate API doubles in this local test.
    for domain in ('caddy','journald','serial'):
        path=tmp_path/'canary-domains'/domain
        path.mkdir(parents=True)
        (path/'trace.json').write_bytes(ip.encode()+ua.encode())
    sources={domain:lambda domain=domain:tree_sources(tmp_path/'canary-domains'/domain) for domain in DOMAINS}
    detected=scan(sources,(ip.encode(),ua.encode()))
    assert detected['status']=='FAIL' and detected['hits']==14
    del sources['bucket']
    assert scan(sources,(ip.encode(),ua.encode()))['status']=='INCOMPLETE'
