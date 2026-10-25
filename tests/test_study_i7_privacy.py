import pytest

from scripts.study_operator.i7_privacy import (
    DOMAINS,
    archive_legacy,
    clean_import,
    scan,
)


def sources():
    return {domain:lambda:iter([('record',[b'clean'])]) for domain in DOMAINS}


def test_zero_hits_requires_every_domain_and_stream_completion():
    result = scan(sources(),(b'203.0.113.77',b'I7-UA-secret'))
    assert result['status'] == 'PASS' and result['hits'] == 0
    partial = sources()
    partial.pop('serial')
    assert scan(partial,(b'ip',b'ua'))['status'] == 'INCOMPLETE'
    def unreadable():
        raise PermissionError('private text')
    partial = sources()
    partial['bucket'] = unreadable
    assert scan(partial,(b'ip',b'ua'))['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('domain',DOMAINS)
def test_each_domain_detects_canary_across_chunk_boundary_without_echo(domain):
    streams = sources()
    streams[domain] = lambda:iter([('private-name',[b'prefix I7-UA-',b'secret suffix'])])
    result = scan(streams,(b'203.0.113.77',b'I7-UA-secret'))
    assert result['status'] == 'FAIL' and result['hits'] == 1
    assert 'secret' not in str(result) and 'private-name' not in str(result)


def test_clean_import_does_not_copy_runtime_and_archive_keeps_original_bytes(tmp_path):
    source, target, private = [tmp_path/n for n in ('source','target','private')]
    (source/'assets').mkdir(parents=True)
    (source/'assets'/'index').write_bytes(b'index')
    (source/'runtime').mkdir()
    legacy = []
    for i in range(9):
        path = source/'runtime'/str(i)
        path.write_bytes(b'original legacy\x00'+bytes([i]))
        legacy.append(path)
    receipt = archive_legacy(legacy,private/'legacy.tar')
    assert receipt['files'] == 9
    import hashlib
    clean_import(source,target,{'assets/index':hashlib.sha256(b'index').hexdigest()})
    assert (target/'assets'/'index').read_bytes() == b'index'
    assert not (target/'runtime').exists()
    assert all(p.exists() for p in legacy)
    import tarfile
    with tarfile.open(private/'legacy.tar') as archive:
        assert [archive.extractfile(m).read() for m in archive.getmembers()] == [p.read_bytes() for p in legacy]


def test_serial_census_pages_all_ports_and_rejects_lost_prefix():
    from scripts.study_operator.i7_privacy import serial_sources
    class Cloud:
        def __init__(self):
            self.lost=False
        def command(self,args,**kwargs):
            start=int(next(a.split('=')[1] for a in args if a.startswith('--start=')))
            assert kwargs['private_output']
            if start == 0:
                return {'start':1 if self.lost else 0,'next':4,'contents':'test'}
            return {'start':4,'next':4,'contents':''}
    cloud=Cloud()
    rows=[(name,b''.join(blocks)) for name,blocks in serial_sources(cloud,{'name':'own','zone':'us-central1-a'})]
    assert len(rows)==4 and all(data==b'test' for _,data in rows)
    cloud.lost=True
    with pytest.raises(ValueError,match='LOST'):
        [(name,b''.join(blocks)) for name,blocks in serial_sources(cloud,{'name':'own','zone':'us-central1-a'})]
