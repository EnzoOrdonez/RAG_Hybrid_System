import hashlib

import pytest


def fixture(tmp_path):
    from scripts.study_operator.i7_admission import REQUIRED
    files={name:hashlib.sha256(name.encode()).hexdigest() for name in REQUIRED}
    files['output/audit/I7_ENVIRONMENT_ANNEX.md']=hashlib.sha256(b'annex').hexdigest()
    anchor={'commit':'a'*40,'annex_path':'output/audit/I7_ENVIRONMENT_ANNEX.md',
            'source_commit':'b'*40,'software_sha256':'c'*64,'files':files}
    contexts={q+'|'+c:([q] if c=='hybrid' else []) for q in
              ('q001','q068','q180','q016','q070','q172') for c in ('hybrid','no_rag')}
    def git(*args):
        if args[0]=='ls-remote':
            return 'a'*40+'\trefs/heads/fix/interview-readiness\n'
        if args[0]=='show':
            name=args[1].split(':',1)[1]
            return b'annex' if name.endswith('.md') and name.startswith('output/') else name.encode()
        if args[0]=='diff':
            return b''
        raise AssertionError(args)
    return anchor,contexts,git


def test_admission_checks_publication_hashes_and_twelve_contexts(tmp_path):
    from scripts.study_operator.i7_admission import validate
    anchor,contexts,git=fixture(tmp_path)
    result=validate(anchor,software_sha256='c'*64,live=contexts,reference=contexts,git=git)
    assert result['status']=='PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED'
    assert result['combinations']==12


@pytest.mark.parametrize('fault',['unpublished','hash','context','missing','software','code'])
def test_admission_rejects_missing_or_changed_evidence(tmp_path,fault):
    from scripts.study_operator.i7_admission import validate
    anchor,contexts,git=fixture(tmp_path)
    live={k:list(v) for k,v in contexts.items()}
    original=git
    if fault=='hash':
        anchor['files'][next(iter(anchor['files']))]='f'*64
    if fault=='context':
        live['q001|hybrid']=['wrong']
    if fault=='missing':
        live.pop('q001|hybrid')
    if fault=='software':
        anchor['software_sha256']='d'*64
    def altered(*args):
        if fault=='unpublished' and args[0]=='ls-remote':
            return 'd'*40+'\trefs/heads/fix/interview-readiness'
        if fault=='code' and args[0]=='diff':
            return b'scripts/changed.py'
        return original(*args)
    with pytest.raises(ValueError):
        validate(anchor,software_sha256='c'*64,live=live,reference=contexts,git=altered)


def test_live_probe_crosses_frozen_retrieval_and_stops_before_generation():
    from types import SimpleNamespace

    from scripts.study_operator.i7_admission import observed_contexts
    from scripts.study_operator.stimulus_calendar import TASKS
    from src.pipeline.rag_pipeline import RAGPipeline
    pipeline=object.__new__(RAGPipeline)
    pipeline.config=SimpleNamespace(retrieval_method='hybrid',retrieval_top_k=1,final_top_k=1,
        query_expansion=False,balance_cross_cloud_providers=False)
    pipeline.query_processor=pipeline._routing_qp=pipeline.reranker=None
    pipeline.hybrid_index=SimpleNamespace(get_chunk=lambda key:{'chunk_id':key,'text':'synthetic'})
    pipeline.retriever=SimpleNamespace(search=lambda q,**kw:[SimpleNamespace(chunk_id=q)])
    pipeline.llm=SimpleNamespace(generate=lambda **kw:pytest.fail('Probe must never generate'))
    def factory(condition):
        return pipeline if condition=='hybrid' else SimpleNamespace(config=SimpleNamespace(retrieval_method='none'),retriever=None)
    result=observed_contexts(factory,{q:{'question':q} for q in TASKS})
    assert len(result)==12 and all(result[q+'|hybrid']==[q] for q in TASKS)
