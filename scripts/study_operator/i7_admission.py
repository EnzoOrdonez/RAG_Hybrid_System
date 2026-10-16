"""Published source, effective environment and I3 context admission. No cloud writes."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.study_operator.stimulus_calendar import TASKS
from scripts.study_operator.stimulus_host_evidence import sha

REQUIRED=(
    'docs/STUDY_GATE_PREREGISTRATION_AMENDMENT_2026-10-05_ITERATION4_SERVICE.md',
    'docs/STUDY_ITERATION5_ACCEPTANCE_PREREGISTRATION.md',
    'output/audit/ITERATION4_STIMULUS_CALENDAR_PREREGISTERED.json',
    'scripts/study_operator/generation_observer.py','scripts/study_operator/rerank_observer.py',
    'scripts/study_operator/stimulus_host.py','scripts/study_operator/stimulus_evidence.py',
    'scripts/study_operator/i7_gate_host.py','scripts/study_operator/i7_gate_daemon.py',
    'scripts/study_operator/i7_browser.py','config/i7-browser-pin.json')


def git_read(*args):
    return subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),*args],timeout=60)


def compare_contexts(live,reference):
    keys={q+'|'+c for q in TASKS for c in ('hybrid','no_rag')}
    if set(live)!=keys or set(reference)!=keys or live!=reference:
        raise ValueError('TWELVE_I3_CONTEXTS_DIFFER')
    if any(not isinstance(v,list) or (k.endswith('|no_rag') and v) for k,v in live.items()):
        raise ValueError('I3_CONTEXT_SCHEMA_DIFFERS')
    return sha(live)


def validate(anchor, *, software_sha256, live, reference, git=git_read):
    commit=anchor['commit']
    if not re.fullmatch('[a-f0-9]{40}',commit):
        raise ValueError('PUBLISHED_ANNEX_COMMIT_REQUIRED')
    remote=git('ls-remote','origin','refs/heads/fix/interview-readiness')
    remote=remote.decode() if isinstance(remote,bytes) else remote
    if remote.split()[0]!=commit:
        raise ValueError('ENZO_PUBLISHED_ANNEX_REQUIRED')
    annex=anchor['annex_path']
    if not re.fullmatch(r'output/audit/[A-Za-z0-9_-]+\.md',annex):
        raise ValueError('PUBLIC_ANNEX_PATH_REQUIRED')
    if set(anchor['files'])!=set(REQUIRED)|{annex}:
        raise ValueError('ALL_OBSERVER_ANCHORS_REQUIRED')
    if git('diff','--name-only',anchor['source_commit'],commit,'--','src','scripts','config'):
        raise ValueError('PUBLISHED_CODE_DIFFERS_FROM_IMAGE')
    for name,expected in anchor['files'].items():
        data=git('show',commit+':'+name)
        data=data.encode() if isinstance(data,str) else data
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('PUBLISHED_ANNEX_FILE_HASH_DIFFERS')
    if anchor['software_sha256']!=software_sha256:
        raise ValueError('ANNEX_EFFECTIVE_ENVIRONMENT_DIFFERS')
    context_hash=compare_contexts(live,reference)
    return {'status':'PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED','commit':commit,'source_commit':anchor['source_commit'],
            'software_sha256':software_sha256,'contexts_sha256':context_hash,'combinations':12,
            'anchor_sha256':sha(anchor)}


def observed_contexts(factory,queries):
    """Stop at the frozen context boundary, before any LLM generation."""
    import src.pipeline.rag_pipeline as module
    class Captured(BaseException):
        pass
    result={}
    original=module.build_context
    try:
        for condition in ('hybrid','no_rag'):
            pipeline=factory(condition)
            if condition=='no_rag':
                if pipeline.config.retrieval_method!='none' or pipeline.retriever is not None:
                    raise ValueError('NO_RAG_MUST_HAVE_NO_RETRIEVER')
                result.update({q+'|no_rag':[] for q in TASKS})
                continue
            for query in TASKS:
                key=query+'|hybrid'
                def capture(chunks,query_type):
                    result[key]=[item['chunk_id'] for item in chunks]
                    raise Captured()
                module.build_context=capture
                try:
                    pipeline.query(queries[query]['question'])
                except Captured:
                    pass
                else:
                    raise ValueError('CONTEXT_BOUNDARY_NOT_REACHED')
    finally:
        module.build_context=original
    if len(result)!=12:
        raise ValueError('TWELVE_LIVE_CONTEXTS_REQUIRED')
    return result


def live_contexts(deployment):
    import os
    if os.environ.get('CLOUDRAG_ISOLATED_APP')!='1' or deployment['purpose']!='technical':
        raise ValueError('ISOLATED_TECHNICAL_CONTEXTS_REQUIRED')
    from scripts import cloud_entrypoint
    from scripts.environment_identity import load,verify
    from scripts.study_operator.service_transition import cold_state
    from src.ui.components.study_pipeline import build_study_pipeline,configure_study_device
    cloud_entrypoint.configure(deployment)
    protocol=cloud_entrypoint.verify(deployment)
    identity=verify(deployment)
    inventory=load(deployment)
    state=Path(deployment['service_state_path'])
    before=cold_state(json.loads(state.read_bytes()),inventory['observed']['boot_id'])
    configure_study_device()
    contexts=observed_contexts(build_study_pipeline,protocol['queries'])
    after=cold_state(json.loads(state.read_bytes()),inventory['observed']['boot_id'])
    if after!=before or verify(deployment)!=identity or load(deployment)!=inventory:
        raise ValueError('CONTEXT_PROBE_ENVIRONMENT_CHANGED')
    return {'mode':'LIVE_NO_GENERATION','inventory':inventory,'contexts':contexts,
            'initial_service_state':before,'final_service_state':after}


def check_files(root):
    from scripts.study_operator.stimulus_evidence import software_projection
    root=Path(root)
    config=json.loads((root/'installation.json').read_bytes())
    proof=json.loads((root/'live-contexts.json').read_bytes())
    reference=Path(config['i3_context_reference_path'])
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=config['i3_context_reference_sha256']:
        raise ValueError('I3_REFERENCE_HASH_DIFFERS')
    anchor=json.loads((root/'published-annex.json').read_bytes())
    if (proof['mode']!='LIVE_NO_GENERATION' or proof['inventory']['image']['image_id']!=config['image_id']
            or anchor['source_commit']!=proof['inventory']['source']['commit']):
        raise ValueError('CURRENT_IMAGE_CONTEXT_PROOF_REQUIRED')
    return validate(anchor,software_sha256=sha(software_projection(proof['inventory'])),
        live=proof['contexts'],reference=json.loads(reference.read_bytes())['rag']['contexts'])


def main(argv=None):
    import argparse
    from scripts.study_operator.stimulus_evidence import software_projection
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--draft',action='store_true')
    parser.add_argument('--anchor-published',action='store_true')
    parser.add_argument('--annex',default='output/audit/I7_ENVIRONMENT_ANNEX.md')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'output/audit/[A-Za-z0-9_-]+\.md',args.annex):
        parser.error('Fixed public audit Markdown path required')
    proof=json.loads((args.root/'live-contexts.json').read_bytes())
    software=sha(software_projection(proof['inventory']))
    if args.draft:
        path=Path(__file__).resolve().parents[2]/args.annex
        path.parent.mkdir(parents=True,exist_ok=True)
        text=('# Anexo efectivo I7\n\n'
            +'Imagen: '+proof['inventory']['image']['image_id']+'\n\n'
            +'Commit de fuente: '+proof['inventory']['source']['commit']+'\n\n'
            +'SHA-256 de software: '+software+'\n\n'
            +'SHA-256 de doce contextos sin generación: '+sha(proof['contexts'])+'\n\n'
            +'Inventario completo privado; este anexo requiere revisión y publicación antes de medir.\n')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(text)
        print(json.dumps({'status':'DRAFT_REQUIRES_ENZO_REVIEW_AND_PUSH','path':args.annex}))
        return
    if args.anchor_published:
        commit=git_read('ls-remote','origin','refs/heads/fix/interview-readiness').decode().split()[0]
        anchor={'commit':commit,'source_commit':proof['inventory']['source']['commit'],
            'annex_path':args.annex,'software_sha256':software,
            'files':{name:hashlib.sha256(git_read('show',commit+':'+name)).hexdigest() for name in (*REQUIRED,args.annex)}}
        path=args.root/'published-annex.json'
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(anchor,stream,indent=2)
    print(json.dumps(check_files(args.root)))


if __name__=='__main__':
    main()
