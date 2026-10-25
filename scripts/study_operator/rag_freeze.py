"""Versioned read-only projection of the I4 freeze generator; no model queries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inherited_contexts(cohort):
    cohort = Path(cohort)
    contexts, references = {}, []
    for path in sorted(cohort.glob('deployment--candidate--gate-cohort-worker--requests--w*-pipeline.json')):
        payload = json.loads(path.read_bytes())
        wid = path.name.split('--')[-1].removesuffix('-pipeline.json')
        window, position = wid.split('-')
        planned_path = cohort / f'deployment--candidate--gate-cohort--window-{window[1:]}--attempts--{position}-result.json'
        planned = json.loads(planned_path.read_bytes())
        key = planned['query_id'] + '|' + planned['condition']
        ids = [row['chunk_id'] for row in payload.get('retrieved_chunks', [])]
        if key in contexts and contexts[key] != ids:
            raise ValueError('Inherited gate itself has context variants')
        contexts[key] = ids
        references.append(dict(path=str(path), sha256=sha(path), result_path=str(planned_path), result_sha256=sha(planned_path)))
    if len(contexts) != 12:
        raise ValueError('Twelve inherited gate combinations required')
    return contexts, references


def compare(baseline, final, *, require_live=False):
    different = sorted(key for key in set(baseline['rag']) | set(final['rag'])
                       if baseline['rag'].get(key) != final['rag'].get(key))
    if different:
        raise ValueError('Frozen RAG changed: ' + ', '.join(different))
    if require_live and not final['provenance'].get('live_contexts_verified'):
        raise ValueError('Live twelve-context proof required; historical context reads cannot grant acceptance')
    return dict(rag_projection_equal=True, live_contexts_verified=final['provenance'].get('live_contexts_verified', False))


def collect(repo, inventory, manifest, cohort, live=None):
    from src.pipeline.pipeline_config import SURVEY_DEPLOY, LLM_ONLY_NO_RAG
    from src.ui.components.study_pipeline import STUDY_NO_RAG
    from src.generation import prompt_templates

    repo = Path(repo)
    prefixes = ('src/retrieval/', 'src/reranking/', 'src/pipeline/', 'src/processing/')
    specific = {'src/generation/prompt_templates.py', 'src/generation/hallucination_detector.py',
                'src/generation/response_formatter.py', 'src/generation/llm_manager.py',
                'src/ui/components/study_pipeline.py', 'src/ui/components/study_service.py'}
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True, timeout=15).splitlines()
    modules = {name: sha(repo/name) for name in names if name.startswith(prefixes) or name in specific}
    expected = json.loads(Path(manifest).read_bytes())['files']
    actual = {name: sha(repo/name) for name in expected}
    if actual != expected:
        raise ValueError('Local weights/index manifests differ from inherited anchor')
    identity = json.loads(Path(inventory).read_bytes())
    contexts, references = inherited_contexts(cohort)
    if live and json.loads(Path(live).read_bytes())['contexts'] != contexts:
        raise ValueError('Live retrieved contexts differ from R1 gate')
    return dict(schema=1, rag=dict(
        recipes={'SURVEY_DEPLOY': SURVEY_DEPLOY.model_dump(), 'LLM_ONLY_NO_RAG': LLM_ONLY_NO_RAG.model_dump(),
                 'effective_STUDY_NO_RAG': STUDY_NO_RAG.model_dump()}, modules=modules,
        prompt_templates={name: hashlib.sha256(value.encode('utf-8')).hexdigest()
                          for name, value in vars(prompt_templates).items() if name.isupper() and isinstance(value, str)},
        options=dict(temperature=0, num_predict=1024, seed=42, num_ctx=4096),
        option_provenance=dict(file='src/ui/components/study_pipeline.py', sha256=sha(repo/'src/ui/components/study_pipeline.py')),
        nli_route='per_claim', artifacts=actual, contexts=contexts, ollama=identity['ollama'],
        numerical_controls=identity['execution_environment']),
        provenance=dict(source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=15).strip(),
            inherited_inventory=str(inventory), inherited_inventory_sha256=sha(inventory),
            inherited_gate_responses=references, live_contexts_verified=bool(live),
            live_contexts_path=str(live) if live else None, live_contexts_sha256=sha(live) if live else None))


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('repo', 'inventory', 'manifest', 'cohort', 'output', 'baseline'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--live')
    args = parser.parse_args(argv)
    result = collect(args.repo, args.inventory, args.manifest, args.cohort, args.live)
    compared = compare(json.loads(Path(args.baseline).read_bytes()), result, require_live=bool(args.live))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(path=args.output, sha256=sha(args.output), **compared)))


if __name__ == '__main__':
    main()
