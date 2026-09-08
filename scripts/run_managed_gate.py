"""Fixed managed-window payload. Never changes services; PowerShell owns restoration."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import gate_job
from scripts import measure_interview_gate as gate


def run(root):
    root = Path(root).resolve()
    identity = gate_job.enter()
    gate.write_new(root / 'worker-identity.json', dict(at=gate.now(), **identity))
    manifest = gate.read_json(root / 'window.json')
    if manifest['build_id'] != gate.git('rev-parse', 'HEAD'):
        raise RuntimeError('Build changed after window preparation')
    if not (root / 'armed.json').exists() or (root / 'restored.json').exists():
        raise RuntimeError('Window is not armed')
    armed = gate.read_json(root / 'armed.json')
    deadline = datetime.fromisoformat(armed['deadline_utc'])
    env = dict(os.environ, CLOUDRAG_MODE='participant', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONHASHSEED='42', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='', CLOUDRAG_MEMORY_TRACE='1',
        CLOUDRAG_MODEL_DIGEST=manifest['model_digest'], CLOUDRAG_BUILD_ID=manifest['build_id'],
        CLOUDRAG_SESSION_DIR=str(Path(tempfile.gettempdir()) / ('cloudrag-managed-technical-' + manifest['id'])),
        CLOUDRAG_ARTIFACT_MANIFEST=str(root / 'deployment-manifest.json'), OLLAMA_HOST='http://localhost:11434')
    os.environ.update(env)

    def command(label, args):
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 120:
            raise TimeoutError('Window deadline approaching; restore before starting another phase')
        gate.write_new(root / 'phases' / f'{uuid.uuid4().hex}.json', dict(at=gate.now(), phase=label))
        with (root / f'{label}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *args], cwd=gate.PROJECT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=remaining - 30)
        if result.returncode:
            raise RuntimeError(f'{label} failed with exit {result.returncode}; inspect durable evidence')

    command('verify-original', ['scripts/check_deployment_artifacts.py', 'verify', '--manifest', manifest['trusted_manifest']])
    command('snapshot', ['scripts/check_deployment_artifacts.py', 'snapshot', '--manifest', env['CLOUDRAG_ARTIFACT_MANIFEST']])
    command('verify-new', ['scripts/check_deployment_artifacts.py', 'verify', '--manifest', env['CLOUDRAG_ARTIFACT_MANIFEST']])
    from scripts.observe_interview_gate import admission
    admission(root / 'admission-before-contrast')
    command('contrast', ['scripts/contrast_interview_observer.py', '--output', str(root / 'contrast')])
    command('cohort-init', ['scripts/measure_interview_gate.py', 'init', '--output', str(root / 'cohort'), '--systems', 'hybrid', '--controlled'])
    command('cohort-run', ['scripts/measure_interview_gate.py', 'run', '--output', str(root / 'cohort')])
    gate.write_new(root / 'payload-complete.json', dict(at=gate.now(), report=gate.summarize(gate.all_records(root / 'cohort'), ('hybrid',))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.root)
    except Exception as exc:
        gate.write_new(args.root / 'payload-error.json', dict(at=gate.now(), error=f'{type(exc).__name__}: {exc}'))
        raise
