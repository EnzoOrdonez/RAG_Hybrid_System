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
    cohort = Path(manifest['cohort']) if manifest.get('cohort') else root / 'cohort'
    bounded = bool(manifest.get('cohort'))
    protocol = gate.read_json(cohort / 'source-manifest.json')['protocol'] if bounded else None
    if bounded:
        gate.recipe(protocol)
        gate.selected_conditions(protocol, manifest.get('system'), manifest.get('phase'))
        if protocol.get('protocol_version') != 2 or tuple(gate.selected_systems(protocol)) != gate.SYSTEMS:
            raise ValueError('Managed bounded window requires the registered three-system cohort')
    artifact_manifest = protocol['artifact_manifest_path'] if bounded else str(root / 'deployment-manifest.json')
    env = dict(os.environ, CLOUDRAG_MODE='participant', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONHASHSEED='42', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='', CLOUDRAG_MEMORY_TRACE='1',
        CLOUDRAG_MODEL_DIGEST=manifest['model_digest'], CLOUDRAG_BUILD_ID=manifest['build_id'],
        CLOUDRAG_SESSION_DIR=str(Path(tempfile.gettempdir()) / ('cloudrag-managed-technical-' + manifest['id'])),
        CLOUDRAG_ARTIFACT_MANIFEST=artifact_manifest, CLOUDRAG_GATE_DEADLINE=deadline.isoformat(),
        CLOUDRAG_GATE_WINDOW_ID=manifest['id'],
        OLLAMA_HOST='http://localhost:11434')
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
    if not bounded:
        command('snapshot', ['scripts/check_deployment_artifacts.py', 'snapshot', '--manifest', env['CLOUDRAG_ARTIFACT_MANIFEST']])
    command('verify-new', ['scripts/check_deployment_artifacts.py', 'verify', '--manifest', env['CLOUDRAG_ARTIFACT_MANIFEST']])
    from scripts.observe_interview_gate import admission
    admission(root / 'admission-before-contrast')
    command('contrast', ['scripts/contrast_interview_observer.py', '--output', str(root / 'contrast')])
    if not bounded:
        command('cohort-init', ['scripts/measure_interview_gate.py', 'init', '--output', str(cohort), '--systems', 'hybrid', '--controlled'])
    selection = ['--system', manifest['system'], '--phase', manifest['phase']] if bounded else []
    command('cohort-run', ['scripts/measure_interview_gate.py', 'run', '--output', str(cohort), *selection])
    protocol = gate.read_json(cohort / 'source-manifest.json')['protocol']
    rows = gate.all_records(cohort)
    remaining = gate.pending(rows, gate.selected_systems(protocol))
    selected_remaining = [slot for slot in remaining if not bounded or
                          slot[:2] == (manifest['system'], manifest['phase'])]
    gate.write_new(root / 'payload-complete.json', dict(at=gate.now(), cohort=str(cohort),
        condition_complete=not selected_remaining, cohort_complete=not remaining,
        pending_condition=selected_remaining,
        report=gate.summarize(rows, gate.selected_systems(protocol), protocol=protocol)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.root)
    except Exception as exc:
        gate.write_new(args.root / 'payload-error.json', dict(at=gate.now(), error=f'{type(exc).__name__}: {exc}'))
        raise
