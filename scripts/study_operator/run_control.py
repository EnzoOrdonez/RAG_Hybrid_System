"""Finite Limited-token jobs with immutable command receipts and handover state.

This runner is for technical commands. Invitation display must never be captured.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from filelock import FileLock

from src.ui.components.session_storage import atomic_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_limited():
    if os.name == 'nt' and ctypes.windll.shell32.IsUserAnAdmin():
        raise PermissionError('This job requires a Limited token; no elevation permitted')


def validate_command(command):
    args = command['argv']
    if not args or any(not isinstance(arg, str) for arg in args):
        raise ValueError('Explicit nonempty argument vector required')
    if any(arg == 'invite' or arg.startswith('--token') for arg in args):
        raise ValueError('Private invitation display cannot use a capturing runner')
    if not 0 < command.get('timeout', 600) <= 3600:
        raise ValueError('Each command requires a finite limit of at most one hour')
    if not command['name'].replace('-', '').replace('_', '').isalnum():
        raise ValueError('Unsafe receipt label')
    return args


class Recorder:
    def __init__(self, root, app, *, agent='Codex', model='GPT-6 runtime family', phase=0):
        self.root, self.app = Path(root), Path(app)
        self.agent, self.model, self.phase = agent, model, phase

    def append(self, name, text):
        with (self.root / name).open('a', encoding='utf-8') as stream:
            stream.write(text + '\n')

    def checkpoint(self, next_action):
        path = self.root / 'STATE.json'
        with FileLock(str(self.root/'state.lock'), timeout=10):
            state = json.loads(path.read_bytes())
            tasks = set(state.get('scheduled_tasks', []))
            for receipt in self.root.glob('*-task-receipt.json'):
                row = json.loads(receipt.read_text(encoding='utf-8-sig'))
                if row.get('task', '').startswith('CloudRAG-I5-'):
                    tasks.add(row['task'])
            state.update(updated_utc=utc(), phase=self.phase, agent=self.agent,
                         model=self.model, next_action=next_action, worker_pid=os.getpid(),
                         scheduled_tasks=sorted(tasks))
            atomic_json(path, state)
            active = []
            for receipt in self.root.glob('*-active.json'):
                row = json.loads(receipt.read_bytes())
                if row.get('status') == 'RUNNING':
                    active.append(row)
            handover = dict(agent=self.agent, model=self.model, phase=self.phase,
                next_action=next_action, deadline_utc=state.get('deadline_utc'),
                resources=state.get('resources', []), cost=state.get('cost', {}),
                resource_intents=state.get('resource_intents', []),
                open_exposures=state.get('open_exposures', {}), tasks=sorted(tasks),
                independent_safety=state.get('independent_safety', {}), active_jobs=active)
            (self.root/'HANDOVER.md').write_text('# Relevo\nNo repetir efectos sin verificar en vivo.\n\n```json\n'
                +json.dumps(handover, indent=2, ensure_ascii=False)+'\n```\n', encoding='utf-8')
        atomic_json(self.root / 'worker-heartbeat.json', dict(at=utc(), pid=os.getpid()))

    def run(self, command):
        args = validate_command(command)
        name = command['name']
        receipt = self.root / (name + '-receipt.json')
        if receipt.exists() or (self.root / (name + '.stdout')).exists():
            raise FileExistsError('Existing command evidence; reconcile before any retry')
        started, begin = utc(), time.monotonic()
        intent = dict(command=args, label=name, phase=self.phase, agent=self.agent,
                      model=self.model, started_utc=started, event='INTENT')
        self.append('COMMANDS.log', json.dumps(intent))
        self.checkpoint('In progress: ' + name + '; inspect its receipt before replay')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                   CLOUDSDK_CORE_DISABLE_FILE_LOGGING='1', CLOUDSDK_CORE_DISABLE_PROMPTS='1',
                   CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED='False')
        error = None
        with (self.root / (name + '.stdout')).open('xb') as out, (self.root / (name + '.stderr')).open('xb') as err:
            child = subprocess.Popen(args, cwd=self.app, stdout=out, stderr=err, env=env)
            try:
                code = child.wait(timeout=command.get('timeout', 600))
            except subprocess.TimeoutExpired:
                error, code = 'TIMEOUT_PRESERVED', 124
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   capture_output=True, timeout=30, check=True)
                else:
                    child.kill()
                child.wait(timeout=30)
        result = dict(intent, event='RESULT', ended_utc=utc(), duration_s=time.monotonic()-begin,
                      exit_code=code, error=error, child_pid=child.pid)
        atomic_json(receipt, result)
        self.append('COMMANDS.log', json.dumps(result))
        self.append('RUN_LOG.md', f'{utc()} | {self.agent} | {self.model} | {name}: exit {code}; {receipt.name}')
        self.checkpoint('Completed ' + name + '; next command is governed by the immutable job plan')
        if code not in command.get('allowed_exit_codes', [0]):
            raise RuntimeError('Command failed; evidence preserved: ' + name)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args(argv)
    require_limited()
    path = Path(args.plan)
    plan = json.loads(path.read_bytes())
    root = Path(plan['root'])
    recorder = Recorder(root, plan['app'], agent=plan['agent'], model=plan['model'], phase=plan['phase'])
    terminal = root / (plan['label'] + '-job.json')
    if terminal.exists():
        raise FileExistsError('Terminal job cannot be automatically resumed')
    started, begin, results = utc(), time.monotonic(), []
    failed = None
    active = root / (plan['label'] + '-active.json')
    atomic_json(active, dict(status='RUNNING', pid=os.getpid(), started_utc=started,
                             task='CloudRAG-I5-' + plan['label'], plan=str(path)))
    try:
        for command in plan['commands']:
            results.append(recorder.run(command))
    except Exception as exc:
        failed = dict(type=type(exc).__name__, message=str(exc))
    outcome = dict(status='PASS' if failed is None else 'FAIL_PRESERVED', started_utc=started,
                   ended_utc=utc(), duration_s=time.monotonic()-begin, results=results, error=failed,
                   plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), limited_token=True)
    atomic_json(terminal, outcome)
    atomic_json(active, dict(status='TERMINAL', pid=os.getpid(), receipt=str(terminal), ended_utc=utc()))
    recorder.checkpoint(plan.get('next_action', 'Inspect ' + terminal.name) if failed is None
                        else 'Diagnose ' + terminal.name + '; no automatic replay')
    return 0 if failed is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
