"""Short Limited-token queue dispatcher; no conversation-dependent scheduler."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from filelock import FileLock

from scripts.study_operator.run_control import require_limited, validate_command


def checked_request(root, app, request):
    root, app = Path(root).resolve(), Path(app).resolve()
    path = Path(request['plan']).resolve()
    if path.parent != root or not path.name.endswith('-plan.json'):
        raise ValueError('Plan must be an immutable own-package file')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != request['plan_sha256']:
        raise ValueError('Queue plan changed')
    plan = json.loads(content)
    if Path(plan['root']).resolve() != root or Path(plan['app']).resolve() != app:
        raise ValueError('Foreign plan scope')
    if not 1 <= request['native_minutes'] <= 180 or type(request['native_minutes']) is not int:
        raise ValueError('Finite native job limit required')
    for command in plan['commands']:
        validate_command(command)
    return path


def dispatch(root, app, *, invoke=subprocess.run):
    root, app = Path(root).resolve(), Path(app).resolve()
    state = json.loads((root/'STATE.json').read_bytes())
    if state['status'] != 'ACTIVE' or datetime.now(timezone.utc) >= datetime.fromisoformat(state['closure_reserved_utc']):
        return dict(status='ADMISSION_CLOSED', dispatched=0)
    queued = root/'task-queue'
    queued.mkdir(exist_ok=True)
    count = 0
    with FileLock(str(root/'task-broker.lock'), timeout=0):
        for request_path in sorted(queued.glob('*.request.json')):
            receipt = request_path.with_suffix('.receipt.json')
            if receipt.exists():
                continue
            request = json.loads(request_path.read_bytes())
            path = checked_request(root, app, request)
            started, begin = datetime.now(timezone.utc).isoformat(), time.monotonic()
            argv = ['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                    str(app/'scripts/study_operator/register_job.ps1'), '-Plan', str(path),
                    '-Minutes', str(request['native_minutes'])]
            child = invoke(argv, capture_output=True, timeout=60)
            result = dict(status='DISPATCHED_LIMITED' if child.returncode == 0 else 'REGISTRATION_FAILED_PRESERVED',
                command=argv, exit_code=child.returncode, phase=2, agent=state['agent'], model=state['model'],
                started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-begin,
                stdout=child.stdout.decode('utf-8', errors='replace'), stderr=child.stderr.decode('utf-8', errors='replace'),
                request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(), plan_sha256=request['plan_sha256'],
                registering_token_limited=True)
            with receipt.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
            with (root/'COMMANDS.log').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result)+'\n')
            count += 1
    return dict(status='QUEUE_RECONCILED', dispatched=count)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--app', required=True)
    args = parser.parse_args(argv)
    require_limited()
    print(json.dumps(dispatch(args.package, args.app)))


if __name__ == '__main__':
    main()
