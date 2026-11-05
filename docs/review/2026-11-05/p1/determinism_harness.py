"""P1: run statistics scripts under several PYTHONHASHSEED values and compare outputs bit for bit.

Each seed gets its own disposable git worktree at BASE, so signed outputs in the real checkout are never
written. Per run we record exit code, sha256(stdout) and sha256 of every file the run created or modified.
Usage: python determinism_harness.py BASE OUT.json SCRIPT [SCRIPT...]   (env: P1_SEEDS, P1_TIMEOUT)
"""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = os.environ.get('P1_PYTHON', str(REPO/'.venv-app/bin/python'))
SEEDS = [s for s in os.environ.get('P1_SEEDS', '0,7,42').split(',')]
TIMEOUT = int(os.environ.get('P1_TIMEOUT', '120'))

def sh(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout

def worktree(base, seed, scratch):
    path = Path(scratch)/f'wt-{seed}'
    if not path.exists():
        sh('git', '-C', str(REPO), 'worktree', 'add', '-q', '--detach', str(path), base)
    return path

def reset(path):
    sh('git', 'checkout', '-q', '--', '.', cwd=path)
    sh('git', 'clean', '-qfdx', '-e', '.venv-app', cwd=path)

def run_one(path, script, seed):
    reset(path)
    env = dict(os.environ, PYTHONHASHSEED=seed, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1',
               PYTHONPATH=str(path), MPLBACKEND='Agg')
    env.pop('PLAYWRIGHT_BROWSERS_PATH', None)
    try:
        name, _, args = script.partition('::')         # 'script.py::--flag value' passes arguments
        r = subprocess.run([PY, name, *args.split()], cwd=path, env=env, capture_output=True, timeout=TIMEOUT)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 'TIMEOUT', e.stdout or b'', e.stderr or b''
    tag = str(path).encode()
    norm = lambda b: b.replace(tag, b'<WT>')          # each seed has its own worktree path
    out = norm(out)
    changed = {}
    for line in sh('git', 'status', '--porcelain', '-uall', cwd=path).splitlines():
        rel = line[3:].strip().strip('"')
        p = path/rel
        if p.is_file() and '__pycache__' not in rel:
            changed[rel] = hashlib.sha256(norm(p.read_bytes())).hexdigest()
    return {'exit': code, 'stdout_sha256': hashlib.sha256(out).hexdigest(), 'files': changed,
            'stdout_head': out.decode(errors='replace')[:4000],
            'stderr_tail': err.decode(errors='replace').strip().splitlines()[-2:]}

def main():
    base, out, scripts = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    scratch = os.environ.get('P1_SCRATCH') or tempfile.mkdtemp(prefix='p1-')
    trees = {s: worktree(base, s, scratch) for s in SEEDS}
    results = json.loads(out.read_text()) if out.exists() else {}
    for script in scripts:
        with cf.ThreadPoolExecutor(len(SEEDS)) as pool:
            runs = dict(zip(SEEDS, pool.map(lambda s: run_one(trees[s], script, s), SEEDS)))
        ok = all(r['exit'] == 0 for r in runs.values())
        same = len({json.dumps([r['stdout_sha256'], r['files']], sort_keys=True) for r in runs.values()}) == 1
        differing = sorted({f for r in runs.values() for f in r['files']
                            if len({rr['files'].get(f) for rr in runs.values()}) > 1})
        results[script] = {'ran_ok': ok, 'deterministic': same if ok else None, 'differing_files': differing,
                           'stdout_differs': len({r['stdout_sha256'] for r in runs.values()}) > 1, 'runs': runs}
        print(script, 'ok' if ok else 'FAIL', 'DET' if same else 'NONDET', differing[:3], flush=True)
        out.write_text(json.dumps(results, indent=1))

if __name__ == '__main__':
    main()
