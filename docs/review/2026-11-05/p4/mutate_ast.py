"""P4b: first-order AST mutation of one module; each mutant is run against the given tests, then restored.

Operators: + <-> -, * -> /, / -> *, < <-> <=, > <-> >=, == <-> !=, and <-> or, int n -> n+1,
float f -> 2f (0.0 -> 1.0), comparison 'in' <-> 'not in', unary 'not' dropped.
Usage: python mutate_ast.py MODULE.py OUT.json TEST [TEST...]   (env MUT_MAX caps mutants, evenly spaced)
"""
import ast, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = os.environ.get('MUT_PYTHON', str(REPO/'.venv-app/bin/python'))
SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
        ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.And: ast.Or, ast.Or: ast.And,
        ast.In: ast.NotIn, ast.NotIn: ast.In}

def sites(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.BoolOp)) and type(node.op) in SWAP:
            out.append((node.lineno, node.col_offset, 'op', type(node.op).__name__))
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in SWAP:
                    out.append((node.lineno, node.col_offset, f'cmp{i}', type(op).__name__))
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool):
            out.append((node.lineno, node.col_offset, 'const', repr(node.value)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((node.lineno, node.col_offset, 'not', 'Not'))
    return sorted(set(out))

class Mutate(ast.NodeTransformer):
    def __init__(self, site):
        self.site, self.done = site, False
    def generic_visit(self, node):
        node = super().generic_visit(node)
        line, col, kind, _ = self.site
        if self.done or getattr(node, 'lineno', None) != line or getattr(node, 'col_offset', None) != col:
            return node
        if kind == 'op' and isinstance(node, (ast.BinOp, ast.BoolOp)) and type(node.op) in SWAP:
            node.op = SWAP[type(node.op)](); self.done = True
        elif kind.startswith('cmp') and isinstance(node, ast.Compare):
            i = int(kind[3:]); node.ops[i] = SWAP[type(node.ops[i])](); self.done = True
        elif kind == 'const' and isinstance(node, ast.Constant) and type(node.value) in (int, float):
            node.value = node.value + 1 if isinstance(node.value, int) else (node.value * 2 if node.value else 1.0); self.done = True
        elif kind == 'not' and isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self.done = True; return node.operand
        return node

def main():
    module, out, tests = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    path = REPO/module
    original = path.read_text(encoding='utf-8')
    tree = ast.parse(original)
    all_sites = sites(tree)
    cap = int(os.environ.get('MUT_MAX', '0')) or len(all_sites)
    step = max(1, len(all_sites) // cap)
    chosen = all_sites[::step][:cap]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONHASHSEED='42', PYTHONUTF8='1')
    env.pop('PLAYWRIGHT_BROWSERS_PATH', None)
    rows = []
    try:
        for site in chosen:
            mutated = Mutate(site).visit(ast.parse(original))
            path.write_text(ast.unparse(mutated), encoding='utf-8')
            try:
                r = subprocess.run([PY, '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider', '-o', 'addopts=--strict-markers',
                                    *tests], cwd=REPO, env=env, capture_output=True, timeout=int(os.environ.get('MUT_TIMEOUT', '180')))
                status = 'killed' if r.returncode else 'survived'
            except subprocess.TimeoutExpired:
                status = 'killed(timeout)'
            rows.append({'line': site[0], 'col': site[1], 'kind': site[2], 'from': site[3], 'status': status})
            print(module, site, status, flush=True)
    finally:
        subprocess.run(['git', 'checkout', '--', module], cwd=REPO, check=True)
    assert path.read_text(encoding='utf-8') == original
    killed = sum(r['status'] != 'survived' for r in rows)
    out.write_text(json.dumps({'module': module, 'tests': tests, 'sites_total': len(all_sites), 'mutants': len(rows),
                               'killed': killed, 'score': round(killed / len(rows), 3) if rows else None, 'rows': rows}, indent=1))
    print('SCORE', module, killed, '/', len(rows))

if __name__ == '__main__':
    main()
