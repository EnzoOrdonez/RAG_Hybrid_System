"""P2: extract reported decimal figures from docs and look for a machine-produced source with the same value.

A figure is 'verificada' when some JSON/CSV under experiments/results or output/ contains a number that rounds
to it at the reported precision; 'sin fuente' otherwise. Only decimals with >=2 decimal places are traced
(integers such as n or dates are too ambiguous for value matching). This is evidence of existence of a value,
not of semantic identity: the hand-checked key figures are listed separately in TRAZABILIDAD.md.
Usage: python trace_figures.py OUT.json DOC [DOC...]
"""
import csv, json, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NUM = re.compile(r'(?<![\w.,])[-−+]?\d+[.,]\d{2,6}(?![\w])')

def numbers_in(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj)); return
    if isinstance(obj, str):
        try: acc.add(float(obj))
        except ValueError: pass
        return
    if isinstance(obj, dict):
        for v in obj.values(): numbers_in(v, acc)
    elif isinstance(obj, list):
        for v in obj: numbers_in(v, acc)

def source_values():
    index = {}
    roots = [REPO/'experiments/results', REPO/'output']
    for root in roots:
        for p in root.rglob('*'):
            if p.suffix not in ('.json', '.csv') or not p.is_file() or p.stat().st_size > 30_000_000:
                continue
            acc = set()
            try:
                if p.suffix == '.json':
                    numbers_in(json.loads(p.read_text(encoding='utf-8')), acc)
                else:
                    with p.open(encoding='utf-8', newline='') as f:
                        for row in csv.reader(f):
                            for cell in row:
                                try: acc.add(float(cell))
                                except ValueError: pass
            except Exception:
                continue
            index[str(p.relative_to(REPO))] = acc
    return index

def rounded(x, places):
    q = Decimal(1).scaleb(-places)
    return Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)

def main():
    out, docs = Path(sys.argv[1]), sys.argv[2:]
    index = source_values()
    by_places = {}
    rows = []
    for doc in docs:
        for lineno, line in enumerate((REPO/doc).read_text(encoding='utf-8').splitlines(), 1):
            for m in NUM.finditer(line):
                text = m.group(0).replace('−', '-').replace(',', '.')
                places = len(text.split('.')[1])
                target = Decimal(text)
                if places not in by_places:
                    by_places[places] = {f: {rounded(v, places) for v in vals if abs(v) < 1e9} for f, vals in index.items()}
                hits = sorted(f for f, vals in by_places[places].items() if target in vals or -target in vals)
                rows.append({'doc': doc, 'line': lineno, 'figure': m.group(0), 'status': 'verificada' if hits else 'sin fuente',
                             'sources': hits[:3], 'n_sources': len(hits), 'context': line.strip()[:160]})
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding='utf-8')
    for doc in docs:
        mine = [r for r in rows if r['doc'] == doc]
        print(doc, len(mine), 'verificadas', sum(r['status'] == 'verificada' for r in mine))

if __name__ == '__main__':
    main()
