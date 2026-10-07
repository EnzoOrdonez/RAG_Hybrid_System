"""Compare full Ruff output without suppressing or relabeling inherited findings."""
import argparse
from collections import Counter
import json
from pathlib import Path


def findings(rows, app):
    app = Path(app).resolve()
    return Counter((Path(row['filename']).resolve().relative_to(app).as_posix(),
                    row['code'], row['message']) for row in rows)


def compare(baseline, current, app):
    prior, now = findings(baseline, app), findings(current, app)
    new, removed = now-prior, prior-now
    return dict(global_ruff='PASS' if not now else 'FAILED',
                new_findings=sum(new.values()), inherited_findings=sum((now & prior).values()),
                removed_findings=sum(removed.values()), baseline_count=sum(prior.values()),
                status='NO_NEW_FINDINGS' if not new else 'NEW_FINDINGS_FAILED',
                new=[dict(path=k[0], code=k[1], message=k[2], count=count) for k, count in sorted(new.items())],
                location_changes_do_not_suppress_additional_occurrences=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'current', 'app', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    result = compare(json.loads(Path(args.baseline).read_bytes()), json.loads(Path(args.current).read_bytes()), args.app)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    return int(bool(result['new_findings']))


if __name__ == '__main__':
    raise SystemExit(main())
