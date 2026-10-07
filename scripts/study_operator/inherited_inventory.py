"""Read-only inherited defects/source census and static freeze guard for I5."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(old, package, app):
    old, package, app = map(Path, (old, package, app))
    baseline = json.loads((old / 'rag_freeze_baseline.json').read_bytes())
    modules = baseline['rag']['modules']
    current = {name: sha(app / name) for name in modules}
    if current != modules:
        raise ValueError('Frozen RAG source changed; stop this branch')
    ruff = json.loads((package / 'baseline02-ruff.stdout').read_bytes())
    classifications = []
    for row in ruff:
        name = Path(row['filename']).relative_to(app).as_posix()
        classifications.append(dict(path=name, code=row['code'], frozen=name in modules,
             disposition='DECLARED_FROZEN_SOURCE' if name in modules else 'DECLARED_OUTSIDE_AUTHORIZED_SERVICE_SCOPE'))
    oneoffs = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(old.parent.glob('iteration4_*.py'))]
    defects = [
        dict(id='D-I4-01', defect='Ruff global red', evidence='baseline02-ruff.stdout',
             status='DECLARED', reason='No new suppression/baseline changes; four findings in frozen RAG, other legacy files outside I5 service scope'),
        dict(id='D-I4-02', defect='Invalid recorded UTC interval aborts final report',
             evidence='close-report-handoff01-wrapper.stderr', status='CORRECTED_WITH_REGRESSION',
             mechanism='Preserve anomalous row; union only valid intervals and expose unlocatable duration'),
        dict(id='D-I4-03', defect='Single-use tools outside repository', status='INVENTORIED_PRESERVED',
             mechanism='Reusable recovery/control/seal/source tools versioned in scripts/study_operator; old files never removed'),
        dict(id='D-I4-04', defect='Component tests do not establish integrated acceptance', status='DECLARED_PENDING_REAL_ACCEPTANCE',
             evidence='REPORT_WORKING05.md', reason='No live final stimulus, privacy, public smoke or new gate; no synthetic GO'),
        dict(id='D-I4-05', defect='Completed scheduled task definitions cannot be retired with Limited token',
             status='BLOQUEADO-HUMANO', evidence='legacy-tasks-retire.stderr',
             reason='HRESULT 0x80070005; no administrator or DACL workaround authorized; preserve XML and prove no future writer'),
        dict(id='D-I4-06', defect='Runtime named-user candidate not paired-verified', status='PENDING_CPU_PAIR',
             reason='Test absent versus named UID10001 before integrating; no local model measurement'),
        dict(id='D-I4-07', defect='Automatic close backup integration not accepted', status='PENDING_INTEGRATED_PROOF',
             reason='Code already invokes backup_session from closed export; test/recover its host relay rather than duplicate trigger')]
    return dict(at=datetime.now(timezone.utc).isoformat(), certainty='SOURCE_AND_LOG_READS_NOT_REAL_ACCEPTANCE',
                frozen_sources_verified=current, inherited_baseline_sha256=sha(old / 'rag_freeze_baseline.json'),
                defects=defects, ruff_findings=classifications,
                ruff_counts=dict(Counter(row['disposition'] for row in classifications)),
                oneoff_sources=oneoffs, no_old_source_or_evidence_removed=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ['old', 'package', 'app', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = collect(args.old, args.package, args.app)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(defects=len(result['defects']), ruff_counts=result['ruff_counts'],
                          old_oneoff_count=len(result['oneoff_sources']), frozen_sources_pass=True)))


if __name__ == '__main__':
    main()
