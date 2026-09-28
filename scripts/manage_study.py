"""Local operator CLI. No model execution, network publication or participant contact."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.study_analysis import analyze, read_exports
from src.ui.components.session_storage import atomic_json
from src.ui.components.study_protocol import ROOT, digest, load_protocol
from src.ui.components.study_sessions import StudySession, StudyStore


def consolidate(store, output):
    store.check()
    output = Path(output).resolve()
    if output.is_relative_to(ROOT.parent.parent) or output.exists():
        raise ValueError('Use a NEW external export directory')
    paths = []
    for checkpoint in sorted(store.root.glob('*/study_checkpoint.json')):
        session = StudySession.load(store, checkpoint.parent.name)
        if session.data['stage'] in ('complete', 'abandoned'):
            paths.append(session.export())
    rows, hashes = read_exports(paths)
    output.mkdir(parents=True)
    atomic_json(output / 'sessions.json', dict(schema_version=3, sessions=rows, source_hashes=hashes))
    atomic_json(output / 'analysis.json', analyze(rows))
    dictionary = ROOT / 'docs/STUDY_DATA_DICTIONARY.md'
    (output / dictionary.name).write_bytes(dictionary.read_bytes())
    atomic_json(output / 'manifest.json', dict(files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--assignments', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--purpose', choices=('study', 'pilot', 'technical'), default='study')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('preflight')
    invite = sub.add_parser('invite')
    invite.add_argument('participant')
    invite.add_argument('--cell', type=int)
    invite.add_argument('--profile', choices=('without_experience', 'with_experience'))
    for name in ('abandon', 'incident'):
        sub.add_parser(name).add_argument('session_id')
    replace = sub.add_parser('replace')
    replace.add_argument('primary')
    replace.add_argument('reserve')
    sub.add_parser('export').add_argument('output')
    args = parser.parse_args(argv)
    store = StudyStore(args.root, load_protocol(args.config, args.assignments), args.purpose)
    if args.command == 'freeze':
        store.freeze()
    else:
        store.check()
    if args.command == 'invite':
        print(store.issue(args.participant, cell=args.cell, profile=args.profile))
    elif args.command == 'abandon':
        store.abandon(args.session_id)
    elif args.command == 'incident':
        session = StudySession.load(store, args.session_id)
        if session is None:
            raise ValueError('Unknown session')
        session.incident()
    elif args.command == 'replace':
        store.replace(args.primary, args.reserve)
    elif args.command == 'export':
        print(consolidate(store, args.output))
    else:
        print(json.dumps(dict(configuration='valid', fingerprint=store.protocol['fingerprint'],
                              deployment='NOT_CHECKED', warmup='NOT_RUN')))


if __name__ == '__main__':
    main()
