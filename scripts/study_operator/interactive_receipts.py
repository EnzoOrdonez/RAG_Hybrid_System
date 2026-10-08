"""Import observed tool durations once without inventing execution timestamps."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.run_control import require_limited


def import_receipts(root):
    root = Path(root).resolve()
    with FileLock(str(root/'state.lock'), timeout=10):
        state = json.loads((root/'STATE.json').read_bytes())
        if state['status'] not in {'ACTIVE', 'CLOSING'} or (root/'MANIFEST_SHA256.jsonl').exists():
            raise ValueError('Unsealed active or closing own package required')
        logfile = root/'COMMANDS.log'
        previous = [json.loads(line) for line in logfile.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        imported = {row['interactive_source']: row['interactive_sha256'] for row in previous if 'interactive_source' in row}
        prepared = []
        for path in sorted((root/'interactive-receipts').glob('*.json')):
            if path.is_symlink() or path.is_junction():
                raise ValueError('Linked tool receipt rejected')
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            name = path.relative_to(root).as_posix()
            if name in imported:
                if imported[name] != digest:
                    raise ValueError('Previously imported tool receipt changed')
                continue
            row = json.loads(content)
            duration = row['duration_s']
            datetime.strptime(row['observed_after_utc'], '%Y-%m-%d %H:%M:%S UTC')
            if (row['tool'] != 'exec_command' or row['agent'] != state['agent'] or row['model'] != state['model']
                    or row['start_timestamp_not_reconstructed'] is not True
                    or type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
                    or type(row['phase']) is not int or not 0 <= row['phase'] <= 6
                    or not isinstance(row['input']['cmd'], str)):
                raise ValueError('Invalid observed tool receipt')
            prepared.append(dict(row, event='RESULT', command=row['input']['cmd'],
                interactive_source=name, interactive_sha256=digest,
                interval_not_locatable=True, excluded_from_interval_union=True))
        with logfile.open('a', encoding='utf-8') as stream:
            for row in prepared:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        return dict(status='INTERACTIVE_DURATIONS_IMPORTED_WITHOUT_RECONSTRUCTION', imported=len(prepared),
            already_imported=len(imported), interval_union_contribution_seconds=0,
            new_unlocatable_duration_seconds=sum(row['duration_s'] for row in prepared),
            not_added_to_phase_union=True, missing_historical_receipts_not_reconstructed=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require_limited()
    root, output = Path(args.package).resolve(), Path(args.output).resolve()
    if output.parent != root or output.exists():
        raise ValueError('Exclusive own-package import receipt required')
    result = import_receipts(root)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
