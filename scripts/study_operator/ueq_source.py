"""Generate source metadata from preserved official downloads, never typed hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from src.ui.components.study_ueq import HANDBOOK_URL, INSTRUCTION, ITEMS, SOURCE_URL
from src.ui.components.study_ueq import SOURCE_FILE, instrument
from src.ui.components.session_storage import atomic_json


def upgrade_protocol(config_path):
    """Add only the approved instrument/version; preserve all prior protocol values."""
    config_path = Path(config_path)
    prior = json.loads(config_path.read_bytes())
    if prior.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported study protocol version')
    source_digest = hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()
    result = dict(prior, schema_version=2, ueq_s=instrument(), ueq_s_source_sha256=source_digest)
    if prior.get('schema_version') == 2 and prior != result:
        raise ValueError('Do not replace an altered existing UEQ-S instrument')
    if prior != result:
        atomic_json(config_path, result)
    return result


def generate(receipt, downloaded, target):
    receipt, downloaded, target = map(Path, (receipt, downloaded, target))
    source = json.loads(receipt.read_bytes())
    rows = {row['name']: row for row in source['files']}
    for name, url in [('UEQS_Items.pdf', SOURCE_URL), ('Handbook.pdf', HANDBOOK_URL)]:
        row = rows[name]
        path = downloaded / name
        if row['url'] != url or row['final_url'] != url or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Official source download identity mismatch')
    instrument = dict(schema_version=1, source_url=SOURCE_URL, source_sha256=rows['UEQS_Items.pdf']['sha256'],
                      handbook_url=HANDBOOK_URL, handbook_sha256=rows['Handbook.pdf']['sha256'],
                      source_language='Spanish', source_page=1, items=[list(pair) for pair in ITEMS],
                      instruction=INSTRUCTION, instruction_source='application UX; not a standardized UEQ-S instruction',
                      positions=list(range(1, 8)), negative_on_left=True,
                      scoring='position minus 4; pragmatic items 1..4; hedonic 5..8; overall all 8')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(instrument, stream, ensure_ascii=False, indent=2)
    return dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source_files_verified=True, literal_items_review_source='official Spanish page 1')


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ['receipt', 'downloaded', 'target']:
        parser.add_argument('--'+name)
    parser.add_argument('--upgrade-protocol')
    args = parser.parse_args(argv)
    if args.upgrade_protocol:
        upgrade_protocol(args.upgrade_protocol)
        print(json.dumps(dict(status='PROTOCOL_SCHEMA2_UEQ_ADDED_OTHER_VALUES_PRESERVED')))
    elif all([args.receipt, args.downloaded, args.target]):
        print(json.dumps(generate(args.receipt, args.downloaded, args.target)))
    else:
        parser.error('Provide downloads/source target or --upgrade-protocol')


if __name__ == '__main__':
    main()
