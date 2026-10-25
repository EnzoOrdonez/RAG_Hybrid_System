"""Hash-bound phase claims with exact immutable command receipts; no verdict synthesis."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from filelock import FileLock

from src.ui.components.session_storage import atomic_json

CERTAINTIES = {'VERIFICADO', 'DECLARADO', 'SUPUESTO', 'ESTIMADO'}
MUTABLE = {'STATE.json', 'HANDOVER.md', 'RUN_LOG.md', 'COMMANDS.log', 'CLAIMS_LEDGER.md', 'claims.json',
           'claim_corrections.json'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def add(root, specifications):
    root = Path(root).resolve()
    actor = json.loads((root/'STATE.json').read_bytes())
    if not actor.get('agent') or not actor.get('model'):
        raise ValueError('Agent and model required in current state')
    with FileLock(str(root/'claims.lock'), timeout=10):
        path = root/'claims.json'
        rows = json.loads(path.read_bytes()) if path.exists() else []
        keys = {row['key'] for row in rows}
        for spec in specifications:
            if spec['key'] in keys:
                prior = next(row for row in rows if row['key'] == spec['key'])
                if prior['statement'] != spec['statement'] or prior['certainty'] != spec['certainty']:
                    raise ValueError('Existing claim cannot be relabeled')
                continue
            if spec['certainty'] not in CERTAINTIES or not spec['evidence']:
                raise ValueError('Explicit certainty and evidence required')
            evidence = []
            for name in spec['evidence']:
                target = (root/name).resolve()
                if not target.is_relative_to(root) or target.name in MUTABLE:
                    raise ValueError('Use immutable own-package evidence; archive mutable checkpoints first')
                evidence.append(dict(path=target.relative_to(root).as_posix(), sha256=sha(target), bytes=target.stat().st_size))
            commands = []
            for name in spec.get('command_receipts', []):
                target = (root/name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Command receipt outside own package')
                value = json.loads(target.read_bytes())
                commands.append(dict(receipt=name, receipt_sha256=sha(target), command=value['command'], exit_code=value['exit_code']))
            if spec['certainty'] == 'VERIFICADO' and not commands:
                raise ValueError('Verified claim needs its own exact command receipt')
            rows.append(dict(id=f'I5-V{len(rows)+1:03}', key=spec['key'], statement=spec['statement'],
                certainty=spec['certainty'], evidence=evidence, commands=commands,
                at=datetime.now(timezone.utc).isoformat(), agent=actor['agent'], model=actor['model']))
            keys.add(spec['key'])
        atomic_json(path, rows)
        render(root, rows)
        return rows


def corrections(root, rows):
    path = Path(root)/'claim_corrections.json'
    values = json.loads(path.read_bytes()) if path.exists() else []
    indices = {row['id']: index for index,row in enumerate(rows)}
    seen = set()
    for value in values:
        old,new = value['retracted_claim'],value['replacement_claim']
        if (old not in indices or new not in indices or indices[old] >= indices[new]
                or old in seen or not value.get('reason') or not value.get('agent')
                or not value.get('model') or not value.get('at')):
            raise ValueError('Invalid or conflicting claim correction')
        seen.add(old)
    return values


def render(root, rows):
    replaced = {r['retracted_claim']:r for r in corrections(root,rows)}
    text = '# Afirmaciones\n\nTodo resultado heredado sigue DECLARADO salvo comprobación propia indicada.\n\n'
    for row in rows:
        text += f'## {row["id"]} · {row["certainty"]}\n\n'
        if row['id'] in replaced:
            change = replaced[row['id']]
            text += ('**RETRACTADA: no usar como resultado.** Sustituida por '+change['replacement_claim']+
                     '. Motivo: '+change['reason']+'\n\n')
        text += row['statement']+'\n\n'
        for item in row['evidence']:
            text += f'- `{item["path"]}` · SHA-256 `{item["sha256"]}`\n'
        text += '\n'
        for command in row['commands']:
            text += 'Comando exacto (vector de argumentos), salida '+str(command['exit_code'])+':\n\n'
            text += '```json\n'+json.dumps(command['command'],ensure_ascii=False)+'\n```\n\n'
    (Path(root)/'CLAIMS_LEDGER.md').write_text(text,encoding='utf-8')


def retract(root, claim_id, replacement_id, reason):
    """Append a correction without rewriting the original claim or its evidence."""
    root = Path(root)
    with FileLock(str(root/'claims.lock'),timeout=10):
        verify(root)
        rows = json.loads((root/'claims.json').read_bytes())
        values = corrections(root,rows)
        prior = next((r for r in values if r['retracted_claim'] == claim_id),None)
        if prior:
            if prior['replacement_claim'] != replacement_id or prior['reason'] != reason:
                raise ValueError('Existing correction cannot be rewritten')
            return prior
        actor = json.loads((root/'STATE.json').read_bytes())
        result = dict(retracted_claim=claim_id,replacement_claim=replacement_id,reason=reason,
                      agent=actor['agent'],model=actor['model'],at=datetime.now(timezone.utc).isoformat())
        # Validate before writing any correction or changing the rendered ledger.
        indices = {r['id']:i for i,r in enumerate(rows)}
        if (claim_id not in indices or replacement_id not in indices
                or indices[claim_id] >= indices[replacement_id] or not reason.strip()):
            raise ValueError('A correction needs a later existing replacement and a reason')
        atomic_json(root/'claim_corrections.json',[*values,result])
        render(root,rows)
        return result


def verify(root):
    root = Path(root)
    rows = json.loads((root/'claims.json').read_bytes())
    for row in rows:
        for item in row['evidence']:
            if sha(root/item['path']) != item['sha256']:
                raise ValueError('Claim evidence changed: ' + row['id'])
        for command in row['commands']:
            if sha(root/command['receipt']) != command['receipt_sha256']:
                raise ValueError('Claim command receipt changed: ' + row['id'])
    withdrawn = {r['retracted_claim'] for r in corrections(root,rows)}
    return dict(claims=len(rows), current_claims=len(rows)-len(withdrawn),
                retracted_claims=sorted(withdrawn),hashes_verified=True, semantic_acceptance_not_inferred=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--specifications')
    args = parser.parse_args(argv)
    if args.specifications:
        add(args.root, json.loads(Path(args.specifications).read_bytes()))
    print(json.dumps(verify(args.root)))


if __name__ == '__main__':
    main()
