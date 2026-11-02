"""Cierre por inventario de recursos I7 (auditoría 2026-11-02, F-01 / B-3 / B-4).

Herramienta del dueño, fuera del pipeline verificado. Por defecto SOLO LEE:
    python closeout.py --sdk <gcloud> --project <id> [--root <raíz I7>] --out <carpeta>
escribe <carpeta>/inventory.json y <carpeta>/closeout-log.jsonl, e imprime la tabla.

Borrado (solo con --delete y el inventario de una lectura previa):
    python closeout.py --sdk <gcloud> --project <id> --out <carpeta> --delete --inventory <carpeta>/inventory.json
Pide, por recurso, una palabra de confirmación distinta (BORRAR-<tipo>-<últimos 6 del id>).
Antes de cada borrado vuelve a describir el recurso y aborta ese recurso si cambió el id o el marcador.
Orden: VM, reserva, discos, snapshots, reglas. Reanudable: un recurso ausente se registra y se sigue.
Nunca toca la regla permanente cloudrag-i5-i6cowork-iap ni recursos sin marcador I7.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PERMANENT = frozenset({'cloudrag-i5-i6cowork-iap'})
KINDS = ('instances', 'reservations', 'disks', 'snapshots', 'firewall-rules')   # orden de borrado
ZONAL = frozenset({'instances', 'reservations', 'disks'})
I7_MARKER = re.compile(r'cloudrag-i7-[0-9a-f]{32}')
CPU_MARKER = re.compile(r'CloudRAG-I5-restore-cpu-(i7cpu-[0-9a-f]{8})')
CPU_NAME = re.compile(r'cloudrag-i5-restore-(i7cpu-[0-9a-f]{8})(-boot|-iap)?')
RETAINED_NAME = re.compile(r'cloudrag-i7-(vm-[0-9a-f]{12}(-boot)?|clean-[0-9a-f]{12})')
NOT_FOUND = re.compile(r'was not found|notFound|HTTPError 404|NOT_FOUND', re.I)


class GcloudError(RuntimeError):
    def __init__(self, argv, code, stderr):
        super().__init__('gcloud exit %s: %s' % (code, ' '.join(argv[1:4])))
        self.not_found = bool(NOT_FOUND.search(stderr or ''))


def subprocess_runner(argv):
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', timeout=600)
    return result.returncode, result.stdout, result.stderr


class Gcloud:
    def __init__(self, sdk, project, runner=subprocess_runner):
        if not re.fullmatch(r'[a-z][a-z0-9-]{4,28}[a-z0-9]', project):
            raise ValueError('PROJECT_ID_REQUIRED')
        self.sdk, self.project, self.runner = sdk, project, runner

    def __call__(self, *args, json_output=True):
        argv = [self.sdk, 'compute', *args, '--project='+self.project, '--quiet']
        if json_output:
            argv.append('--format=json')
        code, stdout, stderr = self.runner(argv)
        if code:
            raise GcloudError(argv, code, stderr)
        return json.loads(stdout) if json_output and stdout.strip() else None


def last(value):
    return str(value or '').rsplit('/', 1)[-1]


def root_scope(root):
    """Marcador exacto y etiqueta de CPU de una raíz I7 (periodo técnico, no el del smoke: F-01)."""
    config = json.loads((Path(root)/'installation.json').read_bytes())
    technical = config.get('period_ids', {}).get('technical', config['period_id'])
    return {'marker': config['resource_marker'], 'cpu_label': 'i7cpu-'+technical[:8]}


def classify(kind, row, scope=None):
    """None si el recurso no es nuestro; si no, ('temporary'|'retained', motivo)."""
    name, description = row.get('name', ''), row.get('description') or ''
    if name in PERMANENT:
        return None
    cpu = CPU_MARKER.fullmatch(description)
    if cpu:
        named = CPU_NAME.fullmatch(name)
        if not named or named[1] != cpu[1] or (scope and cpu[1] != scope['cpu_label']):
            return None
        expected = {'instances': None, 'disks': '-boot', 'firewall-rules': '-iap'}
        if kind not in expected or named[2] != expected[kind]:
            return None
        return 'temporary', 'CPU '+cpu[1]
    if I7_MARKER.fullmatch(description) and name.startswith('cloudrag-i7-'):
        if scope and description != scope['marker']:
            return None
        if kind == 'firewall-rules':
            return None
        return ('retained' if RETAINED_NAME.fullmatch(name) else 'temporary'), 'I7 '+description[12:20]
    return None


def confirmation_word(kind, resource_id):
    return 'BORRAR-%s-%s' % (kind.replace('-rules', ''), str(resource_id)[-6:])


def inventory(gcloud, scope=None):
    rows = []
    for kind in KINDS:
        for row in gcloud(kind, 'list') or []:
            verdict = classify(kind, row, scope)
            if verdict is None:
                continue
            size = row.get('sizeGb') or row.get('diskSizeGb') or row.get('storageBytes')
            rows.append({'kind': kind, 'name': row['name'], 'id': str(row['id']), 'zone': last(row.get('zone')) or None,
                         'status': row.get('status'), 'size': size, 'created': row.get('creationTimestamp'),
                         'description': row.get('description'), 'class': verdict[0], 'scope': verdict[1],
                         'deletion_protection': bool(row.get('deletionProtection')),
                         'confirm': confirmation_word(kind, row['id'])})
    return rows


class Log:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, **event):
        event = dict(event, at=datetime.now(timezone.utc).isoformat())
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True)+'\n')
        return event


def zone_flag(row):
    return ['--zone='+row['zone']] if row['kind'] in ZONAL else []


def consumers(gcloud, reservation_name):
    return [vm for vm in gcloud('instances', 'list') or []
            if reservation_name in ((vm.get('reservationAffinity') or {}).get('values') or [])]


def delete_one(gcloud, row, log, scope=None):
    kind, name = row['kind'], row['name']
    try:
        current = gcloud(kind, 'describe', name, *zone_flag(row))
    except GcloudError as error:
        if error.not_found:
            return log(resource=name, kind=kind, id=row['id'], result='ALREADY_ABSENT')
        return log(resource=name, kind=kind, id=row['id'], result='DESCRIBE_ERROR_SKIPPED')
    if str(current.get('id')) != row['id'] or classify(kind, current, scope) is None:
        return log(resource=name, kind=kind, id=row['id'], observed_id=str(current.get('id')),
                   result='ABORTED_IDENTITY_CHANGED')
    if kind == 'instances' and current.get('status') != 'TERMINATED':
        return log(resource=name, kind=kind, id=row['id'], status=current.get('status'), result='ABORTED_VM_NOT_TERMINATED')
    if kind == 'reservations':
        active = [vm['name'] for vm in consumers(gcloud, name)]
        if active:
            return log(resource=name, kind=kind, id=row['id'], consumers=active, result='ABORTED_RESERVATION_HAS_CONSUMER')
    try:
        if kind == 'instances' and current.get('deletionProtection'):
            gcloud('instances', 'update', name, *zone_flag(row), '--no-deletion-protection', json_output=False)
            log(resource=name, kind=kind, id=row['id'], result='DELETION_PROTECTION_REMOVED')
        extra = ['--keep-disks=all'] if kind == 'instances' else []
        gcloud(kind, 'delete', name, *zone_flag(row), *extra, json_output=False)
    except GcloudError as error:
        return log(resource=name, kind=kind, id=row['id'], result='DELETE_ERROR_RETRY_LATER' if not error.not_found
                   else 'ALREADY_ABSENT')
    listed = gcloud(kind, 'list') or []
    gone = all(str(r.get('id')) != row['id'] for r in listed)
    return log(resource=name, kind=kind, id=row['id'], result='DELETED_VERIFIED' if gone else 'DELETE_SENT_UNVERIFIED')


def delete(gcloud, rows, log, *, ask=input, include_retained=False, scope=None):
    results = []
    for kind in KINDS:
        for row in [r for r in rows if r['kind'] == kind]:
            if row['name'] in PERMANENT:
                results.append(log(resource=row['name'], kind=kind, result='PERMANENT_NEVER_TOUCHED'))
                continue
            if row['class'] == 'retained' and not include_retained:
                results.append(log(resource=row['name'], kind=kind, id=row['id'], result='RETAINED_SKIPPED'))
                continue
            word = confirmation_word(kind, row['id'])
            answer = ask('Borrar %s %s (id %s). Escribe %s para confirmar: ' % (kind, row['name'], row['id'], word))
            if answer.strip() != word:
                results.append(log(resource=row['name'], kind=kind, id=row['id'], result='NOT_CONFIRMED_SKIPPED'))
                continue
            results.append(delete_one(gcloud, row, log, scope))
    return results


def table(rows):
    lines = ['%-15s %-46s %-20s %-11s %-8s %-25s %-9s %s' % ('tipo', 'nombre', 'id', 'estado', 'tamaño', 'creado', 'clase', 'confirmación')]
    for r in rows:
        lines.append('%-15s %-46s %-20s %-11s %-8s %-25s %-9s %s' % (r['kind'], r['name'], r['id'], r['status'] or '-',
                     r['size'] or '-', (r['created'] or '-')[:25], r['class'], r['confirm']))
    return '\n'.join(lines)


def main(argv=None, *, runner=subprocess_runner, ask=input, out=sys.stdout):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sdk', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--root', type=Path, help='raíz I7: restringe al marcador exacto y a la CPU de esa raíz')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delete', action='store_true')
    parser.add_argument('--inventory', type=Path, help='inventory.json de una lectura previa (obligatorio con --delete)')
    parser.add_argument('--include-retained', action='store_true', help='permite borrar VM/disco GPU y snapshot limpio')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = Log(args.out/'closeout-log.jsonl')
    gcloud = Gcloud(args.sdk, args.project, runner)
    scope = root_scope(args.root) if args.root else None
    if not args.delete:
        rows = inventory(gcloud, scope)
        target = args.out/'inventory.json'
        if target.exists():
            target = args.out/('inventory-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with target.open('x', encoding='utf-8') as stream:
            json.dump({'project': args.project, 'scope': scope, 'resources': rows}, stream, ensure_ascii=False, indent=1)
        log(mode='READ_ONLY', inventory=str(target), resources=len(rows))
        print(table(rows), file=out)
        print('\nInventario: %s (solo lectura; nada borrado)' % target, file=out)
        return 0
    if not args.inventory:
        parser.error('--delete requiere --inventory de una lectura previa')
    saved = json.loads(args.inventory.read_bytes())
    if saved.get('project') != args.project or saved.get('scope') != scope:
        parser.error('El inventario es de otro proyecto o de otro alcance')
    log(mode='DELETE', inventory=str(args.inventory), resources=len(saved['resources']))
    results = delete(gcloud, saved['resources'], log, ask=ask, include_retained=args.include_retained, scope=scope)
    for row in results:
        print('%-32s %-46s %s' % (row['result'], row['resource'], row.get('id', '')), file=out)
    bad = [r for r in results if r['result'] in ('ABORTED_IDENTITY_CHANGED', 'DESCRIBE_ERROR_SKIPPED',
                                                    'DELETE_ERROR_RETRY_LATER', 'DELETE_SENT_UNVERIFIED',
                                                    'ABORTED_VM_NOT_TERMINATED', 'ABORTED_RESERVATION_HAS_CONSUMER')]
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
