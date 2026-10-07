"""New operator entry point. Plain invitations only reach the interactive console."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.lifecycle import Operator
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.region_scope import US_L4_ZONES
from scripts.study_operator.run_control import require_limited
from scripts.study_operator.service_gateway import save_state


def parser():
    value = argparse.ArgumentParser(description='Operador CloudRAG iteración 5; sin participantes en ensayos.')
    value.add_argument('--root',default='C:/CloudRAG/operator-iteration5')
    sub = value.add_subparsers(dest='operation',required=True)
    start = sub.add_parser('start')
    start.add_argument('--purpose',required=True,choices=('study','technical','smoke','rehearsal','pilot'))
    for name in ('status','preflight','stop','diagnostics','ip-reserve','ip-release','failback','export-anonymized'):
        sub.add_parser(name)
    invite = sub.add_parser('invite')
    invite.add_argument('code')
    invite.add_argument('--cell',type=int,choices=(1,2,3,4))
    invite.add_argument('--profile',choices=('with_experience','without_experience'))
    revoke = sub.add_parser('revoke')
    revoke.add_argument('code')
    tls = sub.add_parser('tls-prepare')
    tls.add_argument('--first-session',required=True)
    alternate = sub.add_parser('failover')
    alternate.add_argument('--zone',required=True,choices=sorted(US_L4_ZONES))
    bootstrap = sub.add_parser('bootstrap')
    bootstrap.add_argument('--zone', required=True, choices=sorted(US_L4_ZONES))
    for name in ('purge-study','withdraw','archive-local'):
        command = sub.add_parser(name)
        if name == 'withdraw':
            command.add_argument('code')
        command.add_argument('--execute',action='store_true',help='Sin este flag solo se simula.')
    restore = sub.add_parser('restore')
    restore.add_argument('code')
    restore.add_argument('--session-id',required=True)
    restore.add_argument('--full-generation',required=True)
    restore.add_argument('--manifest-generation',required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    require_limited()
    root = Path(args.root).resolve()
    if root != Path('C:/CloudRAG/operator-iteration5').resolve():
        raise OperatorError('Raíz no autorizada. Usa C:/CloudRAG/operator-iteration5; los operadores anteriores quedan congelados.')
    root.mkdir(parents=True,exist_ok=True)
    run = root/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    with FileLock(str(root/'operator.lock'),timeout=0):
        installation = json.loads((root/'installation.json').read_text(encoding='utf-8'))
        cloud = Cloud(installation['gcloud'],installation['project'],run)
        operator = Operator(root,cloud)
        operation = args.operation
        if operation == 'invite':
            if not sys.stdout.isatty():
                raise OperatorError('invite exige consola interactiva sin redirección. No se imprime ni se guarda ningún token.')
            token = operator.invite(args.code,cell=args.cell,profile=args.profile)
            save_state(run/'receipt.json',dict(status='INVITATION_HASH_REGISTERED',ttl_hours=24,token_not_persisted=True))
            print('Token (se muestra una sola vez): '+token)
            return 0
        if operation in {'withdraw','purge-study','archive-local'}:
            confirmation = None
            if args.execute and operator.state.get('purpose') == 'study':
                if not sys.stdin.isatty():
                    raise OperatorError('El borrado study exige confirmar en consola. No uses stdin redirigido.')
                confirmation = input('Escribe '+(args.code if operation == 'withdraw' else
                                                'ARCHIVAR' if operation == 'archive-local' else 'PURGAR')+': ')
            result = operator.delete_sessions(operation,code=getattr(args,'code',None),
                dry_run=not args.execute,confirmation=confirmation)
        elif operation == 'start':
            result = operator.start(args.purpose)
        elif operation == 'tls-prepare':
            result = operator.tls_prepare(args.first_session)
        elif operation == 'failover':
            result = operator.failover(args.zone)
        elif operation == 'bootstrap':
            from scripts.study_operator.bootstrap import bootstrap

            result = bootstrap(operator, args.zone)
        elif operation == 'revoke':
            operator.preflight()
            result = operator.bridge(dict(operation='revoke',participant_id=args.code),private=True)
        elif operation == 'restore':
            result = operator.restore(args.code,args.session_id,full_generation=args.full_generation,
                                      manifest_generation=args.manifest_generation)
        else:
            result = getattr(operator,operation.replace('-','_'))()
        save_state(run/'receipt.json',result)
        print(json.dumps(result,ensure_ascii=False))
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OperatorError,OSError,ValueError,KeyError,TimeoutError) as error:
        message = str(error) if isinstance(error,OperatorError) else 'Operación incompleta. Conserva los recibos, ejecuta status y revisa installation.json y el runbook.'
        print(message,file=sys.stderr)
        raise SystemExit(2)
