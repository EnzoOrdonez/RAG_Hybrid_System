"""Owner-side cold jobs: no model execution; private coded evidence never printed."""
import json
from pathlib import Path
import re

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.stimulus_host_evidence import sha


def run(operator, operation, *, boot_index=None):
    if operator.state.get('purpose') != 'technical':
        raise OperatorError('El ensayo del estímulo exige technical y un periodo vacío; study no se mide.')
    if operator.observed()['status'] != 'RUNNING':
        raise OperatorError('La VM no está encendida. Conserva el intento terminal; no repitas un arranque sin revisar su evidencia.')
    if operation == 'stimulus-start':
        if type(boot_index) is not int or not 1 <= boot_index <= 12:
            raise OperatorError('Selecciona el arranque 1 a 12 del calendario anclado.')
        operator.preflight()
        return operator.bridge(dict(operation=operation,boot_index=boot_index),private=True)
    if operation == 'stimulus-status':
        return operator.bridge(dict(operation=operation),private=True)
    if operation != 'stimulus-collect':
        raise OperatorError('Operación de estímulo desconocida. Revisa el runbook.')
    result = operator.bridge(dict(operation='stimulus-evidence'),private=True)
    proof = result['proof']
    boot = proof['inventory']['observed']['boot_id']
    if not re.fullmatch('[a-f0-9-]{36}',boot) or sha(proof) != result['proof_sha256']:
        raise OperatorError('La descarga del estímulo no coincide con su hash; conserva el disco y no sustituya el intento.')
    root = Path(operator.root)/'private'/'stimulus'/'coded-P999'
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    path = root/(boot+'.json')
    if path.exists():
        if sha(json.loads(path.read_bytes())) != result['proof_sha256']:
            raise OperatorError('Ya existe evidencia distinta para ese arranque. No se reemplaza ni se reanuda.')
    else:
        save_state(path,proof)
    if sha(json.loads(path.read_bytes())) != result['proof_sha256']:
        raise OperatorError('La descarga local no se confirmó. Conserva el disco y no confirme el apagado.')
    operator.bridge(dict(operation='stimulus-ack',proof_sha256=result['proof_sha256']),private=True)
    return dict(status='STIMULUS_DOWNLOADED_VERIFIED_NOT_ACCEPTANCE',path=str(path),
                proof_sha256=result['proof_sha256'],boot_index=proof['boot_index'],calls=len(proof['rows']),
                private_content_not_logged=True,acceptance_not_inferred=True)
