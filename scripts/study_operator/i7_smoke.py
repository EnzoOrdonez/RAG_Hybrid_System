"""Validated, fully automatic owner browser smoke; tokens stay in memory."""
import hashlib
import json
import sys
from pathlib import Path


def validate_receipt(path):
    from scripts.study_operator.i7_browser import PIN
    source=Path(__file__).with_name('i7_browser.py')
    value=json.loads(Path(path).read_bytes())
    pin=json.loads(PIN.read_bytes())
    platform='win32' if sys.platform=='win32' else 'linux'
    if (value.get('status')!='BROWSER_EXPORT_VERIFIED' or value.get('local_validation') is not True
            or value['automator']['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest()
            or value['automator']['pin_sha256']!=hashlib.sha256(PIN.read_bytes()).hexdigest()
            or value['automator']['browser_sha256']!=pin['platforms'][platform]['browser_sha256']):
        raise ValueError('CURRENT_LOCAL_BROWSER_VALIDATION_REQUIRED')
    return value


def smoke_participant(config):
    # Same rule the host seals into deployment.json for the app (CLOUDRAG_SMOKE_PARTICIPANT).
    from scripts.study_operator.deployment import smoke_participant as checked
    return checked(config)


def run(operator, *, user_agent=None, ip=None):
    from scripts.study_operator.i7_browser import run as browse
    from scripts.study_operator.i7_workflows import wait_ready
    from src.ui.components.study_protocol import verify_draw
    validate_receipt(operator.root/'browser-validation.json')
    code=smoke_participant(operator.config)
    operator.stop()
    try:
        operator.start('smoke')  # a timed-out 'instances start' may still boot the VM
        wait_ready(operator)
        identity={'boot_id':operator.state['ready']['boot_id'],'image_id':operator.config['image_id'],
                  'source_commit':operator.config['commit']}
        token=operator.invite(code,cell=1,profile='without_experience')
        def observe():
            return operator.bridge({'operation':'i7-smoke-observe','identity':identity},private=True)
        result=browse('https://'+operator.config['hostname'],token,
                      verify_draw(operator.config['local_reviewed_config']),identity,observe,
                      user_agent=user_agent,ip=ip,participant=code,
                      failure_dir=operator.root/'private'/'browser-failures')
        return operator.receipt('smoke',dict(result,boot_id=identity['boot_id']))
    except BaseException:
        # Success keeps the VM RUNNING: restore and withdraw-purge need maintenance on it.
        operator.stop()
        raise
