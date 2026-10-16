"""Validated, fully automatic owner browser smoke; tokens stay in memory."""
import hashlib
import json
from pathlib import Path


def validate_receipt(path):
    from scripts.study_operator.i7_browser import PIN
    source=Path(__file__).with_name('i7_browser.py')
    value=json.loads(Path(path).read_bytes())
    if (value.get('status')!='BROWSER_EXPORT_VERIFIED' or value.get('local_validation') is not True
            or value['automator']['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest()
            or value['automator']['pin_sha256']!=hashlib.sha256(PIN.read_bytes()).hexdigest()):
        raise ValueError('CURRENT_LOCAL_BROWSER_VALIDATION_REQUIRED')
    return value


def run(operator, *, user_agent=None, ip=None):
    from scripts.study_operator.i7_browser import run as browse
    from scripts.study_operator.i7_workflows import wait_ready
    from src.ui.components.study_protocol import verify_draw
    validate_receipt(operator.root/'browser-validation.json')
    operator.stop()
    operator.start('smoke')
    wait_ready(operator)
    identity={'boot_id':operator.state['ready']['boot_id'],'image_id':operator.config['image_id'],
              'source_commit':operator.config['commit']}
    token=operator.invite('P999',cell=1,profile='without_experience')
    def observe():
        return operator.bridge({'operation':'i7-smoke-observe','identity':identity},private=True)
    result=browse('https://'+operator.config['hostname'],token,
                  verify_draw(operator.config['local_reviewed_config']),identity,observe,
                  user_agent=user_agent,ip=ip)
    return operator.receipt('smoke',dict(result,boot_id=identity['boot_id']))
