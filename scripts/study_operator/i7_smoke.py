"""Start a synthetic browser smoke and verify its actual export and GCS backup."""
import time
import webbrowser


def run(operator, *, open_browser=webbrowser.open):
    from scripts.study_operator.i7_workflows import wait_ready
    operator.stop()
    operator.start('smoke')
    wait_ready(operator)
    token = operator.invite('P999',cell=1,profile='without_experience')
    # Token never enters receipts, process argv, a script or a log file.
    print('Invitación sintética, solo consola privada: '+token,flush=True)
    if not open_browser('https://'+operator.config['hostname']+'/'):
        raise ValueError('SYNTHETIC_BROWSER_NOT_OPENED')
    deadline = time.monotonic()+3600
    while time.monotonic() < deadline:
        result = operator.bridge({'operation':'i7-smoke-status'},private=True)
        if result.get('status') == 'SMOKE_COMPLETE_VERIFIED':
            return operator.receipt('smoke',result)
        operator.sleep(5)
    raise TimeoutError('SYNTHETIC_BROWSER_SMOKE_INCOMPLETE')
