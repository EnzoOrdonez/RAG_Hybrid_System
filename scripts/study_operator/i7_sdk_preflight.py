"""Local SDK capabilities only; no authentication or provider operations."""
import json
import os
import shutil
import subprocess


def verify_beta(sdk=None, *, invoke=subprocess.run):
    sdk = sdk or shutil.which('gcloud.cmd') or shutil.which('gcloud')
    try:
        if not sdk:
            raise ValueError('missing SDK')
        result = invoke([str(sdk), 'components', 'list', '--only-local-state', '--format=json', '--quiet'],
            capture_output=True, timeout=15, env=dict(os.environ,
                CLOUDSDK_CORE_DISABLE_FILE_LOGGING='1', CLOUDSDK_CORE_DISABLE_PROMPTS='1',
                CLOUDSDK_CORE_DISABLE_USAGE_REPORTING='true',
                CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK='true', PYTHONUTF8='1'))
        if result.returncode:
            raise ValueError('component query failed')
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError('component rows')
        beta = [row for row in rows if row.get('id') == 'beta']
        if (len(beta) != 1 or not isinstance(beta[0].get('current_version_string'), str)
                or not beta[0]['current_version_string'].strip()):
            raise ValueError('beta not installed')
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        raise ValueError('GCLOUD_BETA_REQUIRED') from error
    return {'status': 'GCLOUD_BETA_INSTALLED', 'beta_version': beta[0]['current_version_string']}
