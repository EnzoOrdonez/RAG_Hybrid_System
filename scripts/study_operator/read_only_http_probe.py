"""Compare SDK POST body sizes using only the project's read-only IAM method."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.run_control import require_limited

URL = 'https://cloudresourcemanager.googleapis.com/v1/projects/pure-loop-474323-a8:getIamPolicy'

# The installed SDK's own Python and transport are used; no dependency changes.
# Unknown synthetic fields are identical in both bodies except for their length.
# A server rejection is useful transport evidence, not a study-system failure.
NATIVE = r'''
import hashlib,json,os,sys,time
from pathlib import Path
settings=json.load(sys.stdin)
sdk=Path(settings['sdk'])
sys.path.insert(0,str(sdk/'lib/third_party'))
sys.path.insert(0,str(sdk/'lib'))
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as sdk_requests
properties.VALUES.core.log_http.Set(False)
if settings.get('dry_run'):
 print(json.dumps(dict(status='SDK_TRANSPORT_IMPORT_PASS',python=sys.version.split()[0],transport_source_sha256=hashlib.sha256(Path(sdk_requests.__file__).read_bytes()).hexdigest(),no_network=True)))
 sys.exit(0)
url=settings['url']
assert url=='https://cloudresourcemanager.googleapis.com/v1/projects/pure-loop-474323-a8:getIamPolicy'
rows=[]
for repetition in (1,2):
 for size in (0,settings['large_bytes']):
  body=json.dumps(dict(options=dict(requestedPolicyVersion=3),_transport_probe='X'*size)).encode()
  session=sdk_requests.GetSession(timeout=20)
  began=time.monotonic()
  try:
   response=session.request('POST',url,data=body,headers={'Authorization':'Bearer '+settings['token'],'Content-Type':'application/json'},allow_redirects=False)
   result=dict(http_status=response.status_code,server_responded=True)
   response.close()
  except Exception as error:
   result=dict(http_status=None,server_responded=False,exception_type=type(error).__name__)
  finally:
   session.close()
  rows.append(dict(result,padding_bytes=size,body_bytes=len(body),repetition=repetition,elapsed_s=time.monotonic()-began))
module=Path(sdk_requests.__file__)
print(json.dumps(dict(status='READ_ONLY_SDK_SIZE_PROBE',rows=rows,python=sys.version.split()[0],transport_source_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),no_iam_or_compute_mutations=True,token_and_policy_not_persisted=True)))
'''


def probe(sdk, token, large_bytes, *, invoke=subprocess.run, dry_run=False):
    sdk = Path(sdk).resolve()
    native = sdk/'platform/bundledpython/python.exe'
    if (not native.is_file() or not (sdk/'lib/googlecloudsdk/core/requests.py').is_file()
            or type(large_bytes) is not int or not 100_000 <= large_bytes <= 200_000):
        raise ValueError('Installed SDK transport and bounded synthetic size required')
    environment = dict(os.environ, CLOUDSDK_CORE_DISABLE_FILE_LOGGING='1', CLOUDSDK_CORE_DISABLE_PROMPTS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    result = invoke([str(native), '-B', '-c', NATIVE], input=json.dumps(dict(sdk=str(sdk), token=token,
        url=URL, large_bytes=large_bytes, dry_run=dry_run)).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, env=environment)
    if result.returncode:
        # Do not forward SDK traceback, proxy configuration or policy contents.
        raise ValueError('READ_ONLY_SDK_PROBE_EXECUTION_FAILED')
    value = json.loads(result.stdout)
    if dry_run:
        if value.get('status') != 'SDK_TRANSPORT_IMPORT_PASS' or value.get('no_network') is not True:
            raise ValueError('SDK_PROBE_RECEIPT_INVALID')
        return value
    if (value['status'] != 'READ_ONLY_SDK_SIZE_PROBE' or len(value['rows']) != 4
            or value['no_iam_or_compute_mutations'] is not True or value['token_and_policy_not_persisted'] is not True
            or [(row['repetition'], row['padding_bytes']) for row in value['rows']]
               != [(r, n) for r in (1, 2) for n in (0, large_bytes)]):
        raise ValueError('SDK_PROBE_RECEIPT_INVALID')
    return value


def validate_paths(package, installation, startup_file, output):
    root, owner = Path(package).resolve(), Path(installation).resolve()
    config = json.loads((owner/'installation.json').read_bytes())
    state = json.loads((owner/'active.json').read_bytes())
    script, destination = Path(startup_file).resolve(), Path(output).resolve()
    if (Path(config['audit_run']).resolve() != root or destination.parent != root
            or destination.exists() or script.name != 'primary-startup.sh'
            or not script.is_relative_to(owner/'runs')
            or hashlib.sha256(script.read_bytes()).hexdigest()
               != state['primary_creation_intent']['startup_sha256']):
        raise ValueError('Owned startup intent and unused package output required')
    return config, script, destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--installation', required=True)
    parser.add_argument('--startup-file', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    require_limited()
    config, script, output = validate_paths(args.package, args.installation, args.startup_file, args.output)
    cloud = Cloud(config['gcloud'], config['project'], Path(args.package)/'sdk-readonly-probe-api')
    token = '' if args.dry_run else cloud.owner_token()
    result = probe(Path(config['gcloud']).parents[1], token, script.stat().st_size, dry_run=args.dry_run)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
