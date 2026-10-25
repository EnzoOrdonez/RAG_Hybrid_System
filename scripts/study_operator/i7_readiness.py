"""Source-bound local verification and explicitly owned remaining preconditions."""
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.study_operator.stimulus_host_evidence import sha


def source_sha256():
    root=Path(__file__).resolve().parents[2]
    names=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard',
        '--','src','scripts','tests','config','requirements-i7-browser.txt','requirements-lock.txt','Dockerfile','pyproject.toml'],text=True).splitlines()
    return sha({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(set(names))})


def evaluate(evidence=None, *, root=None,november_authorized=False):
    blockers=[]
    try:
        if not evidence:
            raise ValueError('LOCAL_VERIFICATION_RECEIPT_REQUIRED')
        value=json.loads(Path(evidence).read_bytes())
        if (value['source_sha256']!=source_sha256() or value['freeze']!={'modules':19,'documents':3}
                or any(value['tests'][platform]['exit_code']!=0 for platform in ('windows','linux','i7'))
                or value.get('four_posix_tests_passed') is not True):
            raise ValueError('LOCAL_VERIFICATION_MISSING_OR_SOURCE_CHANGED')
        for check in value['tests'].values():
            if hashlib.sha256(Path(check['log_path']).read_bytes()).hexdigest()!=check['log_sha256']:
                raise ValueError('LOCAL_VERIFICATION_LOG_CHANGED')
    except (OSError,ValueError,KeyError) as error:
        blockers.append({'owner':'code','code':'LOCAL_VERIFICATION_RECEIPT_REQUIRED_OR_CHANGED',
                         'reason':type(error).__name__})
    if root and (Path(root)/'published-annex.json').is_file():
        try:
            from scripts.study_operator.i7_admission import check_files
            check_files(root)
        except (OSError,ValueError,KeyError):
            blockers.append({'owner':'Enzo','code':'PUBLISHED_EFFECTIVE_ANNEX_REQUIRED'})
    else:
        blockers.append({'owner':'Enzo','code':'PUBLISHED_EFFECTIVE_ANNEX_REQUIRED'})
    if not november_authorized:
        blockers.append({'owner':'Enzo','code':'NOVEMBER_PHASE_B_AUTHORIZATION_REQUIRED'})
    return {'status':'LOCAL_RUNBOOK_PARTIAL' if blockers else 'LOCAL_RUNBOOK_READY',
            'blockers':blockers,'cloud_measured':False}
