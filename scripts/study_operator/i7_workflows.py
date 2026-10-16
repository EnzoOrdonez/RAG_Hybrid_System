"""Independent owner workflows; receipts survive the agent's session."""
import getpass
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

from scripts.study_operator.i7 import tls_three
from scripts.study_operator.i7_preparation import file_sha
from scripts.study_operator.service_gateway import save_state


def archive_tools(source, destination, commit):
    source, destination = Path(source), Path(destination)/commit
    files = {p.relative_to(source).as_posix():p for p in sorted((source/'scripts').rglob('*.py'))}
    hashes = {name:file_sha(path) for name,path in files.items()}
    if destination.exists():
        if json.loads((destination/'manifest.json').read_bytes()) != hashes:
            raise ValueError('ARCHIVED_TOOL_VERSION_CHANGED')
        if any(file_sha(destination/name) != digest for name,digest in hashes.items()):
            raise ValueError('ARCHIVED_TOOL_BYTES_CHANGED')
    else:
        destination.mkdir(parents=True,exist_ok=False)
        for relative,path in files.items():
            target = destination/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
        save_state(destination/'manifest.json',hashes)
    return {'commit':commit,'files':len(files),'manifest_sha256':file_sha(destination/'manifest.json')}


def verify_tool_source(operator):
    source = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head != operator.config['commit']:
        raise ValueError('OPERATOR_SOURCE_COMMIT_DIFFERS')
    modified = subprocess.run(['git','-C',str(source),'status','--porcelain','--untracked-files=all',
                               '--','scripts','src','config'],
                               check=False,capture_output=True,timeout=60)
    if modified.returncode or modified.stdout.strip():
        raise ValueError('OPERATOR_SOURCE_BYTES_MODIFIED')
    return archive_tools(source,operator.root/'tool-versions',head)


def check_tls_sequence(attempts):
    rows = sorted(attempts,key=lambda r:r['start_requested_utc'])
    if len(rows) < 3 or any(r.get('status') != 'READY' for r in rows[-3:]):
        raise ValueError('THREE_CONSECUTIVE_SUCCESSFUL_BOOTS_REQUIRED')
    return tls_three(rows)


def wait_ready(operator):
    from scripts.study_operator.policy import ReadyPending
    began = time.monotonic()
    while time.monotonic()-began < 900:
        try:
            return operator.preflight()
        except ReadyPending:
            operator.sleep(5)
    raise TimeoutError('OWNER_READY_900S')


def series(operator):
    """Only complete cold boots permit continuation; terminal attempts cannot replay."""
    from scripts.study_operator.stimulus_evidence import analyze_directory
    journal = operator.root/'attempts'
    journal.mkdir(exist_ok=True)
    try:
        for index in range(1,13):
            record = journal/f'stimulus-{index:02d}.json'
            completion_path = record.with_suffix('.complete.json')
            if record.exists():
                existing = json.loads(completion_path.read_bytes()) if completion_path.exists() else {}
                if existing.get('status') != 'COMPLETE':
                    raise ValueError('TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW')
                continue
            with record.open('x',encoding='utf-8') as stream:
                json.dump({'status':'INTENT','boot_index':index},stream)
            operator.start()
            wait_ready(operator)
            result = operator.bridge({'operation':'stimulus-start','boot_index':index},private=True)
            if result.get('status') != 'STIMULUS_STARTED_NOT_ACCEPTANCE':
                raise ValueError('STIMULUS_START_UNCONFIRMED')
            deadline = time.monotonic()+7200
            while time.monotonic() < deadline:
                state = operator.bridge({'operation':'stimulus-status'},private=True)['result']['status']
                if state == 'BOOT_COMPLETE_UNANALYZED':
                    break
                if state == 'INCOMPLETE_TERMINAL':
                    raise ValueError('STIMULUS_TERMINAL')
                operator.sleep(2)
            else:
                raise TimeoutError('STIMULUS_OWNER_DEADLINE')
            receipt = operator.collect('stimulus')
            operator.stop()
            # Keep the initial intent and an independent completion receipt.
            completion = {'status':'COMPLETE','boot_index':index,'download':receipt}
            with completion_path.open('x',encoding='utf-8') as stream:
                json.dump(completion,stream,sort_keys=True)
        output = operator.root/'stimulus-acceptance.json'
        if output.exists():
            raise ValueError('ACCEPTANCE_ALREADY_EXISTS')
        from src.ui.components.study_protocol import verify_draw
        result = analyze_directory(operator.root/'private'/'stimulus'/'coded-P999',
                                   verify_draw(operator.config['local_reviewed_config']),output)
        return {'status':'SERIES_COMPLETE','acceptance':result}
    finally:
        operator.stop()
        operator.guard()


def privacy(operator):
    from scripts.study_operator.gcs import Storage
    from scripts.study_operator.i7_privacy import (
        bucket_sources,
        inject,
        scan,
        serial_sources,
    )
    ip = getpass.getpass('IP propia, solo en memoria: ')
    user_agent = 'CloudRAG-I7-'+secrets.token_hex(24)
    inject(operator.config['hostname'],ip,user_agent)
    from scripts.study_operator.i7_smoke import run as browser_smoke
    operator.delete_sessions('purge-study',dry_run=True)
    operator.delete_sessions('purge-study',dry_run=False)
    journey=browser_smoke(operator,user_agent=user_agent,ip=ip)
    if not all(journey['canary_transport'].values()):
        raise ValueError('FULL_BROWSER_WEBSOCKET_CANARY_REQUIRED')
    guest = operator.bridge({'operation':'i7-privacy','ip':ip,'user_agent':user_agent},private=True,timeout=7200)
    storage = Storage(operator.config['sessions_bucket'],operator.cloud.owner_token,
        creation_anchor=operator.config.get('sessions_bucket_creation'))
    def serial():
        yield from serial_sources(operator.cloud,operator.selected())
    owner = scan({'serial':serial,'bucket':lambda:bucket_sources(storage)},
                 (ip.encode(),user_agent.encode()))
    domains = {**{k:v for k,v in guest['domains'].items() if k not in ('serial','bucket')},
               **{k:owner['domains'][k] for k in ('serial','bucket')}}
    total = sum(d['hits'] for d in domains.values())
    status = 'INCOMPLETE' if any(not d['complete'] for d in domains.values()) else 'FAIL' if total else 'PASS'
    return operator.receipt('privacy',{'status':status,'hits':total,'domains':domains,'exclusions':[],
        'journey':{'session_id':journey['session_id'],'identity':journey['identity'],
                   'export_sha256':journey['export_sha256'],'transport':journey['canary_transport']}})


def schedule(operator, *, invoke=subprocess.run):
    if sys.platform != 'win32':
        raise ValueError('WINDOWS_SCHEDULER_REQUIRED')
    source = Path(__file__).resolve().parents[2]
    version = archive_tools(source,operator.root/'tool-versions',operator.config['commit'])
    def quote(value):
        return "'"+str(value).replace("'","''")+"'"
    args = subprocess.list2cmdline(['-m','scripts.study_operator.i7','--root',str(operator.root),
        '--sdk',str(operator.cloud.sdk),'--live','series'])
    script = operator.root/'scheduled-series.ps1'
    if script.exists():
        raise ValueError('SCHEDULED_VERSION_ALREADY_EXISTS')
    task = 'CloudRAG-I7-'+operator.config['period_id']
    content = ("$ErrorActionPreference = 'Stop'\n"
        +"$action = New-ScheduledTaskAction -Execute "+quote(sys.executable)+" -Argument "+quote(args)
        +" -WorkingDirectory "+quote(source)+"\n"
        +"$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)\n"
        +"$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -StartWhenAvailable\n"
        +"Register-ScheduledTask -TaskName "+quote(task)+" -Action $action -Trigger $trigger -Settings $settings -RunLevel Limited | Out-Null\n")
    with script.open('x',encoding='utf-8') as stream:
        stream.write(content)
    result = invoke(['powershell','-NoProfile','-NonInteractive','-File',str(script)],
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
    if result.returncode:
        raise ValueError('TASK_REGISTRATION_FAILED')
    return operator.receipt('task',{'status':'SERIES_SCHEDULED','name':task,
        'script_sha256':file_sha(script),'tool_version':version})


def dispatch(operator, operation):
    if operation == 'series':
        return series(operator)
    if operation == 'schedule':
        return schedule(operator)
    if operation == 'privacy':
        return privacy(operator)
    if operation == 'close':
        stopped = operator.stop()
        deleted = operator.guard() if not operator.state['reservation'].get('deleted_verified') else {
            'status':'PRIOR_RESERVATION_CLOSURE_RECORDED'}
        from scripts.study_operator.i7_cpu_pipeline import cleanup_cpu
        cleaned = cleanup_cpu(operator)
        return {'status':'CLOSED','vm':stopped,'reservation':deleted,'cpu_cleanup':cleaned}
    raise ValueError('UNKNOWN_I7_WORKFLOW')
