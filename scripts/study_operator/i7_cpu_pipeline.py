"""November CPU qualification and specific-reservation restore orchestration."""
import base64
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from scripts.study_operator.i7_preparation import file_sha
from scripts.study_operator.service_gateway import save_state


def build(operator, spec):
    from scripts.study_operator.i7_streamlit_preflight import verify_before_cpu, verify_recipe
    preflight = verify_before_cpu()
    operator.receipt('streamlit-prebuild',preflight)
    from scripts.study_operator.build_context import prepare
    from scripts.study_operator.cloud_client import Cloud
    from scripts.study_operator.cpu_build import Build
    from scripts.study_operator.cpu_restoration import Controller, verified_closure
    from scripts.study_operator.pricing import disk_quote_archive, quote_archive
    root = operator.root/'cpu'
    original = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
    remote = subprocess.check_output(['git','-C',str(original),'ls-remote','origin',
                                      'refs/heads/fix/interview-readiness'],text=True).split()[0]
    if head != operator.config['commit'] or remote != head:
        raise ValueError('PUBLISHED_I7_SOURCE_REQUIRED')
    if root.exists():
        raise ValueError('CPU_ATTEMPT_EXISTS_RECONCILE_WITHOUT_REPLAY')
    reference = Path(spec['inherited_package'])
    verified_closure(reference)
    prices = Path(spec['price_archive'])
    quote = quote_archive(prices,'e2-standard-2','us-central1')
    gpu_quote = quote_archive(prices,'g2-standard-4','us-central1')
    pd_quote = disk_quote_archive(prices,'us-central1')
    if (gpu_quote['catalog_receipt_sha256'] != operator.config['rates']['price_receipt_sha256']
            or float(gpu_quote['usd_per_hour']) != operator.config['rates']['compute_usd_h']):
        raise ValueError('I7_PRICE_RECEIPT_OR_RATE_CHANGED')
    # Upper covers 6 CPU hours and two 180 GiB owned PDs through 72h.
    exposure = 6*float(quote['usd_per_hour'])+360*float(pd_quote['estimated_usd_gib_h'])*72
    operator.admit_paid(exposure)
    root.mkdir()
    for name in ('iteration4-recovered-seal02.json','legacy-seal-retry-pin.json','rag_freeze_baseline.json'):
        shutil.copyfile(reference/name,root/name)
    shutil.copytree(prices,root/'official-compute-skus')
    save_state(root/'STATE.json',{'status':'ACTIVE','resources':[],'cloud_cutoff_usd':20,
        'independent_closure_verified':True,
        'closure_reserved_utc':(operator.now()+timedelta(hours=72)).isoformat(),
        'cost':{'estimated_spend_usd':operator.state.get('spent_usd',0),
                'reserved_retention_and_closure_usd':3}})
    operator.state['open_exposure_usd'] = exposure
    operator.persist()
    source = json.loads(Path(spec['source_installation']).read_bytes())
    # Old host code is never deployed. Frozen image/artifacts/weights are checked;
    # the new host source is extracted and qualified from the newly built image.
    source['host_infrastructure_sha256'] = {}
    cloud = Cloud(operator.cloud.sdk,operator.cloud.project,root/'runs')
    controller = Controller(root,cloud,label='i7cpu01')
    inventory = {'resources':{'snapshots':cloud.command(['compute','snapshots','list'])}}
    source_snapshots=[s for s in inventory['resources']['snapshots']
                      if s['name']=='cloudrag-i6-cierre-disco-20261016']
    if len(source_snapshots)!=1 or source_snapshots[0].get('storageLocations')!=['us-central1']:
        raise ValueError('COST_SCOPE_REQUIRES_CENTRAL1_SOURCE_SNAPSHOT')
    resources = controller.prepare(inventory,source,snapshot_name='cloudrag-i6-cierre-disco-20261016')
    restored = controller.probe_and_stop(resources,source)
    if restored['status'] != 'CPU_RESTORATION_VERIFIED':
        raise ValueError('CPU_SOURCE_RESTORATION_NOT_VERIFIED')
    # Reserve more build scratch, only on this owned CPU disk.
    cloud.command(['compute','disks','resize',resources['disk']['name'],'--zone=us-central1-a','--size=180GB'])
    repo = root/'source'
    subprocess.run(['git','clone','--no-hardlinks','--single-branch','--branch','fix/interview-readiness',
                    str(original),str(repo)],check=True,capture_output=True,timeout=600)
    def run(command):
        result = subprocess.run(command['argv'],capture_output=True,timeout=command['timeout'],check=False)
        (root/(command['name']+'.stdout')).write_bytes(result.stdout)
        (root/(command['name']+'.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise ValueError('CLEAN_BUILD_CONTEXT_COMMAND_FAILED')
    context = prepare(repo,spec['vendor'],root,'finali7',run=run,python=sys.executable)
    verify_recipe(root/'build-context-finali7/repository')
    if context['requirements_lock_sha256'] != preflight['files']['requirements-lock.txt']:
        raise ValueError('I7_BUILD_CONTEXT_LOCK_CHANGED')
    builder = Build(root,cloud,cpu_label='i7cpu01')
    builder.launch('finali7',context,Path(__file__).with_name('linux_build.py').read_bytes())
    deadline = time.monotonic()+7200
    while time.monotonic() < deadline:
        proof = builder.collect('finali7')
        if proof['status'] != 'BUILD_NOT_TERMINAL_OR_OWNER_RECOVERY_REQUIRED':
            break
        operator.sleep(15)
    else:
        raise TimeoutError('CPU_BUILD_NATIVE_LIMIT')
    if proof['status'] != 'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED':
        raise ValueError('CPU_BUILD_NOT_QUALIFIED')
    operator.config.update(image_id=proof['image_id'],host_code='/srv/cloudrag/iteration5/code-finali7')
    operator.persist()
    verify_image_or_stop_cpu(operator)
    receipt = clean_snapshot(operator,cloud,resources,proof)
    operator.state['spent_usd'] = operator.state.get('spent_usd',0)+exposure
    operator.state['cpu_cost_basis']='CONSERVATIVE_EXPOSURE_NOT_INVOICE'
    operator.state['open_exposure_usd'] = 0
    operator.persist()
    return operator.receipt('cpu-build',receipt)


def clean_snapshot(operator, cloud, resources, build_proof):
    """An offline secondary clone; originals and the CPU source remain untouched."""
    vm, disk = resources['vm'],resources['disk']
    marker = operator.config['resource_marker']
    suffix = operator.config['period_id'][:12]
    stage, target, final = ['cloudrag-i7-'+name+'-'+suffix for name in ('stage','service','clean')]
    zone = 'us-central1-a'
    from scripts.study_operator.i7_reconcile import observed,record_intent
    for name in (stage,final):
        if cloud.command(['compute','snapshots','list','--filter=name='+name]):
            raise ValueError('I7_SNAPSHOT_INTENT_EXISTS')
    save_state(operator.root/'clean-disk-intent.json',{'cpu_id':str(vm['id']),'cpu_disk_id':str(disk['id']),
               'stage':stage,'target':target,'snapshot':final,'marker':marker})
    stage_intent=record_intent(operator,'snapshots',stage)
    cloud.command(['compute','snapshots','create',stage,'--source-disk='+disk['name'],
        '--source-disk-zone='+zone,'--storage-location=us-central1','--description='+marker],timeout=600)
    stage_receipt=cloud.command(['compute','snapshots','describe',stage])
    observed(operator,stage_intent,stage_receipt)
    disk_intent=record_intent(operator,'disks',target,zone,source_snapshot_id=str(stage_receipt['id']))
    cloud.command(['compute','disks','create',target,'--source-snapshot='+stage,'--zone='+zone,
        '--type=pd-balanced','--description='+marker],timeout=600)
    target_receipt = cloud.command(['compute','disks','describe',target,'--zone='+zone])
    observed(operator,disk_intent,target_receipt)
    stage_receipt = cloud.command(['compute','snapshots','describe',stage])
    if (stage_receipt.get('storageLocations')!=['us-central1'] or target_receipt['description'] != marker or
            str(target_receipt['sourceSnapshotId']) != str(stage_receipt['id'])):
        raise ValueError('OWN_SERVICE_CLONE_IDENTITY_CHANGED')
    cloud.command(['compute','instances','attach-disk',vm['name'],'--zone='+zone,
                   '--disk='+target,'--device-name='+target,'--mode=rw'])
    cloud.command(['compute','instances','start',vm['name'],'--zone='+zone],timeout=300)
    script = operator.config['host_code']+'/scripts/study_operator/i7_service_disk.py'
    def rpc(request):
        payload = cloud.command(['compute','ssh',vm['name'],'--zone='+zone,'--tunnel-through-iap',
            '--ssh-key-expire-after=10m','--command=sudo -n python3 -B '+shlex.quote(script)],
            input_data=json.dumps(request).encode(),json_output=False,private_output=True,timeout=1800)
        return json.loads(payload)
    job = {'cpu_id':str(vm['id']),'disk_name':target,
           'legacy_archive':'/srv/cloudrag/iteration7-legacy-'+suffix+'.tar'}
    try:
        rpc(dict(job,operation='mount'))
        archived = rpc(dict(job,operation='archive'))
        data = base64.b64decode(archived.pop('content_base64'),validate=True)
        if hashlib.sha256(data).hexdigest() != archived['sha256'] or archived['files'] != 9:
            raise ValueError('LEGACY_ARCHIVE_DOWNLOAD_DIFFERS')
        private = operator.root/'private'/'legacy'
        private.mkdir(parents=True,exist_ok=True)
        archive = private/'i3-nine-originals.tar'
        with archive.open('xb') as stream:
            stream.write(data)
        if file_sha(archive) != archived['sha256']:
            raise ValueError('LEGACY_LOCAL_READBACK_DIFFERS')
        images = rpc(dict(job,operation='export-images',images=[operator.config[k]
                          for k in ('image_id','ollama_image','caddy_image')]))
        prepared = rpc(dict(job,operation='sanitize',installation=operator.config,
            legacy_archive_ack_sha256=archived['sha256'],**images))
        rpc(dict(job,operation='unmount'))
        cloud.command(['compute','instances','stop',vm['name'],'--zone='+zone],timeout=600)
        stopped = cloud.command(['compute','instances','describe',vm['name'],'--zone='+zone])
        if str(stopped['id']) != str(vm['id']) or stopped['status'] != 'TERMINATED':
            raise ValueError('CPU_CLEANING_STOP_NOT_VERIFIED')
        final_intent=record_intent(operator,'snapshots',final)
        cloud.command(['compute','snapshots','create',final,'--source-disk='+target,
            '--source-disk-zone='+zone,'--storage-location=us-central1','--description='+marker],timeout=600)
        snapshot = cloud.command(['compute','snapshots','describe',final])
        observed(operator,final_intent,snapshot)
        if (snapshot['status'] != 'READY' or snapshot['description'] != marker
                or snapshot.get('storageLocations')!=['us-central1']):
            raise ValueError('CLEAN_SERVICE_SNAPSHOT_NOT_READY')
        operator.config.update(prepared['installation'])
        operator.state['qualified_service_snapshot'] = snapshot
        operator.state['legacy_archive_sha256']=archived['sha256']
        operator.state['cpu_resources'] = {'vm':vm,'source_disk':disk,'service_disk':target_receipt,
                                           'stage_snapshot':stage_receipt}
        operator.persist()
        return {'status':'I7_CPU_IMAGE_AND_CLEAN_DISK_PREPARED','build':build_proof,
                'snapshot_id':str(snapshot['id']),'legacy_archive':archived,'cloud_gpu_validated':False}
    finally:
        cloud.command(['compute','instances','stop',vm['name'],'--zone='+zone],timeout=600)


def verify_image_or_stop_cpu(operator):
    """An image rejection can only perform identity-checked CPU shutdown."""
    from scripts.study_operator.i7_streamlit_preflight import verify
    try:
        result = verify(operator.root)
    except ValueError as error:
        from scripts.study_operator.cpu_build import Build
        builder = Build(operator.root/'cpu',operator.cloud,cpu_label='i7cpu01')
        try:
            cpu = builder.stopped(builder.resource())
        except Exception as stop_error:
            operator.receipt('streamlit-admission',{'status':'STREAMLIT_BLOCKED_CPU_STOP_UNCONFIRMED',
                'reason':str(error),'stop_error':type(stop_error).__name__})
            raise ValueError('STREAMLIT_BLOCKED_CPU_STOP_UNCONFIRMED') from stop_error
        operator.receipt('streamlit-admission',{'status':'STREAMLIT_BLOCKED_CPU_STOPPED',
            'reason':str(error),'cpu_id':str(cpu['id']),'cpu_status':cpu['status']})
        raise
    operator.receipt('streamlit-admission',result)
    return result


def gpu_restore(operator):
    verify_image_or_stop_cpu(operator)
    from scripts.study_operator.startup import write_startup
    from scripts.study_operator.i7_reconcile import observed,record_intent
    if any(row['status']=='UNKNOWN' for row in operator.state.get('creation_intents',[])):
        raise ValueError('UNKNOWN_CREATION_RECONCILE_BEFORE_CAPACITY')
    snapshot = operator.state['qualified_service_snapshot']
    actual = operator.cloud.command(['compute','snapshots','describe',snapshot['name']])
    if (str(actual['id']) != str(snapshot['id']) or actual['status'] != 'READY'
            or actual.get('storageLocations')!=['us-central1']):
        raise ValueError('QUALIFIED_SERVICE_SNAPSHOT_CHANGED')
    # Capacity is held before restoring the GPU disk, never while another GPU is active.
    instances = operator.cloud.command(['compute','instances','list'])
    if any(v.get('guestAccelerators') and v['status'] != 'TERMINATED' for v in instances):
        raise ValueError('ANOTHER_GPU_ACTIVE')
    operator.reserve(12)
    reservation = operator.state['reservation']
    zone = operator.config['zone']
    name = 'cloudrag-i7-vm-'+operator.config['period_id'][:12]
    disk = name+'-boot'
    disk_intent=record_intent(operator,'disks',disk,zone,source_snapshot_id=str(snapshot['id']))
    operator.cloud.command(['compute','disks','create',disk,'--zone='+zone,'--type=pd-balanced',
        '--source-snapshot='+snapshot['name'],'--description='+operator.config['resource_marker']],timeout=600)
    restored_disk = operator.cloud.command(['compute','disks','describe',disk,'--zone='+zone])
    observed(operator,disk_intent,restored_disk)
    if (str(restored_disk['sourceSnapshotId']) != str(snapshot['id'])
            or restored_disk['description'] != operator.config['resource_marker']):
        raise ValueError('GPU_BOOT_DISK_SOURCE_CHANGED')
    config = dict(operator.config,instance_id='0',purpose='technical',native_stop_s=10800,
        start_requested_utc=operator.now().isoformat(),
        native_deadline_utc=(operator.now()+timedelta(hours=3)).isoformat())
    startup = operator.root/'first-gpu-startup.sh'
    write_startup(startup,config,discover_instance=True)
    vm_intent=record_intent(operator,'instances',name,zone,source_snapshot_id=str(snapshot['id']),role='primary_gpu')
    operator.cloud.command(['compute','instances','create',name,'--zone='+zone,'--machine-type=g2-standard-4',
        '--disk=name='+disk+',boot=yes,auto-delete=no','--deletion-protection','--max-run-duration=3h',
        '--instance-termination-action=STOP','--reservation-affinity=specific','--reservation='+reservation['name'],
        '--maintenance-policy=TERMINATE','--service-account='+config['service_account'],
        '--scopes=https://www.googleapis.com/auth/devstorage.read_write','--network='+config['network'],
        '--subnet='+config['subnet'],'--address='+config['static_ip'],
        '--tags=cloudrag-i3-managed',
        '--description='+config['resource_marker'],'--metadata-from-file=startup-script='+str(startup),
        '--metadata=enable-guest-attributes=TRUE'],timeout=600)
    vm = operator.cloud.command(['compute','instances','describe',name,'--zone='+zone])
    observed(operator,vm_intent,vm)
    if vm['description'] != config['resource_marker']:
        raise ValueError('NEW_GPU_OWNERSHIP_CHANGED')
    operator.config.update(primary_vm={'name':name,'id':str(vm['id']),'zone':zone},native_stop_s=10800)
    operator.state.update(purpose='technical',boot_started_utc=config['start_requested_utc'])
    operator.persist()
    return operator.receipt('gpu-restore',{'status':'GPU_RESTORED_NOT_READY','id':str(vm['id']),
                                          'source_snapshot_id':str(snapshot['id'])})


def cleanup_cpu(operator):
    """Only successfully qualified, explicitly recorded temporary CPU resources."""
    resources = operator.state.get('cpu_resources')
    if not resources:
        return {'status':'NO_QUALIFIED_CPU_CLEANUP_PLAN'}
    if operator.state.get('cpu_cleanup_verified'):
        return {'status':'PRIOR_CPU_CLEANUP_RECORDED'}
    archive=operator.root/'private'/'legacy'/'i3-nine-originals.tar'
    if (not archive.is_file() or file_sha(archive)!=operator.state.get('legacy_archive_sha256')):
        raise ValueError('PRIVATE_LEGACY_ARCHIVE_REQUIRED_BEFORE_CPU_CLEANUP')
    vm = resources['vm']
    zone = 'us-central1-a'
    current = operator.cloud.command(['compute','instances','describe',vm['name'],'--zone='+zone])
    if (str(current['id']) != str(vm['id']) or current['description'] != vm['description']
            or not vm['name'].startswith('cloudrag-i5-restore-i7cpu')):
        raise ValueError('CPU_CLEANUP_IDENTITY_CHANGED')
    if current['status'] != 'TERMINATED':
        raise ValueError('CPU_CLEANUP_REQUIRES_STOPPED_VM')
    operator.cloud.command(['compute','instances','update',vm['name'],'--zone='+zone,'--no-deletion-protection'])
    operator.cloud.command(['compute','instances','delete',vm['name'],'--zone='+zone,'--keep-disks=all'],timeout=300)
    for resource in (resources['source_disk'],resources['service_disk']):
        actual = operator.cloud.command(['compute','disks','describe',resource['name'],'--zone='+zone])
        if str(actual['id']) != str(resource['id']) or actual['description'] != resource['description']:
            raise ValueError('CPU_DISK_CLEANUP_IDENTITY_CHANGED')
        operator.cloud.command(['compute','disks','delete',resource['name'],'--zone='+zone],timeout=300)
    stage = resources['stage_snapshot']
    actual = operator.cloud.command(['compute','snapshots','describe',stage['name']])
    if str(actual['id']) != str(stage['id']) or actual['description'] != operator.config['resource_marker']:
        raise ValueError('STAGING_SNAPSHOT_CLEANUP_IDENTITY_CHANGED')
    operator.cloud.command(['compute','snapshots','delete',stage['name']],timeout=300)
    cpu_state = json.loads((operator.root/'cpu'/'STATE.json').read_bytes())
    for resource in cpu_state['resources']:
        if resource['type'] == 'firewall':
            actual = operator.cloud.command(['compute','firewall-rules','describe',resource['name']])
            if str(actual['id']) != str(resource['id']) or actual['description'] != resource['ownership_marker']:
                raise ValueError('TEMPORARY_IAP_RULE_IDENTITY_CHANGED')
            operator.cloud.command(['compute','firewall-rules','delete',resource['name']])
    lists = {kind:operator.cloud.command(['compute',kind,'list']) for kind in
             ('instances','disks','snapshots','firewall-rules')}
    removed_ids = {str(resources[k]['id']) for k in ('vm','source_disk','service_disk','stage_snapshot')}
    removed_ids.update(str(r['id']) for r in cpu_state['resources'] if r['type'] == 'firewall')
    if any(str(r['id']) in removed_ids for rows in lists.values() for r in rows):
        raise ValueError('CPU_RESOURCE_ABSENCE_NOT_VERIFIED')
    operator.state['cpu_cleanup_verified'] = True
    operator.persist()
    return operator.receipt('cpu-cleanup',{'status':'CPU_TEMPORARY_RESOURCES_ABSENT',
                            'retained_clean_snapshot_id':str(operator.state['qualified_service_snapshot']['id'])})
