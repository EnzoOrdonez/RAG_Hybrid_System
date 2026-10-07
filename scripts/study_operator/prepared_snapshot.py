"""Prepare a retained regional snapshot only after all mutable study copies are empty."""
import hashlib

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.service_gateway import save_state


def prepare(operator, certificate_sha256):
    if not operator.state.get('snapshot_empty_verified'):
        raise OperatorError('La instantánea exige STOP y todos los periodos vacíos verificados. Purga y repite tls-prepare.')
    selected = operator.selected()
    observed = operator.observed()
    if observed['status'] != 'TERMINATED' or not operator.state.get('snapshot_empty_verified'):
        raise OperatorError('La instantánea exige STOP y todos los periodos vacíos verificados. Purga y repite tls-prepare.')
    key = hashlib.sha256((operator.config['image_id']+'\0'+operator.config['hostname']+'\0'+certificate_sha256).encode()).hexdigest()
    name = 'cloudrag-i5-ready-'+key[:24]
    source = observed['disks'][0]['source']
    disk_name = source.rsplit('/',1)[-1]
    disk = operator.cloud.command(['compute','disks','describe',disk_name,'--zone='+selected['zone']])
    if disk.get('selfLink') != source or not str(disk.get('id','')).isdigit():
        raise OperatorError('Disco de contingencia distinto del observado. Conserva los recursos y revisa su identidad.')
    existing = operator.cloud.command(['compute','snapshots','list','--filter=name='+name])
    prior = operator.config.get('prepared_snapshot',{})
    intent = operator.state.get('snapshot_creation_intent',{})
    if not existing:
        # Public rate is for compressed bytes;100GiB for7days is a conservative creation reserve.
        operator.reserve_cost('snapshot-'+key[:12],100*.000068493*24*7)
        intent = dict(name=name,source_disk_id=str(disk['id']),image_id=operator.config['image_id'],
                      certificate_sha256=certificate_sha256,hostname=operator.config['hostname'],
                      ownership_marker='CloudRAG-I5-ready-'+key,requested_utc=operator.now().isoformat())
        operator.state['snapshot_creation_intent'] = intent
        operator.persist()
        operator.cloud.command(['compute','snapshots','create',name,'--source-disk='+disk_name,
            '--source-disk-zone='+selected['zone'],'--storage-location=us-central1',
            '--description='+intent['ownership_marker']],timeout=600)
    snapshot = operator.cloud.command(['compute','snapshots','describe',name])
    owned = ((prior.get('name') == name and prior.get('id') == str(snapshot.get('id')))
             or (intent.get('name') == name and snapshot.get('description') == intent.get('ownership_marker')))
    if (not owned or snapshot.get('status') != 'READY' or snapshot.get('storageLocations') != ['us-central1']
            or str(snapshot.get('sourceDiskId')) != str(disk['id'])):
        raise OperatorError('Instantánea no READY, ajena o con otro disco. Conserva el intento y verifica su recibo; no se recrea a ciegas.')
    result = dict(name=name,id=str(snapshot['id']),source_disk_id=str(disk['id']),
        source_vm_id=selected['id'],image_id=operator.config['image_id'],
        hostname=operator.config['hostname'],certificate_sha256=certificate_sha256,
        storage_bytes=int(snapshot['storageBytes']),created_utc=snapshot['creationTimestamp'],
        ownership_marker=snapshot.get('description'),
        idle_usd_day=int(snapshot['storageBytes'])/2**30*.000068493*24,
        session_data='ALL_I4_PERIODS_EMPTY_VERIFIED',retain_at_closure=True)
    operator.config['prepared_snapshot'] = result
    resources = operator.state.setdefault('snapshots',[])
    if not any(row['id'] == result['id'] for row in resources):
        resources.append(result)
    operator.state.pop('snapshot_creation_intent',None)
    operator.state.pop('snapshot_empty_verified',None)
    operator.persist()
    save_state(operator.cloud.root/'prepared-snapshot.json',result)
    return result
