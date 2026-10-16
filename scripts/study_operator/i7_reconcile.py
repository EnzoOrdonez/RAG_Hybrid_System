"""Read-only identity reconciliation of CPU, VM, disk and snapshot UNKNOWN effects."""
import json
import re


def record_intent(operator,kind,name,zone=None, *, source_snapshot_id=None,role=None):
    rows=operator.state.setdefault('creation_intents',[])
    if any(row['kind']==kind and row['name']==name for row in rows):
        raise ValueError('RESOURCE_INTENT_EXISTS_RECONCILE_WITHOUT_REPLAY')
    row={'kind':kind,'name':name,'zone':zone,'marker':operator.config['resource_marker'],
         'source_snapshot_id':source_snapshot_id,'role':role,'status':'UNKNOWN'}
    rows.append(row)
    operator.persist()
    return row


def observed(operator,intent,value):
    if value.get('description')!=intent['marker'] or value['name']!=intent['name']:
        raise ValueError('CREATED_RESOURCE_IDENTITY_DIFFERS')
    intent.update(status='IDENTITY_RESOLVED',id=str(value['id']))
    operator.persist()


def intents(operator):
    recorded=operator.state.get('creation_intents',[])
    rows=[r for r in recorded if r['status']=='UNKNOWN']
    known={(r['kind'],r['name']) for r in recorded if r['status']!='UNKNOWN'}
    cpu=operator.root/'cpu'/'STATE.json'
    if cpu.is_file():
        state=json.loads(cpu.read_bytes())
        kinds={'vm':'instances','disk':'disks','firewall':'firewall-rules','snapshot':'snapshots'}
        for row in state.get('resource_intents',[]):
            if any(r['type']==row['type'] and r['name']==row['name'] for r in state.get('resources',[])):
                continue
            rows.append(dict(kind=kinds[row['type']],name=row['name'],zone=row.get('zone'),
                marker=row['ownership_marker'],source_snapshot_id=row.get('source_snapshot_id'),status='UNKNOWN'))
    clean=operator.root/'clean-disk-intent.json'
    if clean.is_file():
        value=json.loads(clean.read_bytes())
        rows.extend({'kind':kind,'name':value[key],'zone':'us-central1-a' if kind=='disks' else None,
                     'marker':value['marker'],'status':'UNKNOWN'} for key,kind in
                    (('stage','snapshots'),('target','disks'),('snapshot','snapshots')) if (kind,value[key]) not in known)
    unique={}
    for row in rows:
        unique.setdefault((row['kind'],row['name']),row)
    return list(unique.values())


def source_matches(operator,intent,value):
    expected=intent.get('source_snapshot_id')
    if not expected:
        return True
    if intent['kind']=='disks':
        return str(value.get('sourceSnapshotId'))==str(expected)
    if intent['kind']=='instances':
        boot=[d for d in value.get('disks',[]) if d.get('boot')]
        if len(boot)!=1:
            return False
        disk=operator.cloud.command(['compute','disks','describe',boot[0]['source'].split('/')[-1],
                                      '--zone='+intent['zone']])
        return str(disk.get('sourceSnapshotId'))==str(expected) and disk.get('description')==intent['marker']
    return False


def reconcile(operator):
    outcomes=[]
    for intent in intents(operator):
        kind=intent['kind']
        name=intent['name']
        if kind not in ('instances','disks','snapshots','firewall-rules') or not re.fullmatch('cloudrag-[a-z0-9-]+',name):
            raise ValueError('MANAGED_RESOURCE_INTENT_REQUIRED')
        command=['gcloud','compute',kind,'describe',name,'--project=pure-loop-474323-a8','--format=json']
        if kind in ('instances','disks'):
            if intent.get('zone') not in ('us-central1-a','us-central1-b','us-central1-c'):
                raise ValueError('CENTRAL1_RECONCILIATION_REQUIRED')
            command.append('--zone='+intent['zone'])
        outcome={'kind':kind,'name':name,'status':'BLOCKED','next_command':command}
        try:
            listed=operator.cloud.command(['compute',kind,'list'])
            if not isinstance(listed,list):
                raise ValueError('SUCCESSFUL_RESOURCE_LIST_REQUIRED')
            found=[r for r in listed if r['name']==name and (kind not in ('instances','disks')
                       or r['zone'].split('/')[-1]==intent['zone'])]
            if (len(found)==1 and found[0].get('description')==intent['marker']
                    and (not intent.get('id') or str(found[0]['id'])==str(intent['id']))
                    and source_matches(operator,intent,found[0])):
                observed=found[0]
                if intent.get('role')=='primary_gpu':
                    from scripts.study_operator.cloud_client import checked_vm
                    checked_vm(observed,name=name,instance_id=str(observed['id']),zone=intent['zone'])
                    operator.config['primary_vm']={'name':name,'id':str(observed['id']),'zone':intent['zone']}
                outcome.update(status='IDENTITY_RESOLVED',id=str(observed['id']))
                intent.update(status='IDENTITY_RESOLVED',id=str(observed['id']))
                operator.state.setdefault('reconciled_resources',{})[kind+'/'+name]=outcome
            else:
                outcome['reason']='ABSENT_OR_IDENTITY_UNRESOLVED_NO_REPLAY'
        except (OSError,RuntimeError,ValueError,KeyError,TypeError):
            outcome['reason']='API_RESULT_UNKNOWN_NO_REPLAY'
        outcomes.append(outcome)
    operator.persist()
    result={'status':'RESOURCES_RECONCILED_BY_IDENTITY' if all(r['status']=='IDENTITY_RESOLVED' for r in outcomes)
            else 'BLOCKED_UNKNOWN_RESOURCE','resources':outcomes,'mutations':0}
    return operator.receipt('resource-reconciliation',result)
