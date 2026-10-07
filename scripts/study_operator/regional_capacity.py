"""US-only capacity after three central rounds and latest exhaustion; never READY."""
import argparse
from datetime import datetime
import hashlib
import ipaddress
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.capacity_order import zones
from scripts.study_operator.capacity_probe import Probe
from scripts.study_operator.cloud_client import Cloud,no_other_gpu
from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.pricing import disk_quote_archive,quote_archive
from scripts.study_operator.region_scope import region as checked_region
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json

SNAPSHOT_TRANSFER_SOURCE = 'https://cloud.google.com/compute/disks-image-pricing#network-charges'


def admission_order(state,latency,catalog):
    rounds = state.get('capacity_rounds',[])
    allowed = {'CAPACITY_EXHAUSTED','L4_RUNNING_OBSERVED_NOT_READY'}
    if (len(rounds)!=3 or any(r['status']!='COMPLETED' or len(r['results'])!=3
            or any(v['status'] not in allowed or not v['stopped_verified'] for v in r['results']) for r in rounds)
            or any(v['status']!='CAPACITY_EXHAUSTED' for v in rounds[-1]['results'])):
        raise ValueError('Three complete central rounds and latest exhaustion required before another region')
    # An earlier, stopped capacity observation does not reserve that capacity.
    # The latest complete round must show no capacity in all three central zones.
    for index,row in enumerate(rounds):
        if (row.get('round') != index+1
                or [v.get('zone') for v in row['results']] != ['us-central1-a','us-central1-b','us-central1-c']):
            raise ValueError('Central round identities or zone census changed')
        at = datetime.fromisoformat(row['at'])
        if at.tzinfo is None or (index and (at-datetime.fromisoformat(rounds[index-1]['at'])).total_seconds()<2700):
            raise ValueError('Three central rounds require aware timestamps and 45-minute separation')
    digest=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest()
    observed=zones(catalog)
    if (latency['status']!='DESCRIPTIVE_REGION_LATENCY_NOT_CAPACITY'
            or latency['catalog_sha256']!=digest or latency['unresolved_regions']):
        raise ValueError('Complete measured region ordering and exact API catalog required')
    expected=sorted((r for r in latency['results'] if r['region']!='us-central1'),
                    key=lambda r:(r['median_ms'],r['region']))
    order=[r['region'] for r in expected]
    if (order!=latency['after_central_rounds_order'] or set(order)!=set(observed)-{'us-central1'}
            or any(r['zones']!=observed[r['region']] or len(r['samples'])!=5
                   or any(s['status']!='OK' for s in r['samples']) for r in expected)):
        raise ValueError('Region ordering or successful samples changed')
    return [(r,observed[r]) for r in order]


def regional_create(name,disk,zone,startup,marker,network,subnet):
    region=checked_region(zone)
    if region=='us-central1' or not subnet.startswith('cloudrag-i5-'+region+'-'):
        raise ValueError('Own prepared US regional subnet required')
    return ['compute','instances','create',name,'--zone='+zone,'--machine-type=g2-standard-4',
        '--provisioning-model=STANDARD','--maintenance-policy=TERMINATE',
        '--disk=name='+disk+',boot=yes,auto-delete=no','--deletion-protection',
        '--max-run-duration=3h','--instance-termination-action=STOP','--no-address',
        '--no-service-account','--no-scopes','--network='+network,'--subnet='+subnet,
        '--metadata-from-file=startup-script='+str(startup),'--metadata=enable-guest-attributes=TRUE',
        '--description='+marker]


class RegionalProbe(Probe):
    def creation_arguments(self,*args):
        return regional_create(*args)

    def extra_exposure(self,zone,state,quote):
        rate=disk_quote_archive(self.root/'official-compute-skus',checked_region(zone))
        self.disk_quote=rate
        hours=max(0,(datetime.fromisoformat(state['closure_reserved_utc'])-self.now()).total_seconds())/3600
        # Official North America snapshot restore transfer: .02 USD/GiB.
        # Reserve100GiB before every disk creation; not a claim of transferred bytes.
        value=100*float(rate['estimated_usd_gib_h'])*hours+2
        with (self.root/'COST_LEDGER.md').open('a',encoding='utf-8') as f:
            f.write('\nBEFORE regional '+zone+': '+json.dumps(rate)+'; snapshot restore transfer upperUSD2 '
                    '(100GiB at .02 USD/GiB), official source '+SNAPSHOT_TRANSFER_SOURCE+
                    ' consulted2026-10-07; sessions remain us-central1, no session traffic during capacity-only probe.\n')
        return value

    def subnet(self,region,network,index):
        name='cloudrag-i5-'+region+'-20261007'
        marker='CloudRAG-I5-regional-'+self.root.name+'-'+region
        observed=self.cloud.command(['compute','networks','subnets','list'])
        matches=[r for r in observed if r['name']==name]
        if matches:
            if len(matches)!=1 or matches[0].get('description')!=marker:
                raise ValueError('Regional subnet already exists outside own scope')
            subnet=matches[0]
        else:
            cidr=ipaddress.ip_network('10.43.'+str(16+index)+'.0/24')
            if any(cidr.overlaps(ipaddress.ip_network(r['ipCidrRange'])) for r in observed
                   if r['network'].endswith('/'+network)):
                raise ValueError('Regional subnet CIDR overlaps study network')
            def intent(state):
                admission(state,sum(r['maximum_usd'] for r in state.get('open_exposures',{}).values()),now=self.now())
                row=dict(type='subnet',name=name,region=region,ownership_marker=marker,disposable=True)
                prior=[r for r in state.setdefault('resource_intents',[]) if r['type']=='subnet' and r['name']==name]
                if prior and prior!=[row]:
                    raise ValueError('Regional subnet intent differs')
                if not prior:
                    state['resource_intents'].append(row)
            self.update(intent)
            with (self.root/'DESTRUCTION_LOG.md').open('a',encoding='utf-8') as f:
                f.write('\nBEFORE own subnet '+name+': disposable regional infrastructure; no flow logs, no session payload. No hourly subnet resource charge; network traffic quoted separately.\n')
            self.cloud.command(['compute','networks','subnets','create',name,'--network='+network,
                '--region='+region,'--range='+str(cidr),'--enable-private-ip-google-access',
                '--no-enable-flow-logs','--description='+marker],timeout=300)
            subnet=self.cloud.command(['compute','networks','subnets','describe',name,'--region='+region])
        if (subnet['region'].split('/')[-1]!=region or not subnet['network'].endswith('/'+network)
                or not subnet.get('privateIpGoogleAccess') or subnet.get('enableFlowLogs')
                or subnet.get('description')!=marker):
            raise ValueError('Regional subnet lacks required scope or private access')
        def record(state):
            row=dict(type='subnet',name=name,id=str(subnet['id']),region=region,
                disposable=True,ownership_marker=marker,idle_usd_day=0)
            prior=[r for r in state['resources'] if r['type']=='subnet' and r['name']==name]
            if prior and prior!=[row]:
                raise ValueError('Regional subnet identity changed')
            if not prior:
                state['resources'].append(row)
        self.update(record)
        return name

    def regional(self,snapshot,network,latency,catalog):
        state=json.loads((self.root/'STATE.json').read_bytes())
        order=admission_order(state,latency,catalog)
        if state.get('regional_capacity'):
            raise ValueError('Regional attempt already exists; reconcile preserved outcomes before explicit retry')
        proof=json.loads((self.root/'cpu-restoration-bootstrap01-proof.json').read_bytes())
        live=self.cloud.command(['compute','snapshots','describe',snapshot['name']])
        if (str(live['id'])!=str(snapshot['id']) or live['status']!='READY'
                or str(live['id'])!=str(proof['source_snapshot_id']) or proof['status']!='CPU_RESTORATION_VERIFIED'
                or proof['synthetic'] is not False or int(live['diskSizeGb'])>100
                or live['storageLocations']!=['us-central1']):
            raise ValueError('Actual qualified central snapshot required before regional restoration')
        entry=dict(at=self.now().isoformat(),status='RUNNING',order=[r for r,_ in order],results=[],
                   not_a_measurement_or_ready=True,source_snapshot_id=str(live['id']))
        self.update(lambda s:s.update(regional_capacity=entry))
        try:
            for index,(region,zone_list) in enumerate(order):
                no_other_gpu(self.cloud.command(['compute','instances','list']),selected_id='NO_RUNNING_GPU')
                subnet=self.subnet(region,network,index)
                quote=quote_archive(self.root/'official-compute-skus','g2-standard-4',region)
                for zone in zone_list:
                    receipt=self.attempt(zone,'regional',live,network,subnet,quote)
                    entry['results'].append(receipt)
                    self.update(lambda s:s.update(regional_capacity=dict(entry)))
                    if receipt['status']=='L4_RUNNING_OBSERVED_NOT_READY':
                        entry.update(status='CAPACITY_OBSERVED_STOPPED_NOT_READY',selected=receipt)
                        return entry
            entry['status']='ALL_US_REGIONS_EXHAUSTED_WAIT_DOCUMENTED'
            return entry
        except Exception:
            entry['status']='FAILED_PRESERVED_RECONCILIATION_REQUIRED'
            raise
        finally:
            self.update(lambda s:s.update(regional_capacity=dict(entry)))
            atomic_json(self.root/'regional-capacity-receipt.json',entry)


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('package','sdk','snapshot','network','latency','catalog'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    require_limited()
    root=Path(args.package)
    with FileLock(str(root/'capacity-probe.lock'),timeout=0):
        probe=RegionalProbe(root,Cloud(args.sdk,'pure-loop-474323-a8',root/'regional-capacity-api'))
        result=probe.regional(json.loads(Path(args.snapshot).read_bytes()),args.network,
            json.loads(Path(args.latency).read_bytes()),json.loads(Path(args.catalog).read_bytes()))
        print(json.dumps(dict(status=result['status'],results=result['results'])))


if __name__=='__main__':
    main()
