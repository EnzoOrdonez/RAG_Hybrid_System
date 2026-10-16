"""Create a separate local November plan from verified catalog/protocol inputs."""
import argparse
import json
import subprocess
import uuid
from pathlib import Path

from scripts.study_operator.deployment import checked_config
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.service_gateway import save_state


def create(root, spec, *, commit):
    from src.ui.components.study_protocol import verify_draw
    root=Path(root)
    source=json.loads(Path(spec['source_installation']).read_bytes())
    prices=Path(spec['price_archive'])
    quote=quote_archive(prices,'g2-standard-4','us-central1')
    receipt=json.loads((prices/'receipt.json').read_bytes())
    protocol=verify_draw(spec['reviewed_config'])
    periods={kind:uuid.uuid4().hex for kind in ('technical','smoke')}
    source.update(iteration=7,commit=commit,operator_commit=commit,purpose='technical',
        period_ids=periods,period_id=periods['technical'],zone='us-central1-a',
        native_stop_s=10800,budget_usd=20,closure_reserve_usd=3,
        resource_marker='cloudrag-i7-'+periods['technical'],
        local_reviewed_config=str(Path(spec['reviewed_config']).resolve()),
        fingerprint=protocol['fingerprint'],cuda_cache_driver='580.178.04',
        cuda_cache='/srv/cloudrag/iteration6-cowork-backups/nv-computecache-580.178.04/ComputeCache',
        rates={'compute_usd_h':float(quote['usd_per_hour']),'verified_utc':receipt['at'],
               'price_receipt_sha256':quote['catalog_receipt_sha256']},
        qualification_status='LOCAL_I7_PLAN_NOT_CLOUD_QUALIFIED')
    for key in ('bootstrap_inputs','bootstrap_inputs_first_boot_only','configuration_provenance',
                'i6_provenance','primary_disk','secondary_vm','secondary_disk','instance_id',
                'final_snapshot','host_infrastructure_sha256','host_infrastructure_commit'):
        source.pop(key,None)
    # Placeholder identity cannot point start/stop at an inherited VM.
    source['primary_vm']={'name':'cloudrag-i7-unallocated','id':'0','zone':'us-central1-a'}
    checked_config(source)
    root.mkdir(parents=True,exist_ok=False)
    save_state(root/'installation.json',source)
    save_state(root/'I7.json',{'iteration':7})
    save_state(root/'active.json',{})
    save_state(root/'phase-b-spec.json',spec)
    return {'status':'LOCAL_I7_ROOT_CREATED_NOT_AUTHORIZATION','cloud_measured':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    print(json.dumps(create(args.root,json.loads(args.spec.read_bytes()),commit=commit)))


if __name__=='__main__':
    main()
