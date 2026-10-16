"""Complete conservative list-price exposure, including retained resources after STOP."""
import argparse
import hashlib
import json
import math
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path

from scripts.study_operator.pricing import ENDPOINT

STORAGE_ENDPOINT='https://cloudbilling.googleapis.com/v1/services/95FF-2EF5-5EA1/skus'
# IDs and units from official catalog archives. Tier maxima avoid free-tier assumptions.
SKU={
    'g2_vcpu':('D18E-3563-415F','h'),'g2_ram':('F0A4-E3D3-33BD','GiBy.h'),
    'l4':('A88A-5A60-E821','h'),'e2_vcpu':('CF4E-A0C7-E3BF','h'),'e2_ram':('F449-33EC-A5EF','GiBy.h'),
    'pd_gib_h':('6AE1-525F-8B80','GiBy.mo'),'snapshot_gib_h':('817F-F5A3-514E','GiBy.mo'),
    'ip_used_h':('C054-7F72-A02E','h'),'ip_unused_h':('66A2-68EA-56BE','h'),
    'egress_gib':('03A5-1B2A-3485','GiBy'),'snapshot_transfer_gib':('EB4E-CD21-DA31','GiBy'),
    'gcs_gib_h':('E5F0-6A5D-7BAD','GiBy.mo'),'gcs_a':('4DBF-185F-A415','count'),
    'gcs_b':('7870-010B-2763','count'),'gcs_egress_gib':('22EB-AAE8-FBCD','GiBy')}
KEYS=tuple(SKU)


def archived_rows(folder,endpoint):
    folder=Path(folder)
    receipt=json.loads((folder/'receipt.json').read_bytes())
    if receipt['source']!=endpoint or receipt['currency']!='USD':
        raise ValueError('USD_LIST_PRICE_ARCHIVE_REQUIRED')
    rows=[]
    for i,page in enumerate(receipt['pages']):
        if page['path']!=f'page-{i:03}.json' or not page['url'].startswith(endpoint+'?'):
            raise ValueError('PRICE_PAGE_PROVENANCE_DIFFERS')
        raw=(folder/page['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=page['sha256'] or len(raw)!=page['bytes']:
            raise ValueError('PRICE_PAGE_HASH_DIFFERS')
        data=json.loads(raw)
        if (i==len(receipt['pages'])-1)==bool(data.get('nextPageToken')):
            raise ValueError('COMPLETE_PRICE_PAGES_REQUIRED')
        rows.extend(data['skus'])
    if not rows or len(rows)!=receipt['skus']:
        raise ValueError('PRICE_CENSUS_DIFFERS')
    return rows,receipt


def list_prices(compute,storage):
    try:
        compute_rows,c=archived_rows(compute,ENDPOINT)
        storage_rows,s=archived_rows(storage,STORAGE_ENDPOINT)
        rows={row['skuId']:row for row in compute_rows+storage_rows}
        rates={}
        for key,(sku_id,unit) in SKU.items():
            row=rows[sku_id]
            pricing=max(row['pricingInfo'],key=lambda p:p['effectiveTime'])
            expression=pricing['pricingExpression']
            if expression['usageUnit']!=unit or row['category']['usageType']!='OnDemand':
                raise ValueError('REVIEWED_SKU_UNIT_REQUIRED')
            values=[]
            for tier in expression['tieredRates']:
                money=tier['unitPrice']
                if money['currencyCode']!='USD':
                    raise ValueError('USD_SKU_REQUIRED')
                values.append(Decimal(money.get('units','0'))+Decimal(money.get('nanos',0))/Decimal(10**9))
            value=max(values)
            if unit=='GiBy.mo':
                hours=Decimal(str(expression['baseUnitConversionFactor']))/Decimal(2**30)/Decimal(3600)
                if hours<=0:
                    raise ValueError('MONTHLY_CONVERSION_REQUIRED')
                value/=hours
            rates[key]={'usd':float(value),'sku_id':sku_id,'description':row['description'],
                'catalog_unit':unit,'basis':'MAXIMUM_TIER_NO_FREE_ALLOWANCE','effective_time':pricing['effectiveTime']}
        return {'status':'PRICED_LIST','basis':'LIST_PRICES_NOT_INVOICE','rates':rates,
            'verified_utc':min(c['at'],s['at']),
            'archive_receipts_sha256':[hashlib.sha256((Path(p)/'receipt.json').read_bytes()).hexdigest() for p in (compute,storage)]}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'status':'UNPRICED','basis':'LIST_PRICES_NOT_INVOICE','rates':{},'reason':type(error).__name__}


def default_quantities(hours=24):
    # Full phase upper: 12h + 5h GPU reservations, 6h CPU; all retained bytes
    # through closure, including inherited snapshots and the existing regional IP.
    return dict(g2_vcpu=4*17,g2_ram=16*17,l4=17,e2_vcpu=2*6,e2_ram=8*6,
        pd_gib_h=(360+180)*hours,snapshot_gib_h=(360+192)*hours,
        ip_used_h=0,ip_unused_h=hours,egress_gib=1,snapshot_transfer_gib=0,
        gcs_gib_h=hours,gcs_a=10000,gcs_b=100000,gcs_egress_gib=1)


def exposure(prices,quantities, *, extra=0):
    rows={}
    try:
        if prices['status']!='PRICED_LIST' or set(prices['rates'])!=set(KEYS) or set(quantities)!=set(KEYS):
            raise ValueError('ALL_SKUS_PRICED_REQUIRED')
        for key in KEYS:
            rate=prices['rates'][key]['usd']
            quantity=quantities[key]
            if any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in (rate,quantity)):
                raise ValueError('FINITE_NONNEGATIVE_COST_REQUIRED')
            rows[key]={'quantity':quantity,'rate_usd':rate,'sku_id':prices['rates'][key]['sku_id'],
                       'upper_usd':rate*quantity}
        if type(extra) not in (float,int) or not math.isfinite(extra) or extra<0:
            raise ValueError('FINITE_EXTRA_EXPOSURE_REQUIRED')
        total=sum(row['upper_usd'] for row in rows.values())+extra+3
        return {'status':'PRICED_LIST','decision':'ADMIT' if total<=20 else 'STOP',
                'total_with_closure_usd':total,'closure_usd':3,'rows':rows,'basis':'EXPOSURE_NOT_INVOICE'}
    except (ValueError,KeyError,TypeError):
        return {'status':'UNPRICED','decision':'STOP','rows':rows,'basis':'EXPOSURE_NOT_INVOICE'}


def allocated_hours(reserved,running):
    from scripts.study_operator.i7 import reservation_charge
    return reservation_charge(reserved,running,1)


def retained_gib_hours(size,created,until):
    return size*(until-datetime.fromisoformat(created.replace('Z','+00:00'))).total_seconds()/3600


def admit(operator):
    now=operator.now()
    started=operator.state.setdefault('budget_started_utc',now.isoformat())
    hours=max(24,(now-datetime.fromisoformat(started)).total_seconds()/3600)
    prices=operator.config.get('cost_plan',{'status':'UNPRICED'})
    result=exposure(prices,default_quantities(hours),extra=operator.state.get('extra_exposure_usd',0))
    operator.state['cost_ledger']=dict(result,at=now.isoformat(),retention_hours=hours,
        reservation_gpu_hours_used=operator.state.get('reservation_gpu_hours_used',0))
    operator.persist()
    if result['decision']!='ADMIT':
        raise ValueError('I7_ALL_SKU_BUDGET_STOP_OR_UNPRICED')
    if hours>24:
        raise ValueError('I7_RETENTION_DEADLINE_CLOSE_RESOURCES')
    verified=datetime.fromisoformat(prices['verified_utc'])
    if verified.tzinfo is None or not 0<=(now-verified).total_seconds()<=7*86400:
        raise ValueError('FRESH_ALL_SKU_LIST_PRICES_REQUIRED')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute',type=Path,required=True)
    parser.add_argument('--storage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prices=list_prices(args.compute,args.storage)
    result={'prices':prices,'exposure':exposure(prices,default_quantities()),
            'recorded_utc':datetime.now(timezone.utc).isoformat()}
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({'status':prices['status'],'decision':result['exposure']['decision']}))


if __name__=='__main__':
    main()
