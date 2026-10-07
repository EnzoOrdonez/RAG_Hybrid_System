"""Bounded official Compute Engine SKU reads; existing auth remains in memory."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.run_control import require_limited

ENDPOINT = 'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus'


def unit_rate(sku):
    infos = sku.get('pricingInfo', [])
    if not infos:
        raise ValueError('SKU has no price')
    info = max(infos, key=lambda row: row['effectiveTime'])
    expression = info['pricingExpression']
    tiers = expression['tieredRates']
    if len(tiers) != 1 or tiers[0].get('startUsageAmount', 0) != 0:
        raise ValueError('Do not silently flatten tiered pricing')
    money = tiers[0]['unitPrice']
    if money['currencyCode'] != 'USD':
        raise ValueError('Expected explicit USD price')
    value = Decimal(money.get('units', '0')) + Decimal(money.get('nanos', 0))/Decimal(10**9)
    return dict(usd_per_usage_unit=str(value), usage_unit=expression['usageUnit'],
                effective_time=info['effectiveTime'])


def candidates(skus, region='us-central1'):
    return [dict(sku_id=row['skuId'], description=row['description'], category=row['category'],
                 **unit_rate(row)) for row in skus
            if region in row.get('serviceRegions', []) and row.get('category', {}).get('usageType') == 'OnDemand'
            and any(word in row.get('description', '').lower() for word in ('e2 instance', 'g2 instance', 'nvidia l4'))]


def machine_quote(skus, machine, region):
    specifications = {'e2-standard-2': ('E2', 2, 8, 0), 'g2-standard-4': ('G2', 4, 16, 1)}
    if machine not in specifications or not region.startswith('us-'):
        raise ValueError('Only reviewed machines and US regions can be quoted')
    family, cores, memory, gpu = specifications[machine]
    selected, total = [], Decimal(0)
    resources = [(rf'{family} Instance Core running in .+', 'CPU', 'h', cores),
                 (rf'{family} Instance Ram running in .+', 'RAM', 'GiBy.h', memory)]
    if gpu:
        resources.append((r'Nvidia L4 GPU running in .+', 'GPU', 'h', gpu))
    for pattern, group, unit, count in resources:
        rows = [row for row in skus if region in row.get('serviceRegions', [])
                and row.get('category', {}).get('usageType') == 'OnDemand'
                and row['category']['resourceGroup'] == group and re.fullmatch(pattern, row['description'])]
        if len(rows) != 1:
            raise ValueError('Ambiguous or missing standard resource SKU: ' + group)
        row = rows[0]
        rate = unit_rate(row)
        if rate['usage_unit'] != unit:
            raise ValueError('Unexpected SKU unit; do not infer conversion')
        total += count * Decimal(rate['usd_per_usage_unit'])
        selected.append(dict(sku_id=row['skuId'], description=row['description'], count=count, **rate))
    return dict(machine=machine, region=region, usd_per_hour=str(total), skus=selected,
                excluded_costs=['disk', 'snapshot', 'network', 'IP', 'storage operations'])


def quote_archive(folder, machine, region):
    folder = Path(folder)
    receipt = json.loads((folder/'receipt.json').read_bytes())
    if receipt.get('source') != ENDPOINT or receipt.get('currency') != 'USD':
        raise ValueError('Official complete USD catalog receipt required')
    rows = []
    for index, page in enumerate(receipt['pages']):
        if page['path'] != f'page-{index:03}.json' or not page['url'].startswith(ENDPOINT+'?'):
            raise ValueError('Unexpected catalog page provenance')
        data = (folder/page['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != page['sha256'] or len(data) != page['bytes']:
            raise ValueError('Official catalog page identity changed')
        payload = json.loads(data)
        if (index == len(receipt['pages'])-1) == bool(payload.get('nextPageToken')):
            raise ValueError('Partial or disconnected catalog pages')
        rows.extend(payload['skus'])
    if len(rows) != receipt['skus']:
        raise ValueError('Catalog count mismatch')
    return dict(catalog_receipt_sha256=hashlib.sha256((folder/'receipt.json').read_bytes()).hexdigest(),
                **machine_quote(rows, machine, region))


def disk_quote_archive(folder, region):
    """Keep the official monthly unit; hourly conversion is an explicit estimate."""
    verified = quote_archive(folder,'g2-standard-4',region)
    rows = [row for page in sorted(Path(folder).glob('page-*.json'))
            for row in json.loads(page.read_bytes())['skus']
            if region in row.get('serviceRegions',[])
            and row.get('category',{}).get('usageType') == 'OnDemand'
            and re.fullmatch(r'Balanced PD Capacity(?: in .+)?',row['description'])]
    if len(rows) != 1:
        raise ValueError('Ambiguous or missing zonal balanced PD SKU')
    rate = unit_rate(rows[0])
    if rate['usage_unit'] != 'GiBy.mo':
        raise ValueError('Explicit monthly storage unit required')
    return dict(region=region,sku_id=rows[0]['skuId'],description=rows[0]['description'],**rate,
        estimated_usd_gib_h=str(Decimal(rate['usd_per_usage_unit'])/Decimal(730)),
        estimated_month_hours=730,catalog_receipt_sha256=verified['catalog_receipt_sha256'])


def download(cloud, destination, *, open_url=urlopen, max_pages=100, seconds=1200):
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    begin, deadline, cursor, rows, pages = time.monotonic(), time.monotonic()+seconds, '', [], []
    token = cloud.owner_token()
    for page in range(max_pages):
        if time.monotonic() >= deadline:
            raise TimeoutError('Official catalog read limit; preserve partial pages')
        url = ENDPOINT + '?' + urlencode(dict(currencyCode='USD', pageSize=5000, pageToken=cursor))
        request = Request(url, headers={'Authorization': 'Bearer '+token})
        with open_url(request, timeout=min(60, max(1, deadline-time.monotonic()))) as response:
            payload = response.read(16000001)
        if len(payload) > 16000000:
            raise ValueError('Official catalog page exceeds bounded read; no resource creation')
        value = json.loads(payload)
        if not isinstance(value.get('skus'), list):
            raise ValueError('Unexpected catalog contract')
        path = destination / f'page-{page:03}.json'
        path.open('xb').write(payload)
        pages.append(dict(path=path.name, url=url, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
        rows.extend(value['skus'])
        cursor = value.get('nextPageToken', '')
        if not cursor:
            result = dict(at=datetime.now(timezone.utc).isoformat(), pages=pages, skus=len(rows),
                          source=ENDPOINT, currency='USD', duration_s=time.monotonic()-begin,
                          candidates=candidates(rows), auth_never_persisted=True)
            with (destination/'receipt.json').open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
            return result
    raise ValueError('Catalog page limit exhausted; partial reads are not a complete price inventory')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdk', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args(argv)
    require_limited()
    cloud = Cloud(args.sdk, 'pure-loop-474323-a8', Path(args.package)/'pricing-catalog-api')
    result = download(cloud, args.destination)
    print(json.dumps(dict(skus=result['skus'], pages=len(result['pages']), candidates=result['candidates'])))


if __name__ == '__main__':
    main()
