"""API-listed US L4 zones and device HTTPS latency; never infer available stock."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from scripts.study_operator.run_control import require_limited


def zones(catalog):
    result = {}
    for row in catalog:
        if row.get('name') != 'nvidia-l4':
            continue
        zone = row['zone'].split('/')[-1]
        if not re.fullmatch(r'us-[a-z]+\d-[a-z]', zone):
            continue
        region = zone.rsplit('-', 1)[0]
        if zone in result.get(region, []):
            raise ValueError('Duplicate L4 zone in API catalog')
        result.setdefault(region, []).append(zone)
    if not result:
        raise ValueError('API has no US L4 catalog; do not invent regions')
    return {key: sorted(value) for key, value in sorted(result.items())}


def request_ms(url):
    begin = time.perf_counter()
    with urlopen(Request(url, headers={'User-Agent': 'GCPing-CLI'}), timeout=10) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError('Latency endpoint rejected request')
        # Match GCPing's end boundary at receipt of HTTP headers. No response
        # body, client address, or headers are written to evidence.
        return (time.perf_counter()-begin)*1000


def order(catalog, endpoints, *, measure=request_ms):
    observed = zones(catalog)
    rows = []
    for region in observed:
        endpoint = endpoints.get(region, {})
        url = endpoint.get('URL', '')
        parsed = urlsplit(url)
        if (endpoint.get('Region') != region or parsed.scheme != 'https' or parsed.username
                or parsed.password or parsed.port or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or not re.fullmatch(re.escape(region)+r'-[a-z0-9-]+\.a\.run\.app', parsed.hostname or '')):
            raise ValueError('Regional endpoint contract differs')
        samples = []
        for index in range(5):
            try:
                value = measure(url.rstrip('/')+'/api/ping')
                if type(value) not in (int, float) or not 0 < value < 10000:
                    raise ValueError('Latency outside timeout boundary')
                samples.append(dict(position=index+1, status='OK', ms=value))
            except (OSError, ValueError) as error:
                samples.append(dict(position=index+1, status='FAILED', error_type=type(error).__name__))
        successful = all(s['status'] == 'OK' for s in samples)
        rows.append(dict(region=region, zones=observed[region], endpoint=url, samples=samples,
                         median_ms=statistics.median(s['ms'] for s in samples) if successful else None))
    measured = sorted((r for r in rows if r['median_ms'] is not None), key=lambda r: (r['median_ms'], r['region']))
    return dict(status='DESCRIPTIVE_REGION_LATENCY_NOT_CAPACITY', at=datetime.now(timezone.utc).isoformat(),
        origin='Enzo Windows device; Lima declared by operator', samples_per_region=5,
        boundary='GET start through HTTP200 headers; DNS/TCP/TLS included; no exclusions',
        catalog_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        endpoints_sha256=hashlib.sha256(json.dumps(endpoints, sort_keys=True).encode()).hexdigest(),
        results=rows, after_central_rounds_order=[r['region'] for r in measured if r['region'] != 'us-central1'],
        unresolved_regions=[r['region'] for r in rows if r['median_ms'] is None], available_stock_not_inferred=True)


def next_round(previous, now):
    if now.tzinfo is None or any(datetime.fromisoformat(r['at']).tzinfo is None for r in previous):
        raise ValueError('Aware UTC round timestamps required')
    if len(previous) >= 3:
        raise ValueError('Central capacity rounds exhausted; use measured US order')
    if previous and (now-max(datetime.fromisoformat(r['at']) for r in previous)).total_seconds() < 45*60:
        raise ValueError('At least45 minutes between central rounds; no immediate retry')
    return len(previous)+1


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('catalog', 'endpoints', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    require_limited()
    result = order(json.loads(Path(args.catalog).read_bytes()), json.loads(Path(args.endpoints).read_bytes()))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], order=result['after_central_rounds_order'],
                          unresolved_regions=result['unresolved_regions'])))


if __name__ == '__main__':
    main()
