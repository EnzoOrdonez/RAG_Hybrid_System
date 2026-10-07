"""Owner-only Admin Activity read; persist policy evidence, never caller metadata."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

METHODS = ('storage.buckets.create', 'storage.buckets.update',
           'storage.buckets.delete', 'storage.setIamPermissions')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collect(token, project, bucket, metadata, *, request=None, seconds=300):
    if project != 'pure-loop-474323-a8' or bucket != 'cloudrag-study-i4-103950017681-20261004':
        raise ValueError('Audit history outside authorized session bucket')
    created = datetime.fromisoformat(metadata['timeCreated'].replace('Z', '+00:00'))
    if created.tzinfo is None or not 0 < seconds <= 600:
        raise ValueError('Bounded, timezone-aware audit history required')
    end = datetime.now(timezone.utc)
    # A create request can precede the bucket's server creation timestamp.
    start = created-timedelta(minutes=5)
    query = ('resource.type="gcs_bucket" AND resource.labels.bucket_name="'+bucket+'" '
        'AND logName="projects/'+project+'/logs/cloudaudit.googleapis.com%2Factivity" '
        'AND timestamp>="'+start.isoformat()+'" AND timestamp<="'+end.isoformat()+'" '
        'AND protoPayload.serviceName="storage.googleapis.com"')
    body = dict(resourceNames=['projects/'+project], filter=query, orderBy='timestamp asc', pageSize=1000)
    deadline, rows, pages, seen = time.monotonic()+seconds, [], [], set()

    def fetch(payload):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Audit history bound reached; preserve all session copies')
        call = Request('https://logging.googleapis.com/v2/entries:list',
            data=json.dumps(payload).encode(), method='POST',
            headers={'Authorization': 'Bearer '+token(), 'Content-Type': 'application/json'})
        with urlopen(call, timeout=min(60, remaining)) as response:
            return json.loads(response.read())

    request = request or fetch
    for _ in range(100):
        if time.monotonic() >= deadline:
            raise TimeoutError('Audit history bound reached; preserve all session copies')
        page = request(body.copy())
        if not isinstance(page, dict) or not isinstance(page.get('entries', []), list):
            raise ValueError('Invalid audit pagination; preserve all copies')
        pages.append(digest(page))
        for entry in page.get('entries', []):
            payload = entry['protoPayload']
            if (entry.get('resource', {}).get('labels', {}).get('bucket_name') != bucket
                    or payload.get('serviceName') != 'storage.googleapis.com'
                    or payload.get('methodName') not in METHODS):
                raise ValueError('Unexpected audit resource or mutation')
            # Authentication, requestMetadata, IPs, headers and arbitrary request
            # content stay in RAM. The original page is represented by its hash.
            rows.append(dict(method=payload['methodName'], server_timestamp=entry['timestamp'],
                server_insert_id=entry['insertId'], status_code=payload.get('status', {}).get('code', 0)))
        cursor = page.get('nextPageToken')
        if not cursor:
            return dict(status='COMPLETE_ADMIN_ACTIVITY_QUERY', bucket=bucket,
                query_start_utc=start.isoformat(), query_end_utc=end.isoformat(),
                pages=len(pages), page_sha256=pages, mutations=rows, complete_pagination=True,
                personal_metadata_not_persisted=True, no_future_policy_changes_inferred=True)
        if not isinstance(cursor, str) or cursor in seen:
            raise ValueError('Repeated audit page token; preserve all copies')
        seen.add(cursor)
        body['pageToken'] = cursor
    raise ValueError('Incomplete audit history; preserve all copies')


def validate(before, after, anchor, history):
    """Conservatively require the exact evidenced creation/setup mutation chain."""
    from scripts.study_operator.gcs import zero_retention_history

    zero_retention_history(anchor)
    for value in (before, after, anchor):
        if (str(value.get('softDeletePolicy', {}).get('retentionDurationSeconds')) != '0'
                or value.get('versioning', {}).get('enabled', False) is not False):
            raise ValueError('Zero soft delete and disabled versioning required')
    keys = ('id', 'name', 'timeCreated', 'metageneration')
    if any(before.get(key) != after.get(key) or before.get(key) != anchor.get(key) for key in keys):
        raise ValueError('Bucket mutation since anchor; preserve copies pending a new audited chain')
    if (history.get('status') != 'COMPLETE_ADMIN_ACTIVITY_QUERY'
            or history.get('complete_pagination') is not True or history.get('bucket') != before['name']):
        raise ValueError('Complete live Cloud Audit Logs history required')
    successful = [row for row in history['mutations'] if row['status_code'] == 0]
    expected = anchor.get('zero_retention_history', [])
    if not expected and str(anchor.get('metageneration')) == '1':
        raise ValueError('Creation audit anchor required in addition to bucket generation one')
    def identity(row):
        return row['method'], row['server_timestamp'], row['server_insert_id']
    if sorted(map(identity, successful)) != sorted(map(identity, expected)):
        raise ValueError('Unanchored or missing mutation in audit history; no deletion admitted')
    return dict(method='LIVE_ZERO_POLICY_AND_COMPLETE_ANCHORED_ADMIN_HISTORY',
        soft_delete_retention_seconds=0, versioning_enabled=False, complete_history=True,
        live_before_sha256=digest(before), live_after_sha256=digest(after),
        unchanged_metageneration=str(before['metageneration']), history=history,
        no_soft_deleted_listing_claim=True)


def main(argv=None):
    from scripts.study_operator.cloud_client import Cloud
    from scripts.study_operator.gcs import Storage
    from scripts.study_operator.run_control import require_limited

    parser = argparse.ArgumentParser(description='Verifica política e historial; no borra objetos.')
    parser.add_argument('--package', required=True)
    parser.add_argument('--sdk', required=True)
    parser.add_argument('--installation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require_limited()
    root, output = Path(args.package).resolve(), Path(args.output).resolve()
    if output.parent != root or output.exists():
        raise ValueError('New own-package technical receipt required')
    source = Path(args.installation).read_bytes()
    config = json.loads(source)
    cloud = Cloud(args.sdk, config['project'], root/'bucket-history-owner-api')
    storage = Storage(config['sessions_bucket'], cloud.owner_token,
        creation_anchor=config['sessions_bucket_creation'],
        policy_history=lambda metadata: collect(cloud.owner_token, config['project'],
            config['sessions_bucket'], metadata))
    proof = storage.verify_zero_retention()
    proof['installation_source_sha256'] = hashlib.sha256(source).hexdigest()
    with output.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(dict(status='LIVE_ZERO_POLICY_HISTORY_VERIFIED',
        successful_mutations=len(proof['history']['mutations']), no_session_objects_read=True)))


if __name__ == '__main__':
    main()
