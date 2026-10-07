"""Generation-bound GCS operations; bearer tokens are never returned or logged."""
import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request


def zero_retention_history(anchor):
    """Only creation and individually evidenced, non-retention setup mutations."""
    if str(anchor.get('metageneration')) == '1':
        return 'CREATION_GENERATION_ONE_ZERO_RETENTION'
    rows = anchor.get('zero_retention_history', [])
    expected = ['storage.buckets.create', 'storage.buckets.update',
                'storage.setIamPermissions', 'storage.setIamPermissions']
    if str(anchor.get('metageneration')) != '4' or [row.get('method') for row in rows] != expected:
        raise ValueError('Soft delete history cannot be proved; preserve all copies')
    for index, row in enumerate(rows):
        command = row.get('command', [])
        valid = ((index == 0 and command[1:4] == ['storage', 'buckets', 'create']
                  and '--soft-delete-duration=0' in command)
                 or (index == 1 and command[1:4] == ['storage', 'buckets', 'update']
                     and '--no-versioning' in command and not any('soft-delete' in part for part in command))
                 or (index > 1 and command[1:4] == ['storage', 'buckets', 'add-iam-policy-binding']
                     and not any('soft-delete' in part for part in command)))
        if (not valid or 'gs://' + anchor['name'] not in command or row.get('exit_code') != 0
                or not row.get('server_timestamp') or not row.get('server_insert_id')
                or any(len(row.get(key, '')) != 64 for key in ('audit_sha256', 'command_receipt_sha256'))):
            raise ValueError('Soft delete history cannot be proved; preserve all copies')
    return 'ANCHORED_ZERO_RETENTION_SETUP_HISTORY'


class Storage:
    def __init__(self, bucket, access_token, *, creation_anchor=None, policy_history=None):
        self.bucket = bucket
        self.access_token = access_token
        self.creation_anchor = creation_anchor
        self.soft_delete_verification = None
        self.policy_history = policy_history
        self.policy_verification = None

    def verify_zero_retention(self):
        from scripts.study_operator.bucket_history import validate

        if self.policy_history is None or self.creation_anchor is None:
            raise ValueError('Live audit policy verifier required; preserve all copies')
        before = json.loads(self.request(''))
        history = self.policy_history(before)
        after = json.loads(self.request(''))
        self.policy_verification = validate(before, after, self.creation_anchor, history)
        return self.policy_verification

    def request(self, path, *, params=None, data=None, method='GET', upload=False):
        base = 'https://storage.googleapis.com/' + ('upload/' if upload else '') + 'storage/v1/b/'
        url = base + self.bucket + path
        if params:
            url += '?' + urllib.parse.urlencode(params)
        headers = {'Authorization': 'Bearer ' + self.access_token(), 'Content-Type': 'application/octet-stream'}
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=headers), timeout=120) as response:
            return response.read()

    def metadata(self, name):
        return json.loads(self.request('/o/' + urllib.parse.quote(name, safe='')))

    def read(self, name, generation):
        return self.request('/o/' + urllib.parse.quote(name, safe=''), params={'alt': 'media', 'generation': generation})

    def objects(self, prefix, *, versions=False, soft_deleted=False):
        params = {'prefix': prefix}
        if versions:
            params['versions'] = 'true'
        if soft_deleted:
            params['softDeleted'] = 'true'
        result = []
        while True:
            try:
                value = json.loads(self.request('/o', params=params))
            except urllib.error.HTTPError as error:
                if not soft_deleted or error.code != 400 or not self.creation_anchor:
                    raise
                failure = json.loads(error.read())
                if 'Soft delete policy is required' not in failure.get('error',{}).get('message',''):
                    raise
                actual = json.loads(self.request(''))
                keys = ('id','name','timeCreated','metageneration')
                if (any(actual.get(key) != self.creation_anchor.get(key) for key in keys)
                        or actual.get('name') != self.bucket
                        or str(actual.get('softDeletePolicy',{}).get('retentionDurationSeconds')) != '0'
                        or str(self.creation_anchor.get('softDeletePolicy',{}).get('retentionDurationSeconds')) != '0'):
                    raise ValueError('Soft delete history cannot be proved; preserve all copies') from None
                method = zero_retention_history(self.creation_anchor)
                # Never call the HTTP400 response an empty API listing.
                self.soft_delete_verification = dict(method=method,
                    api_list_status='HTTP400_POLICY_REQUIRED',live_metadata=actual,
                    creation_anchor=self.creation_anchor,no_soft_delete_history_verified=True)
                return []
            if soft_deleted:
                self.soft_delete_verification = dict(method='API_LIST',api_list_status='SUCCESS')
            result.extend(value.get('items', []))
            if not value.get('nextPageToken'):
                return result
            params['pageToken'] = value['nextPageToken']

    def put(self, name, data):
        try:
            metadata = json.loads(self.request('/o', params={'uploadType': 'media', 'name': name, 'ifGenerationMatch': 0},
                                               data=data, method='POST', upload=True))
        except urllib.error.HTTPError as exc:
            if exc.code != 412:
                raise
            metadata = self.metadata(name)
        generation = metadata['generation']
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(self.read(name, generation)).hexdigest() != digest:
            raise ValueError('Remote immutable object differs; preserve both copies')
        return {'object': name, 'generation': generation, 'sha256': digest, 'bytes': len(data)}

    def delete(self, name, generation):
        self.request('/o/' + urllib.parse.quote(name, safe=''), method='DELETE',
                     params={'generation': generation, 'ifGenerationMatch': generation})

    def create_technical(self, name, data):
        """Creator-only write. The owner independently downloads this generation."""
        if not name.startswith('iteration4/') or '..' in name.split('/'):
            raise ValueError('Technical object outside permitted prefix')
        metadata = json.loads(self.request('/o', params={'uploadType':'media', 'name':name, 'ifGenerationMatch':0},
            data=data, method='POST', upload=True))
        checksum = base64.b64encode(hashlib.md5(data).digest()).decode()
        if (metadata.get('name') != name or not str(metadata.get('generation','')).isdigit()
                or int(metadata.get('size',-1)) != len(data) or metadata.get('md5Hash') != checksum):
            raise ValueError('Upload receipt checksum differs')
        return dict(object=name,generation=str(metadata['generation']),sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),server_created=metadata.get('timeCreated'),owner_download_verification_pending=True)
