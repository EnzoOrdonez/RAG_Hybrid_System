"""Generation-bound GCS operations; bearer tokens are never returned or logged."""
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request


class Storage:
    def __init__(self, bucket, access_token):
        self.bucket = bucket
        self.access_token = access_token

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
            value = json.loads(self.request('/o', params=params))
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
