"""Stage hash-bound public study configuration without broadening guest IAM."""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath


FILES = {'reviewed/study.json', 'reviewed/assignments.csv', 'reviewed/draw_seal.json',
         'reviewed/task_evidence.json', 'deployment-artifacts.json', 'service-preregistration.md'}
PREFIX = '/srv/cloudrag/iteration5/inputs/'


def assemble(reviewed, manifest, preregistration):
    reviewed = Path(reviewed)
    sources = {name: reviewed/name.split('/')[-1] for name in FILES if name.startswith('reviewed/')}
    sources.update({'deployment-artifacts.json': Path(manifest), 'service-preregistration.md': Path(preregistration)})
    rows = []
    for name, source in sorted(sources.items()):
        if source.is_symlink() or source.is_junction() or not source.is_file():
            raise ValueError('Reviewed public input must be a regular file')
        data = source.read_bytes()
        rows.append(dict(name=name, sha256=hashlib.sha256(data).hexdigest(),
                         base64=base64.b64encode(data).decode('ascii')))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = dict(root=PREFIX+digest, files=rows)
    validate(result)
    return result


def validate(value):
    root = PurePosixPath(value['root'])
    tail = str(root).removeprefix(PREFIX)
    if (str(root) != PREFIX+tail or len(tail) != 64 or any(c not in '0123456789abcdef' for c in tail)
            or {row['name'] for row in value['files']} != FILES or len(value['files']) != len(FILES)):
        raise ValueError('Only the six reviewed public inputs can be staged')
    total = 0
    for row in value['files']:
        if not isinstance(row['base64'], str) or len(row['base64']) > 200000:
            raise ValueError('Public input encoding exceeds metadata budget')
        data = base64.b64decode(row['base64'], validate=True)
        total += len(data)
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Staged public input hash changed')
    if total > 150000:
        raise ValueError('Reviewed public inputs exceed metadata budget')
    digest = hashlib.sha256(json.dumps(value['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if tail != digest:
        raise ValueError('Public input inventory hash changed')
    return value


def staging_program(config):
    value = config.get('bootstrap_inputs')
    if value is None:
        return ''
    validate(value)
    root = value['root']
    if (config['reviewed_config'] != root+'/reviewed'
            or config['artifact_manifest'] != root+'/deployment-artifacts.json'
            or config['preregistration_file'] != root+'/service-preregistration.md'):
        raise ValueError('Guest input paths differ from the reviewed inventory')
    return guest_source()+"\nmaterialize(c['bootstrap_inputs'])\n"


def guest_source():
    # Standard library only: final host code and image remain unchanged. Never
    # put private session, invitation or credential files in this allowlist.
    return '''import hashlib,os,base64,json
def materialize(value):
 prefix='/srv/cloudrag/iteration5/inputs/'
 digest=hashlib.sha256(json.dumps(value['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if value['root']!=prefix+digest:raise ValueError('PUBLIC_INPUT_ROOT_CHANGED')
 root=Path(value['root'])
 allowed={'reviewed/study.json','reviewed/assignments.csv','reviewed/draw_seal.json','reviewed/task_evidence.json','deployment-artifacts.json','service-preregistration.md'}
 if {r['name'] for r in value['files']}!=allowed or len(value['files'])!=6:raise ValueError('PUBLIC_INPUT_ALLOWLIST_CHANGED')
 prepared={}
 for row in value['files']:
  if not isinstance(row['base64'],str) or len(row['base64'])>200000:raise ValueError('PUBLIC_INPUT_TOO_LARGE')
  data=base64.b64decode(row['base64'],validate=True)
  if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('PUBLIC_INPUT_HASH_CHANGED')
  path=root/row['name']
  if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('PUBLIC_INPUT_SYMLINK')
  prepared[path]=data
 if sum(map(len,prepared.values()))>150000:raise ValueError('PUBLIC_INPUT_TOO_LARGE')
 for path,data in prepared.items():
  if path.exists() and (not path.is_file() or path.read_bytes()!=data):raise ValueError('PUBLIC_INPUT_ALREADY_DIFFERS')
 for path,data in prepared.items():
  path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
  if not path.exists():
   with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
   os.chmod(path,0o600)
   fd=os.open(path.parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
'''
