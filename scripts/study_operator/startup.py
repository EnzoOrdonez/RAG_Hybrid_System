"""Bounded host startup, also usable before a newly created VM has its ID."""
import base64
import json


def host_key_publication(public_root='/etc/ssh'):
    """Publish existing public keys for instances enabled after first boot."""
    return ('import base64,urllib.request\n'
        'published=0\n'
        'for algorithm,filename in (("ssh-ed25519","ssh_host_ed25519_key.pub"),("ecdsa-sha2-nistp256","ssh_host_ecdsa_key.pub"),("ssh-rsa","ssh_host_rsa_key.pub")):\n'
        ' key=Path('+repr(public_root)+')/filename\n'
        ' if key.is_symlink():raise ValueError("PUBLIC_HOST_KEY_SYMLINK")\n'
        ' if not key.is_file():continue\n'
        ' fields=key.read_text(encoding="ascii").split()\n'
        ' if len(fields)<2 or fields[0]!=algorithm or len(fields[1])>16384:raise ValueError("INVALID_PUBLIC_HOST_KEY")\n'
        ' wire=base64.b64decode(fields[1],validate=True)\n'
        ' prefix=len(algorithm).to_bytes(4,"big")+algorithm.encode()\n'
        ' if not wire.startswith(prefix) or len(wire)<=len(prefix)+4 or base64.b64encode(wire).decode()!=fields[1]:raise ValueError("INVALID_PUBLIC_HOST_KEY_WIRE")\n'
        ' request=urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hostkeys/"+algorithm,data=fields[1].encode(),headers={"Metadata-Flavor":"Google"},method="PUT")\n'
        ' with urllib.request.urlopen(request,timeout=5) as response:response.read()\n'
        ' published+=1\n'
        'if not published:raise ValueError("NO_PUBLIC_HOST_KEYS")\n')


def write_startup(path, config, *, discover_instance=False):
    from scripts.study_operator.startup_inputs import staging_program

    staging = staging_program(config)
    encoded = base64.b64encode(json.dumps(config).encode()).decode()
    discovery = ''
    if discover_instance:
        discovery = (
            'import urllib.request\n'
            'def metadata(field):\n'
            ' r=urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/"+field,headers={"Metadata-Flavor":"Google"})\n'
            ' with urllib.request.urlopen(r,timeout=5) as s:return s.read().decode()\n'
            'assert metadata("zone").split("/")[-1]==c["zone"]\n'
            'assert metadata("machine-type").split("/")[-1]=="g2-standard-4"\n'
            'c["instance_id"]=metadata("id")\n'
        )
    path.write_text('#!/bin/bash\nset -euo pipefail\npython3 - <<\'PY\'\n'
        'import base64,json,subprocess,os\nfrom pathlib import Path\n'
        'root=Path("/srv/cloudrag/iteration5");root.mkdir(exist_ok=True)\n'
        'c=json.loads(base64.b64decode('+repr(encoded)+'))\n'+discovery+staging+host_key_publication()+
        'p=root/"launch-config.json"\np.write_text(json.dumps(c))\nos.chmod(p,0o600)\n'
        'subprocess.Popen(["python3","-B","-m","scripts.study_operator.host_runtime",'
        '"--settings",str(p)],cwd=c["host_code"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n'
        'PY\n',encoding='utf-8',newline='\n')
    return path
