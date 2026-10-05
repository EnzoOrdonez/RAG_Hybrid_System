"""Bounded host startup, also usable before a newly created VM has its ID."""
import base64
import json


def write_startup(path, config, *, discover_instance=False):
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
        'root=Path("/srv/cloudrag/iteration4");root.mkdir(exist_ok=True)\n'
        'c=json.loads(base64.b64decode('+repr(encoded)+'))\n'+discovery+
        'p=root/"launch-config.json"\np.write_text(json.dumps(c))\nos.chmod(p,0o600)\n'
        'subprocess.Popen(["python3","-B","-m","scripts.study_operator.host_runtime",'
        '"--settings",str(p)],cwd=c["host_code"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n'
        'PY\n',encoding='utf-8',newline='\n')
    return path
