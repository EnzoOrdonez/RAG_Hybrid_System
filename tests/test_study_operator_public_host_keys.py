import base64
import hashlib
import pytest
from scripts.study_operator.windows_ssh import api_host_key_flags


def key(algorithm='ssh-ed25519'):
    data=len(algorithm).to_bytes(4,'big')+algorithm.encode()+(32).to_bytes(4,'big')+b'\x01'*32
    return dict(namespace='hostkeys',key=algorithm,value=base64.b64encode(data).decode()),data


def test_authenticated_api_wire_key_produces_exact_public_pin():
    row,data=key()
    digest=base64.b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
    assert api_host_key_flags({'queryValue':{'items':[row]}})==['--ssh-flag=-hostkey','--ssh-flag=SHA256:'+digest]


@pytest.mark.parametrize('mutation',['duplicate','wire_type','newline','unknown','empty'])
def test_untrusted_host_key_data_fails_closed(mutation):
    row,_=key()
    rows=[row]
    if mutation=='duplicate':
        rows.append(row.copy())
    elif mutation=='wire_type':
        row['key']='ssh-rsa'
    elif mutation=='newline':
        row['value']+='\nINJECTION'
    elif mutation=='unknown':
        row['key']='UNTRUSTED_TYPE'
    elif mutation=='empty':
        rows=[]
    with pytest.raises(ValueError):
        api_host_key_flags({'queryValue':{'items':rows}})
