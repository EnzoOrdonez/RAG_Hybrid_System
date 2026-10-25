"""Owner-side deployed restore check; private bytes travel only through RPC stdin."""
import base64
import hashlib


def probe(bridge, request):
    if request.get('operation') != 'restore':
        raise ValueError('RESTORE_REQUEST_REQUIRED')
    expected = hashlib.sha256(base64.b64decode(request['full_session_base64'],validate=True)).hexdigest()
    if expected != request['objects']['full_session.json']['sha256']:
        raise ValueError('RESTORE_SOURCE_HASH_DIFFERS')
    maintenance = bridge({'operation':'maintenance'})
    if maintenance.get('status') != 'MAINTENANCE' or maintenance.get('admission_closed') is not True:
        raise ValueError('RESTORE_MAINTENANCE_NOT_VERIFIED')
    result = bridge(request)
    if (result.get('status') != 'RESTORED_VERIFIED' or result.get('export_sha256') != expected
            or result.get('new_app_instance') is not True or result.get('original_store_unchanged') is not True
            or result.get('invitation_revoked') is not True):
        raise ValueError('RESTORE_DEPLOYMENT_NOT_VERIFIED')
    return {'status':'RESTORE_DEPLOYMENT_VERIFIED','export_sha256':expected,
            'recovery_store_id':result['recovery_store_id'],'original_store_unchanged':True,
            'invitation_revoked':True}
