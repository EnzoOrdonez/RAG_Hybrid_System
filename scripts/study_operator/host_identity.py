"""Cloud identity from a host-owned read-only receipt, without metadata access."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re


def read_vm_receipt(path, *, zone, machine_type, instance_id=None, current_boot=None, now=None):
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    actual_boot = current_boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    observed = datetime.fromisoformat(receipt['observed_utc'])
    if (receipt.get('schema_version') != 1 or receipt.get('boot_id') != actual_boot
            or observed.tzinfo is None or observed > (now or datetime.now(timezone.utc)) + timedelta(seconds=5)):
        raise ValueError('Host receipt does not belong to this boot')
    identity = receipt['instance']
    if (set(identity) != {'id', 'zone', 'machine-type'}
            or not re.fullmatch('[0-9]+', identity['id'])
            or identity['zone'].split('/')[-1] != zone
            or identity['machine-type'].split('/')[-1] != machine_type
            or (instance_id and identity['id'] != str(instance_id))):
        raise ValueError('Host cloud identity differs from deployment')
    return dict(identity, boot_id=actual_boot)
