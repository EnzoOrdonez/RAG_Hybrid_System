from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.study_operator.host_identity import read_vm_receipt


def receipt(tmp_path):
    value = dict(schema_version=1, boot_id='boot-fixture', observed_utc=datetime.now(timezone.utc).isoformat(),
                 instance={'id': '12345', 'zone': 'projects/123/zones/us-central1-a',
                           'machine-type': 'projects/123/machineTypes/g2-standard-4'})
    path = tmp_path / 'host.json'
    path.write_text(json.dumps(value))
    return path, value


def read(path):
    return read_vm_receipt(path, zone='us-central1-a', machine_type='g2-standard-4',
                           instance_id='12345', current_boot='boot-fixture')


def test_host_receipt_replaces_metadata_with_checked_current_boot(tmp_path):
    path, value = receipt(tmp_path)
    actual = read(path)
    assert actual == dict(value['instance'], boot_id='boot-fixture')


@pytest.mark.parametrize('field,value', [('id', '54321'), ('zone', 'us-central1-b'),
                                       ('machine-type', 'n1-standard-4')])
def test_wrong_live_vm_identity_rejected(tmp_path, field, value):
    path, record = receipt(tmp_path)
    record['instance'][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity'):
        read(path)


def test_prior_boot_and_future_observation_rejected(tmp_path):
    path, record = receipt(tmp_path)
    for changes in [dict(boot_id='previous-boot'), dict(observed_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())]:
        path.write_text(json.dumps(dict(record, **changes)))
        with pytest.raises(ValueError, match='boot'):
            read(path)
