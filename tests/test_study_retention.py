import copy

import pytest

from scripts.study_operator.retention import LISTS, ORIGINAL, census, digest, plan, projection


def listing():
    resources = dict(vms=[dict(id='1', name=ORIGINAL, status='TERMINATED', deletionProtection=True,
                               disks=[dict(source='disk-url', autoDelete=False)]),
                         dict(id='6', name='cloudrag-i5-cpu', status='TERMINATED')],
        disks=[dict(id='2', name=ORIGINAL, selfLink='disk-url', sizeGb='100'),
               dict(id='5', name='cloudrag-i5-cpu-boot', selfLink='cpu-disk-url', sourceSnapshotId='3')],
        snapshots=[dict(id='3', name='cloudrag-i4-final', status='READY', storageBytes=str(50*2**30)),
                   dict(id='4', name='cloudrag-i4-redundant', status='READY', storageBytes='1000')],
        ips=[], firewalls=[], subnets=[])
    return resources


def test_census_uses_actual_sdk_subnet_group_and_keeps_private_metadata_in_memory():
    calls, rows = [], listing()
    rows['vms'][0]['metadata'] = {'ssh-keys': 'PRIVATE_CLIENT_CANARY'}

    class Cloud:
        def command(self, command, **options):
            calls.append(command)
            assert options['private_output']
            for kind, expected in LISTS.items():
                if command == expected:
                    return rows[kind]
            return dict(quotas=[])

    result = census(Cloud(), {'historical': 'state'})
    assert ['compute', 'networks', 'subnets', 'list'] in calls
    assert ['compute', 'subnets', 'list'] not in calls
    assert 'PRIVATE_CLIENT_CANARY' not in str(result)
    assert result['protected_ids'] == dict(vm='1', disk='2')
    assert result['listing_sha256'] == digest(result['resources'])


def fixture_inventory():
    rows = listing()
    return dict(resources=rows, listing_sha256=digest(rows), protected_ids=dict(vm='1', disk='2'))


def restoration():
    return dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='3',
                all_expected_files_verified=True, image_config_verified=True, restored_disk_id='5', cpu_vm_id='6',
                model_manifest_and_blobs_verified=True, source=dict(files=19), artifacts=dict(files=79))


def inherited():
    return [dict(type='snapshot', id='4', name='cloudrag-i4-redundant', inherited=True)]


def test_no_deletion_plan_from_snapshot_name_or_synthetic_receipt():
    inv = fixture_inventory()
    for receipt in ({}, dict(restoration(), synthetic=True), dict(restoration(), source_snapshot_id='4')):
        with pytest.raises(ValueError, match='restoration'):
            plan(inv, '3', receipt, inherited_resources=inherited())
    result = plan(inv, '3', restoration(), inherited_resources=inherited())
    assert [row['id'] for row in result['removals']] == ['4']
    assert result['buckets_and_file_evidence_never_deleted']
    modified = copy.deepcopy(inv)
    modified['resources']['disks'][0]['id'] = 'different'
    with pytest.raises(ValueError, match='changed'):
        plan(modified, '3', restoration(), inherited_resources=inherited())


def test_unknown_resource_and_running_test_vm_are_never_disposed():
    inv = fixture_inventory()
    inv['resources']['vms'].append(dict(id='7', name='third-party', status='RUNNING'))
    inv['listing_sha256'] = digest(inv['resources'])
    with pytest.raises(ValueError, match='outside'):
        plan(inv, '3', restoration(), inherited_resources=inherited())
    inv['resources']['vms'][-1]['name'] = 'cloudrag-i4-test'
    inv['listing_sha256'] = digest(inv['resources'])
    with pytest.raises(ValueError, match='stopped'):
        plan(inv, '3', restoration(), inherited_resources=inherited()+[
            dict(type='vm', id='7', name='cloudrag-i4-test', inherited=True)])


@pytest.mark.parametrize('change', [dict(model_manifest_and_blobs_verified=False),
                                  dict(source=dict(files=18)), dict(artifacts=dict(files=78))])
def test_partial_or_model_unverified_receipt_cannot_qualify_retention(change):
    with pytest.raises(ValueError, match='restoration'):
        plan(fixture_inventory(), '3', dict(restoration(), **change), inherited_resources=inherited())


def test_matching_namespace_without_inherited_id_is_not_deletion_authority():
    with pytest.raises(ValueError, match='outside'):
        plan(fixture_inventory(), '3', restoration(), inherited_resources=[])


def test_forecast_keeps_margins_and_wait_separate_and_prices_session_disk():
    result = projection(fixture_inventory(), '3', pd_usd_gib_h=.000136986,
        snapshot_usd_gib_h=.000068493, bucket_usd_day=.006,
        initial_spend_usd=12, margins_usd=5, qualification_gpu_hours=12, gpu_usd_h=.706832255)
    assert result['estimated_post_cleanup_idle_usd_day'] < .45
    assert result['scenarios'][0]['below_own_cutoff']
    assert not result['scenarios'][1]['below_own_cutoff']
    assert not result['scenarios'][-1]['below_own_cutoff']
    assert result['scenarios'][0]['upper_with_margins_usd'] == result['scenarios'][0]['estimated_spend_usd']+5
    assert result['assumptions']['additional_disk_gib'] == 100
    assert result['assumptions']['associated_IP_session_days'] == 24
