from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.capacity_order import next_round, order, zones


def catalog():
    return [dict(name='nvidia-l4', zone='zones/'+zone) for zone in
            ('us-central1-a', 'us-east1-b', 'us-west1-a', 'europe-west4-a')]


def endpoints():
    return {region: dict(Region=region, URL='https://'+region+'-example-uc.a.run.app')
            for region in ('us-central1', 'us-east1', 'us-west1')}


def test_api_catalog_us_only_measured_median_order_and_five_positions():
    def measure(url):
        assert url.endswith('/api/ping')
        return 80 if 'central1' in url else 100 if 'east1' in url else 50

    result = order(catalog(), endpoints(), measure=measure)
    assert result['after_central_rounds_order'] == ['us-west1', 'us-east1']
    assert all(len(row['samples']) == 5 for row in result['results'])
    assert result['available_stock_not_inferred']
    assert set(zones(catalog())) == {'us-central1', 'us-west1', 'us-east1'}


def test_failed_samples_are_retained_and_do_not_rank_region_as_fastest():
    def measure(url):
        if 'west1' in url:
            raise OSError('CLIENT_ADDRESS_PRIVATE_CANARY')
        return 80

    result = order(catalog(), endpoints(), measure=measure)
    assert result['unresolved_regions'] == ['us-west1']
    assert result['after_central_rounds_order'] == ['us-east1']
    assert 'PRIVATE_CANARY' not in str(result)


def test_unknown_or_plaintext_endpoint_and_duplicate_zones_are_rejected():
    values = endpoints()
    values['us-west1']['URL'] = 'http://unreviewed.example'
    with pytest.raises(ValueError, match='contract'):
        order(catalog(), values, measure=lambda _: 50)
    with pytest.raises(ValueError, match='Duplicate'):
        zones(catalog()+[catalog()[0]])


def test_round_spacing_and_maximum_are_independent_of_conversation_pause():
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    first = [dict(at=now.isoformat())]
    assert next_round([], now) == 1
    with pytest.raises(ValueError, match='45 minutes'):
        next_round(first, now+timedelta(minutes=44, seconds=59))
    assert next_round(first, now+timedelta(minutes=45)) == 2
    with pytest.raises(ValueError, match='exhausted'):
        next_round(first*3, now+timedelta(hours=3))
