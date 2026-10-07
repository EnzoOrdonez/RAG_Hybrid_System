import io
import json

import pytest

from scripts.study_operator.pricing import candidates, download, machine_quote, unit_rate, disk_quote_archive


def sku():
    return dict(skuId='fixture', description='E2 Instance Core running in Americas',
                serviceRegions=['us-central1'], category=dict(usageType='OnDemand'),
                pricingInfo=[dict(effectiveTime='2026-10-01T00:00:00Z', pricingExpression=dict(
                    usageUnit='h', tieredRates=[dict(startUsageAmount=0, unitPrice=dict(currencyCode='USD', nanos=21000000))]))])


def test_decimal_rates_and_no_spot_or_wrong_region():
    row = sku()
    assert unit_rate(row)['usd_per_usage_unit'] == '0.021'
    assert len(candidates([row])) == 1
    assert candidates([dict(row, category=dict(usageType='Preemptible'))]) == []
    assert candidates([dict(row, serviceRegions=['europe-west1'])]) == []
    row['pricingInfo'][0]['pricingExpression']['tieredRates'].append(dict(startUsageAmount=10))
    with pytest.raises(ValueError, match='tiered'):
        unit_rate(row)


def test_pagination_raw_sha_and_credentials_not_persisted(tmp_path):
    class Cloud:
        def owner_token(self):
            return 'PRIVATE_FIXTURE_SECRET'

    calls = []
    def open_url(request, **kwargs):
        assert request.get_header('Authorization') == 'Bearer PRIVATE_FIXTURE_SECRET'
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(dict(skus=[sku()], nextPageToken='second' if len(calls) == 1 else '')).encode())

    root = tmp_path/'catalog'
    result = download(Cloud(), root, open_url=open_url)
    assert result['skus'] == 2 and len(result['pages']) == 2
    assert 'pageToken=second' in calls[1]
    assert all('PRIVATE_FIXTURE_SECRET' not in path.read_text() for path in root.iterdir())
    assert all(len(page['sha256']) == 64 for page in result['pages'])
    with pytest.raises(FileExistsError):
        download(Cloud(), root, open_url=open_url)


def test_machine_quote_selects_standard_sk_us_and_units_only():
    core = sku()
    core['category']['resourceGroup'] = 'CPU'
    ram = json.loads(json.dumps(core))
    ram.update(skuId='ram', description='E2 Instance Ram running in Americas')
    ram['category']['resourceGroup'] = 'RAM'
    ram['pricingInfo'][0]['pricingExpression'].update(usageUnit='GiBy.h')
    ram['pricingInfo'][0]['pricingExpression']['tieredRates'][0]['unitPrice']['nanos'] = 3000000
    quote = machine_quote([core, ram], 'e2-standard-2', 'us-central1')
    assert quote['usd_per_hour'] == '0.066'  # 2*.021 + 8*.003
    with pytest.raises(ValueError, match='Ambiguous'):
        machine_quote([core, core, ram], 'e2-standard-2', 'us-central1')
    with pytest.raises(ValueError, match='US'):
        machine_quote([core, ram], 'e2-standard-2', 'europe-west1')


def test_regional_disk_quote_keeps_monthly_unit_and_rejects_ambiguity(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.study_operator.pricing.quote_archive',lambda *a:dict(catalog_receipt_sha256='c'*64))
    row=sku()
    row.update(description='Balanced PD Capacity in Northern Virginia',serviceRegions=['us-east4'])
    row['pricingInfo'][0]['pricingExpression']['usageUnit']='GiBy.mo'
    page=tmp_path/'page-000.json'
    page.write_text(json.dumps(dict(skus=[row])))
    result=disk_quote_archive(tmp_path,'us-east4')
    assert result['usage_unit']=='GiBy.mo' and result['estimated_month_hours']==730
    assert float(result['estimated_usd_gib_h'])==pytest.approx(.021/730)
    with pytest.raises(ValueError,match='missing'):
        disk_quote_archive(tmp_path,'us-west1')
    page.write_text(json.dumps(dict(skus=[row,row])))
    with pytest.raises(ValueError,match='Ambiguous'):
        disk_quote_archive(tmp_path,'us-east4')
