import pytest

from scripts.study_operator.cli import parser


@pytest.mark.parametrize('argv,operation', [(['status'],'status'), (['diagnostics'],'diagnostics'), (['start','--purpose','study'],'start'),
    (['invite','P01'],'invite'), (['invite','P999','--cell','1','--profile','without_experience'],'invite'),
    (['withdraw','P01'],'withdraw'), (['withdraw','P01','--execute'],'withdraw'),
    (['purge-study'],'purge-study'), (['archive-local'],'archive-local'),
    (['archive-local','--execute'],'archive-local'), (['ip-reserve'],'ip-reserve'), (['ip-release'],'ip-release'),
    (['tls-prepare','--first-session','2026-11-01'],'tls-prepare'),
    (['failover','--zone','us-central1-c'],'failover'), (['failback'],'failback')])
def test_human_operator_commands_parse_without_extra_information(argv,operation):
    result = parser().parse_args(argv)
    assert result.operation == operation
    if operation in {'withdraw','purge-study','archive-local'}:
        assert result.execute == ('--execute' in argv)


def test_cloud_region_and_profiles_are_restricted():
    for argv in [['failover','--zone','europe-west4-a'],['invite','P999','--profile','invented']]:
        with pytest.raises(SystemExit):
            parser().parse_args(argv)


def test_operator5_root_and_api_listed_us_zone_parse():
    result = parser().parse_args(['failover', '--zone', 'us-west4-c'])
    assert result.root == 'C:/CloudRAG/operator-iteration5'
    assert result.zone == 'us-west4-c'


def test_bootstrap_configuration_and_private_iap_parse():
    assert parser().parse_args(['configure-bootstrap', '--config', 'derived.json']).config == 'derived.json'
    for operation in ('iap-prepare', 'iap-release'):
        assert parser().parse_args([operation]).operation == operation
