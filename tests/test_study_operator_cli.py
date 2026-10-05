import pytest

from scripts.study_operator.cli import parser


@pytest.mark.parametrize('argv,operation', [(['status'],'status'), (['start','--purpose','study'],'start'),
    (['invite','P01'],'invite'), (['invite','P999','--cell','1','--profile','without_experience'],'invite'),
    (['withdraw','P01'],'withdraw'), (['withdraw','P01','--execute'],'withdraw'),
    (['purge-study'],'purge-study'), (['ip-reserve'],'ip-reserve'), (['ip-release'],'ip-release'),
    (['tls-prepare','--first-session','2026-11-01'],'tls-prepare'),
    (['failover','--zone','us-central1-c'],'failover'), (['failback'],'failback')])
def test_human_operator_commands_parse_without_extra_information(argv,operation):
    result = parser().parse_args(argv)
    assert result.operation == operation
    if operation in {'withdraw','purge-study'}:
        assert result.execute == ('--execute' in argv)


def test_cloud_region_and_profiles_are_restricted():
    for argv in [['failover','--zone','us-east1-b'],['invite','P999','--profile','invented']]:
        with pytest.raises(SystemExit):
            parser().parse_args(argv)
