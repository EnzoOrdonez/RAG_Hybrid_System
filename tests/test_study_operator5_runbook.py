from pathlib import Path
import shlex

from scripts.study_operator.cli import parser


RUNBOOK = Path(__file__).resolve().parents[1]/'scripts/study_operator/RUNBOOK_ITERATION5.md'


def test_each_literal_operator_command_is_accepted_by_the_installed_cli_parser():
    substitutions = {'$config.zone':'us-west1-a', '$alternateZone':'us-central1-b',
        '$participantCode':'P999', '$firstSession':'2026-10-20', '$sessionId':'synthetic-session',
        '$fullGeneration':'123', '$manifestGeneration':'456', '$bootIndex':'1'}
    assert '$bootIndex = 1' in RUNBOOK.read_text(encoding='utf-8')
    operations = set()
    for line in RUNBOOK.read_text(encoding='utf-8').splitlines():
        prefix = '& C:/CloudRAG/operator-iteration5/operator.ps1 '
        if not line.startswith(prefix):
            continue
        values = [substitutions.get(value, value) for value in shlex.split(line[len(prefix):])]
        if values == ['--help']:
            continue
        result = parser().parse_args(values)
        operations.add(result.operation)
    assert {'bootstrap','iap-prepare','iap-release','ip-reserve','tls-prepare','start','preflight','invite',
        'diagnostics','stop','status','revoke','restore','export-anonymized','withdraw','purge-study',
        'archive-local','failover','failback','ip-release','stimulus-start','stimulus-status',
        'stimulus-collect'} <= operations


def test_runbook_preserves_ethics_privacy_and_actual_iteration5_paths():
    text = ' '.join(RUNBOOK.read_text(encoding='utf-8').split())
    assert 'C:/CloudRAG/operator-iteration4/' not in text
    assert 'C:/CloudRAG/operator-iteration3/' not in text
    assert 'no crea ese registro ni invita a personas' in text
    assert 'UEQ-S inmediatamente' in text
    assert 'soft delete=0' in text and 'historial de auditoría sin cambios' in text
    assert 'SIN transcripción' in text and 'No repitas `bootstrap`' in text
    assert 'pendiente de ensayo literal completo' in text
