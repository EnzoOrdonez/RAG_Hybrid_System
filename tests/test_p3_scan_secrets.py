"""P3 / R-03: tokens que el escáner no reconocía, casos negativos y presencia de material de clave sin leerlo.

Los valores se construyen por concatenación para que este archivo no contenga ningún token literal.
"""
import pathlib
import subprocess
from pathlib import Path

import pytest

from scripts import scan_secrets as ss

POSITIVES = {
    'google_oauth_token': 'ya29.' + 'a0Ae' + 'X' * 60,
    'google_api_key': 'AIza' + 'S' * 35,
    'huggingface_token': 'hf_' + 'A' * 34,
    'github_fine_grained_token': 'github_pat_' + '1' * 22 + '_' + 'b' * 59,
    'github_app_or_oauth_token': 'gho_' + 'c' * 36,
    'slack_token': 'xox' + 'b-' + '1234567890-abcdef',
    'gcp_service_account_key_id': '"private_key_id": "' + 'f' * 40 + '"',
}
NEGATIVES = [
    'ya29.' + 'short',                         # demasiado corto
    'AIza' + 'S' * 34,                         # una letra menos
    'AIza' + 'S' * 36,                         # una letra más (no termina donde debe)
    'hf_' + 'A' * 20,                          # corto
    'the hf_hub cache path',                   # prosa
    'github_pat_' + 'x' * 10,                  # truncado
    'gho_' + 'c' * 35,                         # longitud distinta
    '"private_key_id": "' + 'f' * 39 + '"',     # no es un id de 40 hex
    'xox' + 'b-short',                         # corto
]


@pytest.mark.parametrize('name,token', sorted(POSITIVES.items()))
def test_new_cloud_token_patterns_are_detected_and_redacted(name, token):
    findings = ss.scan_text('value = %s\n' % token, Path('notes.md'))
    assert [f.pattern for f in findings] == [name]
    assert len(findings[0].preview) == 4 and findings[0].preview != token


@pytest.mark.parametrize('text', NEGATIVES)
def test_near_misses_are_not_reported(text):
    assert ss.scan_text(text + '\n', Path('notes.md')) == []


def git(cwd, *args):
    subprocess.run(['git', '-c', 'user.name=p3', '-c', 'user.email=p3', '-c', 'commit.gpgsign=false', *args],
                   cwd=cwd, check=True, capture_output=True)


def repo_with_key_material(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path/'.gitignore').write_text('.env\n')
    (tmp_path/'.env').write_text('IGNORED=1\n')                    # ignorado: nunca se lista
    (tmp_path/'deploy').mkdir()
    (tmp_path/'deploy'/'service.pem').write_text('not read\n')
    (tmp_path/'config').mkdir()
    (tmp_path/'config'/'.env.production').write_text('not read\n')
    (tmp_path/'README.md').write_text('ok\n')
    git(tmp_path, 'add', '-f', '.gitignore', 'deploy/service.pem', 'config/.env.production', 'README.md')
    git(tmp_path, 'commit', '-q', '-m', 'x')
    return tmp_path


@pytest.mark.parametrize('use_git', [True, False])
def test_key_material_is_reported_by_presence_without_being_opened(tmp_path, monkeypatch, use_git):
    root = repo_with_key_material(tmp_path)
    opened = []
    real_open = pathlib.Path.open

    def guarded(self, *args, **kwargs):
        if ss.is_key_material(Path(self.name)):
            opened.append(self.name)
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, 'open', guarded)
    findings = ss.scan_repository(root, use_git=use_git)
    keys = sorted(str(f.path) for f in findings if f.pattern == ss.KEY_MATERIAL_PATTERN)
    expected = ['config/.env.production', 'deploy/service.pem'] + ([] if use_git else ['.env'])
    assert keys == sorted(expected)
    assert opened == []                                           # el contenido nunca se lee
    assert all(f.preview == '' and f.line == 0 for f in findings if f.pattern == ss.KEY_MATERIAL_PATTERN)


def test_report_counts_key_material(tmp_path):
    findings = [ss.Finding(pattern=ss.KEY_MATERIAL_PATTERN, path=Path('a.pem'), line=0, preview='')]
    assert 'key_material_file: 1' in ss.format_report(findings)
