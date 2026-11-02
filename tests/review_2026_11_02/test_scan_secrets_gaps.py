"""Reproductores de la auditoría 2026-11-02: R-03 (S1), huecos de la capa 1 (scripts/scan_secrets.py).

Valores sintéticos construidos por concatenación para que el propio escáner no los marque en este
archivo. xfail(strict=True) hasta aplicar docs/review/2026-11-02/patches/003-*.patch (D-04).
"""
import pytest

from scripts import scan_secrets

KEY_HEADER = "-----BEGIN " + "PRIVATE KEY-----\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\n"

CASES = {
    "pem_file_skipped": ("deploy.pem", KEY_HEADER),
    "key_file_skipped": ("server.key", KEY_HEADER),
    "credentials_json_skipped": ("credentials.json", '{"private_key": "' + KEY_HEADER.replace("\n", "\\n") + '"}'),
    "env_variant_skipped": (".env.production", "DB_PASSWORD=" + "hunter2" * 3 + "\n"),
    "gcp_oauth_token": ("notes.md", "token: ya29." + "a0AfH6SMB" + "x" * 60 + "\n"),
    "google_api_key": ("notes.md", "key = AIza" + "SyD" + "x" * 32 + "\n"),
    "hf_token": ("notes.md", "HF_TOKEN=hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345" + "\n"),
    "db_password_suffix": ("settings.py", "db_password = '" + "hunter2" * 3 + "'\n"),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.xfail(strict=True, reason="R-03 (S1): la capa 1 no detecta este caso o lo omite en silencio")
def test_layer1_reports_case(tmp_path, case):
    name, content = CASES[case]
    root = tmp_path / "repo"
    root.mkdir()
    (root / name).write_text(content, encoding="utf-8")
    assert scan_secrets.main(["--root", str(root)]) != 0


def test_layer1_control_plain_password_is_detected(tmp_path):
    """Control: el patrón existente sí bloquea (el fallo de arriba no es del arnés del test)."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "settings.py").write_text("password = '" + "hunter2" * 3 + "'\n", encoding="utf-8")
    assert scan_secrets.main(["--root", str(root)]) != 0
