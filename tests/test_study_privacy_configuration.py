import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_free_query_explicitly_excludes_personal_and_employer_confidential_data():
    config = json.loads((ROOT / 'config/study.example.json').read_text(encoding='utf-8'))
    assert config['free_instruction'] == (
        'Ahora escribe tú una consulta sobre algo de AWS, Azure o GCP que te interese o que hayas necesitado buscar alguna vez. '
        'Escríbela preferentemente en inglés, el idioma de la documentación. '
        'No incluyas datos personales ni información confidencial de tu empleador.'
    )


def test_csrf_and_cors_protection_enabled_without_usage_telemetry():
    config = tomllib.loads((ROOT / '.streamlit/config.toml').read_text(encoding='utf-8'))
    assert config['server']['enableXsrfProtection'] is True
    assert config['server']['enableCORS'] is True
    assert config['browser']['gatherUsageStats'] is False
