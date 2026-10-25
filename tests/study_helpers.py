"""Technical study fixtures; SUS placeholders are never proposed as validated wording."""
import csv
import json

from src.ui.components.study_protocol import ROOT, QUOTAS, PROFILES, load_protocol


def configured(tmp_path):
    config = json.loads((ROOT / 'config/study.example.json').read_text(encoding='utf-8'))
    config['labels'] = {'A': 'hybrid', 'B': 'no_rag'}
    config_path, csv_path = tmp_path / 'config.json', tmp_path / 'assignments.csv'
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['participant_id', 'role', 'cell', 'profile'])
        index = 1
        for cell, quota in QUOTAS.items():
            for profile, n in zip(PROFILES, quota):
                for _ in range(n):
                    writer.writerow([f'P{index:02d}', 'primary', cell, profile])
                    index += 1
        writer.writerow(['P21', 'reserve', 1, PROFILES[0]])
    return config_path, csv_path, load_protocol(config_path, csv_path)
