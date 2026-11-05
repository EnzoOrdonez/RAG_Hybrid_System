"""P3 / R-16: inventory of identifying data in tracked files (emails, IPs, GCP project ids, personal paths,
hostnames, service accounts, token-like strings). Reports counts per category and file; never prints the
full value of anything token-like (first 4 chars only). Usage: python inventory_identifiers.py OUT.json [REV]
"""
import ipaddress, json, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CATS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'ipv4': re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])'),
    'gcp_project_id': re.compile(r'\b[a-z][a-z-]{4,28}-\d{6}-[a-z0-9]{2}\b'),   # GCP auto-generated project id shape
    'service_account': re.compile(r'\b[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com\b'),
    'windows_user_path': re.compile(r'(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s"\']+'),
    'posix_home_path': re.compile(r'/(?:home|Users)/[A-Za-z0-9._-]+'),
    'sslip_host': re.compile(r'\b(?:\d{1,3}[.-]){3}\d{1,3}\.sslip\.io\b'),
    'google_oauth_token': re.compile(r'\bya29\.[A-Za-z0-9_-]{20,}'),
    'google_api_key': re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    'hf_token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
    'github_token': re.compile(r'\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,})\b'),
}
SAFE_IP = lambda ip: (ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_link_local or ip.is_reserved
                      or any(ip in ipaddress.ip_network(n) for n in ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24')))
SAFE_EMAIL = re.compile(r'(?i)@(?:example\.(?:com|org|net)|anthropic\.com|users\.noreply\.github\.com)$|^noreply@')
SEVERITY = {'google_oauth_token': 'S0', 'google_api_key': 'S0', 'hf_token': 'S0', 'github_token': 'S0',
            'service_account': 'S1', 'gcp_project_id': 'S2', 'email': 'S2', 'ipv4_public': 'S2',
            'sslip_host': 'S2', 'windows_user_path': 'S3', 'posix_home_path': 'S3'}

def main():
    out = Path(sys.argv[1]); rev = sys.argv[2] if len(sys.argv) > 2 else 'HEAD'
    names = subprocess.run(['git', '-C', str(REPO), 'ls-tree', '-r', '--name-only', rev], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    per_cat, per_file, distinct = Counter(), defaultdict(Counter), defaultdict(set)
    for name in names:
        try:
            text = subprocess.run(['git', '-C', str(REPO), 'show', f'{rev}:{name}'], capture_output=True,
                                  check=True).stdout.decode('utf-8')
        except (UnicodeDecodeError, subprocess.CalledProcessError):
            continue
        for cat, rx in CATS.items():
            for m in rx.finditer(text):
                value, key = m.group(0), cat
                if cat == 'ipv4':
                    try:
                        ip = ipaddress.ip_address(value)
                    except ValueError:
                        continue
                    if SAFE_IP(ip):
                        continue
                    key = 'ipv4_public'
                if cat == 'email' and (SAFE_EMAIL.search(value) or value.endswith('.iam.gserviceaccount.com')):
                    continue
                if cat == 'posix_home_path' and value in ('/home/user', '/Users/runner'):
                    continue
                per_cat[key] += 1; per_file[name][key] += 1
                distinct[key].add(value[:4] + '…')
    report = {'rev': rev, 'files_scanned': len(names),
              'by_category': {k: {'occurrences': v, 'files': sum(1 for f in per_file.values() if f[k]),
                                  'distinct_values': len(distinct[k]), 'severity': SEVERITY.get(k, 'S3')}
                              for k, v in per_cat.most_common()},
              'by_file': {f: dict(c) for f, c in sorted(per_file.items(), key=lambda kv: -sum(kv[1].values()))},
              'review_folders': {k: sum(c[k] for f, c in per_file.items() if f.startswith('docs/review/'))
                                 for k in per_cat}}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in report['by_category'].items():
        print(k, v)

if __name__ == '__main__':
    main()
