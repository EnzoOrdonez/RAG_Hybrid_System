"""Read-only equality check against the downloaded I7 image package census."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/'requirements-lock.txt'
RECIPE_PIN = ROOT/'config/i7-build-recipe-pin.json'
RECIPE_FILES = {'Dockerfile', 'requirements-lock.txt', 'scripts/locked_vendor_requirements.py',
                'scripts/study_operator/build_context.py', 'scripts/study_operator/linux_build.py'}


def _pin(text):
    pins = re.findall(r'^streamlit==([^\s;]+)\s*$', text, flags=re.MULTILINE | re.IGNORECASE)
    if len(pins) != 1:
        raise ValueError('STREAMLIT_EXACT_PIN_REQUIRED')
    return pins[0]


def verify_recipe(source=None):
    """Check exact LF source bytes, including the lock consumed by Docker."""
    try:
        pin = json.loads(RECIPE_PIN.read_bytes())
        source = Path(source) if source is not None else ROOT
        actual = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in RECIPE_FILES}
        if set(pin['files']) != RECIPE_FILES or actual != pin['files']:
            raise ValueError('Recipe changed')
        return {'recipe_sha256':hashlib.sha256(RECIPE_PIN.read_bytes()).hexdigest(), 'files':actual}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError('I7_BUILD_RECIPE_HASH_MISMATCH') from error


def verify_before_cpu():
    try:
        owner, lock = version('streamlit'), _pin(LOCK.read_text(encoding='utf-8'))
    except (OSError, ValueError, PackageNotFoundError) as error:
        raise ValueError('STREAMLIT_OWNER_LOCK_REQUIRED') from error
    if owner != lock:
        raise ValueError('STREAMLIT_VERSION_MISMATCH')
    return dict(verify_recipe(), status='STREAMLIT_BUILD_INPUTS_MATCH',
                owner_version=owner, lock_version=lock)


def verify(root):
    """Never infer installed image packages from the source requirements file."""
    try:
        owner = version('streamlit')
        lock = _pin(LOCK.read_text(encoding='utf-8'))
        root = Path(root)
        installation = json.loads((root/'installation.json').read_bytes())
        proof = json.loads((root/'cpu/linux-build-finali7-download-proof.json').read_bytes())
        if (installation['iteration'] != 7
                or proof['status'] != 'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED'
                or proof['image_id'] != installation['image_id']
                or proof['commit'] != installation['commit']
                or not re.fullmatch(r'sha256:[a-f0-9]{64}', proof['image_id'])):
            raise ValueError('I7_IMAGE_BUILD_IDENTITY_REQUIRED')
        directory = root/'cpu/linux-build-finali7-evidence'
        files = {}
        for name in ('packages.stdout', 'packages.receipt.json'):
            rows = [r for r in proof['files'] if r['object'].split('/')[-1] == name]
            data = (directory/name).read_bytes()
            if (len(rows) != 1 or len(data) != rows[0]['bytes']
                    or hashlib.sha256(data).hexdigest() != rows[0]['sha256']):
                raise ValueError('I7_IMAGE_PACKAGE_CENSUS_CHANGED')
            files[name] = data
        receipt = json.loads(files['packages.receipt.json'])
        command = receipt['command']
        if (receipt['exit_code'] != 0 or command[:2] != ['docker','run']
                or command[-4:] != [proof['image_id'],'-m','pip','freeze']):
            raise ValueError('I7_IMAGE_PACKAGE_CENSUS_FAILED')
        image = _pin(files['packages.stdout'].decode('utf-8'))
        if owner != lock or image != lock:
            raise ValueError('STREAMLIT_VERSION_MISMATCH')
        return {'status':'STREAMLIT_VERSIONS_MATCH', 'owner_version':owner,
                'lock_version':lock, 'image_version':image, 'image_id':proof['image_id'],
                'packages_sha256':hashlib.sha256(files['packages.stdout']).hexdigest()}
    except (OSError, TypeError, ValueError, KeyError, PackageNotFoundError) as error:
        if isinstance(error, ValueError) and str(error) == 'STREAMLIT_VERSION_MISMATCH':
            raise
        raise ValueError('QUALIFIED_I7_STREAMLIT_EVIDENCE_REQUIRED') from error
