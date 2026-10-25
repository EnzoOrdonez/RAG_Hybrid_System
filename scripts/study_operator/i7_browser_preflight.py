"""Read-only owner browser admission, before constructing any cloud client."""
import argparse
import json
import os
from pathlib import Path

from scripts.study_operator.i7_browser import PIN, runtime_identity


def verify():
    try:
        from playwright.sync_api import sync_playwright
        import playwright
        location=os.environ.get('PLAYWRIGHT_BROWSERS_PATH','')
        root=Path(location)
        if not location or not root.is_absolute() or not root.is_dir():
            raise ValueError('Explicit absolute PLAYWRIGHT_BROWSERS_PATH required')
        pin=json.loads(PIN.read_bytes())
        manifest=json.loads((Path(playwright.__file__).parent/'driver/package/browsers.json').read_bytes())
        chromium=next(row for row in manifest['browsers'] if row['name']=='chromium')
        if chromium['revision']!=pin['chromium_revision']:
            raise ValueError('Chromium revision differs from pin')
        with sync_playwright() as driver:
            executable=Path(driver.chromium.executable_path).resolve()
            if not executable.is_relative_to(root.resolve()/('chromium-'+pin['chromium_revision'])):
                raise ValueError('Chromium path outside pinned revision directory')
            identity=runtime_identity(driver)
        return dict(identity,status='PINNED_BROWSER_READY',chromium_revision=chromium['revision'],
                    executable_path=str(executable),browsers_path=str(root.resolve()))
    except (ImportError,OSError,ValueError,KeyError,StopIteration) as error:
        raise ValueError('PINNED_BROWSER_RUNTIME_REQUIRED') from error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result=verify()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    main()
