import hashlib
import json

import pytest

from scripts.study_operator.inherited_inventory import collect


def test_freeze_inventory_is_read_only_and_refuses_changed_source(tmp_path):
    old, app, package = [tmp_path / name for name in ['old', 'app', 'package']]
    for root in [old, app, package]:
        root.mkdir()
    (app / 'frozen.py').write_bytes(b'frozen')
    (old / 'rag_freeze_baseline.json').write_text(json.dumps(dict(rag=dict(modules={
        'frozen.py': hashlib.sha256(b'frozen').hexdigest()}))))
    (package / 'baseline02-ruff.stdout').write_text(json.dumps([
        dict(filename=str(app / 'frozen.py'), code='F401')]))
    (tmp_path / 'iteration4_single_use.py').write_bytes(b'preserve')
    result = collect(old, package, app)
    assert result['ruff_counts'] == {'DECLARED_FROZEN_SOURCE': 1}
    assert len(result['oneoff_sources']) == 1
    assert (app / 'frozen.py').read_bytes() == b'frozen'
    (app / 'frozen.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen'):
        collect(old, package, app)
    assert (app / 'frozen.py').read_bytes() == b'changed'
