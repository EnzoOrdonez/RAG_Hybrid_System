import hashlib
import json

import pytest

from scripts.locked_vendor_requirements import REVISION, VCS_LINE, prepare


def fixture(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "module.py").write_text("value = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                revision=REVISION,
                files={
                    "module.py": hashlib.sha256(
                        (vendor / "module.py").read_bytes()
                    ).hexdigest()
                },
            )
        )
    )
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "# original lock\n" + VCS_LINE + "\nnumpy==2.4.2\ntorch==2.10.0+cu126\n"
    )
    return lock, vendor, manifest, tmp_path / "output.txt"


def test_only_source_transport_changes_and_original_lock_survives(tmp_path):
    lock, vendor, manifest, output = fixture(tmp_path)
    before = lock.read_bytes()
    prepare(lock, vendor, manifest, output)
    assert lock.read_bytes() == before
    assert output.read_text().splitlines() == [
        "# original lock",
        "-e " + vendor.as_posix(),
        "numpy==2.4.2",
        "torch==2.10.0+cu126",
    ]
    with pytest.raises(FileExistsError):
        prepare(lock, vendor, manifest, output)


@pytest.mark.parametrize("fault", ["revision", "content", "extra", "missing", "lock"])
def test_vendor_tampering_never_produces_installable_lock(tmp_path, fault):
    lock, vendor, manifest, output = fixture(tmp_path)
    if fault == "revision":
        manifest.write_text(manifest.read_text().replace(REVISION, "0" * 40))
    elif fault == "content":
        (vendor / "module.py").write_text("value = 2\n")
    elif fault == "extra":
        (vendor / "extra.py").write_text("unexpected")
    elif fault == "missing":
        data = json.loads(manifest.read_text())
        data["files"]["missing.py"] = "0" * 64
        manifest.write_text(json.dumps(data))
    else:
        lock.write_text(lock.read_text().replace(VCS_LINE, "-e untrusted"))
    with pytest.raises(ValueError):
        prepare(lock, vendor, manifest, output)
    assert not output.exists()
