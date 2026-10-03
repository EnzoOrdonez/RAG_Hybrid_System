"""Change transport only for the exact private dependency in the immutable lock."""

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "5db68deb53e6e88a668670101d8372a5cff0dcf7"
VCS_LINE = (
    "-e git+https://github.com/EnzoOrdonez/thesis-paper-agents@"
    + REVISION
    + "#egg=thesis_paper_agents"
)


def prepare(lock, vendor, manifest, output):
    vendor = Path(vendor).resolve()
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if expected.get("revision") != REVISION or not expected.get("files"):
        raise ValueError("Vendor revision differs from the locked Git commit")
    actual = {
        p.relative_to(vendor).as_posix() for p in vendor.rglob("*") if p.is_file()
    }
    if actual != set(expected["files"]):
        raise ValueError("Vendor file inventory differs")
    for name, digest in expected["files"].items():
        path = (vendor / name).resolve()
        if not path.is_relative_to(vendor):
            raise ValueError("Vendor path escapes its snapshot")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Vendor content differs from the verified snapshot")
    lines = Path(lock).read_text(encoding="utf-8").splitlines()
    if lines.count(VCS_LINE) != 1:
        raise ValueError("Unexpected private dependency in the lock")
    result = ["-e " + vendor.as_posix() if line == VCS_LINE else line for line in lines]
    with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("lock", "vendor", "manifest", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(args.lock, args.vendor, args.manifest, args.output)
