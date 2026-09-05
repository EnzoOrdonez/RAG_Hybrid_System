"""Offline bundle manifest generation/verification. Does not download or load models."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.deployment_artifacts import build_manifest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        # Output must be outside protected data; create exclusively, never overwrite.
        destination = args.manifest.resolve()
        for directory in ("data", "experiments", "output", "paper"):
            if destination.is_relative_to((args.root / directory).resolve()):
                parser.error("Write the manifest outside data/evidence directories")
        manifest = build_manifest(args.root)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    else:
        print(verify_manifest(args.root, args.manifest))


if __name__ == "__main__":
    main()
