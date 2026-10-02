"""Literal corpus inspection and fail-closed resealing; never runs retrieval or LLMs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.components.study_protocol import ROOT, digest, load_protocol, verify_draw

OFFICIAL = {
    "aws": ("docs.aws.amazon.com",),
    "azure": ("learn.microsoft.com",),
    "gcp": ("cloud.google.com", "docs.cloud.google.com"),
}


def literal_search(chunks, terms, provider=None):
    if not terms or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("Explicit nonempty literal terms required")
    return [
        dict(
            chunk_id=cid,
            provider=c["cloud_provider"],
            service=c["service_name"],
            url=c["url_source"],
            text=c["text"],
        )
        for cid, c in sorted(chunks.items())
        if (provider is None or c["cloud_provider"] == provider)
        and all(t.casefold() in c["text"].casefold() for t in terms)
    ]


def validate_evidence(chunks, protocol, evidence):
    ids = protocol["config"]["tasks"]["T1"] + protocol["config"]["tasks"]["T2"]
    if evidence.get("method") != "literal-corpus-review-v1" or set(
        evidence.get("tasks", {})
    ) != set(ids):
        raise ValueError("Evidence must cover exactly all six selected tasks")
    if evidence.get("corpus_sha256") != digest(
        ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
    ):
        raise ValueError("Corpus evidence identity mismatch")
    for qid in ids:
        item = evidence["tasks"][qid]
        if (
            item.get("verdict") != "DIRECT_ANSWER"
            or not item.get("rationale")
            or not item.get("fragments")
        ):
            raise ValueError("Missing reviewed direct answer: " + qid)
        providers = set()
        for fragment in item["fragments"]:
            chunk = chunks[fragment["chunk_id"]]
            excerpt = fragment.get("excerpt", "")
            if not excerpt.strip() or excerpt not in chunk["text"]:
                raise ValueError("Evidence excerpt is not literal: " + qid)
            provider = chunk["cloud_provider"]
            if urlsplit(chunk["url_source"]).hostname not in OFFICIAL[provider]:
                raise ValueError("Evidence is not official provider documentation")
            providers.add(provider)
        if not set(protocol["queries"][qid]["cloud_providers"]) <= providers:
            raise ValueError(
                "Each task provider needs direct official evidence: " + qid
            )
    for index in range(3):
        pair_ids = [protocol["config"]["tasks"][name][index] for name in ("T1", "T2")]
        if set(protocol["queries"][pair_ids[0]]["cloud_providers"]) != set(
            protocol["queries"][pair_ids[1]]["cloud_providers"]
        ):
            raise ValueError("Task pair providers changed")
        pair = [
            evidence["tasks"][protocol["config"]["tasks"][name][index]]
            for name in ("T1", "T2")
        ]
        if not pair[0].get("template") or pair[0]["template"] != pair[1].get(
            "template"
        ):
            raise ValueError("Task pair template mismatch")


def reseal(original, destination, tasks, evidence):
    """Create a separate reviewed revision; preserve labels, assignments and old seal."""
    original, destination = Path(original), Path(destination)
    old = verify_draw(original)
    if destination.exists():
        raise FileExistsError("Never overwrite an existing seal")
    chunks = json.loads(
        (ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json").read_text(
            encoding="utf-8"
        )
    )
    config = dict(old["config"], tasks=tasks)
    # Validate structurally and semantically before publishing a new seal.
    candidate = dict(old, config=config)
    validate_evidence(chunks, candidate, evidence)
    destination.mkdir(parents=True)
    evidence_path = destination / "task_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config["task_evidence_sha256"] = digest(evidence_path)
    source_csv = original / "assignments.csv"
    shutil.copyfile(source_csv, destination / "assignments.csv")
    if digest(source_csv) != digest(destination / "assignments.csv"):
        raise ValueError("Assignment copy differs")
    (destination / "study.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    new = load_protocol(destination / "study.json", destination / "assignments.csv")
    seal = dict(
        schema_version=1,
        seeds=config["randomization"],
        hashes=new["hashes"],
        fingerprint=new["fingerprint"],
    )
    (destination / "draw_seal.json").write_text(
        json.dumps(seal, indent=2), encoding="utf-8"
    )
    (destination / "task_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return verify_draw(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", action="append", required=True)
    parser.add_argument("--provider", choices=tuple(OFFICIAL))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
    chunks = json.loads(path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT.parent.parent):
        parser.error("Write evidence outside the checkout")
    with output.open("x", encoding="utf-8") as f:
        json.dump(
            dict(
                method="literal-corpus-review-v1",
                corpus_sha256=digest(path),
                terms=args.term,
                matches=literal_search(chunks, args.term, args.provider),
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
