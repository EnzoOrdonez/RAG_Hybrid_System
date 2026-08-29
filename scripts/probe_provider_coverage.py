"""Describe provider coverage using frozen exp17 retrieval selections only.

This probe never runs retrieval. It recomputes strict coverage from the provider labels
persisted in exp17, which were derived from ``chunk_map[chunk_id].cloud_provider``. The
original exp17 ``*_cov`` flags were conditional on providers available in the candidate
pool; this audit instead requires every provider mentioned by the evaluation query.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data" / "evaluation" / "test_queries.json"
DEFAULT_RETRIEVAL = (
    ROOT / "experiments" / "results" / "exp17_crosscloud_balanced" / "retrieval_ids.json"
)
DEFAULT_OUTPUT_JSON = ROOT / "output" / "audit" / "provider_coverage_probe.json"
DEFAULT_OUTPUT_MD = ROOT / "output" / "audit" / "provider_coverage_probe.md"
ARMS = {"baseline": "base_prov", "balanced": "bal_prov"}
PROVIDERS = ("aws", "azure", "gcp")
PROVIDER_PATTERNS = {
    "aws": re.compile(r"\b(?:aws|amazon web services)\b", re.IGNORECASE),
    "azure": re.compile(r"\bazure\b", re.IGNORECASE),
    "gcp": re.compile(r"\b(?:gcp|google cloud|gke)\b", re.IGNORECASE),
}


class ProbeDataError(ValueError):
    """The frozen inputs lack information needed for this descriptive probe."""


def _mentioned_providers(query: Mapping) -> list[str]:
    question = str(query.get("question", ""))
    declared = {
        str(provider).strip().lower()
        for provider in query.get("cloud_providers", [])
        if str(provider).strip().lower() in PROVIDERS
    }
    detected = {
        provider for provider, pattern in PROVIDER_PATTERNS.items() if pattern.search(question)
    }
    return [provider for provider in PROVIDERS if provider in declared | detected]


def _multicloud_queries(queries: Sequence[Mapping]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for query in queries:
        qid = str(query.get("query_id", "")).strip()
        providers = _mentioned_providers(query)
        says_across = bool(re.search(r"\bacross\b", str(query.get("question", "")), re.I))
        if len(providers) >= 2 or says_across:
            if not qid:
                raise ProbeDataError("A multi-cloud query has no query_id")
            if len(providers) < 2:
                raise ProbeDataError(
                    f"{qid} says 'across' but fewer than two providers can be identified"
                )
            selected[qid] = {
                "question": query.get("question", ""),
                "providers": providers,
            }
    return selected


def _analyse_arm(
    *,
    query_rows: Mapping[str, Mapping],
    retrieval_rows: Mapping[str, Mapping],
    provider_field: str,
) -> tuple[dict, list[dict]]:
    full_coverage = 0
    chunks_total = 0
    chunks_outside = 0
    per_query = []
    for qid, query in query_rows.items():
        if qid not in retrieval_rows:
            raise ProbeDataError(f"Frozen retrieval has no row for multi-cloud query {qid}")
        raw_providers = retrieval_rows[qid].get(provider_field)
        if not isinstance(raw_providers, list) or not raw_providers:
            raise ProbeDataError(f"{qid} has no persisted provider list in {provider_field}")
        providers = [str(provider).strip().lower() for provider in raw_providers]
        if any(not provider for provider in providers):
            raise ProbeDataError(f"{qid} contains an empty provider label in {provider_field}")

        wanted = set(query["providers"])
        covered = wanted.issubset(providers)
        outside = sum(provider not in wanted for provider in providers)
        full_coverage += int(covered)
        chunks_total += len(providers)
        chunks_outside += outside
        per_query.append(
            {
                "query_id": qid,
                "mentioned_providers": query["providers"],
                "chunk_providers": providers,
                "full_provider_coverage": covered,
                "chunks_outside_query_providers": outside,
            }
        )

    n_queries = len(query_rows)
    return (
        {
            "queries_with_full_provider_coverage": full_coverage,
            "provider_coverage_fraction": round(full_coverage / n_queries, 4),
            "chunks_total": chunks_total,
            "chunks_outside_query_providers": chunks_outside,
            "outside_provider_chunk_rate": round(chunks_outside / chunks_total, 4),
        },
        per_query,
    )


def render_markdown(report: Mapping) -> str:
    lines = [
        "# Probe descriptivo de cobertura por proveedor",
        "",
        "Este probe es descriptivo y queda fuera de toda familia de contraste/BH. No "
        "reejecuta retrieval: usa las selecciones congeladas de exp17.",
        "",
        f"Queries multi-nube detectadas: **{report['n_multicloud_queries']}**.",
        "",
        "| Brazo persistido | Cobertura estricta | Fracción | Chunks fuera de los proveedores de la query | Tasa |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm_name, metrics in report["arms"].items():
        lines.append(
            f"| {arm_name} | {metrics['queries_with_full_provider_coverage']}/"
            f"{report['n_multicloud_queries']} | {metrics['provider_coverage_fraction']:.4f} | "
            f"{metrics['chunks_outside_query_providers']}/{metrics['chunks_total']} | "
            f"{metrics['outside_provider_chunk_rate']:.4f} |"
        )
    lines += [
        "",
        "Cobertura estricta significa que el top-k contiene al menos un chunk de cada "
        "proveedor mencionado. La tasa de desajuste cuenta chunks cuyo `cloud_provider` "
        "persistido no pertenece al conjunto de proveedores de la query.",
        "",
        "Fuente: `experiments/results/exp17_crosscloud_balanced/retrieval_ids.json`; "
        "queries: `data/evaluation/test_queries.json`.",
    ]
    return "\n".join(lines) + "\n"


def run_probe(
    *,
    queries_path: Path = DEFAULT_QUERIES,
    retrieval_path: Path = DEFAULT_RETRIEVAL,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> dict:
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8-sig"))
    retrieval = json.loads(Path(retrieval_path).read_text(encoding="utf-8"))
    query_rows = _multicloud_queries(queries)
    if not query_rows:
        raise ProbeDataError("No multi-cloud queries were detected")
    persisted_rows = {
        str(row.get("qid", "")): row for row in retrieval.get("per_query", [])
    }

    arms = {}
    per_query = {}
    for arm_name, provider_field in ARMS.items():
        arms[arm_name], per_query[arm_name] = _analyse_arm(
            query_rows=query_rows,
            retrieval_rows=persisted_rows,
            provider_field=provider_field,
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generated_by": "scripts/probe_provider_coverage.py",
        "descriptive_only": True,
        "bh_family": None,
        "queries_source": str(Path(queries_path)),
        "retrieval_source": str(Path(retrieval_path)),
        "retrieval_experiment": retrieval.get("experiment_id"),
        "n_multicloud_queries": len(query_rows),
        "arms": arms,
        "per_query": per_query,
    }
    Path(output_json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    Path(output_md).write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    try:
        report = run_probe(
            queries_path=args.queries,
            retrieval_path=args.retrieval,
            output_json=args.output_json,
            output_md=args.output_md,
        )
    except (FileNotFoundError, json.JSONDecodeError, ProbeDataError) as exc:
        print(f"BLOQUEO: información persistida insuficiente: {exc}")
        return 2
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
