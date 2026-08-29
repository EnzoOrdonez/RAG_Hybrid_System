import json

from scripts import probe_provider_coverage


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_probe_reports_strict_coverage_and_provider_mismatch(tmp_path) -> None:
    queries_path = tmp_path / "queries.json"
    retrieval_path = tmp_path / "retrieval.json"
    output_json = tmp_path / "probe.json"
    output_md = tmp_path / "probe.md"
    _write_json(
        queries_path,
        [
            {
                "query_id": "q1",
                "question": "Compare AWS and Azure",
                "cloud_providers": ["aws", "azure"],
            },
            {
                "query_id": "q2",
                "question": "Compare services across AWS, Azure, and GCP",
                "cloud_providers": [],
            },
            {
                "query_id": "q3",
                "question": "Explain AWS Lambda",
                "cloud_providers": ["aws"],
            },
        ],
    )
    _write_json(
        retrieval_path,
        {
            "experiment_id": "fixture",
            "per_query": [
                {
                    "qid": "q1",
                    "base_prov": ["aws", "aws", "gcp"],
                    "bal_prov": ["aws", "azure"],
                },
                {
                    "qid": "q2",
                    "base_prov": ["aws", "azure"],
                    "bal_prov": ["aws", "azure", "gcp"],
                },
            ],
        },
    )

    report = probe_provider_coverage.run_probe(
        queries_path=queries_path,
        retrieval_path=retrieval_path,
        output_json=output_json,
        output_md=output_md,
    )

    assert report["n_multicloud_queries"] == 2
    assert report["arms"]["baseline"] == {
        "queries_with_full_provider_coverage": 0,
        "provider_coverage_fraction": 0.0,
        "chunks_total": 5,
        "chunks_outside_query_providers": 1,
        "outside_provider_chunk_rate": 0.2,
    }
    assert report["arms"]["balanced"] == {
        "queries_with_full_provider_coverage": 2,
        "provider_coverage_fraction": 1.0,
        "chunks_total": 5,
        "chunks_outside_query_providers": 0,
        "outside_provider_chunk_rate": 0.0,
    }
    assert output_json.exists()
    assert "0/2" in output_md.read_text(encoding="utf-8")
    assert "2/2" in output_md.read_text(encoding="utf-8")


def test_cli_declares_block_when_frozen_retrieval_lacks_a_query(tmp_path, capsys) -> None:
    queries_path = tmp_path / "queries.json"
    retrieval_path = tmp_path / "retrieval.json"
    output_json = tmp_path / "probe.json"
    output_md = tmp_path / "probe.md"
    _write_json(
        queries_path,
        [
            {
                "query_id": "q-missing",
                "question": "Compare AWS and Azure",
                "cloud_providers": ["aws", "azure"],
            }
        ],
    )
    _write_json(retrieval_path, {"experiment_id": "fixture", "per_query": []})

    exit_code = probe_provider_coverage.main(
        [
            "--queries",
            str(queries_path),
            "--retrieval",
            str(retrieval_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert exit_code == 2
    assert "BLOQUEO: información persistida insuficiente" in capsys.readouterr().out
    assert not output_json.exists()
    assert not output_md.exists()
