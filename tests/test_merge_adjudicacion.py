import csv
import json
from pathlib import Path

import pytest

from scripts import merge_adjudicacion_tandaC


FIELDNAMES = ["idx", "query_id", "claim", "juicio_humano", "comentario"]


def _write_csv(path: Path, n_rows: int = 12) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        for idx in range(1, n_rows + 1):
            writer.writerow(
                {
                    "idx": idx,
                    "query_id": f"q{idx:03d}",
                    "claim": f"claim {idx}",
                    "juicio_humano": "dudoso",
                    "comentario": "comentario previo" if idx == 1 else "",
                }
            )


def _adjudication_payload() -> dict:
    mapping = {f"J-{idx:02d}": f"A-{idx:03d}" for idx in range(1, 10)}
    judgments = {
        key: {
            "juicio": "correcto" if idx % 2 else "incorrecto",
            "comentario": f"razón final {idx}",
            "ts": "2026-08-29T12:00:00-05:00",
        }
        for idx, key in enumerate(mapping, start=1)
    }
    return {
        "formato": "gold_v4_adjudicacion",
        "mapeo": mapping,
        "juicios": judgments,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def test_merge_updates_nine_rows_and_creates_pristine_backup(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    original = csv_path.read_bytes()
    json_path.write_text(json.dumps(_adjudication_payload()), encoding="utf-8")

    backup = merge_adjudicacion_tandaC.merge_adjudication(
        json_path=json_path,
        csv_path=csv_path,
        audit_dir=tmp_path,
        date_stamp="2026-08-29",
    )

    assert backup.read_bytes() == original
    assert csv_path.read_bytes().startswith(bytes.fromhex("efbbbf"))
    rows = _read_csv(csv_path)
    assert rows[0]["juicio_humano"] == "correcto"
    assert rows[0]["comentario"] == "adjudicado: razón final 1 [comentario previo]"
    assert rows[1]["juicio_humano"] == "incorrecto"
    assert rows[1]["comentario"] == "adjudicado: razón final 2"


def test_merge_aborts_before_writing_when_a_j_key_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    original = csv_path.read_bytes()
    payload = _adjudication_payload()
    del payload["juicios"]["J-09"]
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(merge_adjudicacion_tandaC.AdjudicationError, match="9/9"):
        merge_adjudicacion_tandaC.merge_adjudication(
            json_path=json_path,
            csv_path=csv_path,
            audit_dir=tmp_path,
            date_stamp="2026-08-29",
        )

    assert csv_path.read_bytes() == original
    assert not (tmp_path / "backups_adjudicacion_2026-08-29").exists()


def test_merge_aborts_before_writing_when_comment_is_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    original = csv_path.read_bytes()
    payload = _adjudication_payload()
    payload["juicios"]["J-04"]["comentario"] = "   "
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(merge_adjudicacion_tandaC.AdjudicationError, match="comentario"):
        merge_adjudicacion_tandaC.merge_adjudication(
            json_path=json_path,
            csv_path=csv_path,
            audit_dir=tmp_path,
            date_stamp="2026-08-29",
        )

    assert csv_path.read_bytes() == original
    assert not (tmp_path / "backups_adjudicacion_2026-08-29").exists()


def test_merge_does_not_change_rows_outside_the_nine_mapped_indices(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    before = _read_csv(csv_path)
    json_path.write_text(json.dumps(_adjudication_payload()), encoding="utf-8")

    merge_adjudicacion_tandaC.merge_adjudication(
        json_path=json_path,
        csv_path=csv_path,
        audit_dir=tmp_path,
        date_stamp="2026-08-29",
    )

    after = _read_csv(csv_path)
    assert after[9:] == before[9:]


def test_merge_rejects_verdict_outside_controlled_vocabulary(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    original = csv_path.read_bytes()
    payload = _adjudication_payload()
    payload["juicios"]["J-06"]["juicio"] = "tal_vez"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(merge_adjudicacion_tandaC.AdjudicationError, match="vocabulario"):
        merge_adjudicacion_tandaC.merge_adjudication(
            json_path=json_path,
            csv_path=csv_path,
            audit_dir=tmp_path,
            date_stamp="2026-08-29",
        )

    assert csv_path.read_bytes() == original


def test_merge_rejects_invalid_a_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    original = csv_path.read_bytes()
    payload = _adjudication_payload()
    payload["mapeo"]["J-03"] = "A-12"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(merge_adjudicacion_tandaC.AdjudicationError, match="A-xxx"):
        merge_adjudicacion_tandaC.merge_adjudication(
            json_path=json_path,
            csv_path=csv_path,
            audit_dir=tmp_path,
            date_stamp="2026-08-29",
        )

    assert csv_path.read_bytes() == original


def test_cli_aborts_cleanly_when_default_json_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(merge_adjudicacion_tandaC, "AUDIT", tmp_path)

    assert merge_adjudicacion_tandaC.main([]) == 2
    assert "No existe todavía" in capsys.readouterr().out


def test_merge_rejects_wrong_export_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "claim_audit_sample_v4.csv"
    json_path = tmp_path / "adjudicacion.json"
    _write_csv(csv_path)
    payload = _adjudication_payload()
    payload["formato"] = "otro_formato"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(merge_adjudicacion_tandaC.AdjudicationError, match="formato"):
        merge_adjudicacion_tandaC.merge_adjudication(
            json_path=json_path,
            csv_path=csv_path,
            audit_dir=tmp_path,
            date_stamp="2026-08-29",
        )
