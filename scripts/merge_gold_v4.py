"""Fusiona el gold humano (export JSON del anotador HTML) a los tres CSV de anotacion.

- output/audit/claim_audit_sample_v4.csv          -> juicio_humano / comentario (150)
- output/audit/claim_audit_sample_v4_stageB.csv   -> juicio_humano / comentario (50)
- output/audit/unsupported_claims_sample_v2.csv   -> human_verdict / human_notes (40)

Reglas duras:
- No toca ninguna otra columna; conserva delimitador (';' para v4, ',' para taxonomia)
  y codificacion (UTF-8 con BOM en los v4, UTF-8 en taxonomia).
- Exige cobertura total (150+50+40), vocabulario exacto correcto/incorrecto/dudoso,
  y comentario no vacio cuando el juicio es 'dudoso' (guia seccion 2, paso 6).
- Hace backup de los tres CSV en output/audit/backups_pre_merge_<fecha>/ antes de
  escribir. Si ya tienen juicios humanos, aborta (el gold no se sobreescribe).

Uso: python scripts/merge_gold_v4.py [ruta_al_json]
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "output" / "audit"
VOC = {"correcto", "incorrecto", "dudoso"}

TARGETS = [
    # (prefijo de claves, csv, col_juicio, col_comentario, delimitador, utf8bom)
    ("A", AUDIT / "claim_audit_sample_v4.csv", "juicio_humano", "comentario", ";", True),
    ("B", AUDIT / "claim_audit_sample_v4_stageB.csv", "juicio_humano", "comentario", ";", True),
    ("T", AUDIT / "unsupported_claims_sample_v2.csv", "human_verdict", "human_notes", ",", False),
]


def key_for(prefix: str, row: dict) -> str:
    idx = int(row["idx"]) if "idx" in row else None
    if prefix == "A":
        return f"A-{idx:03d}"
    if prefix == "B":
        return f"B-{idx:02d}"
    raise ValueError("la taxonomia no tiene idx; se asigna por orden de fila")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        AUDIT.glob("gold_v4_juicios_enzo_*.json"), key=lambda p: p.stat().st_mtime
    )
    payload = json.loads(src.read_text(encoding="utf-8"))
    juicios = payload["juicios"]
    print(f"fuente: {src.name} ({len(juicios)} juicios)")

    backup_dir = AUDIT / f"backups_pre_merge_{datetime.now():%Y-%m-%d}"
    backup_dir.mkdir(exist_ok=True)

    for prefix, path, col_j, col_c, delim, bom in TARGETS:
        enc = "utf-8-sig" if bom else "utf-8"
        with path.open(encoding=enc, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=delim))
            fields = list(rows[0].keys()) if rows else []
        if any((r.get(col_j) or "").strip() for r in rows):
            raise SystemExit(f"ABORTA: {path.name} ya tiene juicios humanos")

        missing = []
        for i, row in enumerate(rows, start=1):
            k = key_for(prefix, row) if prefix != "T" else f"T-{i:02d}"
            j = juicios.get(k)
            if not j:
                missing.append(k)
                continue
            if j["juicio"] not in VOC:
                raise SystemExit(f"ABORTA: {k} juicio invalido {j['juicio']!r}")
            if j["juicio"] == "dudoso" and not (j.get("comentario") or "").strip():
                raise SystemExit(f"ABORTA: {k} dudoso sin comentario (guia lo exige)")
            row[col_j] = j["juicio"]
            row[col_c] = (j.get("comentario") or "").strip()
        if missing:
            raise SystemExit(f"ABORTA: faltan {len(missing)} claves en {path.name}: {missing[:5]}")

        shutil.copy2(path, backup_dir / path.name)
        with path.open("w", encoding=enc, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter=delim)
            w.writeheader()
            w.writerows(rows)
        print(f"OK {path.name}: {len(rows)} juicios fusionados (backup en {backup_dir.name}/)")

    print("Merge completo: 150 A + 50 B + 40 T.")


if __name__ == "__main__":
    main()
