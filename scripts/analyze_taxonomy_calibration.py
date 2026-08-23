"""Analyze the human calibration of exp18's unsupported-claim taxonomy.

The annotation contract comes from ``docs/GUIA_ANOTACION_GOLD_V4.md``: verdicts are
exactly ``correcto``, ``incorrecto`` or ``dudoso``.  This script never invents or
silently coerces judgments.  It refuses incomplete (>20% blank) or out-of-vocabulary
annotations before computing estimates.

Sampling estimates use the supplied inclusion probability, with Horvitz--Thompson
weight ``1 / inclusion_prob``.  The report is written through ``guard_write`` and will
not replace an existing audit artifact.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.signed_evidence import guard_write


DEFAULT_INPUT = PROJECT_ROOT / "output" / "audit" / "unsupported_claims_sample_v2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "audit" / "taxonomy_calibration_report.md"
CONTROLLED_VERDICTS = ("correcto", "incorrecto", "dudoso")
MAX_BLANK_FRACTION = 0.20


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a calibration CSV without altering verdict text beyond outer whitespace."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV sin cabecera")
        required = {"stratum", "stratum_size", "inclusion_prob", "human_verdict"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"faltan columnas requeridas: {', '.join(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]


def _unexpected(values: Iterable[str]) -> Counter[str]:
    allowed = set(CONTROLLED_VERDICTS)
    return Counter(value for value in values if value and value not in allowed)


def _raise_unexpected(column: str, values: Iterable[str]) -> None:
    bad = _unexpected(values)
    if bad:
        details = ", ".join(f"{value!r}: {count}" for value, count in sorted(bad.items()))
        raise ValueError(
            f"valores inesperados en {column}: {details}; vocabulario permitido: "
            f"{', '.join(CONTROLLED_VERDICTS)}"
        )


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> tuple[float, float]:
    """Return raw agreement and ordinary multiclass Cohen's kappa."""
    if len(first) != len(second):
        raise ValueError("las dos series de test-retest deben tener igual longitud")
    if not first:
        raise ValueError("no hay pares completos para calcular test-retest")

    n = len(first)
    raw = sum(a == b for a, b in zip(first, second)) / n
    counts_a, counts_b = Counter(first), Counter(second)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n)
        for label in CONTROLLED_VERDICTS
    )
    if expected == 1.0:
        raise ValueError("kappa indefinido: ambas anotaciones tienen una sola categoría")
    return raw, (raw - expected) / (1.0 - expected)


def analyze_rows(
    rows: Sequence[Mapping[str, str]], *, expected_population: int | None = None
) -> dict:
    """Validate rows and return HT estimates, Kish n_eff and optional retest metrics."""
    if not rows:
        raise ValueError("el CSV no contiene filas")

    verdicts = [(row.get("human_verdict") or "").strip() for row in rows]
    blank_count = sum(not value for value in verdicts)
    blank_fraction = blank_count / len(rows)
    if blank_fraction > MAX_BLANK_FRACTION:
        raise ValueError(
            "el humano aún no termina: "
            f"{blank_count}/{len(rows)} human_verdict vacíos "
            f"({blank_fraction:.1%}, máximo permitido {MAX_BLANK_FRACTION:.0%})"
        )
    _raise_unexpected("human_verdict", verdicts)

    stratum_sizes: dict[str, int] = {}
    completed: list[dict] = []
    missing_ht_total = 0.0
    for line_no, (row, verdict) in enumerate(zip(rows, verdicts), start=2):
        stratum = (row.get("stratum") or "").strip()
        if not stratum:
            raise ValueError(f"fila {line_no}: stratum vacío")
        try:
            size = int(row.get("stratum_size") or "")
            probability = float(row.get("inclusion_prob") or "")
        except ValueError as exc:
            raise ValueError(f"fila {line_no}: stratum_size/inclusion_prob inválidos") from exc
        if size <= 0 or not 0.0 < probability <= 1.0:
            raise ValueError(f"fila {line_no}: tamaño o probabilidad fuera de rango")
        previous = stratum_sizes.setdefault(stratum, size)
        if previous != size:
            raise ValueError(f"stratum_size inconsistente para {stratum!r}")

        weight = 1.0 / probability
        if verdict:
            completed.append(
                {"stratum": stratum, "verdict": verdict, "weight": weight}
            )
        else:
            missing_ht_total += weight

    population = sum(stratum_sizes.values())
    if expected_population is not None and population != expected_population:
        raise ValueError(
            f"los estratos suman {population}, pero se esperaban {expected_population}"
        )

    weights = [item["weight"] for item in completed]
    if not weights:
        raise ValueError("no hay juicios completos para analizar")
    kish = sum(weights) ** 2 / sum(weight * weight for weight in weights)

    by_stratum: dict[str, dict[str, float]] = defaultdict(
        lambda: {label: 0.0 for label in CONTROLLED_VERDICTS}
    )
    totals = {label: 0.0 for label in CONTROLLED_VERDICTS}
    for item in completed:
        by_stratum[item["stratum"]][item["verdict"]] += item["weight"]
        totals[item["verdict"]] += item["weight"]

    stratum_results = {}
    for stratum in sorted(stratum_sizes):
        size = stratum_sizes[stratum]
        estimates = by_stratum[stratum]
        stratum_results[stratum] = {
            "stratum_size": size,
            "estimated_totals": dict(estimates),
            "rates": {label: estimates[label] / size for label in CONTROLLED_VERDICTS},
        }

    retest_present = any("human_verdict_retest" in row for row in rows)
    retest = None
    if retest_present:
        retest_values = [(row.get("human_verdict_retest") or "").strip() for row in rows]
        _raise_unexpected("human_verdict_retest", retest_values)
        pairs = [
            (first, second)
            for first, second in zip(verdicts, retest_values)
            if first and second
        ]
        if pairs:
            raw, kappa = cohen_kappa(
                [first for first, _ in pairs], [second for _, second in pairs]
            )
            retest = {"n_pairs": len(pairs), "raw_agreement": raw, "cohen_kappa": kappa}

    found = Counter(value for value in verdicts if value)
    return {
        "n_rows": len(rows),
        "n_completed": len(completed),
        "n_blank": blank_count,
        "blank_fraction": blank_fraction,
        "population": population,
        "kish_n_eff": kish,
        "verdict_counts": {label: found[label] for label in CONTROLLED_VERDICTS},
        "strata": stratum_results,
        "ht_totals": totals,
        "ht_rates": {label: totals[label] / population for label in CONTROLLED_VERDICTS},
        "missing_ht_total": missing_ht_total,
        "retest_column_present": retest_present,
        "retest": retest,
    }


def render_markdown(result: Mapping) -> str:
    """Render a transparent, human-readable audit report."""
    lines = [
        "# Calibración humana de la taxonomía exp18",
        "",
        "Vocabulario controlado: `correcto`, `incorrecto`, `dudoso`.",
        "",
        "## Cobertura y ponderación",
        "",
        f"- Filas: {result['n_rows']}",
        f"- Juicios completos: {result['n_completed']}",
        f"- Juicios vacíos: {result['n_blank']} ({result['blank_fraction']:.1%})",
        f"- Población representada: {result['population']}",
        f"- n efectivo de Kish: {result['kish_n_eff']:.3f}",
        f"- Masa HT sin juicio: {result['missing_ht_total']:.3f}",
        "",
        "Los totales usan peso Horvitz–Thompson `1 / inclusion_prob`. No se imputan ni "
        "renormalizan juicios vacíos.",
        "",
        "## Valores encontrados",
        "",
    ]
    for label in CONTROLLED_VERDICTS:
        lines.append(f"- `{label}`: {result['verdict_counts'][label]}")

    lines.extend([
        "",
        "## Tasas HT por estrato",
        "",
        "| Estrato | N | correcto | incorrecto | dudoso |",
        "|---|---:|---:|---:|---:|",
    ])
    for stratum, values in result["strata"].items():
        rates = values["rates"]
        lines.append(
            f"| {stratum} | {values['stratum_size']} | {rates['correcto']:.3%} | "
            f"{rates['incorrecto']:.3%} | {rates['dudoso']:.3%} |"
        )

    lines.extend([
        "",
        f"## Extrapolación HT a los {result['population']}",
        "",
        "| Veredicto | Total estimado | Tasa poblacional |",
        "|---|---:|---:|",
    ])
    for label in CONTROLLED_VERDICTS:
        lines.append(
            f"| {label} | {result['ht_totals'][label]:.3f} | "
            f"{result['ht_rates'][label]:.3%} |"
        )

    lines.extend(["", "## Test-retest", ""])
    if not result["retest_column_present"]:
        lines.append("No existe la columna `human_verdict_retest`; no se calcula acuerdo.")
    elif result["retest"] is None:
        lines.append("La columna `human_verdict_retest` existe, pero no hay pares completos.")
    else:
        retest = result["retest"]
        lines.extend([
            f"- Pares completos: {retest['n_pairs']}",
            f"- Acuerdo crudo: {retest['raw_agreement']:.3%}",
            f"- κ de Cohen: {retest['cohen_kappa']:.4f}",
        ])
    return "\n".join(lines) + "\n"


def write_report(input_path: Path, output_path: Path, *, expected_population: int = 759) -> dict:
    """Analyze ``input_path`` and create, never replace, ``output_path``."""
    result = analyze_rows(read_rows(input_path), expected_population=expected_population)
    output_path = Path(output_path)
    if output_path.exists():
        raise SystemExit(f"REFUSING to overwrite existing audit report: {output_path}")
    guard_write(output_path).write_text(render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = write_report(args.input, args.output, expected_population=759)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Reporte creado: {args.output} | completos={result['n_completed']}/"
        f"{result['n_rows']} | Kish n_eff={result['kish_n_eff']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
