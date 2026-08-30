"""Compute light, descriptive uncertainty summaries from persisted audit artifacts.

This module does not rerun retrieval, generation, NLI or HHEM.  Every calculation is
descriptive and outside the project's Benjamini--Hochberg families.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_taxonomy_calibration import analyze_rows, read_rows
from src.utils.signed_evidence import guard_write


DEFAULT_AUDIT_DIR = PROJECT_ROOT / "output" / "audit"
DEFAULT_EXP18_DIR = PROJECT_ROOT / "experiments" / "results" / "exp18_evidence_ceiling"
DEFAULT_OUTPUT_JSON = DEFAULT_AUDIT_DIR / "descriptive_cis.json"
DEFAULT_OUTPUT_MD = DEFAULT_AUDIT_DIR / "descriptive_cis.md"


LABELS = ("correcto", "incorrecto", "dudoso")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes/total fuera de rango para Wilson")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return centre - half_width, centre + half_width


def exact_two_sided_binomial(successes: int, total: int) -> float:
    """Return the exact two-sided p-value for H0: p=0.5."""
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("successes/total fuera de rango para binomial exacta")
    tail = min(successes, total - successes)
    probability = sum(math.comb(total, k) for k in range(tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def _cohen_kappa(first: list[str], second: list[str]) -> float:
    total = len(first)
    if total == 0 or total != len(second):
        raise ValueError("pares inválidos para kappa")
    observed = sum(a == b for a, b in zip(first, second)) / total
    count_first, count_second = Counter(first), Counter(second)
    expected = sum(
        count_first[label] * count_second[label] / total**2 for label in LABELS
    )
    if expected == 1.0:
        raise ValueError("kappa indefinido")
    return (observed - expected) / (1.0 - expected)


def analyze_retest(retest: Mapping, derived: Mapping) -> dict:
    """Reconstruct the original/retest table without reading the full human gold file."""
    mapping = retest.get("mapeo") or {}
    judgments = retest.get("juicios") or {}
    if set(mapping) != set(judgments):
        raise ValueError("mapeo y juicios de tanda C no contienen las mismas claves")

    disagreements = {}
    for item in derived.get("discordantes", []):
        key = item.get("tandaC")
        if key in disagreements:
            raise ValueError(f"discordante duplicado: {key}")
        disagreements[key] = item

    original: list[str] = []
    repeated: list[str] = []
    for key in sorted(mapping):
        repeat_label = (judgments[key].get("juicio") or "").strip()
        if repeat_label not in LABELS:
            raise ValueError(f"veredicto retest inválido en {key}: {repeat_label!r}")
        if key in disagreements:
            item = disagreements[key]
            if item.get("ref") != mapping[key] or item.get("retest") != repeat_label:
                raise ValueError(f"discordancia inconsistente en {key}")
            original_label = item.get("gold")
        else:
            original_label = repeat_label
        if original_label not in LABELS:
            raise ValueError(f"veredicto original inválido en {key}: {original_label!r}")
        original.append(original_label)
        repeated.append(repeat_label)

    if derived.get("n") != len(original):
        raise ValueError("n del resultado derivado no coincide con tanda C")
    index = {label: position for position, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for first, second in zip(original, repeated):
        matrix[index[first]][index[second]] += 1
    agreement_count = sum(a == b for a, b in zip(original, repeated))
    return {
        "labels": list(LABELS),
        "n": len(original),
        "matrix": matrix,
        "agreement_count": agreement_count,
        "raw_agreement": agreement_count / len(original),
        "cohen_kappa": _cohen_kappa(original, repeated),
        "original_marginals": dict(Counter(original)),
        "retest_marginals": dict(Counter(repeated)),
    }


def _read_verdict_rows(path, key_column: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {key_column, "juicio_humano"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: faltan columnas {sorted(required)}")
        result = {}
        for line, row in enumerate(reader, start=2):
            key = (row.get(key_column) or "").strip()
            verdict = (row.get("juicio_humano") or "").strip()
            if not key or key in result:
                raise ValueError(f"{path}:{line}: clave vacía o duplicada")
            if verdict not in LABELS:
                raise ValueError(f"{path}:{line}: juicio_humano inválido: {verdict!r}")
            result[key] = verdict
    return result


def analyze_evidence_set(stage_a_path, stage_b_path) -> dict:
    """Compare the 50 paired stage-A and stage-B human judgments."""
    stage_a = _read_verdict_rows(stage_a_path, "idx")
    stage_b = _read_verdict_rows(stage_b_path, "stage_a_idx")
    missing = sorted(set(stage_b) - set(stage_a))
    if missing:
        raise ValueError(f"stage B referencia idx inexistentes en A: {missing}")

    index = {label: position for position, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    pairs = [(stage_a[key], stage_b[key]) for key in sorted(stage_b)]
    for first, second in pairs:
        matrix[index[first]][index[second]] += 1

    a_correct_b_correct = sum(a == b == "correcto" for a, b in pairs)
    a_correct_b_other = sum(a == "correcto" and b != "correcto" for a, b in pairs)
    a_other_b_correct = sum(a != "correcto" and b == "correcto" for a, b in pairs)
    a_other_b_other = sum(a != "correcto" and b != "correcto" for a, b in pairs)
    flip_count = sum(a != b for a, b in pairs)
    towards_correct = a_other_b_correct
    other_direction = flip_count - towards_correct
    return {
        "labels": list(LABELS),
        "n": len(pairs),
        "matrix": matrix,
        "flip_count": flip_count,
        "flip_rate": flip_count / len(pairs),
        "flip_wilson95": list(wilson_interval(flip_count, len(pairs))),
        "binary_correct_rest": {
            "a_correct_b_correct": a_correct_b_correct,
            "a_correct_b_other": a_correct_b_other,
            "a_other_b_correct": a_other_b_correct,
            "a_other_b_other": a_other_b_other,
        },
        "mcnemar_exact_p": exact_two_sided_binomial(
            a_other_b_correct, a_correct_b_other + a_other_b_correct
        ),
        "all_flips_towards_correct": towards_correct,
        "all_flips_other_direction": other_direction,
        "all_flips_direction_binomial_p": exact_two_sided_binomial(
            towards_correct, flip_count
        ),
    }


def analyze_taxonomy(path, *, expected_population: int = 759) -> dict:
    """Summarize taxonomy annotations through the existing HT implementation."""
    analyzed = analyze_rows(read_rows(path), expected_population=expected_population)
    return {
        "n": analyzed["n_rows"],
        "population": analyzed["population"],
        "counts": analyzed["verdict_counts"],
        "ht_totals": analyzed["ht_totals"],
        "ht_rates": analyzed["ht_rates"],
        "kish_n_eff": analyzed["kish_n_eff"],
    }


def analyze_thresholds(
    index_path, scores_dir, thresholds: tuple[float, ...] = (0.4, 0.5, 0.6)
) -> dict:
    """Count persisted exp18 claims whose best pool score is at or below each tau."""
    index_path, scores_dir = Path(index_path), Path(scores_dir)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"índice inaccesible o inválido: {index_path}") from exc
    counts = {f"{threshold:.1f}": 0 for threshold in thresholds}
    n_claims = 0
    for query_id, metadata in sorted(index.items()):
        matrix_path = scores_dir / f"{query_id}.npy"
        if not matrix_path.is_file():
            raise ValueError(f"matriz ausente para {query_id}: {matrix_path}")
        matrix = np.load(matrix_path, allow_pickle=False)
        expected_shape = tuple(metadata.get("shape") or ())
        claims = metadata.get("claims") or []
        if matrix.ndim != 2 or matrix.shape != expected_shape or matrix.shape[0] != len(claims):
            raise ValueError(f"matriz/índice inconsistente para {query_id}")
        best = matrix.max(axis=1)
        n_claims += len(claims)
        for threshold in thresholds:
            counts[f"{threshold:.1f}"] += int((best <= threshold).sum())
    return {
        "n_queries": len(index),
        "n_claims": n_claims,
        "comparator": "best_over_pool <= tau",
        "counts": counts,
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON inaccesible o inválido: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"se esperaba un objeto JSON: {path}")
    return value


def _latest_retest_file(audit_dir: Path) -> Path:
    matches = sorted(audit_dir.glob("gold_v4_tandaC_enzo_*.json"))
    if not matches:
        raise ValueError(
            f"no existe gold_v4_tandaC_enzo_*.json en {audit_dir}; "
            "no se puede reconstruir la matriz test-retest"
        )
    return matches[-1]


def compute_report(audit_dir: Path, exp18_dir: Path) -> dict:
    """Validate every persisted input and return the complete descriptive report."""
    audit_dir, exp18_dir = Path(audit_dir), Path(exp18_dir)
    evidence = analyze_evidence_set(
        audit_dir / "claim_audit_sample_v4.csv",
        audit_dir / "claim_audit_sample_v4_stageB.csv",
    )
    retest_path = _latest_retest_file(audit_dir)
    retest = analyze_retest(
        _read_json(retest_path),
        _read_json(audit_dir / "gold_v4_tandaC_resultado.json"),
    )
    retest["agreement_wilson95"] = list(
        wilson_interval(retest["agreement_count"], retest["n"])
    )
    taxonomy = analyze_taxonomy(
        audit_dir / "unsupported_claims_sample_v2.csv", expected_population=759
    )
    taxonomy["interpretive_note"] = (
        "Con n efectivo de Kish reducido, una tasa HT pequeña puede depender de muy "
        "pocos casos observados; se reportan también los conteos no ponderados."
    )
    thresholds = analyze_thresholds(
        exp18_dir / "selection_scores_v2_index.json",
        exp18_dir / "selection_scores_v2",
    )
    return {
        "metadata": {
            "descriptive_only": True,
            "bh_family": None,
            "scope": "agregación CPU de artefactos persistidos; sin recomputar verificadores",
        },
        "evidence_set_sensitivity": evidence,
        "intra_annotator": retest,
        "taxonomy": taxonomy,
        "threshold_sensitivity": thresholds,
        "sources": {
            "stage_a": str(audit_dir / "claim_audit_sample_v4.csv"),
            "stage_b": str(audit_dir / "claim_audit_sample_v4_stageB.csv"),
            "retest": str(retest_path),
            "retest_derived": str(audit_dir / "gold_v4_tandaC_resultado.json"),
            "taxonomy": str(audit_dir / "unsupported_claims_sample_v2.csv"),
            "exp18_index": str(exp18_dir / "selection_scores_v2_index.json"),
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _interval(value: list[float]) -> str:
    return f"[{_percent(value[0])}, {_percent(value[1])}]"


def _matrix_markdown(matrix: list[list[int]]) -> list[str]:
    return [
        "| original \\ segunda | correcto | incorrecto | dudoso |",
        "|---|---:|---:|---:|",
        *[
            f"| {label} | {row[0]} | {row[1]} | {row[2]} |"
            for label, row in zip(LABELS, matrix)
        ],
    ]


def render_markdown(report: Mapping) -> str:
    """Render the validated calculations with restrained interpretations."""
    evidence = report["evidence_set_sensitivity"]
    retest = report["intra_annotator"]
    taxonomy = report["taxonomy"]
    thresholds = report["threshold_sensitivity"]
    lines = [
        "# Intervalos y pruebas descriptivas",
        "",
        "Todos los resultados de este documento son **descriptivos y están fuera de las familias BH**. "
        "Se agregan artefactos ya persistidos; no se reejecutó generación ni verificación.",
        "",
        "## Sensibilidad al conjunto de evidencia (etapas A → B)",
        "",
        f"- Flips: {evidence['flip_count']}/{evidence['n']} "
        f"({_percent(evidence['flip_rate'])}); IC95 Wilson {_interval(evidence['flip_wilson95'])}.",
        "",
        *_matrix_markdown(evidence["matrix"]),
        "",
        "Al reducir `correcto` frente a las demás categorías, la tabla pareada es:",
        "",
        "| | B correcto | B otro |",
        "|---|---:|---:|",
        f"| A correcto | {evidence['binary_correct_rest']['a_correct_b_correct']} | "
        f"{evidence['binary_correct_rest']['a_correct_b_other']} |",
        f"| A otro | {evidence['binary_correct_rest']['a_other_b_correct']} | "
        f"{evidence['binary_correct_rest']['a_other_b_other']} |",
        "",
        f"McNemar exacto correcto/resto (discordantes "
        f"{evidence['binary_correct_rest']['a_other_b_correct']} vs "
        f"{evidence['binary_correct_rest']['a_correct_b_other']}): "
        f"p={evidence['mcnemar_exact_p']:.6f}.",
        f"Como resumen direccional de **todos** los flips, "
        f"{evidence['all_flips_towards_correct']} fueron hacia `correcto` y "
        f"{evidence['all_flips_other_direction']} tuvieron otra dirección; binomial exacta "
        f"p={evidence['all_flips_direction_binomial_p']:.6f}. Esta última no es McNemar ni "
        "una prueba confirmatoria.",
        "",
        "## Confiabilidad intra-anotador",
        "",
        f"- Acuerdo: {retest['agreement_count']}/{retest['n']} "
        f"({_percent(retest['raw_agreement'])}); IC95 Wilson "
        f"{_interval(retest['agreement_wilson95'])}.",
        f"- κ de Cohen: {retest['cohen_kappa']:.4f}.",
        "",
        *_matrix_markdown(retest["matrix"]),
        "",
        "La matriz se reconstruyó con la tanda C y el archivo derivado de discordancias; "
        "no se abrió el archivo completo de juicios humanos.",
        "",
        "## Taxonomía de claims `unsupported@0.5`",
        "",
        "| veredicto | conteo no ponderado | tasa HT |",
        "|---|---:|---:|",
        *[
            f"| {label} | {taxonomy['counts'][label]} | {_percent(taxonomy['ht_rates'][label])} |"
            for label in LABELS
        ],
        "",
        f"n={taxonomy['n']}; población={taxonomy['population']}; n efectivo de Kish="
        f"{taxonomy['kish_n_eff']:.3f}. {taxonomy['interpretive_note']}",
        "",
        "## Sensibilidad descriptiva a τ",
        "",
        f"Se leyeron {thresholds['n_claims']} claims de {thresholds['n_queries']} queries con "
        f"la regla inclusiva `{thresholds['comparator']}`.",
        "",
        "| τ | claims `unsupported@τ` |",
        "|---:|---:|",
        *[f"| {tau} | {count} |" for tau, count in thresholds["counts"].items()],
        "",
        "Estos conteos son una sensibilidad de umbral, no una nueva familia inferencial.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping, output_json: Path, output_md: Path) -> None:
    """Write both reports only after the caller has computed and validated everything."""
    output_json, output_md = Path(output_json), Path(output_md)
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(report)
    guard_write(output_json).write_text(json_text, encoding="utf-8")
    guard_write(output_md).write_text(markdown_text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--exp18-dir", type=Path, default=DEFAULT_EXP18_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    try:
        report = compute_report(args.audit_dir, args.exp18_dir)
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(report, args.output_json, args.output_md)
    print(f"Escritos {args.output_json} y {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
