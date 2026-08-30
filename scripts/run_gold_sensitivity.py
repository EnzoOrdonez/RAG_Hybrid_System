"""Run gold-verifier sensitivity variants around Tanda-C adjudication."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Set
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Callable, Sequence


JUDGE_MAP = {"correcto": 1, "incorrecto": 0}
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "output" / "audit"
EXP15 = ROOT / "experiments" / "results" / "exp15_ablation_nli"
GOLD_CSV = "claim_audit_sample_v4.csv"
TANDA_C_RESULT = "gold_v4_tandaC_resultado.json"


@dataclass(frozen=True)
class PreparedClaim:
    idx: int
    stratum: str
    weight: float
    predictions: Mapping[str, int]


def build_label_variants(
    *,
    current_labels: Mapping[int, str],
    discordant_indices: Set[int],
    adjudication: Mapping[int, str] | None,
    pre_adjudication_labels: Mapping[int, str] | None = None,
) -> dict[str, dict[int, str]]:
    """Build label maps for the available sensitivity variants."""

    if adjudication is None:
        return {
            "sin_adjudicados": {
                idx: label
                for idx, label in current_labels.items()
                if idx not in discordant_indices
            }
        }
    post = dict(current_labels)
    post.update(adjudication)
    pre = dict(pre_adjudication_labels or current_labels)
    return {
        "post_adjudicacion": post,
        "sin_adjudicados": {
            idx: label for idx, label in post.items() if idx not in discordant_indices
        },
        "pre_adjudicacion": pre,
    }


def score_variant(
    labels: Mapping[int, str],
    claims: Sequence[PreparedClaim],
    *,
    candidates: Sequence[str],
    weighted_kappa_fn: Callable[[list[int], list[int], list[float]], float],
) -> dict:
    """Score one label variant with the imported gold-analysis kappa function."""

    used = [claim for claim in claims if labels.get(claim.idx) in JUDGE_MAP]
    human = [JUDGE_MAP[labels[claim.idx]] for claim in used]
    weights = [claim.weight for claim in used]
    anchor_indices = [
        index for index, claim in enumerate(used) if claim.stratum == "random_anchor"
    ]
    results = {}
    for candidate in candidates:
        predicted = [claim.predictions[candidate] for claim in used]
        weighted = weighted_kappa_fn(human, predicted, weights)
        anchor = weighted_kappa_fn(
            [human[index] for index in anchor_indices],
            [predicted[index] for index in anchor_indices],
            [1.0] * len(anchor_indices),
        )
        results[candidate] = {
            "kappa_weighted": None if math.isnan(weighted) else round(weighted, 4),
            "kappa_anchor": None if math.isnan(anchor) else round(anchor, 4),
        }
    return {"n_used": len(used), "candidates": results}


def add_deltas(scored_variants: dict[str, dict]) -> None:
    """Add per-candidate deltas against the post-adjudication variant in place."""

    baseline = scored_variants.get("post_adjudicacion", {}).get("candidates", {})
    for variant in scored_variants.values():
        for candidate, metrics in variant["candidates"].items():
            reference = baseline.get(candidate)
            for metric, delta_name in (
                ("kappa_weighted", "delta_weighted"),
                ("kappa_anchor", "delta_anchor"),
            ):
                current = metrics.get(metric)
                base = reference.get(metric) if reference else None
                metrics[delta_name] = (
                    None if current is None or base is None else round(current - base, 4)
                )


def _read_gold_rows(path: Path) -> tuple[list[dict[str, str]], dict[int, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    labels = {
        int(row["idx"]): (row.get("juicio_humano") or "").strip().lower()
        for row in rows
        if (row.get("juicio_humano") or "").strip()
    }
    return rows, labels


def _indices_from_refs(refs: Sequence[str]) -> set[int]:
    try:
        indices = {int(ref.removeprefix("A-")) for ref in refs}
    except (AttributeError, ValueError) as exc:
        raise RuntimeError("Las referencias de tanda C deben usar A-xxx") from exc
    if len(refs) != 9 or len(indices) != 9:
        raise RuntimeError("Se esperaban exactamente 9 referencias discordantes únicas")
    return indices


def load_tanda_c_indices(path: Path) -> set[int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return _indices_from_refs([item["ref"] for item in document["discordantes"]])


def load_adjudication(path: Path) -> dict[int, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    mapping = document["mapeo"]
    judgments = document["juicios"]
    if len(mapping) != 9 or set(mapping) != set(judgments):
        raise RuntimeError("La adjudicación debe contener las mismas 9 claves en mapeo y juicios")
    return {
        int(target.removeprefix("A-")): judgments[key]["juicio"]
        for key, target in mapping.items()
    }


def latest_adjudication(audit_dir: Path) -> Path | None:
    candidates = sorted(
        audit_dir.glob("gold_v4_adjudicacion_enzo_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load_pre_adjudication_backup(audit_dir: Path) -> dict[int, str]:
    candidates = sorted(
        audit_dir.glob(f"backups_adjudicacion_*/{GOLD_CSV}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(
            "El CSV ya parece adjudicado, pero no existe backup de adjudicación para "
            "reconstruir las etiquetas previas."
        )
    return _read_gold_rows(candidates[-1])[1]


def prepare_claims() -> tuple[list[PreparedClaim], list[str], Callable]:
    """Prepare persisted verifier labels using imported gold-analysis machinery."""

    from scripts import analyze_gold_v4 as gold

    meta = json.loads((AUDIT / "claim_audit_sample_v4_meta.json").read_text(encoding="utf-8"))
    probabilities = {tag: gold.ens.load_nli(tag) for tag in gold.ens.NLI_MEMBERS}
    hhem = gold.ens.load_hhem() if gold.ens.HAS_HHEM else {}
    claims = json.loads((EXP15 / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]

    prediction_rows = []
    for row in meta["rows"]:
        config, query_id, claim_idx = row["config"], row["query_id"], row["claim_idx"]
        members = {
            tag: probabilities[tag][config][query_id][claim_idx]
            for tag in gold.ens.NLI_MEMBERS
        }
        hhem_query = hhem.get(config, {}).get(query_id, [])
        hhem_claim = hhem_query[claim_idx] if claim_idx < len(hhem_query) else []
        predictions = {
            candidate: 1 if gold.ens.label_one(candidate, members, hhem_claim) == "supported" else 0
            for candidate in gold.CANDIDATES
        }
        prediction_rows.append((row, predictions))

    pool_keys, pool_flags = gold.build_pool_flags(
        probabilities,
        claims,
        gold.load_seen_v3(),
    )
    inclusion, _, _ = gold.inclusion_probs(pool_flags)
    inclusion_by_key = dict(zip(pool_keys, inclusion))

    prepared = []
    for row, predictions in prediction_rows:
        key = (row["config"], row["query_id"], row["claim_idx"])
        probability = inclusion_by_key.get(key, 0)
        if probability <= 0:
            raise RuntimeError(
                f"El idx {row['idx']} tiene probabilidad de inclusión cero; "
                "el muestreo y el meta divergieron."
            )
        prepared.append(
            PreparedClaim(
                idx=row["idx"],
                stratum=row["stratum"],
                weight=1.0 / probability,
                predictions=predictions,
            )
        )
    return prepared, list(gold.CANDIDATES), gold.weighted_kappa


def render_markdown(payload: Mapping) -> str:
    def display(value: object) -> object:
        return "[PENDIENTE-ADJUDICACION]" if value is None else value

    lines = ["# Gold v4 — sensibilidad a la adjudicación de tanda C", ""]
    lines.append(payload["declaracion"])
    lines += [
        "",
        "| Variante | Candidato | n usado | κ ponderada | Δκ | κ anchor | Δκ anchor |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant_name, variant in payload["variantes"].items():
        for candidate, metrics in variant["candidates"].items():
            lines.append(
                f"| {variant_name} | {candidate} | {variant['n_used']} | "
                f"{display(metrics['kappa_weighted'])} | "
                f"{display(metrics['delta_weighted'])} | "
                f"{display(metrics['kappa_anchor'])} | "
                f"{display(metrics['delta_anchor'])} |"
            )
    return "\n".join(lines) + "\n"


def run(audit_dir: Path = AUDIT, adjudication_path: Path | None = None) -> dict:
    rows, current_labels = _read_gold_rows(audit_dir / GOLD_CSV)
    resolved_adjudication = adjudication_path or latest_adjudication(audit_dir)

    if resolved_adjudication is None:
        discordant = load_tanda_c_indices(audit_dir / TANDA_C_RESULT)
        variants = build_label_variants(
            current_labels=current_labels,
            discordant_indices=discordant,
            adjudication=None,
        )
        declaration = (
            "[PENDIENTE-ADJUDICACION] No existe JSON de adjudicación: se reporta "
            "solo la variante que excluye los nueve discordantes de tanda C; "
            "los deltas contra post-adjudicación no están disponibles."
        )
    else:
        final_labels = load_adjudication(resolved_adjudication)
        discordant = set(final_labels)
        adjudicated_flags = [
            (row.get("comentario") or "").strip().lower().startswith("adjudicado:")
            for row in rows
            if int(row["idx"]) in discordant
        ]
        if any(adjudicated_flags) and not all(adjudicated_flags):
            raise RuntimeError("El CSV contiene una adjudicación parcial; se rechaza el análisis")
        pre_labels = (
            _load_pre_adjudication_backup(audit_dir)
            if all(adjudicated_flags)
            else None
        )
        variants = build_label_variants(
            current_labels=current_labels,
            discordant_indices=discordant,
            adjudication=final_labels,
            pre_adjudication_labels=pre_labels,
        )
        declaration = (
            "Se reportan etiquetas post-adjudicación, exclusión de los nueve ítems y "
            "estado pre-adjudicación reconstruido sin re-anotar."
        )

    prepared, candidates, weighted_kappa_fn = prepare_claims()
    scored = {
        name: score_variant(
            labels,
            prepared,
            candidates=candidates,
            weighted_kappa_fn=weighted_kappa_fn,
        )
        for name, labels in variants.items()
    }
    add_deltas(scored)
    payload = {
        "generated_by": "scripts/run_gold_sensitivity.py",
        "adjudicacion_disponible": resolved_adjudication is not None,
        "idx_adjudicados": [f"A-{idx:03d}" for idx in sorted(discordant)],
        "declaracion": declaration,
        "variantes": scored,
    }
    (audit_dir / "gold_v4_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "gold_v4_sensitivity.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicacion", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(adjudication_path=args.adjudicacion)
    print(payload["declaracion"])
    print("Escritos: output/audit/gold_v4_sensitivity.json y .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
