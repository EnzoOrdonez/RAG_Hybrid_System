from scripts import run_gold_sensitivity


def _cohen_kappa(human: list[int], verifier: list[int], weights: list[float]) -> float:
    assert weights
    observed = sum(w for h, v, w in zip(human, verifier, weights) if h == v) / sum(weights)
    human_rate = sum(h * w for h, w in zip(human, weights)) / sum(weights)
    verifier_rate = sum(v * w for v, w in zip(verifier, weights)) / sum(weights)
    expected = human_rate * verifier_rate + (1 - human_rate) * (1 - verifier_rate)
    return (observed - expected) / (1 - expected)


def test_without_adjudication_only_builds_exclusion_variant() -> None:
    current = {idx: "correcto" if idx % 2 else "incorrecto" for idx in range(1, 13)}
    discordant = {2, 5, 9}

    variants = run_gold_sensitivity.build_label_variants(
        current_labels=current,
        discordant_indices=discordant,
        adjudication=None,
    )

    assert list(variants) == ["sin_adjudicados"]
    assert variants["sin_adjudicados"] == {
        idx: label for idx, label in current.items() if idx not in discordant
    }


def test_adjudication_builds_post_excluded_and_pre_variants() -> None:
    current = {1: "incorrecto", 2: "correcto", 3: "dudoso", 4: "correcto"}
    final = {1: "correcto", 3: "incorrecto"}

    variants = run_gold_sensitivity.build_label_variants(
        current_labels=current,
        discordant_indices=set(final),
        adjudication=final,
    )

    assert variants == {
        "post_adjudicacion": {
            1: "correcto",
            2: "correcto",
            3: "incorrecto",
            4: "correcto",
        },
        "sin_adjudicados": {2: "correcto", 4: "correcto"},
        "pre_adjudicacion": current,
    }


def test_score_variant_reports_weighted_and_anchor_kappa() -> None:
    claims = [
        run_gold_sensitivity.PreparedClaim(1, "random_anchor", 1.0, {"good": 1, "bad": 0}),
        run_gold_sensitivity.PreparedClaim(2, "random_anchor", 1.0, {"good": 0, "bad": 1}),
        run_gold_sensitivity.PreparedClaim(3, "near_threshold", 2.0, {"good": 1, "bad": 0}),
        run_gold_sensitivity.PreparedClaim(4, "near_threshold", 2.0, {"good": 0, "bad": 1}),
    ]
    labels = {1: "correcto", 2: "incorrecto", 3: "correcto", 4: "incorrecto"}

    result = run_gold_sensitivity.score_variant(
        labels,
        claims,
        candidates=["good", "bad"],
        weighted_kappa_fn=_cohen_kappa,
    )

    assert result["n_used"] == 4
    assert result["candidates"]["good"] == {
        "kappa_weighted": 1.0,
        "kappa_anchor": 1.0,
    }
    assert result["candidates"]["bad"] == {
        "kappa_weighted": -1.0,
        "kappa_anchor": -1.0,
    }


def test_deltas_are_relative_to_post_or_null_when_post_is_unavailable() -> None:
    scored = {
        "post_adjudicacion": {
            "candidates": {"hhem": {"kappa_weighted": 0.30, "kappa_anchor": 0.35}}
        },
        "sin_adjudicados": {
            "candidates": {"hhem": {"kappa_weighted": 0.28, "kappa_anchor": 0.31}}
        },
    }

    run_gold_sensitivity.add_deltas(scored)

    assert scored["post_adjudicacion"]["candidates"]["hhem"]["delta_weighted"] == 0.0
    assert scored["sin_adjudicados"]["candidates"]["hhem"]["delta_weighted"] == -0.02
    assert scored["sin_adjudicados"]["candidates"]["hhem"]["delta_anchor"] == -0.04

    partial = {"sin_adjudicados": scored["sin_adjudicados"]}
    run_gold_sensitivity.add_deltas(partial)
    assert partial["sin_adjudicados"]["candidates"]["hhem"]["delta_weighted"] is None


def test_markdown_marks_unavailable_deltas_as_pending() -> None:
    payload = {
        "declaracion": "[PENDIENTE-ADJUDICACION] falta el JSON",
        "variantes": {
            "sin_adjudicados": {
                "n_used": 10,
                "candidates": {
                    "hhem": {
                        "kappa_weighted": 0.3,
                        "delta_weighted": None,
                        "kappa_anchor": 0.4,
                        "delta_anchor": None,
                    }
                },
            }
        },
    }

    markdown = run_gold_sensitivity.render_markdown(payload)

    assert "None" not in markdown
    assert markdown.count("[PENDIENTE-ADJUDICACION]") == 3
