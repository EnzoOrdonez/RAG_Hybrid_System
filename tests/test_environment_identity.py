"""Generated inventories are external, immutable and checked against live values."""

import copy

import pytest

from scripts import environment_identity as inventory
from scripts import study_gate_environment as environment
from src.ui.components.session_storage import atomic_json


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    observed = {
        "schema_version": 1,
        "observed": {"build": "a" * 40},
        "image": {"image_id": "sha256:" + "b" * 64},
        "artifacts": {"files": {"weights": "c" * 64}},
    }
    monkeypatch.setattr(inventory, "snapshot", lambda _: copy.deepcopy(observed))
    config = inventory.generate({}, tmp_path / "environment_identity.json")
    return config, observed


def test_inventory_is_only_source_for_preflight_and_report(sealed):
    config, observed = sealed
    assert environment.identity(config)["build"] == observed["observed"]["build"]
    assert inventory.report(config)["inventory"]["artifacts"] == observed["artifacts"]
    with pytest.raises(FileExistsError):
        inventory.generate(config, config["environment_identity"])


def test_altered_inventory_is_rejected_even_when_live_fields_match(sealed):
    config, observed = sealed
    atomic_json(config["environment_identity"], observed)
    with pytest.raises(ValueError, match="seal changed"):
        inventory.verify(config)


@pytest.mark.parametrize("field", ["observed", "image", "artifacts"])
def test_live_identity_drift_rejects_admission(sealed, field):
    config, observed = sealed
    observed[field] = {"changed": True}
    with pytest.raises(ValueError, match="Live runtime"):
        environment.identity(config)


def test_inventory_cannot_be_written_inside_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(inventory, "ROOT", tmp_path / "project" / "app")
    with pytest.raises(ValueError, match="outside"):
        inventory.generate({}, tmp_path / "environment_identity.json")
