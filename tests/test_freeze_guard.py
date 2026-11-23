import json

import pytest

from src import config, experiment


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_DIR", tmp_path)
    monkeypatch.setattr(experiment, "rules_sha256", lambda: "rules-A")
    monkeypatch.setattr(experiment, "code_sha256", lambda: "code-A")
    return tmp_path


def make_run(root, name, phase, rules="rules-A", code="code-A", complete=True, valid=True):
    run = root / name
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "phase": phase, "rules_sha256": rules, "code_sha256": code, "dataset_sha256_valid": valid}))
    if complete:
        (run / ("val_check.csv" if phase == experiment.VAL_PHASE else "summary.csv")).write_text("x")
    return run


def test_final_needs_a_completed_val_run(runs):
    with pytest.raises(RuntimeError, match="completed VAL run"):
        experiment.final_guard("missing")
    make_run(runs, "val", experiment.VAL_PHASE, complete=False)
    with pytest.raises(RuntimeError, match="completed VAL run"):
        experiment.final_guard("val")


def test_final_accepts_unchanged_frozen_state(runs):
    make_run(runs, "val", experiment.VAL_PHASE)
    assert experiment.final_guard("val") == {"val_run": "val", "rules_sha256": "rules-A", "code_sha256": "code-A"}


@pytest.mark.parametrize("field", ["rules", "code"])
def test_final_refuses_when_rules_or_code_changed_after_val(runs, field):
    make_run(runs, "val", experiment.VAL_PHASE, **{field: "stale"})
    with pytest.raises(RuntimeError, match=f"{field} changed"):
        experiment.final_guard("val")


def test_final_refuses_a_second_test_pass_for_the_same_frozen_state(runs):
    make_run(runs, "val", experiment.VAL_PHASE)
    make_run(runs, "first_final", experiment.FINAL_PHASE)
    with pytest.raises(RuntimeError, match="already run once"):
        experiment.final_guard("val")


def test_crashed_final_does_not_burn_the_single_test_pass(runs):
    make_run(runs, "val", experiment.VAL_PHASE)
    make_run(runs, "crashed_final", experiment.FINAL_PHASE, complete=False)
    assert experiment.final_guard("val")["val_run"] == "val"


def test_a_non_val_run_cannot_freeze(runs):
    make_run(runs, "sweep", "development sweep")
    (runs / "sweep" / "val_check.csv").write_text("x")
    with pytest.raises(RuntimeError, match="not a VAL run"):
        experiment.final_guard("sweep")
