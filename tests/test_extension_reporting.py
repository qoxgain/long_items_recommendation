"""Evidence-boundary checks for extension reports without expensive model fitting."""

from copy import deepcopy
import importlib.metadata
import json
from types import SimpleNamespace

import numpy as np
import pytest

from saet_repro.evaluation import evaluate, stratified_summaries
from saet_repro.io import sha256
from scripts.summarize_extensions import (
    Audit, aggregate_rows, ensure_unique, ridge_diagnostic_rows, validate_dependencies, validate_population,
)


def ranking_fixture():
    """Include repeated events, excluded prior users, and an ineligible item."""
    return dict(
        target_item_ids=np.array(["private-item-a", "private-item-b", "private-item-c"]),
        target_user_ids=np.array(["private-user-a", "private-user-b", "private-user-c", "private-user-d"]),
        target_degree=np.array([1, 1, 0]),
        target_observed=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]),
        target_positive=np.array([[0, 0, 1], [2, 0, 1], [2, 0, 1], [1, 1, 1], [3, 2, 1]]),
        target_tail80=np.array([True, True, False]),
        target_tail50=np.array([True, False, False]),
        expected_n_candidates=np.array([3, 3, 4]),
        expected_n_positive=np.array([1, 0, 1]),
    )


def evaluated_fixture():
    catalog = ranking_fixture()
    result = evaluate(catalog, 1, lambda items: np.ones((len(items), 4)))
    return catalog, result


def test_dependencies_validate_current_nested_code_and_package_versions(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    record = {"protocol": {"identities": [
        {"code_sha256": {"source.py": sha256(source)}},
        {"dependencies": {"numpy": importlib.metadata.version("numpy")}},
    ]}}
    validate_dependencies(record, root=tmp_path)


@pytest.mark.parametrize("changed", ["code_bytes", "declared_hash", "dependency"])
def test_nested_tampered_dependencies_do_not_enter_a_report(tmp_path, changed):
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    identity = {
        "code_sha256": {"source.py": sha256(source)},
        "dependencies": {"numpy": importlib.metadata.version("numpy")},
    }
    record = {"frozen": [{"identity": identity}]}
    if changed == "code_bytes":
        source.write_text("version = 2\n")
    elif changed == "declared_hash":
        identity["code_sha256"]["source.py"] = "0" * 64
    else:
        identity["dependencies"]["numpy"] = "0.0.0-tampered"
    with pytest.raises(ValueError):
        validate_dependencies(record, root=tmp_path)


def test_uniqueness_uses_the_full_declared_key_and_rejects_duplicate_evidence():
    rows = [
        dict(method="target", origin=2015, seed=11, stratum="tail80", ndcg20=.2),
        dict(method="target", origin=2015, seed=22, stratum="tail80", ndcg20=.2),
        dict(method="target", origin=2016, seed=11, stratum="tail80", ndcg20=.2),
    ]
    keys = ("method", "origin", "seed", "stratum")
    ensure_unique(rows, keys)
    with pytest.raises(ValueError):
        ensure_unique(rows + [dict(rows[0], ndcg20=.9)], keys)


def test_population_accepts_the_complete_actual_evaluator_schema():
    catalog, result = evaluated_fixture()
    np.testing.assert_array_equal(result["n_candidates"], [3, 3, 4])
    np.testing.assert_array_equal(result["n_positive"], [1, 0, 1])
    np.testing.assert_array_equal(result["requested"], [True, True, False])
    np.testing.assert_array_equal(result["evaluated"], [True, False, False])
    validate_population(result, catalog)
    validate_population(result, catalog, reference=deepcopy(result))


@pytest.mark.parametrize("field", ["n_candidates", "n_positive"])
def test_population_rejects_changed_candidate_and_positive_counts(field):
    catalog, result = evaluated_fixture()
    result[field][0] += 1
    with pytest.raises(ValueError):
        validate_population(result, catalog)


@pytest.mark.parametrize("field,index", [
    ("eligible_catalog", 2),
    ("requested", 0),
    ("requested", 2),
    ("evaluated", 0),
    ("evaluated", 1),
    ("evaluated", 2),
])
def test_population_rejects_partial_requests_and_changed_masks(field, index):
    catalog, result = evaluated_fixture()
    result[field][index] = ~result[field][index]
    with pytest.raises(ValueError):
        validate_population(result, catalog)


@pytest.mark.parametrize("changed", ["reordered", "different", "duplicate"])
def test_population_rejects_mismatched_or_duplicate_item_ids(changed):
    catalog, result = evaluated_fixture()
    if changed == "reordered":
        result["item_id"] = result["item_id"][::-1].copy()
    elif changed == "different":
        result["item_id"][0] = "unknown-item"
    else:
        # Matching catalogs still cannot legitimize duplicate item evidence.
        result["item_id"][1] = result["item_id"][0]
        catalog["target_item_ids"][1] = catalog["target_item_ids"][0]
    with pytest.raises(ValueError):
        validate_population(result, catalog)


@pytest.mark.parametrize("field", [
    "item_id", "n_candidates", "n_positive", "eligible_catalog", "requested", "evaluated",
])
def test_shared_population_must_match_the_reference_exactly(field):
    catalog, result = evaluated_fixture()
    reference = deepcopy(result)
    if field == "item_id":
        reference[field] = reference[field][::-1].copy()
    elif reference[field].dtype.kind == "b":
        reference[field][0] = ~reference[field][0]
    else:
        reference[field][0] += 1
    with pytest.raises(ValueError):
        validate_population(result, catalog, reference=reference)


@pytest.mark.parametrize("field,value", [
    ("stage", np.array(2)),
    ("candidate_mode", np.array("sampled_users")),
])
def test_population_rejects_the_wrong_query_stage_or_candidate_mode(field, value):
    catalog, result = evaluated_fixture()
    result[field] = value
    with pytest.raises(ValueError):
        validate_population(result, catalog)


ORIGINS = (2015, 2016, 2017)
SEEDS = (11, 22, 33)


def summary_row(method, origin, seed, value, size):
    catalog, result = evaluated_fixture()
    row = dict(stratified_summaries(catalog, result)["tail80"])
    for name in row:
        if name.startswith(("ndcg", "recall", "precision", "coverage", "conditional_hit_rate")):
            if name != "precision_denominator":
                row[name] = value
    row.update(method=method, origin=origin, seed=seed, stratum="tail80",
               items_catalog=size, items_evaluated=size, items_eligible=size,
               items_requested=size, items_with_positive=size,
               items_without_positive=0, n_positive=size)
    return row


def complete_rows(method="target"):
    return [summary_row(method, year, seed, value, size)
            for year, value, size in zip(ORIGINS, (.1, .4, .7), (2, 100, 1000))
            for seed in SEEDS]


def test_aggregation_averages_seeds_then_origins_with_equal_origin_weights():
    rows = complete_rows()
    # The unequal seed values retain the first origin's mean of .1.
    for row, value in zip(rows[:3], (.0, .1, .2)):
        row["ndcg20"] = value
    by_origin, overall = aggregate_rows(rows, origins=ORIGINS, seeds=SEEDS)
    assert len(by_origin) == 3
    assert {row["origin"] for row in by_origin} == set(ORIGINS)
    assert len(overall) == 1
    assert overall[0]["ndcg20"] == pytest.approx(.4)
    assert overall[0]["n_origins"] == 3
    assert all(row["n_seeds"] == len(SEEDS) for row in by_origin)


def test_incomplete_seed_matrix_is_not_averaged_or_promoted_to_overall():
    by_origin, overall = aggregate_rows(complete_rows()[:-1], origins=ORIGINS, seeds=SEEDS)
    assert {row["origin"] for row in by_origin} == {2015, 2016}
    assert overall == []


def test_incomplete_origin_matrix_is_not_averaged():
    rows = [row for row in complete_rows() if row["origin"] != 2017]
    by_origin, overall = aggregate_rows(rows, origins=ORIGINS, seeds=SEEDS)
    assert len(by_origin) == 2
    assert overall == []


def test_partial_method_does_not_hide_an_independent_complete_method():
    rows = complete_rows("complete") + complete_rows("partial")[:-1]
    by_origin, overall = aggregate_rows(rows, origins=ORIGINS, seeds=SEEDS)
    assert len(by_origin) == 5
    assert [row["method"] for row in overall] == ["complete"]


@pytest.mark.parametrize("method", ["activity", "old/activity"])
def test_deterministic_activity_is_one_seedless_run_per_origin(method):
    rows = [summary_row(method, year, None, value, size)
            for year, value, size in zip(ORIGINS, (.1, .4, .7), (2, 100, 1000))]
    by_origin, overall = aggregate_rows(rows, origins=ORIGINS, seeds=SEEDS)
    assert len(by_origin) == 3
    assert all(row["n_seeds"] == 1 for row in by_origin)
    assert overall[0]["ndcg20"] == pytest.approx(.4)


def test_aggregation_rejects_duplicate_runs_before_computing_means():
    rows = complete_rows()
    with pytest.raises(ValueError):
        aggregate_rows(rows + [dict(rows[0], ndcg20=.9)], origins=ORIGINS, seeds=SEEDS)


def test_aggregation_of_no_evidence_is_empty():
    assert aggregate_rows([], origins=ORIGINS, seeds=SEEDS) == ([], [])


def test_seed_dependent_hit_counts_are_averaged_without_changing_population_counts():
    rows = complete_rows()
    for row, hits in zip(rows, (0, 1, 2, 3, 4, 5, 6, 7, 8)):
        row["items_hit20"] = hits
    by_origin, overall = aggregate_rows(rows, origins=ORIGINS, seeds=SEEDS)
    assert [row["items_hit20"] for row in by_origin] == [1, 4, 7]
    assert overall[0]["items_hit20_sum_over_origins"] == 12


def test_protocol_nested_dependency_changes_are_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    record = {"protocol": {"code_sha256": {"source.py": sha256(source)}}}
    source.write_text("version = 2\n")
    with pytest.raises(ValueError, match="Frozen code SHA256"):
        validate_dependencies(record, root=tmp_path)


def target_trajectory_fixture(tmp_path, final=False):
    """A small evidence fixture exercises receipts without fitting a model."""
    from saet_repro.extension_target import _json_hash

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    prefix = "cds_2015_d0" if final else "cds_2015_d30"
    data_path = data_dir / f"{prefix}.npz"
    data_path.write_bytes(b"frozen data fixture")
    seed, epochs = (11, 6) if final else (101, 50)
    directory = tmp_path / "activity_mean" / ("final" if final else "hpo") / (
        prefix if final else "trial_00") / f"seed_{seed}"
    directory.mkdir(parents=True)
    identity = {"target_protocol_sha256": "target-freeze", "global_protocol_sha256": "global-freeze"}
    run = {**identity, "prefix": prefix, "data_file_sha256": sha256(data_path),
           "phase": "final" if final else "historical_selection"}
    config = {"activity_mode": "mean"}
    frozen = {"seed": seed, "config": config, "run_identity": run,
              "evaluate_each_epoch": not final}
    history = [{"epoch": epoch, "training_seconds": 2., "evaluation_seconds_shared": 0. if final else 12.,
                "evaluation_trajectory_count": 36, "feature_bank_seconds_shared": 3.,
                "feature_bank_trajectory_count": 12} for epoch in range(1, epochs + 1)]
    (directory / "frozen.json").write_text(json.dumps(frozen))
    (directory / "history.json").write_text(json.dumps(history))
    names = ["frozen.json", "history.json", "last_state.npz", "last_state.sha256.json"]
    names += (["model.npz", "test.npz", "result.json"] if final else
              [f"checkpoint_{epoch:03d}.{suffix}" for epoch in (6, 20, 50)
               for suffix in ("npz", "json")])
    for name in names:
        if not (directory / name).exists():
            (directory / name).write_bytes(b"fixture")
    marker = {"artifacts": {name: sha256(directory / name) for name in names},
              "epochs": epochs, "target_protocol_sha256": identity["target_protocol_sha256"],
              "identity_sha256": _json_hash(frozen),
              "identity": {**run, "seed": seed, "config": config, "epochs": epochs}}
    (directory / "complete.json").write_text(json.dumps(marker))
    audit = Audit.__new__(Audit)
    audit.args = SimpleNamespace(data_dir=data_dir)
    audit.protocol = {"stage_b": {"hpo_prefix": "cds_2015_d30"}}
    audit.hashes, audit.curves, audit.budget = {}, [], []
    kwargs = dict(directory=directory, method="activity_mean", seed=seed, config=config,
                  identity=identity, trial=None if final else 0, final=final, epochs=epochs)
    return audit, kwargs, marker


@pytest.mark.parametrize("final", [False, True])
def test_target_trajectory_actual_receipt_schemas_are_accepted(tmp_path, final):
    audit, kwargs, _ = target_trajectory_fixture(tmp_path, final)
    audit.b_history(**kwargs)
    assert audit.budget[0]["historical_evaluation_seconds"] == pytest.approx(0 if final else kwargs["epochs"] / 3)
    assert audit.budget[0]["feature_bank_seconds_allocated"] == pytest.approx(kwargs["epochs"] / 4)


def test_target_missing_required_checkpoint_cannot_enter_a_report(tmp_path):
    audit, kwargs, marker = target_trajectory_fixture(tmp_path)
    del marker["artifacts"]["checkpoint_020.npz"]
    (kwargs["directory"] / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="prescribed artifacts"):
        audit.b_history(**kwargs)


def test_target_final_completion_cannot_claim_a_different_seed(tmp_path):
    audit, kwargs, marker = target_trajectory_fixture(tmp_path, final=True)
    marker["identity"]["seed"] = 22
    (kwargs["directory"] / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="completion identity"):
        audit.b_history(**kwargs)


def ridge_diagnostic_fixture():
    common = dict(n=10, zero_mse=.3, scaled_mse=1., label_rms=.5,
                  sign_accuracy=.6, positive_precision=.5,
                  predicted_positive_fraction=.4, label_positive_fraction=.5)
    trials = [dict(alpha=alpha, mse=mse, **common) for alpha, mse in ((.1, .2), (1., .1), (10., .1))]
    return {"n_validation_labels": 10, "diagnostics": {
        "full": {"selected_alpha": 1., "validation": {k: v for k, v in trials[1].items() if k != "alpha"},
                 "trials": trials}}}


def test_ridge_diagnostics_preserve_every_trial_and_first_exact_tie():
    rows = ridge_diagnostic_rows(ridge_diagnostic_fixture(), "activity_both", 2018, ("full",), (.1, 1., 10.))
    assert [row["alpha"] for row in rows] == [.1, 1., 10.]
    assert [row["selected"] for row in rows] == [False, True, False]
    assert all(row["row_kind"] == "ridge_validation" and row["offset"] == 60 for row in rows)
    assert [row["mse"] for row in rows] == [.2, .1, .1]


@pytest.mark.parametrize("corruption", ["missing_trial", "wrong_winner", "wrong_validation", "wrong_population", "missing_precision"])
def test_ridge_diagnostic_evidence_inconsistencies_are_rejected(corruption):
    document = ridge_diagnostic_fixture()
    selected = document["diagnostics"]["full"]
    if corruption == "missing_trial":
        selected["trials"].pop()
    elif corruption == "wrong_winner":
        selected["selected_alpha"] = 10.
    elif corruption == "wrong_validation":
        selected["validation"]["mse"] = .001
    elif corruption == "wrong_population":
        selected["trials"][0]["n"] = 9
    else:
        selected["trials"][0]["positive_precision"] = None
    with pytest.raises(ValueError):
        ridge_diagnostic_rows(document, "activity_both", 2018, ("full",), (.1, 1., 10.))


def test_ridge_zero_positive_predictions_keep_precision_undefined():
    document = ridge_diagnostic_fixture()
    selected = document["diagnostics"]["full"]
    for trial in selected["trials"]:
        trial["predicted_positive_fraction"] = 0.
        trial["positive_precision"] = None
    selected["validation"] = {k: v for k, v in selected["trials"][1].items() if k != "alpha"}
    rows = ridge_diagnostic_rows(document, "activity_both", 2018, ("full",), (.1, 1., 10.))
    assert all(row["positive_precision"] is None for row in rows)
