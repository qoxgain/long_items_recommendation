"""Evidence-boundary tests for reporting, independent of expensive fitting."""
from collections import Counter
import json

import numpy as np
import pytest

from scripts.summarize_results import (
    ROOT, DETERMINISTIC, METRICS, POLICIES, SEEDS, YEARS, aggregate_runs, build_report,
    expected_cells, make_comparisons, validate_evaluation, validate_hpo_selection,
    validate_utility_selection,
)
from saet_repro.evaluation import evaluate, stratified_summaries
from saet_repro.io import sha256, write_json


def test_expected_matrix_is_explicit_and_has_full_frozen_budgets(tmp_path):
    cells = expected_cells(tmp_path / "data", tmp_path / "results")
    counts = Counter(cell.group for cell in cells)
    assert counts == {
        "data_prefixes": 18, "targets_selected": 5, "targets_hpo": 60,
        "targets_final": 75, "baselines_selected": 3, "baselines_hpo": 36,
        "baselines_final": 45, "deterministic_final": 6,
        "utility_historical": 12, "utility_ridges": 3,
        "utility_primary": 105, "utility_mechanism": 180,
        "utility_selected": 1, "utility_hpo": 73,
    }
    assert len({(cell.group, cell.name) for cell in cells}) == len(cells)
    assert {cell.seed for cell in cells if cell.group == "deterministic_final"} == {None}
    assert {cell.seed for cell in cells if cell.group == "targets_final"} == set(SEEDS)


def summary_row(method, origin, seed, value, size=10):
    return dict(method=method, origin=origin, seed=seed, stratum="tail80",
                items_catalog=size, items_evaluated=size, items_without_positive=0,
                n_positive=size, **{metric: value for metric in METRICS})


def test_means_require_complete_seeds_and_weight_origins_equally():
    rows = [summary_row("activity_mean", year, seed, value, size)
            for year, value, size in ((2015, .1, 2), (2016, .4, 100), (2017, .7, 1000))
            for seed in SEEDS]
    origins, overall = aggregate_runs(rows)
    assert len(origins) == 3
    assert overall[0]["ndcg20"] == pytest.approx(.4)
    assert overall[0]["n_runs"] == 15
    assert overall[0]["origin_weighting"] == "equal"
    origins, overall = aggregate_runs(rows[:-1])
    assert len(origins) == 2
    assert overall == []
    deterministic = [summary_row("cosine", year, None, value)
                     for year, value in zip(YEARS, (.1, .4, .7))]
    _, overall = aggregate_runs(deterministic)
    assert overall[0]["n_runs"] == 3
    assert overall[0]["ndcg20"] == pytest.approx(.4)
    with pytest.raises(ValueError, match="Duplicate run"):
        aggregate_runs(rows + [rows[0]])
    rows[-1]["items_evaluated"] -= 1
    with pytest.raises(ValueError, match="population"):
        aggregate_runs(rows)


def test_reporting_pairs_deterministic_baseline_without_exporting_item_ids():
    paired = {}
    rows = []
    for year in YEARS:
        paired["cosine", year, None] = dict(item_id=np.array(["private-item"]), ndcg20=np.array([.3]))
        rows.append(summary_row("cosine", year, None, .3))
        for seed in SEEDS:
            paired["activity_mean", year, seed] = dict(item_id=np.array(["private-item"]), ndcg20=np.array([.2]))
            rows.append(summary_row("activity_mean", year, seed, .2))
    _, overall = aggregate_runs(rows)
    effects = make_comparisons(overall, paired, 50)
    assert len(effects) == 1
    assert effects[0]["mean_difference"] == pytest.approx(.1)
    assert effects[0]["n_seed_pairs"] == 15
    assert effects[0]["n_item_origins"] == 3
    assert "private-item" not in json.dumps(effects)
    assert "items" not in effects[0]
    assert effects[0]["origin_consistency"] == dict(positive=3, negative=0, zero=0)


def test_selection_must_match_trial_evidence_winner_and_checkpoint(tmp_path):
    directory = tmp_path / "search"
    directory.mkdir()
    trials = []
    for index in range(12):
        config = {"method": "activity_mean", "trial_parameter": index}
        score = .2 if index in (3, 7) else .1
        trials.append(dict(trial=index, score=score, config=config))
        write_json(directory / f"trial_{index:02d}.json", dict(
            method="activity_mean", seed=101, config=config,
            summary={"1": {"tail80": {"ndcg20": score}}},
            frozen={"seed": 101, "config": config}, model_sha256="winning-checkpoint-hash"))
    selected = dict(trials[3], trials=trials, frozen={"seed": 101, "config": trials[3]["config"]})
    checked = []
    def verify_hash(path, expected):
        checked.append((path, expected))
    validate_hpo_selection(tmp_path, selected, "targets", "activity_mean", verify_hash)
    assert checked == [(tmp_path / "selected_model.npz", "winning-checkpoint-hash")]
    selected["trial"] = 7
    with pytest.raises(ValueError, match="HPO winner"):
        validate_hpo_selection(tmp_path, selected, "targets", "activity_mean", verify_hash)
    selected["trial"] = 3
    selected["trials"][0]["score"] = .9
    with pytest.raises(ValueError, match="actual trial results"):
        validate_hpo_selection(tmp_path, selected, "targets", "activity_mean", verify_hash)


def test_utility_selection_requires_complete_trial_evidence_and_prespecified_ties(tmp_path):
    (tmp_path / "search").mkdir()
    configurations, trials, policies = {}, [], {}
    for policy in POLICIES:
        for index in range(1 if policy == "zero" else 12):
            name = "zero" if policy == "zero" else f"{policy}_{index:02d}"
            config = dict(policy=policy, radius=0. if policy == "zero" else .1, temperature=index + 1.)
            configurations[name] = config
            score = .2 if index in (3, 7) else .1
            trials.append(dict(name=name, configuration=config, score=score))
            write_json(tmp_path / "search" / f"{name}.json", dict(
                name=name, configuration=config, summary={"1": {"tail80": {"ndcg20": score}}}))
        policies[policy] = configurations["zero" if policy == "zero" else f"{policy}_03"]
    document, frozen = dict(trials=trials, policies=policies), dict(configurations=configurations)
    validate_utility_selection(tmp_path, document, frozen)
    policies["full"] = configurations["full_07"]
    with pytest.raises(ValueError, match="HPO winner"):
        validate_utility_selection(tmp_path, document, frozen)
    policies["full"] = configurations["full_03"]
    trials[-1]["score"] = .9
    with pytest.raises(ValueError, match="frozen trial evidence"):
        validate_utility_selection(tmp_path, document, frozen)


def ranking_fixture():
    return dict(target_item_ids=np.array(["private-item-a", "private-item-b"]),
                target_user_ids=np.array(["private-user-a", "private-user-b", "private-user-c"]),
                target_degree=np.array([1, 1]),
                target_observed=np.array([[0, 0, 0], [1, 1, 0]]),
                target_positive=np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1]]),
                target_tail80=np.array([True, True]), target_tail50=np.array([True, False]))


def test_full_candidate_arrays_are_recomputed_and_partial_or_fabricated_metrics_fail(tmp_path):
    data = ranking_fixture()
    result = evaluate(data, 1, lambda items: np.ones((len(items), 3)))
    path = tmp_path / "test.npz"
    np.savez(path, **result)
    stored = {"1": stratified_summaries(data, result)}
    actual, paired = validate_evaluation(path, data, stored)
    assert actual["tail80"]["coverage20"] == .5
    assert len(paired["ndcg20"]) == 1
    stored["1"]["tail80"]["ndcg20"] = .123
    with pytest.raises(ValueError, match="differs from arrays"):
        validate_evaluation(path, data, stored)
    result["requested"][0] = False
    np.savez(path, **result)
    with pytest.raises(ValueError, match="entire warm catalog"):
        validate_evaluation(path, data)
    result["requested"][0] = True
    result["ndcg20"][1] = 0.
    np.savez(path, **result)
    with pytest.raises(ValueError, match="remain undefined"):
        validate_evaluation(path, data)


def publish_deterministic_fixture(tmp_path):
    data_dir, result_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    data = ranking_fixture()
    for year in YEARS:
        prefix = f"cds_{year}_d0"
        path = data_dir / (prefix + ".npz")
        np.savez(path, **data)
        write_json(path.with_suffix(".json"), dict(
            data_sha256=sha256(path), code_sha256=sha256(ROOT / "saet_repro/data.py"),
            offset_days=0, fit_start=0, fit_end=365, query_end=395, elapsed_seconds=.2,
            counts={"target": {"users": 3, "items": 2}, "source": {"users": 2, "items": 3}}))
        for method in DETERMINISTIC:
            directory = result_dir / "targets" / prefix / method / "seed_0"
            directory.mkdir(parents=True)
            result = evaluate(data, 1, lambda items: np.ones((len(items), 3)))
            np.savez(directory / "test.npz", **result)
            write_json(directory / "result.json", dict(method=method, seed=0,
                       summary={"1": stratified_summaries(data, result)}))
            write_json(directory / "frozen.json", dict(method=method, data_sha256=sha256(path),
                       target_code_sha256=sha256(ROOT / "saet_repro/target.py")))
    return data_dir, result_dir


def test_partial_snapshot_is_honest_public_and_strict_mode_does_not_overwrite(tmp_path):
    data_dir, results_dir = publish_deterministic_fixture(tmp_path)
    output_dir = tmp_path / "report"
    summary = build_report(data_dir, results_dir, output_dir, allow_partial=True, n_bootstrap=50)
    assert summary["complete"] is False
    assert summary["status"]["deterministic_final"]["completed"] == 6
    assert summary["status"]["data_prefixes"]["completed"] == 3
    assert {row["method"] for row in summary["overall"]} == set(DETERMINISTIC)
    assert not summary["comparisons"]
    assert summary["historical_values_used_in_new_results"] is False
    original = (output_dir / "REPRODUCTION_REPORT.md").read_text()
    assert "实验尚未完成" in original
    assert "原第二轮 A 阶段" in original
    for path in output_dir.iterdir():
        text = path.read_text()
        assert "private-user" not in text
        assert "private-item" not in text
        assert "NaN" not in text
    with pytest.raises(ValueError, match="matrix is incomplete"):
        build_report(data_dir, results_dir, output_dir)
    assert (output_dir / "REPRODUCTION_REPORT.md").read_text() == original
    # Corrupt a saved result; it must be marked invalid rather than enter means.
    result_path = results_dir / "targets/cds_2015_d0/cosine/seed_0/result.json"
    document = json.loads(result_path.read_text())
    document["summary"]["1"]["tail80"]["ndcg20"] += .1
    write_json(result_path, document)
    summary = build_report(data_dir, results_dir, output_dir, allow_partial=True, n_bootstrap=50)
    assert summary["status"]["deterministic_final"]["invalid"] == 1
    assert {row["method"] for row in summary["overall"]} == {"activity"}
