"""Regression tests for exact target continuation and historical selection."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from saet_repro import extension_target as extension
from saet_repro.evaluation import evaluate
from saet_repro.target import Config, FeatureStore, Model, train_target
from test_target import tiny_data


@pytest.fixture(autouse=True)
def isolated_training(monkeypatch):
    monkeypatch.delenv("SAET_FEATURE_CACHE", raising=False)
    # File changes elsewhere in the checkout must not affect a unit-test run.
    monkeypatch.setattr(extension, "code_identity", lambda: {
        "schema": extension.SCHEMA,
        "code_sha256": {"test_source": "version_one"},
        "dependencies": {"numpy": np.__version__},
    })


def query_data():
    data = tiny_data()
    data["target_positive"] = np.array([
        [2, 0, 1], [1, 1, 1], [1, 2, 1], [1, 3, 1], [0, 4, 1],
    ], dtype=np.int64)
    data["target_degree"] = np.bincount(data["target_train"][:, 1], minlength=5)
    data["target_tail80"] = np.array([True, True, True, True, False])
    data["target_tail50"] = np.array([True, True, True, False, False])
    return data


def assert_same_model(first, second):
    np.testing.assert_array_equal(first.theta, second.theta)
    np.testing.assert_array_equal(first.beta, second.beta)
    assert first.alpha == second.alpha


def deterministic_history(history):
    timing = {
        "training_seconds", "feature_bank_seconds_shared", "evaluation_seconds_shared",
        "invocation_elapsed_seconds", "peak_rss_mib",
    }
    return [{key: value for key, value in row.items() if key not in timing}
            for row in history]


GRID_CASES = [
    pytest.param(replace(config, batch_size=2),
                 id=f"{method}-lr{config.lr}-rho{config.rho}-tau{config.tau}")
    for method in extension.METHODS for config in extension.grid(method)
]


def test_grid_retains_all_twelve_predeclared_trials_per_method():
    expected = [(lr, rho, tau) for lr in (.001, .003, .01)
                for rho in (0., .0001) for tau in (.1, .3)]
    for method in extension.METHODS:
        configs = extension.grid(method)
        assert [(config.lr, config.rho, config.tau) for config in configs] == expected
        assert all(config.method == method for config in configs)


@pytest.mark.parametrize("config", GRID_CASES)
def test_six_epoch_grid_matches_original_parameters_losses_and_counts(config):
    data = query_data()
    snapshots = []

    def remember_original(epoch, model):
        snapshots.append((model.theta.copy(), model.beta.copy(), model.alpha))

    original, history = train_target(
        data, config, seed=101, epochs=6, validation_callback=remember_original,
    )

    def check_epoch(epoch, active):
        state = active[0]
        theta, beta, alpha = snapshots[epoch - 1]
        np.testing.assert_array_equal(state.model.theta, theta)
        np.testing.assert_array_equal(state.model.beta, beta)
        assert state.model.alpha == alpha

    state, = extension.train_trajectories(
        data, [extension.Trajectory(config, 101)], epochs=6, callback=check_epoch,
    )
    assert_same_model(state.model, original)
    assert len(state.history) == 6
    for old, new in zip(history, state.history):
        for key in ("epoch", "loss", "visited_items", "pair_count", "updates",
                    "theta_norm", "alpha", "beta"):
            assert new[key] == old[key]
        assert new["visited_items"] == 5
        assert new["positive_count"] == 10
        assert new["pair_count"] == 20
        assert new["updates"] == 3
        assert new["cumulative_updates"] == 3 * new["epoch"]


@pytest.mark.parametrize("method", extension.METHODS)
def test_split_resume_preserves_parameters_adam_rng_and_history(tmp_path, method):
    data = query_data()
    config = Config(method=method, lr=.003, rho=.0001, tau=.3,
                    batch_size=2, max_positive=1, max_negative=1)
    full, = extension.train_trajectories(
        data, [extension.Trajectory(config, 202)], 8, evaluate_each_epoch=True,
    )
    directory = tmp_path / "resumed"
    spec = extension.Trajectory(config, 202, directory)
    partial, = extension.train_trajectories(
        data, [spec], 3, evaluate_each_epoch=True, checkpoints=(3, 6, 8),
    )
    assert partial.epoch == 3 and partial.step == 9
    resumed_epochs = []
    resumed, = extension.train_trajectories(
        data, [spec], 8, evaluate_each_epoch=True, checkpoints=(3, 6, 8),
        callback=lambda epoch, active: resumed_epochs.append(epoch),
    )
    assert resumed_epochs == [4, 5, 6, 7, 8]
    assert_same_model(resumed.model, full.model)
    np.testing.assert_array_equal(resumed.m, full.m)
    np.testing.assert_array_equal(resumed.v, full.v)
    assert resumed.rng.bit_generator.state == full.rng.bit_generator.state
    assert resumed.epoch == full.epoch == 8
    assert resumed.step == full.step == 24
    assert resumed.cumulative_update_norm == full.cumulative_update_norm
    assert deterministic_history(resumed.history) == deterministic_history(full.history)
    for epoch in (3, 6, 8):
        assert (directory / f"checkpoint_{epoch:03d}.npz").is_file()
        assert (directory / f"checkpoint_{epoch:03d}.json").is_file()
    repeated_epochs = []
    repeated, = extension.train_trajectories(
        data, [spec], 8, evaluate_each_epoch=True,
        callback=lambda epoch, active: repeated_epochs.append(epoch),
    )
    assert repeated_epochs == []
    assert_same_model(repeated.model, full.model)
    np.testing.assert_array_equal(repeated.m, full.m)
    np.testing.assert_array_equal(repeated.v, full.v)


@pytest.mark.parametrize("changed", [
    "caller_identity", "config", "seed", "code", "fit_features", "query_positive",
    "degree", "tail80", "tail50", "item_ids", "evaluation_mode",
])
def test_stale_identity_is_quarantined_and_recomputed(tmp_path, monkeypatch, changed):
    data = query_data()
    config, seed = Config(batch_size=2), 101
    identity = {"experiment": "original"}
    directory = tmp_path / "trajectory"
    extension.train_trajectories(
        data, [extension.Trajectory(config, seed, directory)], 1,
        identity=identity, evaluate_each_epoch=True,
    )
    evaluate_each_epoch = True
    if changed == "caller_identity":
        identity = {"experiment": "changed"}
    elif changed == "config":
        config = replace(config, lr=.003)
    elif changed == "seed":
        seed = 202
    elif changed == "code":
        monkeypatch.setattr(extension, "code_identity", lambda: {"schema": "changed_source"})
    elif changed == "fit_features":
        data["target_x"][0, 0] += .125
    elif changed == "query_positive":
        data["target_positive"][0, 0] = 3
    elif changed == "degree":
        data["target_degree"][0] += 1
    elif changed in ("tail80", "tail50"):
        data[f"target_{changed}"][0] = False
    elif changed == "item_ids":
        data["target_item_ids"][0] = "q"
    elif changed == "evaluation_mode":
        evaluate_each_epoch = False
    visited = []
    state, = extension.train_trajectories(
        data, [extension.Trajectory(config, seed, directory)], 1,
        identity=identity, evaluate_each_epoch=evaluate_each_epoch,
        callback=lambda epoch, active: visited.append(epoch),
    )
    assert visited == [1]
    assert state.epoch == 1
    stale, = tmp_path.glob("trajectory.stale_*")
    assert (stale / "last_state.npz").is_file()
    reason = json.loads((stale / "stale_reason.json").read_text())
    assert "identity" in reason["reason"]
    assert (directory / "last_state.npz").is_file()


def test_damaged_state_checksum_restarts_training(tmp_path):
    data = query_data()
    spec = extension.Trajectory(Config(batch_size=2), 101, tmp_path / "trajectory")
    expected, = extension.train_trajectories(data, [spec], 2)
    with (spec.directory / "last_state.npz").open("ab") as handle:
        handle.write(b"damaged state")
    epochs = []
    restored, = extension.train_trajectories(
        data, [spec], 2, callback=lambda epoch, active: epochs.append(epoch),
    )
    assert epochs == [1, 2]
    assert_same_model(expected.model, restored.model)
    stale, = tmp_path.glob("trajectory.stale_*")
    reason = json.loads((stale / "stale_reason.json").read_text())
    assert "checksum" in reason["reason"]


@pytest.mark.parametrize("damaged", ("model", "manifest_identity", "missing_checkpoint"))
def test_damaged_checkpoint_is_quarantined_even_when_last_state_is_valid(tmp_path, damaged):
    data = query_data()
    spec = extension.Trajectory(Config(batch_size=2), 101, tmp_path / "trajectory")
    expected, = extension.train_trajectories(data, [spec], 2, checkpoints=(1, 2))
    checkpoint = spec.directory / "checkpoint_001.npz"
    if damaged == "model":
        with checkpoint.open("ab") as handle:
            handle.write(b"damaged checkpoint")
    elif damaged == "manifest_identity":
        path = checkpoint.with_suffix(".json")
        manifest = json.loads(path.read_text())
        manifest["identity_sha256"] = "stale identity"
        path.write_text(json.dumps(manifest))
    else:
        checkpoint.unlink()
    epochs = []
    restored, = extension.train_trajectories(
        data, [spec], 2, checkpoints=(1, 2),
        callback=lambda epoch, active: epochs.append(epoch),
    )
    assert epochs == [1, 2]
    assert_same_model(expected.model, restored.model)
    stale, = tmp_path.glob("trajectory.stale_*")
    assert (stale / "stale_reason.json").is_file()
    assert (spec.directory / "checkpoint_001.npz").is_file()


def test_compatible_trials_share_one_bank_per_seed_and_one_evaluation_pass(monkeypatch):
    data = query_data()
    configs = [Config(batch_size=2), Config(lr=.01, rho=0., tau=.3, batch_size=3)]
    trajectories = [extension.Trajectory(configs[0], 101),
                    extension.Trajectory(configs[1], 101),
                    extension.Trajectory(configs[0], 202)]
    bank_calls, evaluated_items = [], []
    make_bank = extension._make_epoch_bank
    all_features = FeatureStore.all_features

    def counted_bank(store, rng):
        bank_calls.append(store)
        return make_bank(store, rng)

    def counted_features(store, item):
        evaluated_items.append(item)
        return all_features(store, item)

    monkeypatch.setattr(extension, "_make_epoch_bank", counted_bank)
    monkeypatch.setattr(FeatureStore, "all_features", counted_features)
    states = extension.train_trajectories(
        data, trajectories, 3, evaluate_each_epoch=True,
    )
    assert len(bank_calls) == 3 * 2
    # Frozen query features are constructed once, then reused across epochs.
    assert evaluated_items == list(range(5))
    assert all(state.model.store is states[0].model.store for state in states)
    assert all(store is states[0].model.store for store in bank_calls)
    for state, expected_shared_count in zip(states, (2, 2, 1)):
        assert all(row["feature_bank_trajectory_count"] == expected_shared_count
                   for row in state.history)
        original, _ = train_target(data, state.spec.config, state.spec.seed, epochs=3)
        assert_same_model(state.model, original)


@pytest.mark.parametrize("method", extension.METHODS)
@pytest.mark.parametrize("save_topk", (False, True))
def test_shared_evaluation_exactly_matches_individual_evaluation(method, save_topk, monkeypatch):
    data = query_data()
    states = extension.train_trajectories(
        data, [extension.Trajectory(Config(method=method, lr=lr), 101)
               for lr in (.001, .01)], 2,
    )
    models = [state.model for state in states]
    expected = [evaluate(data, 1, model.score_all, ks=(1, 2, 20), save_topk=save_topk)
                for model in models]
    store = models[0].store
    all_features = store.all_features
    visited = []

    def counted_features(item):
        visited.append(item)
        return all_features(item)

    monkeypatch.setattr(store, "all_features", counted_features)
    actual = extension.evaluate_models(data, models, ks=(1, 2, 20), save_topk=save_topk)
    assert visited == list(range(5))
    for first, second in zip(expected, actual):
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_different_representations_and_stores_cannot_be_shared():
    data = query_data()
    configs = [Config(method=method) for method in extension.METHODS]
    with pytest.raises(ValueError, match="representations"):
        extension.train_trajectories(
            data, [extension.Trajectory(config, 101) for config in configs], 1,
        )
    models = [Model(configs[0], FeatureStore(data, configs[0])) for _ in range(2)]
    with pytest.raises(ValueError, match="FeatureStore"):
        extension.evaluate_models(data, models)


def selection_states():
    return [SimpleNamespace(
        spec=extension.Trajectory(config, seed),
        history=[{"epoch": epoch, "historical_tail80_ndcg20": score}
                 for epoch, score in ((6, .2), (7, .99), (20, .3), (50, .4))],
    ) for config in (Config(lr=.001), Config(lr=.003))
        for seed in extension.SELECTION_SEEDS]


def test_selection_uses_only_declared_checkpoints_and_all_seeds():
    records = extension.selection_records(selection_states())
    assert len(records) == 6
    assert {row["epochs"] for row in records} == {6, 20, 50}
    assert {row["trial"] for row in records} == {0, 1}
    assert all(set(row["seed_scores"]) == {"101", "202", "303"} for row in records)
    assert extension.select_checkpoint(records)["epochs"] == 50
    with pytest.raises(ValueError, match="all declared seeds"):
        extension.selection_records(selection_states()[:-1])
    states = selection_states()
    states[0].history = states[0].history[1:]
    with pytest.raises(ValueError, match="Missing"):
        extension.selection_records(states)


@pytest.mark.parametrize("invalid", (None, float("nan"), float("inf")))
def test_selection_rejects_undefined_endpoint(invalid):
    states = selection_states()
    states[0].history[0]["historical_tail80_ndcg20"] = invalid
    with pytest.raises(ValueError, match="Undefined"):
        extension.selection_records(states)


def test_selection_ties_prefer_earlier_epoch_then_lower_grid_trial():
    records = [{"trial": trial, "epochs": epoch, "score": .5}
               for trial, epoch in ((2, 20), (1, 50), (1, 6), (0, 6))]
    for ordered in (records, records[::-1]):
        winner = extension.select_checkpoint(ordered)
        assert (winner["trial"], winner["epochs"]) == (0, 6)
    with pytest.raises(ValueError, match="No selectable"):
        extension.select_checkpoint([])


@pytest.mark.parametrize("score20,score50,expected,threshold,near_zero", [
    (.5, .505, True, .005, False),
    (.5, .504999, False, .005, False),
    (0., .0001, True, .0001, True),
    (0., .00009999, False, .0001, True),
    (.2, .2, False, .002, False),
])
def test_extension_to_100_uses_predeclared_relative_or_near_zero_threshold(
        score20, score50, expected, threshold, near_zero):
    records = [{"trial": 0, "epochs": epoch, "score": score}
               for epoch, score in ((6, 0.), (20, score20), (50, score50))]
    decision = extension.extension_decision(records)
    assert decision["extend_to_100"] is expected
    assert decision["threshold"] == threshold
    assert decision["near_zero_denominator"] is near_zero
    assert decision["same_config_epoch20_score"] == score20


def test_extension_compares_winner_with_its_own_epoch20_and_requires_epoch50_win():
    records = [{"trial": trial, "epochs": epoch, "score": score}
               for trial, epoch, score in ((0, 6, .7), (0, 20, .8), (0, 50, .805),
                                          (1, 6, .05), (1, 20, .1), (1, 50, .7))]
    decision = extension.extension_decision(records)
    assert decision["same_config_epoch20_score"] == .8
    assert decision["extend_to_100"] is False
    records[0]["score"] = .9
    decision = extension.extension_decision(records)
    assert decision["winner_at_50_stage"]["epochs"] == 6
    assert decision["extend_to_100"] is False

@pytest.mark.parametrize('method', ['activity_mean', 'activity_both'])
def test_bounded_query_cache_preserves_training_and_metrics(method):
    data = query_data()
    specs = [extension.Trajectory(Config(method=method, batch_size=2), 101),
             extension.Trajectory(Config(method=method, lr=.01, batch_size=3), 202)]
    cached = extension.train_trajectories(data, specs, 3, evaluate_each_epoch=True)
    uncached = extension.train_trajectories(data, specs, 3, evaluate_each_epoch=True,
                                           historical_cache_max_bytes=0)
    for left, right in zip(cached, uncached):
        np.testing.assert_array_equal(left.model.theta, right.model.theta)
        np.testing.assert_array_equal(left.model.beta, right.model.beta)
        assert left.model.alpha == right.model.alpha
        assert [x['historical_summary'] for x in left.history] == [x['historical_summary'] for x in right.history]
        assert left.history[-1]['historical_feature_cache_items'] == 5
        assert right.history[-1]['historical_feature_cache_bytes'] == 0


def test_query_cache_budget_and_store_identity():
    data = query_data()
    store = FeatureStore(data, Config(method='activity_both'))
    features = store.all_features(0)
    budget = sum(a.nbytes for a in features)
    cache = extension.HistoricalFeatureCache(store, budget)
    for item in (0, 1, 0, 1):
        actual = cache.get(item)
        expected = store.all_features(item)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    assert len(cache) == 1
    assert cache.nbytes <= budget
    other = Model(Config(method='activity_both'), FeatureStore(data, Config(method='activity_both')))
    with pytest.raises(ValueError, match='FeatureStore'):
        extension.evaluate_models(data, [other], feature_cache=cache)
