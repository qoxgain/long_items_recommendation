"""Numerical parity, continuation and frozen B1/B2 execution contracts."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from saet_repro import extension_cf as extension
from saet_repro.collaborative import CFConfig, CollaborativeModel, fit_edges, past_edges, search_space, train_cf
from saet_repro.io import json_safe, sha256, write_json
from test_collaborative import perturb_future, tiny_data


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extension_cf_runner_test", ROOT / "scripts/run_extension_cf.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def stable_runtime(monkeypatch):
    torch.set_num_threads(1)
    identity = lambda: {"schema": extension.SCHEMA, "code_sha256": {"test": "v1"},
                        "dependencies": {"torch": torch.__version__}}
    monkeypatch.setattr(extension, "code_identity", identity)
    monkeypatch.setattr(runner, "code_identity", identity)


def same_parameters(first, second):
    assert set(first.state_dict()) == set(second.state_dict())
    for key, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)


def deterministic_history(rows):
    ignored = {"training_seconds", "evaluation_seconds", "invocation_elapsed_seconds", "peak_rss_mib"}
    return json_safe([{key: value for key, value in row.items() if key not in ignored} for row in rows])


GRID_CASES = [pytest.param(replace(config, batch_size=4), id=f"{method}-{trial}")
              for method in ("mf_bpr", "lightgcn", "simplex")
              for trial, config in enumerate(extension.grid(method))]


@pytest.mark.parametrize("config", GRID_CASES)
def test_every_native_grid_matches_original_six_epoch_parameters_losses_counts(config):
    data = tiny_data()
    original, history = train_cf(data, config, 101, epochs=6)
    state = extension.train_trajectory(data, config, config.method, 101, 6, evaluate_each_epoch=True)
    same_parameters(state.model, original)
    for old, new in zip(history, state.history):
        assert all(new[key] == value for key, value in old.items())
        assert new["positive_edges"] == len(fit_edges(data))
        assert new["pair_count"] == len(fit_edges(data)) * (10 if config.method == "simplex" else 1)
        assert new["updates"] == 3
        assert new["cumulative_updates"] == new["epoch"] * 3
        assert new["gradient_norm_mean"] > 0 and new["update_norm_sum"] > 0


def test_aligned_grid_and_full_task_budget_preserve_native_grids():
    assert extension.grid("lightgcn_user") == search_space("lightgcn")
    for method in extension.METHODS:
        assert len(extension.grid(method)) == 12
    tasks = runner.task_manifest(extension.METHODS)
    assert len(tasks["hpo"]) == 144 and len(tasks["final"]) == 60
    assert {row["seed"] for row in tasks["hpo"]} == {101, 202, 303}


def test_negative_users_exclude_prefit_low_ratings_and_keep_future_uniformly():
    data = tiny_data()
    sampler = extension.UniformNegativeUserSampler(data)
    items = np.repeat(np.arange(6), 2000)
    sampled = sampler.sample(items, 3, np.random.default_rng(17))
    forbidden = {tuple(row) for row in past_edges(data)}
    assert all((int(user), int(item)) not in forbidden
               for item, users in zip(items, sampled) for user in users)
    # Item 4 has users 0 (low-rating, pre-fit), 2 and 3 in its past.
    np.testing.assert_array_equal(np.unique(sampled[items == 4]), [1])
    values, counts = np.unique(sampled[items == 0], return_counts=True)
    np.testing.assert_array_equal(values, [1, 2])
    assert np.all(np.abs(counts - 3000) < 180)
    assert 1 in sampled[items == 0]  # Future positive must remain available.
    other = extension.UniformNegativeUserSampler(perturb_future(data))
    np.testing.assert_array_equal(sampled, other.sample(items, 3, np.random.default_rng(17)))
    saturated = {**data, "target_train": np.vstack([data["target_train"], [[u, 5] for u in range(4)]])}
    with pytest.raises(ValueError, match="no unseen negative user"):
        extension.UniformNegativeUserSampler(saturated).sample([5], 1, np.random.default_rng(1))


def test_fixed_item_bpr_formula_and_gradients_match_independent_expression():
    torch.manual_seed(9)
    config = CFConfig(method="lightgcn", embedding_dim=4, layers=2, l2=.03)
    model = CollaborativeModel(tiny_data(), config).double()
    users, items, negatives = torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4]), torch.tensor([[3], [2], [1]])
    loss = extension.user_bpr_objective(model, users, items, negatives)
    all_users, all_items = model.propagated()
    margins = ((all_users[users] - all_users[negatives[:, 0]]) * all_items[items]).sum(dim=1)
    expected = torch.nn.functional.softplus(-margins).mean()
    expected += config.l2 * sum(t.square().sum() for t in
                               (model.user_embedding(users), model.user_embedding(negatives),
                                model.item_embedding(items))) / (2 * len(users))
    torch.testing.assert_close(loss, expected, rtol=1e-14, atol=1e-14)
    loss.backward()
    for parameter in model.parameters():
        index = np.unravel_index(int(parameter.grad.abs().argmax()), parameter.shape)
        derivative, original = float(parameter.grad[index]), float(parameter[index].detach())
        observed = []
        for delta in (1e-5, -1e-5):
            with torch.no_grad():
                parameter[index] = original + delta
            observed.append(float(extension.user_bpr_objective(model, users, items, negatives).detach()))
        with torch.no_grad():
            parameter[index] = original
        assert abs(derivative) > 1e-8
        np.testing.assert_allclose((observed[0] - observed[1]) / 2e-5, derivative, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("method", extension.METHODS)
def test_resume_matches_uninterrupted_adam_sampling_rng_and_history(method, tmp_path):
    config = replace(extension.grid(method)[0], embedding_dim=4, batch_size=4)
    direct = extension.train_trajectory(tiny_data(), config, method, 202, 6, evaluate_each_epoch=True)
    draws = (random.random(), np.random.random(5), torch.rand(5), direct.rng.random(5))
    extension.train_trajectory(tiny_data(), config, method, 202, 2, directory=tmp_path, evaluate_each_epoch=True)
    resumed = extension.train_trajectory(tiny_data(), config, method, 202, 6,
                                         directory=tmp_path, evaluate_each_epoch=True)
    same_parameters(direct.model, resumed.model)
    assert deterministic_history(direct.history) == deterministic_history(resumed.history)
    assert direct.step == resumed.step == 18
    assert direct.optimizer.state_dict()["param_groups"] == resumed.optimizer.state_dict()["param_groups"]
    for index, values in direct.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][index][key], rtol=0, atol=0)
    assert random.random() == draws[0]
    np.testing.assert_array_equal(np.random.random(5), draws[1])
    torch.testing.assert_close(torch.rand(5), draws[2], rtol=0, atol=0)
    np.testing.assert_array_equal(resumed.rng.random(5), draws[3])
    assert (tmp_path / "checkpoint_006.npz").exists()


@pytest.mark.parametrize("artifact", ["history.json", "frozen.json", "last_state.npz", "checkpoint_006.npz"])
def test_corruption_quarantines_and_recomputes_identical_model(artifact, tmp_path):
    directory = tmp_path / "trajectory"
    config = CFConfig(embedding_dim=4, batch_size=4)
    first = extension.train_trajectory(tiny_data(), config, "mf_bpr", 101, 6, directory=directory)
    (directory / artifact).write_bytes(b"corrupted")
    restored = extension.train_trajectory(tiny_data(), config, "mf_bpr", 101, 6, directory=directory)
    same_parameters(first.model, restored.model)
    assert deterministic_history(first.history) == deterministic_history(restored.history)
    stale, = tmp_path.glob("trajectory.stale_*")
    assert (stale / artifact).read_bytes() == b"corrupted"
    assert "checksum mismatch" in json.loads((stale / "stale_reason.json").read_text())["reason"]


@pytest.mark.parametrize("change", ["identity", "fit", "users", "items"])
def test_resume_binds_protocol_fit_and_cohort_even_without_runner(change, tmp_path):
    data, directory = tiny_data(), tmp_path / "trajectory"
    config = CFConfig(embedding_dim=4)
    extension.train_trajectory(data, config, "mf_bpr", 101, 2, directory=directory)
    changed = {key: value.copy() for key, value in data.items()}
    identity = None
    if change == "identity":
        identity = {"protocol": "changed"}
    elif change == "fit":
        changed["target_train"] = changed["target_train"][:-1]
    else:
        key = "target_user_ids" if change == "users" else "target_item_ids"
        changed[key] = changed[key][::-1]
    state = extension.train_trajectory(changed, config, "mf_bpr", 101, 2,
                                       directory=directory, identity=identity)
    direct = extension.train_trajectory(changed, config, "mf_bpr", 101, 2, identity=identity)
    same_parameters(state.model, direct.model)
    assert len(list(tmp_path.glob("trajectory.stale_*"))) == 1


def trajectories(extended=False):
    return [{"trial": trial, "seed": seed, "config": asdict(extension.grid("mf_bpr")[trial]),
             "history": [{"epoch": epoch, "historical_tail80_ndcg20":
                          .1 + .001 * trial + .01 * (epoch == 50) + .02 * (epoch == 100)
                          + (seed - 202) * 1e-5}
                         for epoch in ((6, 20, 50, 100) if extended else (6, 20, 50))]}
            for trial in range(12) for seed in extension.SELECTION_SEEDS]


def test_selection_uses_three_seed_means_declared_epochs_and_tie_order():
    rows = trajectories()
    records = extension.selection_records(rows)
    assert len(records) == 36
    winner = extension.select_checkpoint(records)
    assert (winner["trial"], winner["epochs"]) == (11, 50)
    assert winner["score"] == np.mean(list(winner["seed_scores"].values()))
    assert extension.extension_decision(records)["extend_to_100"]
    tied = [{"trial": t, "epochs": e, "score": .2} for t in (7, 2) for e in (50, 20, 6)]
    assert extension.select_checkpoint(tied) == {"trial": 2, "epochs": 6, "score": .2}
    # Non-checkpoint results must never win selection.
    for row in rows:
        row["history"].append({"epoch": 49, "historical_tail80_ndcg20": 1.})
    assert extension.selection_records(rows) == records


@pytest.mark.parametrize("fault", ["missing_seed", "duplicate", "config", "missing_epoch", "nonfinite"])
def test_selection_rejects_incomplete_or_inconsistent_matrix(fault):
    rows = trajectories()
    if fault == "missing_seed":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "config":
        rows[1]["config"] = {}
    elif fault == "missing_epoch":
        rows[0]["history"].pop()
    else:
        rows[0]["history"][0]["historical_tail80_ndcg20"] = float("nan")
    with pytest.raises(ValueError):
        extension.selection_records(rows)


@pytest.mark.parametrize("earlier,later,extend", [(0., 1e-4, True), (0., np.nextafter(1e-4, 0), False),
                                                (1e-8, 1.02e-8, True), (.1, .101, True),
                                                (.1, .1009, False)])
def test_extension_threshold_relative_and_strict_near_zero_boundary(earlier, later, extend):
    records = [{"trial": 0, "epochs": 20, "score": earlier}, {"trial": 0, "epochs": 50, "score": later}]
    decision = extension.extension_decision(records)
    assert decision["extend_to_100"] == extend
    assert decision["near_zero_denominator"] == (abs(earlier) < 1e-8)


def test_runner_extends_entire_family_grid_and_never_selects_partial(tmp_path, monkeypatch):
    calls = []
    def fake_run(data, path, method, trial, seed, epochs, directory, protocol, retry_failed):
        calls.append((trial, seed, epochs))
        row = next(row for row in trajectories(extended=True) if row["trial"] == trial and row["seed"] == seed)
        return {**row, "complete_sha256": f"{trial}-{seed}-{epochs}"}
    monkeypatch.setattr(runner, "run_historical", fake_run)
    path = tmp_path / "data.npz"
    path.write_bytes(b"synthetic")
    selected = runner.run_hpo({}, path, tmp_path, "mf_bpr", {})
    assert len(calls) == 72
    assert set(calls) == {(t, s, e) for t in range(12) for s in (101, 202, 303) for e in (50, 100)}
    assert selected["epochs"] == 100 and len(selected["trials"]) == 48
    broken = tmp_path / "broken"
    def fail_one(*args, **kwargs):
        if args[3:5] == (0, 101):
            raise RuntimeError("synthetic failure")
        return fake_run(*args, **kwargs)
    monkeypatch.setattr(runner, "run_historical", fail_one)
    with pytest.raises(RuntimeError, match="no selection produced"):
        runner.run_hpo({}, path, broken, "mf_bpr", {})
    assert not (broken / "mf_bpr/selected.json").exists()
    assert json.loads((broken / "mf_bpr/hpo_status.json").read_text())["complete_matrix"] is False


def test_historical_completion_rejects_omitted_artifacts_and_recovers_history(tmp_path):
    data, path, directory = tiny_data(), tmp_path / "data.npz", tmp_path / "hpo"
    np.savez(path, **data)
    args = (data, path, "mf_bpr", 0, 101, 2, directory, {"protocol": "test"})
    original = runner.run_historical(*args, pilot=True)
    complete = json.loads((directory / "complete.json").read_text())
    assert runner.verify_complete(directory, complete["run_identity"], 2) is not None
    incomplete = {**complete, "artifact_sha256": {}}
    write_json(directory / "complete.json", incomplete)
    assert runner.verify_complete(directory, complete["run_identity"], 2) is None
    write_json(directory / "complete.json", complete)
    write_json(directory / "history.json", [])
    recovered = runner.run_historical(*args, pilot=True)
    assert deterministic_history(original["history"]) == deterministic_history(recovered["history"])
    assert len(json.loads((directory / "history.json").read_text())) == 2
    assert len(list(tmp_path.glob("hpo.stale_*"))) == 1


@pytest.mark.parametrize("fault", [None, "coverage", "identity", "budget", "artifacts", "winner"])
def test_selection_validation_recomputes_complete_grid_provenance(tmp_path, monkeypatch, fault):
    path = tmp_path / "data.npz"
    path.write_bytes(b"synthetic")
    protocol = {"family_protocol_sha256": "synthetic-family"}
    def persisted_run(data, path, method, trial, seed, epochs, directory, protocol, retry_failed):
        row = next(row for row in trajectories(extended=True) if row["trial"] == trial and row["seed"] == seed)
        identity = {**protocol, **runner.code_identity(), "method": method, "trial": trial,
                    "seed": seed, "role": "hpo", "config": row["config"], "data_sha256": sha256(path)}
        directory.mkdir(parents=True, exist_ok=True)
        for name in runner.required_artifacts(identity, epochs):
            (directory / name).write_bytes(b"synthetic artifact")
        write_json(directory / "history.json", row["history"])
        runner.mark_complete(directory, identity, epochs, runner.required_artifacts(identity, epochs))
        return {**row, "complete_sha256": sha256(directory / "complete.json")}
    monkeypatch.setattr(runner, "run_historical", persisted_run)
    selected = runner.run_hpo({}, path, tmp_path, "mf_bpr", protocol)
    assert runner.validate_selection(tmp_path, "mf_bpr", protocol) == selected
    if fault is None:
        return
    family = tmp_path / "mf_bpr"
    if fault == "coverage":
        value = selected["trajectory_complete_sha256"].pop("trial_00/seed_101")
        selected["trajectory_complete_sha256"]["trial_12/seed_101"] = value
    elif fault == "winner":
        selected["score"] += .1
        selected["criterion_value"] = selected["score"]
    else:
        name = "trial_00/seed_101"
        complete_path = family / "hpo" / name / "complete.json"
        completion = json.loads(complete_path.read_text())
        if fault == "identity":
            completion["run_identity"]["config"]["lr"] = 1.
        elif fault == "budget":
            completion["epochs"] = 50
        else:
            completion["artifact_sha256"] = {}
        write_json(complete_path, completion)
        selected["trajectory_complete_sha256"][name] = sha256(complete_path)
    write_json(family / "selected.json", selected)
    write_json(family / "selected.sha256.json", {"sha256": sha256(family / "selected.json")})
    with pytest.raises(ValueError):
        runner.validate_selection(tmp_path, "mf_bpr", protocol)


def test_runner_does_not_extend_when_epoch20_wins(tmp_path, monkeypatch):
    calls = []
    def fake_run(data, path, method, trial, seed, epochs, directory, protocol, retry_failed):
        calls.append(epochs)
        row = next(row for row in trajectories() if row["trial"] == trial and row["seed"] == seed)
        for metric in row["history"]:
            metric["historical_tail80_ndcg20"] = .3 if metric["epoch"] == 20 else .1
        return {**row, "complete_sha256": f"{trial}-{seed}-{epochs}"}
    monkeypatch.setattr(runner, "run_historical", fake_run)
    path = tmp_path / "data.npz"
    path.write_bytes(b"synthetic")
    selected = runner.run_hpo({}, path, tmp_path, "mf_bpr", {})
    assert calls == [50] * 36 and selected["epochs"] == 20
    assert {row["epochs"] for row in selected["trials"]} == {6, 20, 50}


def test_final_query_gate_precedes_scoring_and_corrupt_evaluation_is_recomputed(tmp_path, monkeypatch):
    data, path = tiny_data(), tmp_path / "cds_2015_d0.npz"
    np.savez(path, **data)
    selected = {"config": asdict(CFConfig(embedding_dim=4, batch_size=4)), "epochs": 2}
    write_json(tmp_path / "mf_bpr/selected.json", selected)
    directory = tmp_path / "mf_bpr/final/cds_2015_d0/seed_11"
    real_evaluate, calls = runner.evaluate, []
    def checked_evaluate(*args, **kwargs):
        receipt = json.loads((directory / "final_query_opened.json").read_text())
        assert receipt["model_sha256"] == sha256(directory / "model.npz")
        assert receipt["identity"]["selected_sha256"] == sha256(tmp_path / "mf_bpr/selected.json")
        assert len(json.loads((directory / "history.json").read_text())) == 2
        calls.append(True)
        return real_evaluate(*args, **kwargs)
    monkeypatch.setattr(runner, "evaluate", checked_evaluate)
    args = (data, path, tmp_path, "mf_bpr", 11, selected, {"protocol": "test"})
    first = runner.run_final(*args)
    assert runner.run_final(*args) == first and len(calls) == 1
    (directory / "test.npz").write_bytes(b"corrupt")
    recovered = runner.run_final(*args)
    assert recovered["summary"] == first["summary"] and len(calls) == 2
    complete = json.loads((directory / "complete.json").read_text())
    assert runner.verify_complete(directory, complete["run_identity"], 2) is not None
