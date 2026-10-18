"""Executable CF protocol checks on a small, genuinely held-out graph."""
from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from saet_repro.collaborative import (
    CFConfig, CollaborativeModel, UniformNegativeSampler, fit_edges,
    history_table, normalized_graph, past_edges, search_space, train_cf,
)


METHODS = ("mf_bpr", "lightgcn", "simplex")


def tiny_data():
    train = np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 3],
                      [2, 2], [2, 4], [3, 0], [3, 4]], dtype=np.int64)
    # A low-rating pre-fit observation must also be forbidden for negatives.
    observed = np.vstack((np.c_[train, np.zeros(len(train), int)], [0, 4, -1]))
    query = np.array([[1, 0, 1], [2, 1, 1], [3, 2, 1]], dtype=np.int64)
    degree = np.bincount(train[:, 1], minlength=6)
    result = dict(
        fit_start=np.array(10), fit_end=np.array(100), query_end=np.array(130),
        target_user_ids=np.array(list("abcd")),
        target_item_ids=np.array(list("uvwxyz")), target_train=train,
        target_observed=np.vstack((observed, query)),
        target_positive=np.vstack((np.c_[train, np.zeros(len(train), int)], query)),
        target_event_users=np.r_[train[:, 0], 0, query[:, 0]],
        target_event_items=np.r_[train[:, 1], 4, query[:, 1]],
        target_event_times=np.r_[np.arange(20, 29), 5, [101, 102, 103]],
        target_event_ratings=np.r_[np.full(len(train), 5), 1, [5, 5, 5]],
        target_all_rating_users=train[:, 0], target_all_rating_items=train[:, 1],
        target_all_rating_times=np.arange(20, 29),
        target_all_rating_ratings=np.full(len(train), 5), target_degree=degree,
        target_tail80=np.array([True, True, True, True, True, False]),
        target_tail50=np.array([False, False, False, True, True, True]),
    )
    return result


def perturb_future(data):
    changed = {key: np.array(value, copy=True) for key, value in data.items()}
    future = changed["target_event_times"] >= int(data["fit_end"])
    changed["target_event_users"][future] = 0
    changed["target_event_items"][future] = 3
    changed["target_event_ratings"][future] = 1
    changed["target_observed"][changed["target_observed"][:, 2] == 1, :2] = [0, 3]
    changed["target_positive"] = changed["target_positive"][changed["target_positive"][:, 2] < 1]
    return changed


def test_prespecified_twelve_trial_grids():
    for method in METHODS:
        grid = search_space(method)
        assert len(grid) == len({str(asdict(row)) for row in grid}) == 12
        assert all(row.method == method for row in grid)
    assert {row.embedding_dim for row in search_space("mf_bpr")} == {32, 64}
    assert {row.layers for row in search_space("lightgcn")} == {1, 2}
    assert all(row.embedding_dim == 64 for row in search_space("lightgcn"))
    assert all(row.negatives == 10 for row in search_space("simplex"))


def test_uniform_negatives_exclude_all_prior_ratings_but_keep_future():
    data = tiny_data()
    sampler = UniformNegativeSampler(data)
    users = np.repeat(np.arange(4), 2000)
    sampled = sampler.sample(users, 3, np.random.default_rng(7))
    forbidden = {tuple(pair) for pair in past_edges(data)}
    assert all((int(u), int(i)) not in forbidden for u, row in zip(users, sampled) for i in row)
    # User 0 has exactly {3,5} available, despite low-rating item 4.
    values, counts = np.unique(sampled[users == 0], return_counts=True)
    np.testing.assert_array_equal(values, [3, 5])
    assert np.all(abs(counts - 3000) < 150)
    assert 0 in sampled[users == 1]  # held-out positive is an eligible training negative
    changed = UniformNegativeSampler(perturb_future(data))
    np.testing.assert_array_equal(sampled, changed.sample(users, 3, np.random.default_rng(7)))


def test_lightgcn_normalization_and_isolated_item():
    data = tiny_data()
    matrix = normalized_graph(data).to_dense().numpy()
    n_users = len(data["target_user_ids"])
    degree = np.r_[np.bincount(data["target_train"][:, 0], minlength=n_users),
                   np.bincount(data["target_train"][:, 1], minlength=6)]
    expected = np.zeros_like(matrix)
    for user, item in fit_edges(data):
        node = n_users + item
        expected[user, node] = expected[node, user] = 1 / np.sqrt(degree[user] * degree[node])
    np.testing.assert_allclose(matrix, expected, atol=1e-7)
    assert np.isfinite(matrix).all() and not matrix[-1].any()


def test_simplex_removes_focal_before_history_cap():
    data = tiny_data()
    config = CFConfig(method="simplex", embedding_dim=3, history_cap=2, history_mix=1.)
    model = CollaborativeModel(data, config)
    np.testing.assert_array_equal(history_table(data, 2)[0], [2, 1, 0])
    with torch.no_grad():
        model.history_transform.weight.copy_(torch.eye(3))
        model.item_embedding.weight[:3].copy_(torch.eye(3))
    actual = model.simplex_users(torch.tensor([0]), torch.tensor([2])).detach().numpy()
    expected = np.array([[1., 1., 0.]]) / np.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_native_objective_gradients_match_finite_differences(method):
    torch.manual_seed(9)
    model = CollaborativeModel(tiny_data(), CFConfig(method=method, embedding_dim=4,
                                 l2=.03, margin=-.5)).double()
    users, positives = torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])
    negatives = torch.tensor([[3, 5], [0, 5], [0, 5]])
    loss = model.objective(users, positives, negatives)
    loss.backward()
    for _, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        # Check the largest active derivative so zero/unused rows cannot pass.
        index = np.unravel_index(int(parameter.grad.abs().argmax()), parameter.shape)
        derivative = float(parameter.grad[index])
        original = float(parameter[index].detach())
        losses = []
        for delta in (1e-5, -1e-5):
            with torch.no_grad():
                parameter[index] = original + delta
            losses.append(float(model.objective(users, positives, negatives).detach()))
        with torch.no_grad():
            parameter[index] = original
        assert abs(derivative) > 1e-8
        np.testing.assert_allclose((losses[0] - losses[1]) / 2e-5, derivative,
                                   rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("method", METHODS)
def test_fixed_epochs_determinism_future_isolation_and_checkpoint(method, tmp_path):
    data = tiny_data()
    config = CFConfig(method=method, embedding_dim=4, batch_size=4, lr=.01, layers=2)
    reported = []
    model, history = train_cf(data, config, seed=11, epochs=6,
                             progress=lambda row: reported.append(row) or "stop")
    assert len(history) == len(reported) == 6
    assert all(row["positive_edges"] == len(fit_edges(data)) for row in history)
    repeated, repeated_history = train_cf(perturb_future(data), config, seed=11, epochs=6)
    assert history == repeated_history
    for key, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, repeated.state_dict()[key], rtol=0, atol=0)
    scores = model.score_all(np.arange(6))
    assert scores.shape == (6, 4) and np.isfinite(scores).all()
    for item in range(6):
        native = model.pair_scores(torch.arange(4), torch.full((4,), item)).detach().numpy()
        np.testing.assert_allclose(scores[item], native, atol=1e-7)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = CollaborativeModel.load(path, perturb_future(data))
    np.testing.assert_array_equal(restored.score_all(np.arange(6)), scores)
    changed = {**data, "target_user_ids": data["target_user_ids"][::-1]}
    with pytest.raises(ValueError, match="cohort or fit history"):
        CollaborativeModel.load(path, changed)
    changed = {**data, "target_train": data["target_train"][:-1]}
    with pytest.raises(ValueError, match="cohort or fit history"):
        CollaborativeModel.load(path, changed)


def test_runner_freezes_before_query_and_rejects_corrupt_resume(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cf_runner", root / "scripts/run_baselines.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tiny_data()
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **data)
    original_evaluate = runner.evaluate
    frozen_path = tmp_path / "frozen.json"
    def assert_frozen(*args, **kwargs):
        assert frozen_path.exists()
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(runner, "evaluate", assert_frozen)
    args = (data, path, CFConfig(method="mf_bpr", embedding_dim=4), 101,
            tmp_path / "model.npz", tmp_path / "result.json", tmp_path / "test.npz",
            frozen_path, {"role": "test_only"}, {"test": "synthetic"}, runner.sha256(path))
    result = runner.run_one(*args, epochs=6)
    assert result["epochs"] == 6
    assert runner.run_one(*args, epochs=6) == result
    (tmp_path / "test.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Evaluation hash mismatch"):
        runner.run_one(*args, epochs=6)
