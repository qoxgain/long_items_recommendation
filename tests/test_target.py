import numpy as np
from saet_repro.target import Config, FeatureStore, Model, pair_loss_and_gradient, train_target, DAY, _raw_activity


def tiny_data():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    users = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3])
    items = np.array([0, 1, 2, 3, 0, 4, 1, 3, 2, 4])
    times = np.arange(10, 20, dtype=np.int64) * DAY
    end = 30 * DAY
    raw = np.array([_raw_activity(times[users == u], end) for u in range(4)])
    data = dict(fit_start=np.array(0), fit_end=np.array(end), query_end=np.array(end + 30 * DAY),
                activity_mean=np.zeros(5), activity_scale=np.ones(5))
    for domain in ("target", "source"):
        values = dict(x=x, user_ids=np.array(["a", "b", "c", "d"]), item_ids=np.array(list("vwxyz")),
                      train=np.column_stack((users, items)),
                      observed=np.column_stack((users, items, np.zeros(len(users), int))),
                      event_users=users, event_items=items, event_times=times, event_ratings=np.full(10, 5),
                      all_rating_users=users, all_rating_items=items, all_rating_times=times,
                      all_rating_ratings=np.full(10, 5), activity=raw, activity_raw=raw)
        data.update({f"{domain}_{k}": v for k, v in values.items()})
    return data


def test_exact_gradient_finite_difference():
    data = tiny_data()
    config = Config(tau=.3, rho=.2)
    store = FeatureStore(data, config)
    model = Model(config, store, theta=np.linspace(-.2, .2, 192), beta=np.arange(5) / 7, alpha=.8)
    batch = store.pair_batch(0, [0, 1], [2, 3])
    _, grad = pair_loss_and_gradient(model, batch)
    for parameter, indexes in (("theta", [0, 63, 64, 128, 191]), ("beta", [0, 4]), ("alpha", [None])):
        for index in indexes:
            epsilon = 1e-5
            old = getattr(model, parameter) if index is None else getattr(model, parameter)[index]
            losses = []
            for shift in (epsilon, -epsilon):
                if index is None:
                    setattr(model, parameter, old + shift)
                else:
                    getattr(model, parameter)[index] = old + shift
                losses.append(pair_loss_and_gradient(model, batch)[0])
            if index is None:
                setattr(model, parameter, old)
                exact = grad[parameter]
            else:
                getattr(model, parameter)[index] = old
                exact = grad[parameter][index]
            np.testing.assert_allclose((losses[0] - losses[1]) / (2 * epsilon), exact, rtol=3e-4, atol=2e-5)


def test_focal_removed_before_cap_and_activity():
    data = tiny_data()
    store = FeatureStore(data, Config(history_cap=2))
    # User 0's latest items are 3,2,1,0. Removing 3 must recover item 1.
    expected = data["target_x"][[2, 1]].sum(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(store.history([0], 3, exclude_focal=True)[0], expected, atol=1e-6)
    batch = store.pair_batch(3, [0], [1])
    times = data["target_all_rating_times"]
    kept = (data["target_all_rating_users"] == 0) & (data["target_all_rating_items"] != 3)
    np.testing.assert_allclose(batch["pos_activity"][0], _raw_activity(times[kept], int(data["fit_end"])))


def test_future_labels_cannot_change_features_or_negatives():
    data = tiny_data()
    original = FeatureStore(data, Config())
    future = {k: np.array(v, copy=True) for k, v in data.items()}
    future["target_event_users"] = np.r_[future["target_event_users"], 3]
    future["target_event_items"] = np.r_[future["target_event_items"], 0]
    future["target_event_times"] = np.r_[future["target_event_times"], int(data["fit_end"]) + DAY]
    future["target_event_ratings"] = np.r_[future["target_event_ratings"], 5]
    future["target_observed"] = np.vstack((future["target_observed"], [3, 0, 1]))
    changed = FeatureStore(future, Config())
    for first, second in zip(original.all_features(0), changed.all_features(0)):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(original.observed_users[0], changed.observed_users[0])
    _, negative = changed.sample_users(0, np.random.default_rng(2))
    assert set(negative) == {2, 3}


def test_norm_zero_history_roundtrip_and_fixed_epochs(tmp_path, monkeypatch):
    data = tiny_data()
    monkeypatch.setenv("SAET_FEATURE_CACHE", str(tmp_path / "cache"))
    config = Config(batch_size=2)
    model, history = train_target(data, config, seed=3, epochs=2,
                                   validation_callback=lambda epoch, model: "stop")
    assert len(history) == 2 and all(row["visited_items"] == 5 for row in history)
    assert np.max(np.linalg.norm(model.store.all_features(0)[1], axis=1)) <= 1.000001
    assert np.all(np.isfinite(model.score_all([0, 1])))
    path = tmp_path / "model.npz"
    model.save(path)
    restored = Model.load(path, data)
    np.testing.assert_array_equal(restored.score_all([0, 1]), model.score_all([0, 1]))
    repeated, _ = train_target(data, config, seed=3, epochs=2)
    np.testing.assert_array_equal(repeated.theta, model.theta)


def test_each_history_variant_and_unbounded_no_activity():
    data = tiny_data()
    for method in ("unbounded_mean", "activity_mean", "activity_decay", "activity_conditional", "activity_both"):
        config = Config(method=method)
        store = FeatureStore(data, config)
        model = Model(config, store)
        batch = store.pair_batch(0, [0, 1], [2, 3])
        loss, gradient = pair_loss_and_gradient(model, batch)
        assert np.isfinite(loss)
        assert np.linalg.norm(store.all_features(0)[1], axis=1).max() <= 1.000001
        if method == "unbounded_mean":
            assert np.all(gradient["beta"] == 0)


def test_empty_pair_items_keep_uniform_item_weight():
    store = FeatureStore(tiny_data(), Config(rho=0))
    model = Model(store.config, store)
    single = store.pair_batch(0, [0], [2])
    loss, gradient = pair_loss_and_gradient(model, single)
    batched = {key: np.stack((value, value)) for key, value in single.items()}
    batched["pos_mask"] = np.array([[True], [False]])
    batched["neg_mask"] = np.ones((2, 1), dtype=bool)
    batched_loss, batched_gradient = pair_loss_and_gradient(model, batched)
    np.testing.assert_allclose(batched_loss, loss / 2)
    for name in gradient:
        np.testing.assert_allclose(batched_gradient[name], gradient[name] / 2)


def test_repeated_focal_events_removed_before_cap():
    data = tiny_data()
    data["target_event_users"] = np.r_[data["target_event_users"], 0, 0, 0]
    data["target_event_items"] = np.r_[data["target_event_items"], 3, 3, 3]
    data["target_event_times"] = np.r_[data["target_event_times"], 22*DAY, 23*DAY, 24*DAY]
    data["target_event_ratings"] = np.r_[data["target_event_ratings"], 5, 5, 5]
    store = FeatureStore(data, Config(history_cap=2))
    expected = data["target_x"][[2, 1]].sum(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(store.history([0], 3, exclude_focal=True)[0], expected, atol=1e-6)
