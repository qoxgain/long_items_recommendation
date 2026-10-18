"""Numerical and temporal invariants of forward-label source transfer."""
import numpy as np
import pytest

from saet_repro.target import Config, FeatureStore, Model, DAY, _raw_activity, pair_loss_and_gradient
from saet_repro.utility import (
    FEATURE_SUBSETS, POLICIES, RidgeUtility, RouteContext, MultiPolicyScorer,
    batch_loss, clip_step, corrected_scores, cosine, forward_labels,
    policy_correction, project_nonnegative_l1, query_batches, select_ridges,
    utility_weights,
)
from saet_repro.evaluation import evaluate, evaluate_many


def tiny_data():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    users = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3])
    items = np.array([0, 1, 2, 3, 0, 4, 1, 3, 2, 4])
    times = np.arange(10, 20, dtype=np.int64) * DAY
    end = 30 * DAY
    raw = np.array([_raw_activity(times[users == u], end) for u in range(4)])
    data = dict(fit_start=np.array(0), fit_end=np.array(end), query_end=np.array(end + 30 * DAY),
                activity_mean=np.zeros(5), activity_scale=np.ones(5))
    fit = np.column_stack((users, items, np.zeros(len(users), int)))
    query = np.array([[2, 0, 1], [1, 2, 1], [0, 4, 1]])
    for domain in ('target', 'source'):
        values = dict(x=x.copy(), user_ids=np.array(['a', 'b', 'c', 'd']),
            item_ids=np.array([f'{domain}_{i}' for i in range(5)]),
            train=np.column_stack((users, items)), observed=np.vstack((fit, query)),
            positive=np.vstack((fit, query)), event_users=np.r_[users, query[:, 0]],
            event_items=np.r_[items, query[:, 1]], event_times=np.r_[times, np.full(3, end + DAY)],
            event_ratings=np.full(13, 5), all_rating_users=users.copy(), all_rating_items=items.copy(),
            all_rating_times=times.copy(), all_rating_ratings=np.full(10, 5),
            activity=raw.copy(), activity_raw=raw.copy(), degree=np.full(5, 2),
            tail80=np.array([True, True, True, True, False]),
            tail50=np.array([True, True, True, False, False]))
        data.update({f'{domain}_{k}': v for k, v in values.items()})
    return data


def base_model(data):
    config = Config(tau=.3)
    return Model(config, FeatureStore(data, config), theta=np.linspace(-.1, .1, 192),
                 beta=np.array([.1, -.05, .01, .1, -.1]), alpha=.8)


class ConstantUtility:
    def __init__(self, value):
        self.value = value

    def predict(self, features, scaled=True):
        return np.full(len(features), self.value)


def constant_ridges(value):
    return {name: ConstantUtility(value) for name in FEATURE_SUBSETS}


def test_projection_satisfies_kkt_and_exact_null():
    rng = np.random.default_rng(8)
    for size in (1, 3, 16):
        for _ in range(30):
            values = rng.normal(size=size)
            weights = project_nonnegative_l1(values)
            assert np.all(weights >= 0) and weights.sum() <= 1. + 1e-12
            positive = weights > 0
            if np.maximum(values, 0).sum() <= 1:
                np.testing.assert_array_equal(weights, np.maximum(values, 0))
            else:
                threshold = values[positive] - weights[positive]
                np.testing.assert_allclose(threshold, threshold[0], atol=1e-12)
                assert np.all(values[~positive] <= threshold[0] + 1e-12)
                np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
    np.testing.assert_array_equal(utility_weights([-1., 0., -2.]), [0., 0., 0.])
    np.testing.assert_array_equal(project_nonnegative_l1([2., 3.], mass=0), [0., 0.])
    with pytest.raises(ValueError):
        project_nonnegative_l1([np.nan])
    with pytest.raises(ValueError):
        project_nonnegative_l1([1.], mass=np.inf)


def test_source_gradient_matches_actual_query_loss_derivative():
    data = tiny_data()
    model = base_model(data)
    batch = model.store.pair_batch(0, [0, 1], [2, 3])
    loss, derivative = pair_loss_and_gradient(model, batch, regularize=False)
    np.testing.assert_allclose(batch_loss(model, batch), loss, rtol=1e-13)
    rng = np.random.default_rng(3)
    direction = rng.normal(size=len(model.theta))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    numerical = (batch_loss(model, batch, epsilon * direction)
                 - batch_loss(model, batch, -epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(numerical, derivative['theta'] @ direction, rtol=1e-6, atol=1e-9)


def test_null_recovers_base_bit_exact_and_freezes_parameters():
    data = tiny_data()
    model = base_model(data)
    original = model.theta.copy(), model.beta.copy(), model.alpha
    context = RouteContext(data, model, source_epochs=2)
    configurations = {p: dict(policy=p, radius=0., temperature=1.) for p in POLICIES}
    scorer = MultiPolicyScorer(context, constant_ridges(1.), configurations)
    expected = model.score_all([0, 1, 2])
    for result in scorer([0, 1, 2]):
        np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(model.theta, original[0])
    np.testing.assert_array_equal(model.beta, original[1])
    assert model.alpha == original[2]
    np.testing.assert_array_equal(context.source.theta, original[0])
    assert context.source.alpha == original[2]
    scores = np.array([np.nextafter(1., 2.), -0.])
    assert corrected_scores(scores, np.ones((2, 3)), np.zeros(3)) is scores


def test_bounded_steps_scores_and_pair_order():
    rng = np.random.default_rng(19)
    gradients = rng.normal(size=(16, 192)) * 4
    features = rng.normal(size=(16, 10))
    phi = rng.normal(size=(40, 192))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    for radius in (0., .01, .05, .1):
        assert np.max(np.linalg.norm(clip_step(gradients, .1, radius), axis=1)) <= radius + 1e-12
        for policy in POLICIES:
            correction, weights = policy_correction(policy, features, gradients,
                                                      constant_ridges(.025), radius)
            assert weights.sum() <= 1. + 1e-12
            assert np.linalg.norm(correction) <= radius + 1e-12
            assert np.max(np.abs(phi @ correction)) <= radius + 1e-12
            original = np.array([0., 2 * radius + 1e-6])
            changed = original + phi[:2] @ correction
            assert changed[1] > changed[0]


def test_mechanisms_keep_null_and_normalize_separately():
    features = np.ones((3, 10))
    gradients = np.eye(3)
    for policy in ('semantic', 'global', 'behavior', 'full', 'predicted_single'):
        for mechanism in ('mass1', 'norm01'):
            correction, weights = policy_correction(policy, features, gradients,
                constant_ridges(-1.), .1, mechanism=mechanism)
            np.testing.assert_array_equal(correction, np.zeros(3))
            np.testing.assert_array_equal(weights, np.zeros(3))
    mass_step, mass_weights = policy_correction('full', features, gradients,
        constant_ridges(.05), .1, mechanism='mass1')
    norm_step, norm_weights = policy_correction('full', features, gradients,
        constant_ridges(.05), .1, mechanism='norm01')
    np.testing.assert_allclose(mass_weights.sum(), 1.)
    np.testing.assert_allclose(norm_weights.sum(), .15)
    np.testing.assert_allclose(np.linalg.norm(norm_step), .1)
    assert np.linalg.norm(mass_step) < .1


def test_future_firewall_source_and_target_and_independent_id_spaces():
    data = tiny_data()
    changed = {key: value.copy() for key, value in data.items()}
    for domain in ('target', 'source'):
        # Future labels, ratings, events and ID alignment cannot enter routing.
        changed[f'{domain}_positive'] = np.array([[3, 0, 1]])
        changed[f'{domain}_observed'] = np.vstack((data[f'{domain}_observed'], [3, 0, 1]))
        mask = changed[f'{domain}_event_times'] >= int(data['fit_end'])
        changed[f'{domain}_event_users'][mask] = 3
        changed[f'{domain}_event_items'][mask] = 1
        changed[f'{domain}_event_ratings'][mask] = 1
    changed['source_user_ids'] = np.array(['different_a', 'different_b', 'different_c', 'different_d'])
    first = RouteContext(data, base_model(data), source_epochs=2)
    second = RouteContext(changed, base_model(changed), source_epochs=2)
    np.testing.assert_array_equal(first.source.beta, second.source.beta)
    for item in range(5):
        for a, b in zip(first.route_inputs(item), second.route_inputs(item)):
            np.testing.assert_array_equal(a, b)
        a = policy_correction('full', *first.route_inputs(item)[1:], constant_ridges(.1), .1)[0]
        b = policy_correction('full', *second.route_inputs(item)[1:], constant_ridges(.1), .1)[0]
        np.testing.assert_array_equal(a, b)


def test_forward_labels_are_actual_loss_difference_and_share_query_pairs():
    data = tiny_data()
    context = RouteContext(data, base_model(data), source_epochs=0)
    labels = forward_labels(data, context)
    assert len(labels['labels']) == 15
    for item, positive, negative, batch in query_batches(data, context):
        indices = np.flatnonzero(labels['item_id'] == item)
        donors, features, gradients = context.route_inputs(item)
        np.testing.assert_array_equal(labels['donor_id'][indices], donors)
        np.testing.assert_array_equal(labels['features'][indices], features)
        expected = [batch_loss(context.base, batch) - batch_loss(context.base, batch, step)
                    for step in clip_step(gradients, .1, .1)]
        np.testing.assert_allclose(labels['labels'][indices], expected, rtol=1e-13)
        assert not set(positive) & set(context.target_store.observed_users[item])
        assert not set(positive) & set(negative)
    zero = forward_labels(data, context, radius=0)
    np.testing.assert_array_equal(zero['labels'], np.zeros(15))
    with pytest.raises(ValueError):
        list(query_batches(data, context, stage=2))


def test_ridge_zero_meaning_scale_validation_and_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(50, 10))
    features[:, 0] = 1.
    for label in (-.2, 0., .3):
        labels = np.full(50, label)
        model = RidgeUtility.fit(features, labels, 'full', .1)
        np.testing.assert_allclose(model.predict(features, scaled=False), labels, atol=1e-12)
        if label:
            np.testing.assert_allclose(model.label_rms, abs(label))
            assert np.all(np.sign(model.predict(features)) == np.sign(label))
        path = tmp_path / f'{label}.npz'
        model.save(path)
        np.testing.assert_array_equal(RidgeUtility.load(path).predict(features), model.predict(features))
    labels = .03 * features[:, 1] - .01 * features[:, 3]
    models, reports = select_ridges(features[:35], labels[:35], features[35:], labels[35:])
    assert set(models) == set(FEATURE_SUBSETS)
    for name, model in models.items():
        assert len(reports[name]['trials']) == 6
        np.testing.assert_allclose(model.label_rms, np.sqrt(np.mean(labels ** 2)))


def test_shared_policy_evaluation_matches_independent_evaluation():
    data = tiny_data()
    model = base_model(data)
    context = RouteContext(data, model, source_epochs=0)
    configs = {'zero': dict(policy='zero', radius=0., temperature=1.),
               'uniform': dict(policy='uniform', radius=.1, temperature=1.),
               'full': dict(policy='full', radius=.1, temperature=.1)}
    ridges = constant_ridges(.1)
    scorer = MultiPolicyScorer(context, ridges, configs)
    shared = evaluate_many(data, 1, scorer, configs)
    for name, config in configs.items():
        single = MultiPolicyScorer(context, ridges, {name: config})
        independent = evaluate(data, 1, lambda ids: single(ids)[0])
        for field in independent:
            np.testing.assert_array_equal(shared[name][field], independent[field])
    expected = evaluate(data, 1, model.score_all)
    np.testing.assert_array_equal(shared['zero']['top_users'], expected['top_users'])


def test_zero_cosine_and_same_catalog_id_excluded_from_donors():
    assert cosine(np.zeros(3), np.ones(3)) == 0.
    data = tiny_data()
    data['source_item_ids'][0] = data['target_item_ids'][0]
    context = RouteContext(data, base_model(data), source_epochs=0)
    assert 0 not in context.donor_items(0)


def test_validation_reports_actual_mixture_loss_not_linearized_gain():
    from scripts.run_utility import validation_mixtures
    data = tiny_data()
    context = RouteContext(data, base_model(data), source_epochs=0)
    ridges = constant_ridges(.1)
    records = validation_mixtures(data, context, ridges)
    batches = {item: batch for item, _, _, batch in query_batches(data, context)}
    for record in records:
        item = record['item_id']
        correction, _ = policy_correction(record['policy'], *context.route_inputs(item)[1:],
                                          ridges, .1)
        expected = batch_loss(context.base, batches[item]) - batch_loss(context.base, batches[item], correction)
        assert record['actual_mixture_gain'] == expected
        assert record['oracle_is_label_only_not_deployable'] is True
    mixtures = [r for r in records if r['policy'] == 'full']
    assert any(abs(r['actual_mixture_gain'] - r['weighted_actual_single_gain']) > 1e-10
               for r in mixtures)


def test_historical_resume_checks_provenance_bytes_and_avoids_disk_banks(tmp_path, monkeypatch):
    import json
    import os
    from types import SimpleNamespace
    from scripts import run_utility as runner
    from saet_repro.data import save_data

    args = SimpleNamespace(data_dir=tmp_path / 'data', result_dir=tmp_path / 'results', wait_inputs=False)
    data_path = args.data_dir / 'cds_2015_d60.npz'
    save_data(data_path, tiny_data())
    unused_cache = str(tmp_path / 'unnecessary_feature_banks')
    monkeypatch.setenv('SAET_FEATURE_CACHE', unused_cache)
    original_train = runner.train_target

    def train_without_disk_banks(*args, **kwargs):
        assert 'SAET_FEATURE_CACHE' not in os.environ
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runner, 'train_target', train_without_disk_banks)
    original = runner.historical_fold(args, 2015, 60)
    assert os.environ['SAET_FEATURE_CACHE'] == unused_cache
    assert not (tmp_path / 'unnecessary_feature_banks').exists()

    def no_second_training(*args, **kwargs):
        raise AssertionError('A complete, verified checkpoint should be reused')

    monkeypatch.setattr(runner, 'train_target', no_second_training)
    resumed = runner.historical_fold(args, 2015, 60)
    for field in original:
        np.testing.assert_array_equal(original[field], resumed[field])
    directory = args.result_dir / 'cds_2015/historical/d60'
    record = json.loads((directory / 'result.json').read_text())
    assert record['artifacts']['labels'] == runner.sha256(directory / 'labels.npz')
    with (directory / 'labels.npz').open('ab') as output:
        output.write(b'changed bytes')
    with pytest.raises(RuntimeError, match='Artifact hash mismatch'):
        runner.historical_fold(args, 2015, 60)
    altered = tiny_data()
    altered['target_x'][0, 0] += .01
    save_data(data_path, altered)
    with pytest.raises(RuntimeError, match='Provenance mismatch'):
        runner.historical_fold(args, 2015, 60)


def test_source_activity_resume_binds_exact_base_model(tmp_path):
    from scripts import run_utility as runner
    from saet_repro.data import save_data

    data = tiny_data()
    model = base_model(data)
    data_path, base_path = tmp_path / 'data.npz', tmp_path / 'base.npz'
    save_data(data_path, data)
    model.save(base_path)
    directory = tmp_path / 'source'
    first = runner.context_for(data, model, directory, 101, data_path=data_path, base_path=base_path)
    second = runner.context_for(data, model, directory, 101, data_path=data_path, base_path=base_path)
    np.testing.assert_array_equal(first.source.beta, second.source.beta)
    assert second.source_fit_reused
    model.theta[0] += .01
    model.save(base_path)
    with pytest.raises(RuntimeError, match='Provenance mismatch'):
        runner.context_for(data, model, directory, 101, data_path=data_path, base_path=base_path)


def test_final_evaluation_budget_has_seven_main_and_twelve_controls():
    from scripts.run_utility import final_configurations

    selected = {policy: dict(policy=policy, radius=0., temperature=1.) for policy in POLICIES}
    configurations = final_configurations(selected)
    assert len(configurations) == 19
    assert sum(name.startswith('main_') for name in configurations) == 7
    assert sum(name.startswith('mass1_') for name in configurations) == 6
    assert sum(name.startswith('norm01_') for name in configurations) == 6
    assert len(configurations) * 3 * 5 == 285
