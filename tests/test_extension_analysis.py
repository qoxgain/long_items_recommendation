"""Behavior tests for the additional Stage A statistics and strict interfaces."""
import numpy as np
import pytest

from saet_repro.extension_statistics import (
    unique_item_cluster_interval, decision_digest, routing_summary, convex_mixture_check,
)
from scripts.analyze_extensions import freeze, secondary_aggregates


def test_cluster_keeps_shared_items_together_and_origin_weights_equal():
    # Paired signs cancel for every shared-cluster draw; independently drawing
    # item-origin rows would produce a nondegenerate interval.
    items = [{'origin': origin, 'item_id': item, 'difference': sign * value}
             for origin, sign in [('2015', 1), ('2016', -1)]
             for item, value in [('a', 1.), ('b', 3.), ('c', 9.)]]
    result = unique_item_cluster_interval(items, n_bootstrap=500)
    assert result['ci'] == [0., 0.]
    assert result == unique_item_cluster_interval(list(reversed(items)), n_bootstrap=500)


def test_cluster_empty_origin_draws_are_rejected_not_reweighted():
    items = [{'origin': 'a', 'item_id': 'x', 'difference': 2.},
             {'origin': 'b', 'item_id': 'y', 'difference': 6.}]
    result = unique_item_cluster_interval(items, n_bootstrap=100)
    assert result['ci'] == [4., 4.]
    assert result['rejected_empty_origin_draws'] > 0
    assert result['n_resamples'] == 100


def test_cluster_rejects_duplicate_rows_and_invalid_budget():
    row = {'origin': 'a', 'item_id': 'x', 'difference': 2.}
    with pytest.raises(ValueError, match='Duplicate'):
        unique_item_cluster_interval([row, row])
    with pytest.raises(ValueError, match='positive integer'):
        unique_item_cluster_interval([row], n_bootstrap=True)


def test_decision_digest_does_not_deduplicate_equal_metric_or_near_decisions():
    assert decision_digest([[0., -0.]]) == decision_digest([[-0., 0.]])
    assert decision_digest([[1., 0.]]) != decision_digest([[0., 1.]])
    assert decision_digest([[1.]]) != decision_digest([[np.nextafter(1., 2.)]])
    assert decision_digest([[1., 2.]]) != decision_digest([[1.], [2.]])


def test_routing_conditions_on_correction_not_mass():
    routing = {'zero': [True, True, False], 'source_mass': [0., 1., .5],
               'correction_norm': [0., 0., .1], 'effective_donors': [0, 2, 1],
               'concentration': [0., .5, 1.]}
    result = routing_summary(routing)
    assert result['zero_action_rate'] == 2 / 3
    assert result['nonzero_mean_source_mass'] == .5
    assert result['n_nonzero_items'] == 1
    assert routing_summary(routing, [True, True, False])['nonzero_mean_source_mass'] is None


def test_convex_actual_loss_bound_allows_numerical_tolerance():
    rows = [{'actual_mixture_gain': 0., 'weighted_actual_single_gain': 0., 'source_mass': 0.},
            {'actual_mixture_gain': .2, 'weighted_actual_single_gain': .1, 'source_mass': .5},
            {'actual_mixture_gain': .1, 'weighted_actual_single_gain': .1 + 1e-13, 'source_mass': 1.}]
    result = convex_mixture_check(rows)
    assert result['passed'] and 0 < result['max_violation'] < 1e-12
    rows[1]['weighted_actual_single_gain'] = .3
    assert not convex_mixture_check(rows)['passed']


def test_analysis_identity_rejects_changed_inputs(tmp_path):
    path = tmp_path / 'frozen.json'
    freeze(path, {'input': 'a'})
    freeze(path, {'input': 'a'})
    with pytest.raises(ValueError, match='identity mismatch'):
        freeze(path, {'input': 'b'})


def test_secondary_aggregation_rejects_incomplete_learned_seed_matrix():
    with pytest.raises(ValueError, match='seed matrix'):
        secondary_aggregates([{'method': 'activity_mean', 'stratum': 'all', 'origin': 2015, 'seed': 11}])
