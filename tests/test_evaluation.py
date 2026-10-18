import json

import numpy as np
import pytest

from saet_repro.evaluation import evaluate, evaluate_many, stratified_summaries, summarize
from saet_repro.io import sha256, write_json


def ranking_data():
    # Deliberately unsorted raw IDs: lexical ties must rank a,b,c,d,e, not
    # users' local array positions. Item 4 is cold and cannot be requested.
    observed = np.array([[0, 0, -1], [1, 0, 0], [1, 0, 0],
                         [0, 2, 0], [1, 2, 0], [2, 2, 0], [4, 2, 0],
                         [3, 0, 1]], dtype=np.int64)
    positive = np.array([[0, 0, 1], [2, 0, 1], [2, 0, 1], [3, 0, 1],
                         [3, 2, 1], [1, 3, 1], [3, 4, 1]], dtype=np.int64)
    return dict(target_user_ids=np.array(["e", "a", "d", "b", "c"]),
                target_item_ids=np.array(list("vwxyz")),
                target_degree=np.array([2, 1, 4, 1, 0]),
                target_observed=observed, target_positive=positive,
                target_tail80=np.array([True, True, False, True, False]),
                target_tail50=np.array([False, True, False, True, False]))


def test_exact_metrics_any_past_rating_unique_positive_and_raw_id_ties():
    data = ranking_data()
    calls = []

    def scorer(indices):
        calls.extend(indices.tolist())
        return np.ones((len(indices), 5))

    result = evaluate(data, 1, scorer, ks=(1, 2, 10), batch_size=2)
    assert calls == [0, 2, 3]
    np.testing.assert_array_equal(result["n_candidates"], [3, 5, 1, 5, 5])
    np.testing.assert_array_equal(result["n_positive"], [2, 0, 1, 1, 1])
    np.testing.assert_array_equal(result["eligible_catalog"], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(result["requested"], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(result["evaluated"], [1, 0, 1, 1, 0])
    # Prefit e and fit a are excluded, but query-time b remains eligible.
    np.testing.assert_array_equal(result["top_users"][0], [3, 4, 2] + [-1] * 7)
    assert result["hits2"][0] == 1
    assert result["recall2"][0] == .5
    assert result["precision2"][0] == .5
    expected = 1 / (1 + 1 / np.log2(3))
    assert result["ndcg2"][0] == pytest.approx(expected)
    assert result["ndcg10"][0] == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    # One available user: precision's denominator is one, even at K=10.
    assert result["precision10"][2] == result["recall10"][2] == result["ndcg10"][2] == 1
    for key in ("ndcg1", "recall2", "precision10", "hits10"):
        assert np.isnan(result[key][[1, 4]]).all()
    assert np.all(result["top_users"][[1, 4]] == -1)


def test_partial_request_and_empty_query_keep_full_catalog():
    data = ranking_data()
    result = evaluate(data, 1, lambda ids: np.ones((len(ids), 5)),
                      item_mask=np.array([False, True, False, True, True]), save_topk=False)
    assert result["top_users"].shape == (5, 0)
    np.testing.assert_array_equal(result["requested"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(result["ranked"], [0, 0, 0, 1, 0])
    summary = summarize(data, result)
    assert summary["items_catalog"] == 5
    assert summary["items_eligible"] == 4
    assert summary["items_requested"] == 2
    assert summary["items_evaluated"] == 1
    assert summary["coverage20"] == .2
    strata = stratified_summaries(data, result, include_degrees=True)
    assert strata["tail50"]["items_catalog"] == 2
    assert strata["tail50"]["evaluated_fraction"] == .5
    assert strata["degree_21_plus"]["items_catalog"] == 0
    assert np.isnan(strata["degree_21_plus"]["ndcg20"])
    empty = evaluate(data, 9, lambda _: pytest.fail("No positives: scorer must not run"))
    assert np.isnan(empty["ndcg20"]).all()
    empty_summary = summarize(data, empty)
    assert np.isnan(empty_summary["ndcg20"])
    assert empty_summary["coverage20"] == 0


def test_shared_policy_scoring_matches_independent_scoring():
    data = ranking_data()
    scores = np.random.default_rng(15).normal(size=(3, 5, 5))
    names = ("zero", "full", "semantic")
    calls = []

    def shared(ids):
        calls.append(ids.copy())
        return {name: scores[index, ids] for index, name in enumerate(names)}

    combined = evaluate_many(data, 1, shared, names, batch_size=2)
    assert len(calls) == 2
    for index, name in enumerate(names):
        independent = evaluate(data, 1, lambda ids: scores[index, ids])
        for key, value in independent.items():
            np.testing.assert_array_equal(combined[name][key], value)


def test_nonfinite_excluded_scores_allowed_but_candidate_scores_rejected():
    data = ranking_data()
    values = np.ones((5, 5))
    values[0, :2] = np.nan  # Both excluded by past events.
    evaluate(data, 1, lambda ids: values[ids])
    values[0, 2] = np.nan
    with pytest.raises(ValueError, match="Nonfinite candidate score"):
        evaluate(data, 1, lambda ids: values[ids])
    with pytest.raises(ValueError, match="Scorer must return numeric shape"):
        evaluate(data, 1, lambda ids: np.ones((len(ids), 4)))
    with pytest.raises(ValueError, match="boolean array"):
        evaluate(data, 1, lambda _: None, item_mask=np.ones(5))


def test_zero_users_and_strict_json_undefined_metrics(tmp_path):
    data = ranking_data()
    data["target_user_ids"] = np.array([], dtype=str)
    data["target_positive"] = data["target_observed"] = np.empty((0, 3), dtype=np.int64)
    result = evaluate(data, 1, lambda _: pytest.fail("Empty users must not be scored"))
    np.testing.assert_array_equal(result["n_candidates"], np.zeros(5))
    path = write_json(tmp_path / "nested" / "summary.json", summarize(data, result))
    assert json.loads(path.read_text())["ndcg20"] is None
    assert "NaN" not in path.read_text()
    known = tmp_path / "known.txt"
    known.write_bytes(b"abc")
    assert sha256(known, chunk_size=1) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
