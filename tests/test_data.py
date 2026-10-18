"""Temporal and representation contracts for independent data prefixes."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from saet_repro.data import (
    DAY, build_domain, build_prefix, deduplicate_reviews, load_data, raw_activity,
    save_data, title_embeddings,
)


FIT_START = 100 * DAY
FIT_END = 465 * DAY
QUERY_END = 495 * DAY


def reviews(rows):
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])


def staged_reviews():
    return reviews([
        ("alice", "A", 5, FIT_START),
        ("alice", "B", 4, 200 * DAY),
        ("alice", "C", 5, FIT_END - DAY // 2),
        ("bob", "B", 5, 130 * DAY),
        ("bob", "C", 4, 300 * DAY),
        ("bob", "D", 5, 400 * DAY),
        ("charlie", "A", 4, 110 * DAY),
        ("charlie", "C", 5, 330 * DAY),
        ("charlie", "E", 5, 460 * DAY),
        ("alice", "D", 1, FIT_START - 1),
        ("alice", "Z", 2, 450 * DAY),
        ("alice", "E", 5, FIT_END),
        ("bob", "A", 2, FIT_END + DAY),
        ("charlie", "D", 4, QUERY_END - 1),
        ("charlie", "B", 5, QUERY_END),
    ])


def test_deduplicate_earliest_globally_before_rating_or_window_filters():
    frame = reviews([
        ("u", "a", 5, FIT_START + DAY),
        ("u", "a", 1, FIT_START - DAY),
        ("u", "b", 4, FIT_END - DAY),
        ("v", "a", 5, FIT_END + DAY),
        ("v", "a", 2, FIT_START),
    ])
    deduped = deduplicate_reviews(frame)
    lookup = {(row.user_id, row.parent_asin): (row.rating, row.timestamp)
              for row in deduped.itertuples()}
    assert lookup == {
        ("u", "a"): (1, FIT_START - DAY),
        ("u", "b"): (4, FIT_END - DAY),
        ("v", "a"): (2, FIT_START),
    }
    shuffled = deduplicate_reviews(frame.sample(frac=1, random_state=9))
    assert {(r.user_id, r.parent_asin): (r.rating, r.timestamp)
            for r in shuffled.itertuples()} == lookup


def test_cohort_hash_selection_is_stable_and_requires_three_fit_positive_items():
    rows = [(f"user-{u:02d}", f"item-{i}", 5, FIT_START + (i + 1) * DAY)
            for u in range(12) for i in range(3)]
    # Query positives cannot complete an otherwise ineligible user's support.
    rows += [("query-only", f"item-{i}", 5,
              FIT_START + DAY if i < 2 else FIT_END) for i in range(3)]
    frame = reviews(rows)
    selected = build_domain(frame, FIT_START, FIT_END, QUERY_END,
                            max_users=5, salt="cds:users")
    by_hash = sorted([f"user-{u:02d}" for u in range(12)],
                     key=lambda uid: hashlib.sha256(f"2026|cds:users|{uid}".encode()).hexdigest())
    assert selected["user_ids"].tolist() == sorted(by_hash[:5])
    reordered = build_domain(frame.sample(frac=1, random_state=8),
                             FIT_START, FIT_END, QUERY_END,
                             max_users=5, salt="cds:users")
    np.testing.assert_array_equal(reordered["user_ids"], selected["user_ids"])
    source = build_domain(frame, FIT_START, FIT_END, QUERY_END)
    assert source["user_ids"].tolist() == [f"user-{u:02d}" for u in range(12)]


def test_stages_all_ratings_and_nonwarm_activity_are_preserved():
    domain = build_domain(staged_reviews(), FIT_START, FIT_END, QUERY_END)
    assert domain["user_ids"].tolist() == ["alice", "bob", "charlie"]
    assert domain["item_ids"].tolist() == list("ABCDE")
    observed = set(map(tuple, domain["observed"]))
    positive = set(map(tuple, domain["positive"]))
    assert (0, 3, -1) in observed and (0, 3, -1) not in positive
    assert (0, 0, 0) in observed  # Inclusive fit start.
    assert (0, 4, 1) in positive  # Inclusive query start.
    assert (1, 0, 1) in observed and (1, 0, 1) not in positive
    assert (2, 3, 1) in positive  # Last millisecond before query end.
    assert (2, 1, 1) not in observed  # Exclusive query end.
    assert len(domain["train"]) == 9
    assert np.all(domain["event_times"] < QUERY_END)
    assert np.all(domain["all_rating_times"] >= FIT_START)
    assert np.all(domain["all_rating_times"] < FIT_END)
    unknown = domain["all_rating_items"] == -1
    np.testing.assert_array_equal(domain["all_rating_users"][unknown], [0])
    np.testing.assert_array_equal(domain["all_rating_ratings"][unknown], [2])
    np.testing.assert_array_equal(domain["all_rating_times"][unknown], [450 * DAY])
    np.testing.assert_allclose(domain["activity_raw"][0], np.log1p([2, 2, 4, .5, 4]), rtol=1e-6)
    np.testing.assert_array_equal(domain["degree"], [2, 2, 3, 1, 1])
    np.testing.assert_array_equal(domain["tail80"], [True, True, False, True, True])
    np.testing.assert_array_equal(domain["tail50"], [True, False, False, True, True])


def test_future_events_cannot_change_cohort_catalog_or_fitted_activity():
    frame = staged_reviews()
    fit_only = frame[frame.timestamp < FIT_END]
    future = reviews([
        ("new-user", f"new-{i}", 5, FIT_END + i * DAY) for i in range(4)
    ] + [("alice", "future-item", 5, FIT_END + DAY),
         ("bob", "future-only", 1, QUERY_END + DAY)])
    baseline = build_domain(fit_only, FIT_START, FIT_END, QUERY_END)
    changed = build_domain(pd.concat([frame, future], ignore_index=True),
                           FIT_START, FIT_END, QUERY_END)
    for name in ("user_ids", "item_ids", "train", "degree", "tail80", "tail50",
                 "all_rating_users", "all_rating_items", "all_rating_times",
                 "all_rating_ratings", "activity_raw"):
        np.testing.assert_array_equal(changed[name], baseline[name], err_msg=name)


def test_activity_fractional_recency_inclusive_windows_and_backward_week_bins():
    cutoff = 400 * DAY
    times = np.array([cutoff - DAY // 2, cutoff - 7 * DAY,
                      cutoff - 7 * DAY - 1, cutoff - 30 * DAY,
                      cutoff - 90 * DAY, cutoff - 365 * DAY], dtype=np.int64)
    actual = raw_activity(np.zeros(len(times), dtype=np.int64), times, 2, cutoff)
    # Exactly seven days old remains in bin 0; one millisecond older is bin 1.
    expected = np.log1p([[4, 5, 6, .5, 5], [0, 0, 0, 365, 0]])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_title_encoder_is_target_only_and_retains_zero_vectors():
    target_titles = ["Blue jazz piano", "Red guitar music", "Piano guitar duet",
                     "", "!!!"]
    first, source, metadata = title_embeddings(
        target_titles, ["Blue piano", "quuxwombat", ""], dimensions=64)
    changed, _, changed_metadata = title_embeddings(
        target_titles, ["entirely different words", "quuxwombat " * 50], dimensions=64)
    assert first.shape == (5, 64) and source.shape == (3, 64)
    assert first.dtype == np.float32 and source.dtype == np.float32
    np.testing.assert_array_equal(first, changed)
    np.testing.assert_array_equal(first[3:], np.zeros((2, 64)))
    np.testing.assert_array_equal(source[1:], np.zeros((2, 64)))
    np.testing.assert_allclose(np.linalg.norm(first[:3], axis=1), 1, atol=1e-6)
    assert isinstance(metadata, dict) and isinstance(changed_metadata, dict)
    for name, expected in {"dimensions": 64, "seed": 2026, "n_iter": 7,
                           "max_features": 30_000, "ngram_range": [1, 2]}.items():
        assert metadata[name] == expected
    assert metadata == changed_metadata


def test_empty_title_corpus_keeps_requested_dimensions_and_all_items():
    target, source, _ = title_embeddings(["", "!!!"], ["source vocabulary"], dimensions=64)
    np.testing.assert_array_equal(target, np.zeros((2, 64), dtype=np.float32))
    np.testing.assert_array_equal(source, np.zeros((1, 64), dtype=np.float32))


def test_prefix_uses_target_statistics_for_both_domains_and_ignores_source_changes():
    target = staged_reviews()
    source = staged_reviews().copy()
    source["user_id"] = "source-" + source["user_id"]
    source["parent_asin"] = "source-" + source["parent_asin"]
    titles = dict(zip("ABCDE", ["blue jazz piano", "red rock guitar",
                               "guitar piano duet", "", "live jazz music"]))
    source_titles = {"source-" + item: title for item, title in titles.items()}
    data = build_prefix(target, source, titles, source_titles, FIT_END)
    assert int(data["fit_start"]) == FIT_START
    assert int(data["fit_end"]) == FIT_END
    assert int(data["query_end"]) == QUERY_END
    np.testing.assert_array_equal(data["cutoffs"],
                                  FIT_END + np.array([0, 30, 60, 90]) * DAY)
    expected_mean = data["target_activity_raw"].mean(axis=0, dtype=np.float64)
    expected_scale = np.maximum(data["target_activity_raw"].std(axis=0, dtype=np.float64), 1e-6)
    np.testing.assert_allclose(data["activity_mean"], expected_mean, rtol=1e-6)
    np.testing.assert_allclose(data["activity_scale"], expected_scale, rtol=1e-6)
    for domain in ("target", "source"):
        np.testing.assert_allclose(
            data[f"{domain}_activity"],
            (data[f"{domain}_activity_raw"] - data["activity_mean"]) / data["activity_scale"],
            atol=1e-6,
        )
    for name in ("degree", "tail80", "tail50"):
        np.testing.assert_array_equal(data[name], data[f"target_{name}"])

    changed_source = pd.concat([source, reviews([
        ("source-alice", f"source-extra-{i}", 1, FIT_END - i - 1)
        for i in range(10)
    ])], ignore_index=True)
    changed = build_prefix(target, changed_source, titles,
                           {key: "different source-only vocabulary" for key in source_titles}, FIT_END)
    assert not np.array_equal(changed["source_activity_raw"], data["source_activity_raw"])
    for name in ("activity_mean", "activity_scale", "target_activity_raw", "target_activity", "target_x"):
        np.testing.assert_array_equal(changed[name], data[name], err_msg=name)


def test_data_roundtrip_and_failed_write_preserve_completed_artifact(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "prefix.npz"
    original = {
        "fit_end": np.array(FIT_END, dtype=np.int64),
        "target_user_ids": np.array(["alice", "用户"]),
        "target_train": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "target_x": np.eye(3, 64, dtype=np.float32),
    }
    save_data(path, original)
    saved_bytes = path.read_bytes()
    restored = load_data(path)
    assert set(restored) == set(original)
    for key, expected in original.items():
        np.testing.assert_array_equal(restored[key], expected)
        assert restored[key].dtype == expected.dtype

    def interrupted(output, **arrays):
        output.write(b"interrupted NPZ")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="simulated storage failure"):
        save_data(path, {"replacement": np.array([42])})
    assert path.read_bytes() == saved_bytes
    np.testing.assert_array_equal(load_data(path)["target_user_ids"], original["target_user_ids"])
