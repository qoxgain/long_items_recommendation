import numpy as np
import pytest

from saet_repro.statistics import paired_effects, paired_item_differences


def records_for(seeds=(11, 22)):
    records = []
    # Unequal origin sizes deliberately distinguish equal-origin from pooled
    # item weighting. Seed offsets cancel in the paired differences.
    for origin, item, difference in ((2015, "a", -.4), (2015, "b", .4), (2016, "a", .2)):
        for index, seed in enumerate(seeds):
            baseline = .45 + .01 * (index % 2)
            records.extend([dict(origin=origin, item_id=item, seed=seed,
                                 method="candidate", ndcg20=baseline + difference),
                            dict(origin=origin, item_id=item, seed=seed,
                                 method="base", ndcg20=baseline)])
    return records


def test_equal_origin_bootstrap_harm_and_leave_one_out():
    result = paired_effects(records_for(), "candidate", "base")
    assert result["n_item_origins"] == 3
    assert result["n_unique_items"] == 2
    assert result["n_seed_pairs"] == 6
    assert result["mean_difference"] == pytest.approx(.1)
    np.testing.assert_allclose(result["ci"], [-.1, .3])
    np.testing.assert_allclose(result["origin_bootstrap"]["ci"], [0, .2], atol=1e-16)
    assert result["harm"]["n_harmed"] == 1
    assert result["harm"]["n_improved"] == 2
    assert result["harm"]["harm_rate"] == .25
    assert result["harm"]["pooled_harm_rate"] == pytest.approx(1 / 3)
    loo = {row["omitted_origin"]: row["mean_difference"]
           for row in result["leave_one_origin_out"]}
    assert loo["2015"] == pytest.approx(.2)
    assert loo["2016"] == pytest.approx(0)


def test_dependent_seed_repetition_does_not_narrow_item_interval():
    first = paired_effects(records_for(), "candidate", "base")
    repeated = paired_effects(records_for((11, 22, 33, 44)), "candidate", "base")
    assert first["n_item_origins"] == repeated["n_item_origins"]
    assert repeated["n_seed_pairs"] == 2 * first["n_seed_pairs"]
    np.testing.assert_array_equal(first["ci"], repeated["ci"])
    assert first["harm"] == repeated["harm"]
    shuffled = records_for()
    np.random.default_rng(123).shuffle(shuffled)
    assert paired_effects(shuffled, "candidate", "base") == first


def test_deterministic_baseline_is_broadcast_then_seeds_are_averaged():
    records = [dict(origin="one", item_id="a", seed=None, method="base", ndcg20=.5),
               dict(origin="one", item_id="a", seed=11, method="candidate", ndcg20=.8),
               dict(origin="one", item_id="a", seed=22, method="candidate", ndcg20=.2)]
    items, diagnostics = paired_item_differences(records, "candidate", "base")
    assert items[0]["n_seeds"] == 2
    assert items[0]["difference"] == pytest.approx(0, abs=1e-16)
    assert not any(diagnostics.values())
    # Use tolerance only for the binary harm labels; differences stay exact.
    result = paired_effects(records, "candidate", "base", harm_tolerance=1e-15)
    assert result["harm"]["n_unchanged"] == 1
    assert result["harm"]["n_harmed"] == 0
    assert np.isnan(result["leave_one_origin_out"][0]["mean_difference"])
    swapped = paired_effects(records, "base", "candidate", harm_tolerance=1e-15)
    assert swapped["mean_difference"] == pytest.approx(-result["mean_difference"])


def test_missing_and_undefined_values_are_excluded_with_counts():
    records = [dict(origin=2015, item_id="a", seed=None, method="base", ndcg20=None),
               dict(origin=2015, item_id="a", seed=11, method="candidate", ndcg20=.5),
               dict(origin=2015, item_id="b", seed=None, method="base", ndcg20=.2)]
    result = paired_effects(records, "candidate", "base")
    assert result["n_item_origins"] == result["n_origins"] == 0
    assert result["n_undefined_item_origins"] == result["n_undefined_seed_pairs"] == 1
    assert result["n_unmatched_item_origins"] == 1
    assert np.isnan(result["mean_difference"])
    assert np.isnan(result["ci"]).all()
    assert result["items"] == []


def test_duplicate_and_unmatched_seed_records_fail_explicitly():
    records = records_for()
    with pytest.raises(ValueError, match="Duplicate paired record"):
        paired_effects(records + [records[0]], "candidate", "base")
    with pytest.raises(ValueError, match="Unmatched learned seed sets"):
        paired_effects(records[1:], "candidate", "base")
    mixed = records + [dict(records[0], seed=None)]
    with pytest.raises(ValueError, match="Mixed deterministic and seeded"):
        paired_effects(mixed, "candidate", "base")
    for kwargs in ({"n_bootstrap": 0}, {"confidence": 1}, {"harm_tolerance": -1}):
        with pytest.raises(ValueError):
            paired_effects(records, "candidate", "base", **kwargs)
