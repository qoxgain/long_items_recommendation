"""Descriptive paired effects without treating optimization seeds as datasets.

Input is a sequence of long records with origin, item_id, seed, method and a
metric (ndcg20 by default). Pair seeds first, average their differences within
each item-origin, then weight origins equally. A deterministic method is marked
with seed=None and is broadcast only against its paired method's seeds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _seed_order(value):
    return type(value).__name__, str(value)


def paired_item_differences(records, method, baseline, metric="ndcg20"):
    """Return paired, seed-averaged item rows and transparent exclusion counts.

    Duplicate method/origin/item/seed rows and unmatched learned seed sets are
    errors. Missing methods and undefined paired values are excluded and counted.
    A row whose metric is None is undefined, as is a NaN or infinite value.
    Deterministic records must use seed=None, never an arbitrary numeric seed.
    """
    if method == baseline:
        raise ValueError("method and baseline must be distinct")
    grouped = {method: defaultdict(dict), baseline: defaultdict(dict)}
    for record in records:
        name = record["method"]
        if name not in grouped:
            continue
        key = (str(record["origin"]), str(record["item_id"]))
        seed = record["seed"]
        if isinstance(seed, np.generic):
            seed = seed.item()
        if seed in grouped[name][key]:
            raise ValueError(f"Duplicate paired record: {name}, {key}, seed={seed}")
        value = record[metric]
        grouped[name][key][seed] = float(value) if value is not None else float("nan")
    method_keys, baseline_keys = set(grouped[method]), set(grouped[baseline])
    common = method_keys & baseline_keys
    diagnostics = {
        "n_unmatched_item_origins": len(method_keys ^ baseline_keys),
        "n_undefined_item_origins": 0,
        "n_undefined_seed_pairs": 0,
    }
    items = []
    for origin, item in sorted(common):
        left, right = grouped[method][origin, item], grouped[baseline][origin, item]
        for name, values in ((method, left), (baseline, right)):
            if None in values and len(values) != 1:
                raise ValueError(f"Mixed deterministic and seeded records: {name}, {(origin, item)}")
        if None in left and None in right:
            seeds = [None]
        elif None in left:
            seeds = sorted(right, key=_seed_order)
        elif None in right:
            seeds = sorted(left, key=_seed_order)
        else:
            if set(left) != set(right):
                raise ValueError(f"Unmatched learned seed sets for {(origin, item)}")
            seeds = sorted(left, key=_seed_order)
        pairs = np.asarray([(left[None if None in left else seed],
                             right[None if None in right else seed])
                            for seed in seeds], dtype=np.float64)
        finite = np.all(np.isfinite(pairs), axis=1)
        diagnostics["n_undefined_seed_pairs"] += int((~finite).sum())
        pairs = pairs[finite]
        if not len(pairs):
            diagnostics["n_undefined_item_origins"] += 1
            continue
        items.append({"origin": origin, "item_id": item, "n_seeds": len(pairs),
                      "method_mean": float(pairs[:, 0].mean()),
                      "baseline_mean": float(pairs[:, 1].mean()),
                      "difference": float((pairs[:, 0] - pairs[:, 1]).mean())})
    return items, diagnostics


def _interval(values, confidence):
    if not len(values):
        return [float("nan"), float("nan")]
    alpha = (1 - confidence) / 2
    return np.quantile(values, [alpha, 1 - alpha]).tolist()


def _origin_summary(origin, rows, tolerance):
    differences = np.asarray([row["difference"] for row in rows])
    harmed, improved = differences < -tolerance, differences > tolerance
    return {"origin": origin, "n_items": len(rows),
            "n_seed_pairs": sum(row["n_seeds"] for row in rows),
            "method_mean": float(np.mean([row["method_mean"] for row in rows])),
            "baseline_mean": float(np.mean([row["baseline_mean"] for row in rows])),
            "mean_difference": float(differences.mean()),
            "n_harmed": int(harmed.sum()), "n_improved": int(improved.sum()),
            "n_unchanged": int((~harmed & ~improved).sum()),
            "harm_rate": float(harmed.mean()), "improvement_rate": float(improved.mean()),
            "unchanged_rate": float((~harmed & ~improved).mean()),
            "mean_harm": float(np.maximum(-differences, 0).mean()),
            "mean_improvement": float(np.maximum(differences, 0).mean()),
            "worst_difference": float(differences.min()),
            "best_difference": float(differences.max())}


def paired_effects(records, method, baseline, metric="ndcg20", n_bootstrap=2000,
                   seed=2026, confidence=.95, harm_tolerance=0.):
    """Compute equal-origin effect, paired-item interval, harm and sensitivities.

    In each bootstrap draw, resample item-origin rows separately within every
    origin, then average the origin means equally. Seeds are averaged *before*
    resampling. Origin resampling and leave-one-origin-out values are additional
    descriptive sensitivity analyses, not confirmatory significance tests.

    ``items`` contains the actual seed-averaged paired records. All means and
    rates in the top-level report use equal origin weights unless named pooled.
    Empty comparisons return undefined (NaN) estimates and intervals.
    """
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not np.isfinite(harm_tolerance) or harm_tolerance < 0:
        raise ValueError("harm_tolerance must be finite and nonnegative")
    n_bootstrap = int(n_bootstrap)
    items, diagnostics = paired_item_differences(records, method, baseline, metric)
    by_origin = defaultdict(list)
    for row in items:
        by_origin[row["origin"]].append(row)
    origins = [_origin_summary(origin, rows, harm_tolerance)
               for origin, rows in sorted(by_origin.items())]
    rng = np.random.default_rng(seed)
    item_draws = np.zeros(n_bootstrap) if origins else np.empty(0)
    # Chunking caps memory while preserving the exact sample sequence.
    for origin in sorted(by_origin):
        values = np.asarray([row["difference"] for row in by_origin[origin]])
        for start in range(0, n_bootstrap, 128):
            stop = min(start + 128, n_bootstrap)
            indices = rng.integers(len(values), size=(stop - start, len(values)))
            item_draws[start:stop] += values[indices].mean(axis=1) / len(origins)
    origin_means = np.asarray([row["mean_difference"] for row in origins])
    if len(origins):
        origin_indices = rng.integers(len(origins), size=(n_bootstrap, len(origins)))
        origin_draws = origin_means[origin_indices].mean(axis=1)
    else:
        origin_draws = np.empty(0)
    equal_mean = lambda key: (float(np.mean([row[key] for row in origins]))
                              if origins else float("nan"))
    counts = {name: sum(row[name] for row in origins)
              for name in ("n_harmed", "n_improved", "n_unchanged")}
    harm = dict(counts, tolerance=float(harm_tolerance))
    for name in ("harm_rate", "improvement_rate", "unchanged_rate", "mean_harm", "mean_improvement"):
        harm[name] = equal_mean(name)
    harm["pooled_harm_rate"] = counts["n_harmed"] / len(items) if items else float("nan")
    leave_one_out = []
    for index, origin in enumerate(origins):
        remaining = np.delete(origin_means, index)
        leave_one_out.append({"omitted_origin": origin["origin"],
                              "n_origins": len(remaining),
                              "mean_difference": float(remaining.mean()) if len(remaining) else float("nan")})
    return {
        "method": method, "baseline": baseline, "metric": metric,
        "n_origins": len(origins), "n_item_origins": len(items),
        "n_unique_items": len({row["item_id"] for row in items}),
        "n_seed_pairs": sum(row["n_seeds"] for row in items),
        **diagnostics,
        "method_mean": equal_mean("method_mean"),
        "baseline_mean": equal_mean("baseline_mean"),
        "mean_difference": equal_mean("mean_difference"),
        "ci": _interval(item_draws, confidence),
        "bootstrap": {"n_resamples": n_bootstrap, "seed": seed,
                      "confidence": confidence, "unit": "item within origin after seed averaging",
                      "origin_weighting": "equal", "interpretation": "descriptive"},
        "harm": harm, "origins": origins,
        "origin_bootstrap": {"ci": _interval(origin_draws, confidence),
                             "unit": "whole origin", "interpretation": "sensitivity only"},
        "leave_one_origin_out": leave_one_out,
        "items": items,
        "limitations": ["Shared-user dependence is not preserved by item resampling.",
                        "Repeated-item dependence across origins and overlapping time windows is not preserved.",
                        "Few origins and exploratory model comparisons do not support confirmatory significance claims."],
    }
