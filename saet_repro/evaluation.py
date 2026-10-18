"""Exact item-to-user ranking on the complete eligible user universe.

Each item excludes users with *any* observed rating before the query stage.
Query positives are deduplicated and filtered through that same exclusion. Ties
are broken by ascending raw user ID. Ranking metrics condition on at least one
eligible query positive; the full warm catalog remains in coverage denominators.

``score_all`` receives local integer item indices and returns [items, users].
Metrics for items not scored are NaN, and their saved user ranks are -1. Precision
uses min(K, n_candidates), the number of possible recommendation slots.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np


def _target(data, name):
    key = "target_" + name
    if key in data:
        return np.asarray(data[key])
    if name in data:
        return np.asarray(data[name])
    raise KeyError(f"Missing target data array {key!r}")


def _mask(mask, size, name="item_mask"):
    if mask is None:
        return np.ones(size, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape != (size,) or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array of shape ({size},)")
    return mask


def _events(data, name, n_users, n_items):
    events = _target(data, name)
    if events.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 3 or events.dtype.kind not in "iu":
        raise ValueError(f"target_{name} must have integer columns user,item,stage")
    if (np.any(events[:, 0] < 0) or np.any(events[:, 0] >= n_users)
            or np.any(events[:, 1] < 0) or np.any(events[:, 1] >= n_items)):
        raise ValueError(f"target_{name} contains indices outside the catalog")
    return events


def _group_codes(codes, n_users, n_items):
    """Compact item adjacency without a dense items-by-users allocation."""
    counts = np.bincount(codes // n_users, minlength=n_items)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return codes % n_users, counts, offsets


def _prepare(data, stage, ks, item_mask):
    if isinstance(stage, bool) or int(stage) != stage:
        raise ValueError("stage must be an integer")
    stage = int(stage)
    ks = tuple(ks)
    if not ks or any(isinstance(k, bool) or int(k) != k or k < 1 for k in ks):
        raise ValueError("ks must contain positive integer cutoffs")
    ks = tuple(sorted(set(int(k) for k in ks)))
    users, items = _target(data, "user_ids"), _target(data, "item_ids")
    if users.ndim != 1 or items.ndim != 1:
        raise ValueError("user_ids and item_ids must be one-dimensional")
    if len(np.unique(users)) != len(users) or len(np.unique(items)) != len(items):
        raise ValueError("user_ids and item_ids must be unique")
    n_users, n_items = len(users), len(items)
    degree = _target(data, "degree")
    if degree.shape != (n_items,):
        raise ValueError("target_degree must have one entry per item")
    eligible = degree > 0
    requested = eligible & _mask(item_mask, n_items)
    observed = _events(data, "observed", n_users, n_items)
    positive = _events(data, "positive", n_users, n_items)
    prior = observed[observed[:, 2] < stage, :2]
    future = positive[positive[:, 2] == stage, :2]
    # Encoded pairs sort by item then local user index, and deduplicate repeats.
    prior_codes = np.unique(prior[:, 1] * n_users + prior[:, 0])
    query_codes = np.unique(future[:, 1] * n_users + future[:, 0])
    query_codes = np.setdiff1d(query_codes, prior_codes, assume_unique=True)
    if n_users:
        prior_users, n_prior, prior_offsets = _group_codes(prior_codes, n_users, n_items)
        positive_users, n_positive, positive_offsets = _group_codes(query_codes, n_users, n_items)
    else:
        prior_users = positive_users = np.empty(0, dtype=np.int64)
        n_prior = n_positive = np.zeros(n_items, dtype=np.int64)
        prior_offsets = positive_offsets = np.zeros(n_items + 1, dtype=np.int64)
    return {
        "items": items, "n_items": n_items, "n_users": n_users,
        "user_order": np.argsort(users.astype(str), kind="stable"),
        "prior_users": prior_users, "prior_offsets": prior_offsets,
        "positive_users": positive_users, "positive_offsets": positive_offsets,
        "n_candidates": n_users - n_prior, "n_positive": n_positive,
        "eligible_catalog": eligible, "requested": requested,
        "to_score": np.flatnonzero(requested & (n_positive > 0)),
        "stage": stage, "ks": ks,
    }


def _empty_result(prepared, save_topk):
    size = prepared["n_items"]
    result = {
        "item_id": prepared["items"].copy(),
        "n_candidates": prepared["n_candidates"].copy(),
        "n_positive": prepared["n_positive"].copy(),
        "eligible_catalog": prepared["eligible_catalog"].copy(),
        "requested": prepared["requested"].copy(),
        "scored": np.zeros(size, dtype=bool),
        "evaluated": np.zeros(size, dtype=bool),
        "ranked": np.zeros(size, dtype=bool),
        "top_users": np.full((size, max(prepared["ks"]) if save_topk else 0),
                             -1, dtype=np.int64),
        "stage": np.asarray(prepared["stage"], dtype=np.int64),
        "candidate_mode": np.asarray("full_eligible_users"),
    }
    for k in prepared["ks"]:
        for metric in ("ndcg", "recall", "precision", "hits"):
            result[f"{metric}{k}"] = np.full(size, np.nan, dtype=np.float64)
    return result


def _rank_row(prepared, result, item, scores):
    allowed = np.ones(prepared["n_users"], dtype=bool)
    start, end = prepared["prior_offsets"][item:item + 2]
    allowed[prepared["prior_users"][start:end]] = False
    candidates = prepared["user_order"][allowed[prepared["user_order"]]]
    if not np.all(np.isfinite(scores[candidates])):
        raise ValueError(f"Nonfinite candidate score for item index {item}")
    # Starting with lexical user order makes a stable score sort deterministic.
    ranking = candidates[np.argsort(-scores[candidates], kind="stable")]
    start, end = prepared["positive_offsets"][item:item + 2]
    positives = prepared["positive_users"][start:end]
    relevant = np.isin(ranking[:max(prepared["ks"])], positives, assume_unique=True)
    discount = 1.0 / np.log2(np.arange(len(relevant), dtype=np.float64) + 2.0)
    dcg = np.cumsum(relevant * discount)
    hits = np.cumsum(relevant)
    n_positive = int(prepared["n_positive"][item])
    for k in prepared["ks"]:
        slots = min(k, len(ranking))
        n_hits = int(hits[slots - 1])
        ideal_slots = min(k, n_positive)
        ideal = float(np.sum(discount[:ideal_slots]))
        result[f"ndcg{k}"][item] = float(dcg[slots - 1] / ideal)
        result[f"recall{k}"][item] = n_hits / n_positive
        result[f"precision{k}"][item] = n_hits / slots
        result[f"hits{k}"][item] = n_hits
    saved = min(result["top_users"].shape[1], len(ranking))
    result["top_users"][item, :saved] = ranking[:saved]
    for name in ("scored", "evaluated", "ranked"):
        result[name][item] = True


def evaluate(data, stage, score_all, ks=(10, 20, 50), save_topk=True,
             batch_size=8, item_mask=None):
    """Evaluate a scorer exactly, retaining one output row per catalog item.

    ``eligible_catalog`` denotes warm fit items (degree > 0), independently of
    future feedback. ``requested`` intersects that catalog with ``item_mask``.
    ``scored``, ``evaluated`` and ``ranked`` indicate successful full-candidate
    scoring of requested items with at least one eligible stage positive. They
    coincide for this evaluator; ``ranked`` remains true with save_topk=False.
    """
    def score_many(indices):
        return np.asarray(score_all(indices))[None, ...]

    return evaluate_many(data, stage, score_many, ("model",), ks, save_topk,
                         batch_size, item_mask)["model"]


def evaluate_many(data, stage, score_many, policy_names, ks=(10, 20, 50),
                  save_topk=True, batch_size=8, item_mask=None):
    """Evaluate policies together with one shared scorer call per item batch.

    ``score_many(indices)`` must return [policies, items, users], or a mapping of
    policy names to [items, users] arrays. The returned mapping has normal
    ``evaluate`` results and can be saved one policy at a time without objects.
    """
    names = tuple(policy_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("policy_names must be nonempty and unique")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    prepared = _prepare(data, stage, ks, item_mask)
    results = {name: _empty_result(prepared, save_topk) for name in names}
    indices = prepared["to_score"]
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        scores = score_many(batch)
        if isinstance(scores, Mapping):
            if set(scores) != set(names):
                raise ValueError("score_many mapping must match policy_names")
            scores = np.stack([np.asarray(scores[name]) for name in names])
        else:
            scores = np.asarray(scores)
        expected = (len(names), len(batch), prepared["n_users"])
        if scores.shape != expected or scores.dtype.kind not in "fiu":
            raise ValueError(f"Scorer must return numeric shape {expected}, got {scores.shape}")
        for policy, name in enumerate(names):
            for row, item in enumerate(batch):
                _rank_row(prepared, results[name], item,
                          np.asarray(scores[policy, row], dtype=np.float64))
    return results


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def summarize(data, result, mask=None):
    """Macro conditional metrics and explicit full-catalog denominators.

    Coverage K is the fraction of selected catalog items with at least one hit,
    including no-positive items in its denominator. Under partial requests this
    is observed hit coverage of the full stratum, not an estimate for unscored
    items. ``evaluated_fraction`` likewise uses all selected catalog rows.
    """
    item_ids = _target(data, "item_ids")
    if not np.array_equal(item_ids, result["item_id"]):
        raise ValueError("Result item IDs do not match the data catalog")
    selected = _mask(mask, len(item_ids), "mask")
    eligible = selected & np.asarray(result["eligible_catalog"], dtype=bool)
    requested = selected & np.asarray(result["requested"], dtype=bool)
    evaluated = selected & np.asarray(result["evaluated"], dtype=bool)
    scored = selected & np.asarray(result["scored"], dtype=bool)
    ranked = selected & np.asarray(result["ranked"], dtype=bool)
    positive = selected & (np.asarray(result["n_positive"]) > 0)
    n_catalog, n_evaluated = int(selected.sum()), int(evaluated.sum())
    summary = {
        "items_catalog": n_catalog,
        "items_eligible": int(eligible.sum()),
        "items_requested": int(requested.sum()),
        "items_with_positive": int(positive.sum()),
        "items_without_positive": int((selected & ~positive).sum()),
        "items_scored": int(scored.sum()),
        "items_evaluated": n_evaluated,
        "items_ranked": int(ranked.sum()),
        "evaluated_fraction": _ratio(n_evaluated, n_catalog),
        "query_positive_fraction": _ratio(int(positive.sum()), n_catalog),
        "n_positive": int(np.asarray(result["n_positive"])[selected].sum()),
        "mean_candidates": (float(np.mean(np.asarray(result["n_candidates"])[selected]))
                            if n_catalog else float("nan")),
        "candidate_mode": str(np.asarray(result["candidate_mode"]).item()),
        "precision_denominator": "min(k,n_candidates)",
    }
    for key in sorted(result):
        match = re.fullmatch(r"(ndcg|recall|precision|hits)(\d+)", key)
        if match is None:
            continue
        values = np.asarray(result[key])[evaluated]
        if np.any(~np.isfinite(values)):
            raise ValueError(f"Evaluated rows contain undefined {key}")
        summary[key] = float(np.mean(values)) if len(values) else float("nan")
        if match.group(1) == "hits":
            k = match.group(2)
            n_hit = int(np.sum(values > 0))
            summary[f"items_hit{k}"] = n_hit
            summary[f"coverage{k}"] = _ratio(n_hit, n_catalog)
            summary[f"conditional_hit_rate{k}"] = _ratio(n_hit, n_evaluated)
    return summary


def stratified_summaries(data, result, include_degrees=False):
    """Return all/tail80/tail50 strata defined from fit history only.

    Optional natural degree strata are 1, 2–5, 6–10, 11–20, and 21 or more.
    Every stratum includes unscored and no-positive items in catalog counts.
    """
    summaries = {"all": summarize(data, result)}
    for name in ("tail80", "tail50"):
        summaries[name] = summarize(data, result, _target(data, name))
    if include_degrees:
        degree = _target(data, "degree")
        for name, lower, upper in (("degree_1", 1, 1), ("degree_2_5", 2, 5),
                                   ("degree_6_10", 6, 10), ("degree_11_20", 11, 20),
                                   ("degree_21_plus", 21, np.inf)):
            summaries[name] = summarize(data, result, (degree >= lower) & (degree <= upper))
    return summaries
