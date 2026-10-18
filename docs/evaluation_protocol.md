# Evaluation and descriptive paired analysis

This is an independent reconstruction. Supplied historical metrics are references,
not observations produced by this code. Evaluation uses the full eligible user
universe; it does not sample ranking negatives.

## Catalog, candidates and undefined metrics

`evaluate(data, stage, score_all, ks=(10,20,50), save_topk=True, batch_size=8,
item_mask=None)` retains one row for every input item. `eligible_catalog` is
`fit_positive_degree > 0` and never depends on query feedback. `requested` is that
warm mask intersected with an optional boolean item mask. Production prepared
catalogs contain only warm items. The distinction remains explicit in saved
results and allows the evaluator to reject cold items even in diagnostic inputs.

For each item, candidates are every cohort user except users with any observed
rating at a stage strictly before the query stage. Exclusion therefore includes
prefit history and ratings below the positive threshold. Query positives are
unique user-item pairs at the requested stage, after applying the same prior
exclusion. Query-stage events do not exclude users. Scores tie by ascending raw
user ID, independent of the order of users in memory.

Only requested items with at least one eligible query positive are scored.
`scored`, `evaluated` and `ranked` coincide for successfully evaluated rows.
`ranked` remains true when rank saving is disabled. Other rows have NaN metrics
and saved top-user indices of -1; they are never represented as measured zero.
`top_users` contains local user indices, padded to the largest K with -1; map
these through `target_user_ids` to recover raw IDs. `save_topk=False` produces
an array with shape `[n_items,0]`. Candidate scores must be finite; scores for
excluded users may be nonfinite because they cannot affect rankings.

`evaluate_many` accepts a shared scorer returning `[policies,items,users]`, or
a policy-name mapping to `[items,users]` arrays. It returns one ordinary result
dictionary per policy and shares all query eligibility preparation. The utility
runner uses this interface to reuse user features across policies without
changing rankings or metric definitions.

## Metric definitions and denominators

For K, let `slots=min(K,n_candidates)`, `hits` be the number of positives among
those slots, and P the number of unique eligible query positives. Then:

- Recall@K = hits / P.
- Precision@K = hits / slots. This convention is explicit when fewer than K
  candidates exist.
- DCG@K = sum of binary relevance / log2(rank+1), with rank starting at one.
- NDCG@K = DCG@K / sum from rank 1 through min(K,P) of 1/log2(rank+1).

Summary metric means are macro averages over evaluated items only. Their
denominator must be reported alongside full catalog counts. Catalog counts
include every supplied row in the selected stratum, including no-positive and
unrequested rows. `items_eligible` separately counts warm rows. In production
these catalog and warm denominators coincide. Hit coverage@K is the number of
evaluated items with at least one hit divided by the entire stratum catalog;
under a partial request this is observed coverage, not an extrapolation to
unscored items. `conditional_hit_rateK` divides that same hit count by evaluated
items. Empty conditional means are NaN, while zero hits over a nonempty catalog
give coverage zero. JSON serialization converts undefined values to null.

Tail80, Tail50, and optional natural degree strata (1, 2–5, 6–10, 11–20, 21+)
are determined from fit positive degree, never query feedback. The evaluator
uses the supplied fixed tail masks. It records candidate counts and the literal
candidate mode `full_eligible_users` in every result.

## Paired effects and dependence

`paired_effects(records, method, baseline, metric='ndcg20', n_bootstrap=2000,
seed=2026)` accepts long records with keys `origin,item_id,seed,method,ndcg20`.
For every common item-origin, it pairs the same optimization seeds and averages
their metric differences first. A deterministic method must use `seed=None`;
its value is broadcast against the other method's seeds. Duplicate records,
mixed deterministic and seeded records, and unequal learned seed sets are
errors. Items absent in one method and nonfinite paired values are excluded
with explicit counts. Partially finite seed sets report their surviving pair
counts in `items`; they are not silently counted as complete runs.

The reported point difference averages item differences within each origin and
then averages origins equally, regardless of their catalog size. For each of
2,000 bootstrap draws, items are resampled with replacement separately within
each origin, followed by the same equal-origin mean. The default 95% interval
uses the 2.5th and 97.5th empirical percentiles with NumPy's linear quantile
interpolation. Input row order cannot change the results. Repeating identical
optimization seeds does not add item observations or narrow the interval.

Harm, improvement and unchanged counts use the seed-averaged item difference;
the default tolerance is exactly zero. Top-level rates average origin rates
equally; `pooled_harm_rate` is separately labeled. Harm magnitudes are the mean
negative part of the difference. Whole-origin resampling and leaving out each
origin in turn provide sensitivity descriptions. No p-values or confirmatory
significance labels are produced. Item bootstrap does not preserve shared-user
dependence, repeated-item dependence across origins, or overlapping-window
dependence. Three development origins and exploratory model selection do not
support reliable confirmatory significance claims.
