# Data protocol

This module reconstructs the Digital Music → CDs and Vinyl experiment from the recorded protocol. It does not recover the original preprocessing code. Reviews and current title metadata come from the complete Amazon Reviews 2023 category files. Download manifests record source URLs, file hashes, and row counts; downloading a sample or stopping a review stream early changes the experiment.

## Independent temporal prefixes

There are 18 independently constructed prefixes: three UTC origins, 2015-01-01, 2016-01-01, and 2017-01-01, each with offsets 270, 180, 90, 60, 30, and 0 days. For each prefix, `fit_end = origin - offset`, `fit_start = fit_end - 365 days`, and `query_end = fit_end + 30 days`. A day is exactly 86,400,000 milliseconds. The fit interval is `[fit_start, fit_end)` and the query interval is `[fit_end, query_end)`.

The d270/d180/d90 prefixes fit the historical utility regression; d60 selects the ridge penalty before all four historical prefixes are refitted. The d30 prefix selects deployment parameters and, for the 2015 origin, target and collaborative model hyperparameters. The d0 prefix supplies the final development evaluation. These are separate cohort, catalog, encoder, and activity-standardization fits; a later prefix is never a source of preprocessing statistics for an earlier one.

## Reviews, cohorts, and catalogs

The raw review schema is `user_id`, `parent_asin`, `rating`, and integer Unix-millisecond `timestamp`. Deduplication retains the earliest review of each user–parent-item pair across the complete review history **before** applying time windows or positive-rating thresholds; source-file order resolves exact timestamp ties. Consequently, an early low rating is not replaced by a later positive review. Ratings of at least 4 are positive. `build_domain` and `build_prefix` accept globally deduplicated review frames; the review loader performs that step before construction.

A user is eligible only with at least three distinct positive items inside the prefix's fit interval. Target users are the first 6,000 eligible users in ascending SHA256 order of `2026|cds:users|<user_id>`; a smaller eligible set is retained in full. Every eligible source user is retained. User sampling does not use query activity, query positives, or future availability. Persisted user IDs and item IDs are sorted lexically, so local integer indexes have a stable interpretation.

The warm catalog contains items with a positive fit interaction from a selected user. Training pairs consist of fit positives on that catalog. Fit-positive item degree determines both tail masks: the lowest `ceil(0.8 * n_items)` and `ceil(0.5 * n_items)` degrees, resolving equal degrees by raw item ID. Query labels cannot change the catalog, training pairs, degrees, or tail membership.

All events for selected users on warm items strictly before `query_end` are retained. Their stage is -1 before `fit_start`, 0 in the fit interval, and 1 in the query interval. `observed` retains every rating; `positive` retains ratings of at least 4. Thus candidate exclusion can use every prior rating, including prefit low ratings. The query has one actual stage. The compatibility `cutoffs` array contains `fit_end`, `fit_end + 30 days`, `fit_end + 60 days`, and `fit_end + 90 days`; its extra boundaries do not authorize additional query data.

Fit-period activity uses selected users' reviews across **all** items and all rating values. The `all_rating_*` arrays preserve these events even for items outside the warm catalog, whose local item index is -1. Missing warm membership must not silently remove a user's activity. These arrays contain only `[fit_start, fit_end)` events.

## Title representation

Each prefix fits a fresh TF–IDF vocabulary on the target warm-item titles only: word unigrams and bigrams, at most 30,000 features. Truncated SVD uses 64 output dimensions, seven iterations, and random seed 2026; small fixtures use a feasible rank and pad the remaining dimensions with zeros. The fitted target vocabulary and SVD transform both target and source titles. Source titles cannot change the fitted vocabulary, projection, or target embeddings.

Embeddings are float32, have 64 columns, and are normalized to unit Euclidean norm when nonzero. Empty titles, titles containing no usable tokens, and source titles with no target-vocabulary overlap retain zero vectors; those items remain in the catalog. There is no source-trained fallback encoder. Current metadata does not provide an as-of publication snapshot, so the historical availability and wording of titles cannot be verified; this is a reconstruction limitation.

## Activity features

For each selected user, let `t` range over their fit-period any-rating timestamps across all items, let `c = fit_end`, and let `age = (c - t) / DAY`. The five raw coordinates are all transformed with `log1p`:

1. Number of events with `age <= 30`.
2. Number of events with `age <= 90`.
3. Number of events with `age <= 365`.
4. Minimum fractional age in days, or 365 when there are no events.
5. Number of distinct backward seven-day bins, `floor((c - 1 - t) / (7 * DAY))`.

The one-millisecond shift in the final coordinate places an event exactly seven days before the cutoff in the most recent bin. Fractional recency must not be rounded to a whole day. Target selected users define the mean and population standard deviation of these five transformed coordinates. Standard deviations are clamped below at `1e-6`. The same target mean and scale standardize target and source activity. Leave-focal-out training features remove focal-item activity before recomputing these coordinates, using the same frozen target standardization.

## NPZ contract and persistence

`save_data(path, data)` writes an NPZ through a temporary file and atomically replaces the destination. `load_data(path)` loads arrays with pickle disabled. Successful replacement yields one complete artifact, and scalar time boundaries remain integer millisecond values.

Global arrays are `fit_start`, `fit_end`, `query_end`, `cutoffs`, `activity_mean`, and `activity_scale`; `degree`, `tail80`, and `tail50` alias the corresponding target values. Both `target_` and `source_` namespaces contain:

| Suffix | Meaning |
| --- | --- |
| `x` | Float32 title embedding, shape `[n_items, 64]` |
| `user_ids`, `item_ids` | Lexically sorted Unicode raw IDs |
| `train` | Integer fit-positive pairs, shape `[n_fit_positive, 2]` |
| `observed`, `positive` | Integer user/item/stage rows, shape `[n_events, 3]` |
| `event_users/items/times/ratings` | Aligned warm-catalog events before query end |
| `all_rating_users/items/times/ratings` | Aligned fit-only any-rating events across all items |
| `degree`, `tail80`, `tail50` | Fit-positive degrees and deterministic tail masks |
| `activity_raw`, `activity` | Five transformed activity coordinates, before/after target standardization |

The tests cover global earliest-event deduplication, temporal boundaries, deterministic cohorts, future-data invariance of fitted state, nonwarm activity, target-only title fitting, activity arithmetic, and NPZ round trips. These checks establish the reconstruction's implemented rules; they are not evidence of equivalence to unavailable historical artifacts.
