# Target-model reconstruction protocol

These implementations independently reconstruct the five R3 families from the
supplied report and analysis. Original source and checkpoints are unavailable.
Historical report values are not measurements from this implementation.

Each scorer is `alpha * dot(h,x) + dot(theta,phi) + dot(beta,a)`. The
`unbounded_mean` control omits the activity term. Alpha starts at one, theta and
beta at zero; all active parameters receive exact gradients and Adam updates
(moments 0.9/0.999, epsilon 1e-8). There is no tanh score bound.

The historical documents do **not** fully specify phi. This reconstruction
explicitly chooses `phi = normalize(concat(h, x*h, abs(x-h)))`, dimension 192
for the supplied 64-dimensional titles. Normalization divides by
`max(L2 norm, 1e-12)`, so zero remains zero and every phi has norm at most one
(up to floating-point rounding). This is a reconstruction assumption, not a
claim about the original implementation. The theta block alone is shared with
source corrections; alpha, beta, and all fit representations stay frozen.

History contains positive (rating >=4) events within the 365-day fit window.
Its latest 50 events are retained, using item ID to resolve timestamp ties.
For training positive users, **all focal-item events are removed before the
50-event cap**. Ordinary mean uses equal weights; decay uses half-life 90 days;
conditional uses `exp(dot(x,item_history)/0.2)`; both multiplies the two weights.
The weighted sum is L2 normalized, and empty histories produce zeros.

Activity uses all fit-window ratings, including nonwarm items: log1p review
counts in 30, 90, and 365 days; log1p fractional days since the latest rating;
and log1p occupied 7-day bins relative to cutoff (`(cutoff-1-time)//7days`).
No remaining rating uses recency 365 days. The complete focal item's rating
events are removed for training positives before recomputing all five values.
Target fit statistics standardize both domains; no query event enters features.

Every warm item appears exactly once in an independently shuffled epoch. For
each, uniformly sample up to five distinct fit-positive users and up to five
distinct negative users, then use their Cartesian pairs. Negative eligibility
excludes **every** user with any rating for that item before the fit cutoff,
including prefit ratings; future ratings never alter eligibility. Empty pair
sets have no data gradient. Average pair losses within each item, then average
items in batches of 32. The loss is
`mean softplus((negative_score-positive_score)/tau) + rho/2 * ||theta||²`.
Only theta receives the declared residual regularizer. Negative sampling,
initialization, and shuffle use NumPy's seeded generator.

`pair_loss_and_gradient(model,batch,regularize=True)` accepts positive/negative
anchor, phi, and activity arrays, optionally a leading item-batch dimension and
padding masks. It returns the scalar loss and `theta`, `beta`, `alpha`
gradients. Utility construction uses `regularize=False` to measure empirical
source/query objectives. `FeatureStore.pair_batch` performs focal removal for
the positive branch; `all_features` and `score_all` use full fit history.

The frozen HPO grid has 12 settings: lr 0.001/0.003/0.01, rho 0/0.0001, tau
0.1/0.3. Each uses seed 101 and exactly six epochs on `cds_2015_d30`. Choose
maximum tail80 macro NDCG@20, with trial order breaking ties, then refit each
family for each 2015/2016/2017 final prefix and seeds 11/22/33/44/55. No callback
can select an epoch or alter the six-epoch budget. Code/config/data hashes are
written before scoring. The runner resumes completed files, and cached feature
banks depend only on data, representation configuration, and sampling seed.

Deterministic comparisons are mean-history title cosine and raw 365-day
all-rating activity count (stored as a monotone log1p count). Both use the same
full eligible user evaluator, with stable user-ID tie breaking.
