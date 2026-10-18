# Collaborative filtering reconstruction protocol

These models are independently implemented task adaptations of MF-BPR,
LightGCN and SimpleX. They are not original-source or original-checkpoint
reproductions. They optimize native user–item objectives; only retrieval
transposes their scores to rank users for each item. Titles, source-domain
items, target-query positives and target-query activity never enter fitting.

## Shared fitting rules

All methods use the deduplicated positive edges from the 365-day target fit
window and the same frozen warm user/item IDs as the other methods. Positives
have rating at least four. Every positive edge is visited once per epoch in a
seeded shuffled order. Negatives are drawn independently and uniformly with
replacement from warm items the user has never rated before `fit_end`, including
low-rating and pre-window exclusions. Future positives remain eligible training
negatives. A user with no available negative causes an explicit error.

Training uses Adam, batch size 1024, normal embedding initialization with
standard deviation 0.1, one CPU thread, deterministic PyTorch operations and
exactly six epochs. All parameters are optimized. No validation callback, early
stopping or best-epoch selection exists. The returned model is the final epoch.

## Model definitions and frozen search spaces

MF-BPR scores a user/item pair by the dot product of their learned embeddings.
It minimizes `mean(softplus(score_negative - score_positive))`, with one sampled
negative per positive. Its twelve trials are embedding dimension `{32,64}` ×
learning rate `{0.001,0.005,0.01}` × L2 weight `{0.00001,0.001}`.

LightGCN builds an undirected bipartite graph from fit-positive edges. Each edge
has weight `1/sqrt(degree_user * degree_item)`. It averages the ego embeddings
and each successive propagated embedding through one or two layers, with no
self-loops, feature transform or activation. The score is the propagated dot
product. It uses the same BPR loss and one negative. Its twelve trials are layer
count `{1,2}` × learning rate `{0.001,0.005,0.01}` × L2 weight `{0.00001,0.001}`,
with embedding dimension fixed at 64. Isolated nodes have zero propagated
messages and retain their scaled ego embedding in the layer average.

SimpleX uses 64-dimensional learned user and item embeddings. A user history is
the most recent 50 distinct fit-positive items, ordered by last fit-positive
timestamp and then lexical item ID. During training, the focal positive is
removed **before** the history cap; keeping 51 potential history items allows
the next item to enter. The mean history embedding is transformed by a learned
bias-free linear map initialized with Xavier uniform. The normalized user
representation is
`normalize((1-history_mix)*user_embedding + history_mix*transform(history_mean))`.
Item embeddings are normalized and the pair score is their cosine similarity.
Zero history means a zero history vector. Serving uses the full capped fit
history, without focal removal. This explicit exclusion is a task adaptation.
The cosine contrastive loss is
`mean((1-score_positive)^2 + negative_weight*mean(relu(score_negative-0.9)^2))`,
with ten uniform negatives per positive. The twelve trials are learning rate
`{0.001,0.005}` × history mix `{0.25,0.5,0.75}` × negative weight `{10,50}`.
Its L2 weight is fixed at `0.00001`.

All losses add L2 times half the sum of the mean squared norms of sampled ego
user, positive-item and negative-item embeddings. SimpleX also adds half the
mean squared entries of the history transform. LightGCN regularizes ego
embeddings, not propagated representations. Negatives are averaged within each
example, so changing their count does not automatically change regularization.

## Selection, final evaluation and outputs

Each family executes its twelve prespecified configurations on `cds_2015_d30`
with seed 101 and six epochs. Its selected configuration maximizes macro item
NDCG@20 among tail80 items with eligible query positives in the next 30 days.
Exact ties use the earliest grid index. The complete search table and selected
configuration are frozen before final evaluation. No final query participates
in configuration, seed, epoch, candidate-universe or tail-set selection.

Each selected family is fitted independently on `cds_2015_d0`, `cds_2016_d0`
and `cds_2017_d0`, using each of seeds 11, 22, 33, 44 and 55, always for six
epochs. The families run sequentially. The default run therefore makes 36 HPO
fits and 45 final fits, with no search-budget reduction. Deterministic cosine
and activity baselines belong to `scripts/run_targets.py`.

Evaluation uses the shared evaluator: for each warm item, rank the complete
target-user cohort excluding anyone who rated that item before the query;
resolve score ties by lexical user ID. Preserve every catalog item in coverage
denominators. Items without eligible positives have undefined ranking metrics,
not artificial zeros. Report NDCG, recall, precision and hit coverage at
10, 20 and 50, with all, tail80 and tail50 summaries. Tail sets derive solely
from fit degrees. Scores have shape `[requested_items, all_users]`.

Outputs are `results/baselines/cds_2015_d30/METHOD/search/trial_NN.*` plus
`METHOD/selected.json`; final outputs are
`results/baselines/cds_YEAR_d0/METHOD/seed_N/{model.npz,frozen.json,test.npz,result.json}`.
Every run writes checkpoint, code/config/data hashes, cohort and fit timestamps
and fixed epoch **before** query scoring. Checkpoints include raw catalog IDs
and a fit-structure fingerprint; loading rejects mismatched IDs or histories
but allows changed future labels. JSON is atomic and completed-run resume
verifies the frozen identity, checkpoint hash and evaluation hash. A partial run
is refit deterministically; a complete run with differing provenance fails
explicitly rather than mixing experiments.

Run the full matrix, waiting for atomically published processed prefixes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/run_baselines.py --wait-for-data
```

`--hpo-only` is a development control and does not produce the final matrix.
`--methods` selects families for operational recovery; the full required result
matrix still includes all three. No CLI option changes the six-epoch budget.

The synthetic tests check finite-difference objective gradients, graph
normalization, exact focal removal before capping, exclusion of all past
ratings, unchanged training under altered future labels, native/transposed
score agreement, six-epoch traversal, checkpoint round trips, cohort mismatch
rejection, pre-scoring provenance and rejection of corrupt resume artifacts.
