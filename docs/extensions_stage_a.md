# Stage A existing-artifact analysis

Run `.venv/bin/python scripts/analyze_extensions.py --help` to inspect its interface,
then `.venv/bin/python scripts/analyze_extensions.py`. The entry point only loads
existing artifacts, reconstructs historical routing from saved checkpoints, and
replays the 15 exact-null final scorers. It never fits parameters or selects using
final queries. The original core modules and reports remain unchanged.

The global `configs/extensions_v1.json` fixes five target comparisons, 2,000
bootstrap draws with seed 2026, and 2015–2017 exploratory scope. A separate
`analysis_frozen.json` binds finalized analysis code and dependencies before the
strict original summarizer or any new analysis runs. Reruns reject changed code,
protocol or original reports; use a new versioned output directory for a revised
analysis. The original strict summarizer validates recorded data/model/evaluation
identities, completion markers and the full required matrix. Its manifest is
preserved, and all input bytes are rehashed after analysis.

Outputs under `reports/extensions_v1/stage_a/` include the four required CSVs,
the complete provenance JSON, the Chinese report, an original-table comparison,
material availability, and an isolated full strict-original report. Secondary
metrics retain explicit full-catalog and eligible-item denominators at run,
origin, and equal-origin levels. Aggregate counts sum origin populations;
aggregate rates remain equal-origin means, not pooled ratios.

The existing paired estimator averages matched optimization seeds first. The
additional sensitivity resamples unique raw item identities across origins,
carries all their records with shared multiplicity, and still averages origins
equally. Draws losing any origin are rejected and counted. Neither interval
accounts for shared-user dependence or turns these development data into an
independent confirmatory test.

Historical configuration deduplication hashes exact correction vectors across
all historically evaluated items, normalizing only signed zero. Equal scores or
equal routing summaries are insufficient. Radius-zero execution is reported
separately from native zero actions at positive radius and the same temperature.
Conditional nonzero routing statistics condition on actual correction vectors,
including the possibility of nonzero weights cancelling to zero correction.

The actual convex-mixture check uses stored d60 actual pair losses and their
weighted single-donor gains with mass at most one. It does not extend to
norm-normalized controls or future ranking metrics. Exact-null acceptance
recomputes every evaluated item's complete user-score row and evaluation arrays
for all 15 canonical origin/seed runs, using saved source coefficients and no
training.
