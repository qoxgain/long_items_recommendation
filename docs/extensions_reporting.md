# Extension reporting

`scripts/summarize_extensions.py` audits saved evidence without fitting models,
opening final-query gates, modifying training caches, or changing frozen
protocols. Its default is strict: every requested cell must validate before any
report file is written.

```bash
# Complete Stage A only; preserve the original analysis directory.
.venv/bin/python scripts/summarize_extensions.py \
  --stages a --output-dir reports/extensions_v1/stage_a_summary

# Final combined report, after all B/C cells finish.
.venv/bin/python scripts/summarize_extensions.py

# Explicitly incomplete progress snapshot in a separate directory.
.venv/bin/python scripts/summarize_extensions.py \
  --allow-partial --output-dir reports/extensions_v1/progress

.venv/bin/python -m pytest -q tests/test_extension_reporting.py
```

The requested B matrix contains 216 historical trajectories, six verified
family selections, and 90 final models. A triggered 100-epoch extension must
cover all 36 trajectories in that family; checkpoints are not independent
experiments. The unchanged six-epoch comparators add 75 fitted evaluations and
three deterministic activity evaluations, explicitly marked as old evidence.

C requires 24 historical label folds, six ridge bundles, two complete
37-configuration historical searches (74 records), the frozen final-query
gate, 120 final policy evaluations, and three deterministic activity
evaluations. Each final model is shared by four policies. Historical/final
configurations, data bytes, code/dependency identities, labels, selected
policies, artifact hashes, candidate counts, evaluation masks, and canonical
zero-policy equality are checked through their saved receipts. Missing,
failed, and invalid evidence remain separate states.

The public CSVs preserve All/Tail50/Tail80 metrics and population denominators.
Metrics average seeds within origin, then weight origins equally. Candidate
and eligibility counts must match across seeds; hit counts depend on fitted
rankings and therefore average across seeds. Shared target historical
evaluation and feature-bank time are allocated across the trajectories that
shared each operation. Timings and process peak RSS are descriptive resource
records, not independent sample counts.

`historical_diagnostics.csv` distinguishes `fold_label` rows from
`ridge_validation` rows. The latter preserve every saved alpha trial for each
feature subset, with a `selected` flag. The audit verifies the complete frozen
alpha grid, validation population, selected minimum-MSE configuration and
exact tie order, and agreement between the selected trial and saved validation
summary. Undefined precision is retained when there are no predicted positives.
These historical loss diagnostics are separate from final ranking endpoints.

`REPORT_ZH.md` and `MANUSCRIPT_RESULT_PATCH.md` contain only verified measured
results. They distinguish the previously viewed 2015–2017 development results
from 2018–2020 same-domain new-time validation. They explicitly identify the
Movies and TV → Books extension as not run because of the resource gate.
Partial mode never averages incomplete seed/origin matrices or imputes missing
results. Reports expose no user IDs, item IDs, rankings, or private attachment
names/hashes; paired item arrays remain in memory.
