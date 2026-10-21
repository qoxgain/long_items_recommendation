# Extension figures

`scripts/plot_extensions.py` renders aggregate results from the strict extension
reporter. It never fits or evaluates a model. The input summary and audit must
both be complete; progress snapshots and missing requested stages are refused.
The script checks complete origin/seed matrices and all historical curves before
using them. It neither fills missing cells nor invents uncertainty from seed SD.

```bash
# Completed stages can be rendered separately while other stages run.
.venv/bin/python scripts/plot_extensions.py \
  --report-dir reports/extensions_v1/stage_a_summary --stages a
.venv/bin/python scripts/plot_extensions.py \
  --report-dir reports/extensions_v1/stage_c_summary --stages c

# After the complete A/B/C matrix passes strict reporting:
.venv/bin/python scripts/summarize_extensions.py
.venv/bin/python scripts/plot_extensions.py
```

Default output is `reports/extensions_v1/figures/`. Each figure has SVG, PDF,
240-dpi PNG, the plotted aggregate CSV, and a caption text file. The manifest
records report-input, plotting-code and output hashes plus dependency versions.
Re-render after a report changes; manifests from separate stage invocations
identify their own inputs.

- **A target paired effects:** the five prespecified comparisons, both item and
  unique-item-cluster 95% descriptive bootstrap intervals.
- **B historical convergence:** twelve configuration curves per family, each
  averaging all three historical seeds. The chosen configuration is blue; its
  shading is optimization-seed SD, not a confidence interval. The selected
  checkpoint comes only from saved historical selection.
- **B final comparison:** complete five-seed, three-origin point estimates
  beside the original six-epoch results. The new user-objective LightGCN has
  no original six-epoch counterpart, so that bar is absent.
- **C outcomes by origin:** all four correction policies on each frozen target
  model, plus deterministic activity. Overlapping curves remain in the plot;
  a zero-radius policy must not appear to have a distinct effect.
- **C paired effects:** the six prespecified comparisons and both interval
  definitions.

A/B are previously viewed development-origin analyses. C is same-domain
new-time validation. No figure implies independent-domain confirmation,
statistical significance from point separation, or a ranking guarantee from a
loss bound. Public figure data contain no raw user IDs, item IDs or rankings.
