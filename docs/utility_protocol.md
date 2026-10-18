# Forward-label utility and bounded transfer protocol

This is an independent reconstruction, not a recovery of the original source or checkpoints. Previously reported numbers are not observations from this implementation. The runner is `scripts/run_utility.py` and the numerical implementation is `saet_repro/utility.py`.

## Independent historical labels

Each origin, 2015, 2016 and 2017, has four independently prepared historical prefixes at offsets 270, 180, 90 and 60 days. Each fits its own target `activity_mean` model for six epochs using learning rate 0.001, preference regularization 0.0001, pair-loss temperature 0.1 and seed 101. These settings are fixed before deployment HPO. Prefix preparation, cohorts, catalogs, target-fitted title encoding and activity standardization follow `data_protocol.md`.

For each fitted target model, the source scorer starts with a copy of its preference vector θ and semantic coefficient α, and a new zero five-coordinate activity coefficient β. Six source epochs visit every source warm item in a seeded random order. Each item samples up to 16 fit-positive users and five negatives; negatives exclude every prior rating. Plain itemwise SGD at the base model's learning rate updates only source β using empirical Cartesian-pair loss without regularization. Target parameters and the copied source θ and α remain unchanged. The source and target have independent local user spaces; no user parameter is copied across domains.

Each target item considers up to 16 source items in descending title similarity, with raw source item ID resolving ties. An identical raw item ID is excluded. Source gradients use up to 16 positive users and five negatives; target support uses up to five positives and five negatives. All support sampling is deterministic for seed, domain and item and uses only the fit prefix. Positive support features remove the focal item from preference history and activity. Only the empirical gradient with respect to θ is eligible for transfer.

The ten routing features are an intercept, text cosine, support-profile cosine, preference-gradient cosine, log1p target and source support counts, target and source profile dispersion, and target and source gradient norms. Counts are the sampled support counts. Features and donor selection never inspect query events.

For a historical query item, sample up to five next-30-day positives that had no prior rating and up to five eligible users without a positive in that query. Every donor shares these query pairs. With source gradient g, form the step δ by projecting −0.1g onto the Euclidean ball of radius 0.1. The label is the actual base pair loss minus the actual pair loss after adding φ·δ to target scores. It is not a gradient dot product. The base target β and α remain fixed. Query information is used only to construct these historical labels and to compute explicitly marked diagnostics.

## Ridge fitting and policy selection

The feature subsets are semantic (intercept and text cosine), global (intercept and source support, dispersion and gradient norm), behavior (semantic, profile cosine, support counts and dispersions), and full (all ten). Each subset tries ridge penalties 0.0001, 0.001, 0.01, 0.1, 1 and 10. Feature mean and standard deviation come from training folds; the intercept is literal and unpenalized. Labels are divided by their positive RMS without subtracting a mean, preserving zero as zero actual improvement.

The d270, d180 and d90 labels train the ridge candidates. The d60 labels select the lowest actual-label MSE, breaking ties by penalty order. Each selected ridge then refits on all four folds with freshly fitted feature normalization and label RMS. The d60 mixture diagnostics use the first-three-fold model, not the four-fold refit; d60 is a selection set, not an untouched test set.

For each diagnostic query batch, `validation_mixtures.json` separately stores the actual mixture loss reduction and the weighted sum of actual single-donor loss reductions. These quantities need not agree because pair loss is nonlinear. Best actual single-donor and best-with-null values are marked label-only oracles and are never deployment policies or selection inputs.

Deployment has seven policies: exact zero, uniform, semantic, global, behavior, full, and predicted single. The four regression policies solve the nonnegative L1-ball objective

`max_w w·predicted_scaled_gain − temperature·||w||²/2`, subject to `w ≥ 0` and `sum(w) ≤ 1`.

This is Euclidean projection of predicted gain divided by temperature onto that ball. All nonpositive predictions give literal zero weights. Uniform assigns mass 1 equally; predicted single assigns mass 1 to the first maximum positive full-model prediction, otherwise zero. Predicted single does not access actual labels.

On the 2015 d30 prefix only, each of the six nonzero policies evaluates the fixed 12 combinations of radius {0, 0.01, 0.05, 0.1} and temperature {0.1, 1, 10}. Exact zero is evaluated once, giving 73 nominal evaluations. Duplicate configurations retain their nominal slots. Tail80 macro item NDCG@20 selects each policy independently, breaking ties by radius then temperature order. Undefined metrics cannot select a configuration. These policy settings are frozen for all final origins and seeds.

## Frozen final evaluation and controls

Each final run uses the exact corresponding canonical target `activity_mean` checkpoint from target HPO and final training, with seeds 11, 22, 33, 44 and 55. It learns a source-only β for that base and applies item-specific preference corrections without updating the target model. A native correction is a nonnegative mixture of individually clipped source steps; mass at most 1 guarantees correction norm at most the selected radius. Since φ has norm at most 1, an individual score changes by at most that radius, and a pair margin greater than twice the radius cannot reverse.

Exact zero bypasses score addition and preserves base values. The runner compares its final zero results, top users and evaluation mask against the saved canonical target results with exact array equality and asserts that target θ, β and α did not change.

Two diagnostic controls are evaluated separately for each of the six nonzero policies at fixed radius 0.1 and temperature 1. `mass1` normalizes a native nonzero weight vector to mass 1 before mixing clipped steps. `norm01` retains the native weights and scales a nonzero final correction to norm 0.1. Both preserve native exact nulls; a cancelling zero correction stays zero. These controls do not select a deployment setting.

Each origin/seed therefore has seven main plus twelve control evaluations. Three origins times five seeds give **285 final policy evaluations**: 105 main and 180 controls. Historical fits and the 73 deployment-selection evaluations are separate from this final count.

## Artifacts, resumption and storage

Frozen records are written before fitting or scoring. They bind data bytes, utility/target/evaluator/runner code, configuration, seed and stage-specific upstream hashes. Source β is bound to the exact base checkpoint and data. Historical labels bind their base and source checkpoints; ridges bind all four label files; deployment selection and final evaluation bind ridge and canonical base checkpoints. Final records also bind the selected policy file.

Completion records contain SHA-256 hashes of their artifacts. A resume validates dependencies and bytes rather than trusting file existence. Changed provenance or modified completed bytes fails explicitly; use a separate result directory for a changed experiment. Incomplete stages can be regenerated from the same frozen inputs. NPZ and JSON publication uses atomic replacement, and a completion marker is published last. Every final policy stores `test.npz`, `routing.npz` and `result.json`; each origin/seed additionally has `complete.json` covering all nineteen policies.

Historical target fitting explicitly disables the optional on-disk feature banks: there is only one target configuration per independent historical prefix, so keeping twelve six-epoch caches provides no cross-trial reuse. An inherited cache setting is restored after the fit. Source β, model, label and ridge checkpoints remain resumable without accumulating these large feature banks.

Tests cover temporal invariance, independent user spaces, actual forward and mixture losses, ridge zero semantics, projection KKT conditions, score and norm bounds, exact-null equivalence, separate control normalizations, shared-versus-independent evaluation, verified resumption, corrupted artifacts, source checkpoint binding and the fixed 285-evaluation final budget.
