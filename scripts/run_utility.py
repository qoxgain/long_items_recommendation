#!/usr/bin/env python3
"""Resume independent historical labels, ridge selection and frozen transfer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from saet_repro.target import Config, Model, train_target
from saet_repro.evaluation import evaluate_many, stratified_summaries
from saet_repro.io import write_json, sha256, json_safe
from saet_repro.utility import (
    FEATURE_NAMES, FEATURE_SUBSETS, POLICIES, RADII, TEMPERATURES, RIDGE_ALPHAS,
    RidgeUtility, RouteContext, MultiPolicyScorer, forward_labels, select_ridges,
    query_batches, batch_loss, policy_correction,
)

HISTORICAL_CONFIG = Config(method='activity_mean', lr=.001, rho=.0001, tau=.1)
OFFSETS = (270, 180, 90, 60)


def freeze(path, expected):
    """Reject changed inputs instead of silently mixing two experiments."""
    path = Path(path)
    expected = json_safe(expected)
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError(f'Provenance mismatch at {path}; use a separate result directory')
    if not path.exists():
        write_json(path, expected)


def completed(path, frozen, artifacts):
    """A completion marker is reusable only with matching dependencies/bytes."""
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('frozen') != json_safe(frozen):
        raise RuntimeError(f'Provenance mismatch at {path}; use a separate result directory')
    if not all(Path(value).exists() for value in artifacts.values()):
        return None
    hashes = {name: sha256(value) for name, value in artifacts.items()}
    if record.get('artifacts') != hashes:
        raise RuntimeError(f'Artifact hash mismatch at {path}')
    return record


def finish(path, frozen, artifacts, **record):
    write_json(path, dict(record, frozen=frozen,
                         artifacts={name: sha256(value) for name, value in artifacts.items()}))


def code_hashes():
    return {name: sha256(ROOT / name) for name in (
        'saet_repro/utility.py', 'saet_repro/target.py',
        'saet_repro/evaluation.py', 'scripts/run_utility.py')}


def wait_for(path, wait):
    path = Path(path)
    while not path.exists():
        if not wait:
            raise FileNotFoundError(path)
        print(f'Waiting for {path}', flush=True)
        time.sleep(15)
    return path


def load_prefix(path, wait):
    wait_for(path, wait)
    from saet_repro.data import load_data
    return load_data(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def provenance(path, **extra):
    return {'data_sha256': sha256(path), 'data_path': str(path),
            'code_sha256': code_hashes(),
            'status': 'independent reconstruction; historical report numbers are not observations',
            **extra}


def context_for(data, model, directory, seed, *, data_path, base_path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'source_activity.npz'
    frozen = provenance(data_path, base_model_sha256=sha256(base_path), seed=seed,
        source_epochs=6, source_positive_cap=16, negative_cap=5,
        theta_and_alpha_frozen=True, source_beta_initial='zero',
        source_beta_optimizer='itemwise SGD at base learning rate, no regularization')
    freeze(directory / 'source_activity.frozen.json', frozen)
    summary_path = directory / 'source_activity.json'
    if completed(summary_path, frozen, {'source_activity': path}):
        with np.load(path, allow_pickle=False) as saved:
            context = RouteContext(data, model, seed=seed, source_beta=saved['beta'])
    else:
        context = RouteContext(data, model, seed=seed)
        save_npz(path, beta=context.source.beta)
        finish(summary_path, frozen, {'source_activity': path}, **context.source_summary())
    return context


def historical_fold(args, year, offset):
    path = args.data_dir / f'cds_{year}_d{offset}.npz'
    directory = args.result_dir / f'cds_{year}' / 'historical' / f'd{offset}'
    label_path = directory / 'labels.npz'
    data = load_prefix(path, args.wait_inputs)
    directory.mkdir(parents=True, exist_ok=True)
    frozen = provenance(path, config=asdict(HISTORICAL_CONFIG), seed=101, epochs=6,
                        role='ridge_validation' if offset == 60 else 'ridge_training',
                        label_radius=.1, label_eta=.1, donor_cap=16,
                        source_positive_cap=16, target_support_cap=5, negative_cap=5)
    freeze(directory / 'training.frozen.json', frozen)
    started = time.monotonic()
    model_path = directory / 'model.npz'
    model_artifacts = {'model': model_path, 'history': directory / 'training.json'}
    if completed(directory / 'training.complete.json', frozen, model_artifacts):
        model = Model.load(model_path, data)
    else:
        # One model per prefix: feature banks cannot be reused across trials.
        # Keeping all twelve banks would waste several GB of the disk budget.
        previous_cache = os.environ.pop('SAET_FEATURE_CACHE', None)
        try:
            model, history = train_target(data, HISTORICAL_CONFIG, seed=101, epochs=6)
        finally:
            if previous_cache is not None:
                os.environ['SAET_FEATURE_CACHE'] = previous_cache
        model.save(model_path)
        write_json(directory / 'training.json', history)
        finish(directory / 'training.complete.json', frozen, model_artifacts)
    context = context_for(data, model, directory, 101, data_path=path, base_path=model_path)
    frozen.update(base_model_sha256=sha256(model_path),
                  source_activity_sha256=sha256(directory / 'source_activity.npz'))
    freeze(directory / 'labels.frozen.json', frozen)
    if completed(directory / 'result.json', frozen, {'labels': label_path}):
        with np.load(label_path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    labels = forward_labels(data, context, radius=.1)
    save_npz(label_path, **labels)
    finish(directory / 'result.json', frozen, {'labels': label_path},
        n_labels=len(labels['labels']), n_query_items=len(labels['query_item_summary']),
        label_positive_fraction=np.mean(labels['labels'] > 0) if len(labels['labels']) else None,
        label_rms=np.sqrt(np.mean(labels['labels'] ** 2)) if len(labels['labels']) else None,
        source=context.source_summary(), elapsed_seconds=time.monotonic() - started)
    print(f'Historical {year} d{offset}: {len(labels["labels"])} forward labels', flush=True)
    return labels


def validation_mixtures(data, context, ridges):
    """Query-dependent diagnostics only; never use these to choose deployment."""
    records = []
    for item, positive, negative, batch in query_batches(data, context):
        donors, features, gradients = context.route_inputs(item)
        if not len(donors):
            continue
        base_loss = batch_loss(context.base, batch)
        from saet_repro.utility import clip_step
        single_gains = [base_loss - batch_loss(context.base, batch, step)
                        for step in clip_step(gradients, .1, .1)]
        for policy in POLICIES:
            correction, weights = policy_correction(policy, features, gradients,
                                                       ridges, .1, temperature=1.)
            records.append({'item_id': item, 'policy': policy,
                'actual_mixture_gain': base_loss - batch_loss(context.base, batch, correction),
                'weighted_actual_single_gain': float(weights @ single_gains),
                'source_mass': float(weights.sum()), 'correction_norm': float(np.linalg.norm(correction)),
                'oracle_single_gain': max(single_gains),
                'oracle_with_null_gain': max(0., max(single_gains)),
                'oracle_is_label_only_not_deployable': True,
                'query_positive_count': len(positive), 'query_negative_count': len(negative)})
    return records


def fit_ridges(args, year):
    directory = args.result_dir / f'cds_{year}' / 'ridges'
    labels = {offset: historical_fold(args, year, offset) for offset in OFFSETS}
    train_x = np.concatenate([labels[d]['features'] for d in OFFSETS[:3]])
    train_y = np.concatenate([labels[d]['labels'] for d in OFFSETS[:3]])
    valid_x, valid_y = labels[60]['features'], labels[60]['labels']
    if not len(train_y) or not len(valid_y):
        raise ValueError(f'{year}: historical training and validation must have labels')
    directory.mkdir(parents=True, exist_ok=True)
    frozen = {'code_sha256': code_hashes(), 'feature_names': FEATURE_NAMES,
        'feature_subsets': FEATURE_SUBSETS, 'alphas': RIDGE_ALPHAS,
        'train_offsets': OFFSETS[:3], 'validation_offset': 60,
        'label_scaling': 'positive training RMS, no mean subtraction',
        'fold_hashes': {str(d): sha256(args.result_dir / f'cds_{year}' / 'historical' / f'd{d}' / 'labels.npz') for d in OFFSETS}}
    freeze(directory / 'frozen.json', frozen)
    artifacts = {s: directory / f'{s}.npz' for s in FEATURE_SUBSETS}
    artifacts['validation_mixtures'] = directory / 'validation_mixtures.json'
    if completed(directory / 'selected.json', frozen, artifacts):
        return {s: RidgeUtility.load(directory / f'{s}.npz') for s in FEATURE_SUBSETS}
    ridges, diagnostics = select_ridges(train_x, train_y, valid_x, valid_y)
    for subset, model in ridges.items():
        model.save(directory / f'{subset}.npz')
    # The diagnostics use first-three-fold coefficients, never the four-fold
    # refit. d60 selects ridge and is explicitly not an untouched test set.
    validation_models = {s: RidgeUtility.fit(train_x, train_y, s, diagnostics[s]['selected_alpha'])
                         for s in FEATURE_SUBSETS}
    valid_path = args.data_dir / f'cds_{year}_d60.npz'
    data = load_prefix(valid_path, args.wait_inputs)
    hist_dir = args.result_dir / f'cds_{year}' / 'historical' / 'd60'
    base = Model.load(hist_dir / 'model.npz', data)
    context = context_for(data, base, hist_dir, 101,
                          data_path=valid_path, base_path=hist_dir / 'model.npz')
    write_json(directory / 'validation_mixtures.json', validation_mixtures(data, context, validation_models))
    finish(directory / 'selected.json', frozen, artifacts, year=year, diagnostics=diagnostics,
        n_train_labels=len(train_y), n_validation_labels=len(valid_y),
        refit_offsets=OFFSETS, validation_is_used_for_selection=True,
        deployment_models_refit_all_four_folds=True)
    print(f'Ridges {year}: {len(train_y)} training + {len(valid_y)} validation labels', flush=True)
    return ridges


def routing_arrays(rows):
    fields = ('item_id', 'source_mass', 'zero', 'correction_norm', 'effective_donors', 'concentration')
    return {field: np.asarray([row[field] for row in rows],
                             dtype=bool if field == 'zero' else np.int64 if field in ('item_id', 'effective_donors') else float)
            for field in fields}


def search(args, ridges):
    directory = args.result_dir / 'cds_2015_d30'
    selected_path = directory / 'selected.json'
    path = args.data_dir / 'cds_2015_d30.npz'
    data = load_prefix(path, args.wait_inputs)
    base_path = args.target_dir / path.stem / 'activity_mean' / 'selected_model.npz'
    # selected.json is published after the selected checkpoint is completely copied.
    wait_for(base_path.parent / 'selected.json', args.wait_inputs)
    model = Model.load(wait_for(base_path, args.wait_inputs), data)
    context = context_for(data, model, directory, 101, data_path=path, base_path=base_path)
    configurations = {'zero': dict(policy='zero', radius=0., temperature=1.)}
    for policy in POLICIES[1:]:
        for index, (radius, temperature) in enumerate(itertools.product(RADII, TEMPERATURES)):
            configurations[f'{policy}_{index:02d}'] = dict(policy=policy, radius=radius, temperature=temperature)
    frozen = provenance(path, base_model_sha256=sha256(base_path),
        ridge_hashes={s: sha256(args.result_dir / 'cds_2015' / 'ridges' / f'{s}.npz') for s in FEATURE_SUBSETS},
        source_activity_sha256=sha256(directory / 'source_activity.npz'),
        configurations=configurations, selection_metric='tail80 macro item NDCG@20',
        nominal_trials_per_nonzero_policy=12, zero_baseline_evaluations=1)
    freeze(directory / 'search.frozen.json', frozen)
    artifacts = {f'{name}_{suffix}': directory / 'search' / f'{name}.{suffix}'
                 for name in configurations for suffix in ('json', 'npz')}
    existing = completed(selected_path, frozen, artifacts)
    if existing:
        return existing['policies']
    scorer = MultiPolicyScorer(context, ridges, configurations)
    started = time.monotonic()
    results = evaluate_many(data, 1, scorer, configurations, save_topk=False)
    records = []
    for name, config in configurations.items():
        summary = stratified_summaries(data, results[name])
        record = {'name': name, 'configuration': config, 'summary': {'1': summary},
                  'routing': scorer.summary(name), 'frozen': frozen}
        save_npz(directory / 'search' / f'{name}.npz', **results[name])
        write_json(directory / 'search' / f'{name}.json', record)
        records.append(record)
    selected = {'zero': configurations['zero']}
    for policy in POLICIES[1:]:
        candidates = [r for r in records if r['configuration']['policy'] == policy]
        def value(record):
            score = record['summary']['1']['tail80']['ndcg20']
            return score if score is not None and np.isfinite(score) else -np.inf
        # candidates preserve increasing radius/temperature trial order.
        if not any(np.isfinite(value(candidate)) for candidate in candidates):
            raise RuntimeError(f'No defined tail80 NDCG@20 for utility HPO prefix {path}')
        selected[policy] = max(candidates, key=value)['configuration']
    finish(selected_path, frozen, artifacts, policies=selected,
        elapsed_seconds=time.monotonic() - started,
        tie_break='first trial in radius then temperature order', selection_prefix=path.stem,
        trials=[{'name': r['name'], 'configuration': r['configuration'],
                 'score': r['summary']['1']['tail80']['ndcg20']} for r in records])
    print(f'Utility deployment selected: {selected}', flush=True)
    return selected


def final_configurations(selected):
    configurations = {f'main_{p}': selected[p] for p in POLICIES}
    for mechanism in ('mass1', 'norm01'):
        for policy in POLICIES[1:]:
            configurations[f'{mechanism}_{policy}'] = dict(policy=policy, radius=.1,
                                                          temperature=1., mechanism=mechanism)
    return configurations


def final_evaluation(args, year, seed, ridges, selected):
    path = args.data_dir / f'cds_{year}_d0.npz'
    directory = args.result_dir / path.stem / f'seed_{seed}'
    configurations = final_configurations(selected)
    data = load_prefix(path, args.wait_inputs)
    base_dir = args.target_dir / path.stem / 'activity_mean' / f'seed_{seed}'
    wait_for(base_dir / 'result.json', args.wait_inputs)
    base_path = wait_for(base_dir / 'model.npz', args.wait_inputs)
    model = Model.load(base_path, data)
    theta, beta, alpha = model.theta.copy(), model.beta.copy(), model.alpha
    context = context_for(data, model, directory, seed, data_path=path, base_path=base_path)
    frozen = provenance(path, base_model_sha256=sha256(base_path), seed=seed,
        ridge_hashes={s: sha256(args.result_dir / f'cds_{year}' / 'ridges' / f'{s}.npz') for s in FEATURE_SUBSETS},
        selected_policy_sha256=sha256(args.result_dir / 'cds_2015_d30' / 'selected.json'),
        source_activity_sha256=sha256(directory / 'source_activity.npz'),
        configurations=configurations, all_target_parameters_frozen=True,
        mechanism_note='separate mass1 and norm0.1 controls; native exact nulls retained; fixed R=.1,T=1')
    freeze(directory / 'evaluation.frozen.json', frozen)
    artifacts = {f'{name}_{filename}': directory / name / filename
                 for name in configurations for filename in ('result.json', 'test.npz', 'routing.npz')}
    if completed(directory / 'complete.json', frozen, artifacts):
        return
    scorer = MultiPolicyScorer(context, ridges, configurations)
    started = time.monotonic()
    results = evaluate_many(data, 1, scorer, configurations)
    np.testing.assert_array_equal(model.theta, theta)
    np.testing.assert_array_equal(model.beta, beta)
    assert model.alpha == alpha
    # Persist a paired, explicit exact-null check against the canonical scorer.
    zero_result = results['main_zero']
    base_test_path = base_dir / 'test.npz'
    with np.load(base_test_path, allow_pickle=False) as baseline:
        for key in ('ndcg10', 'ndcg20', 'ndcg50', 'recall20', 'top_users', 'evaluated'):
            np.testing.assert_array_equal(zero_result[key], baseline[key])
    for name, config in configurations.items():
        output = directory / name
        save_npz(output / 'test.npz', **results[name])
        save_npz(output / 'routing.npz', **routing_arrays(scorer.diagnostics[name]))
        write_json(output / 'result.json', {'method': name, 'year': year, 'seed': seed,
            'configuration': config, 'summary': {'1': stratified_summaries(data, results[name])},
            'routing': scorer.summary(name), 'frozen': frozen,
            'exact_zero_matches_canonical_target': True,
            'shared_19_policy_elapsed_seconds': time.monotonic() - started})
    finish(directory / 'complete.json', frozen, artifacts, policies=list(configurations),
           exact_zero_matches_canonical_target=True, elapsed_seconds=time.monotonic() - started)
    print(f'Utility {year}/{seed}: 7 main + 12 matched mechanism evaluations complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--target-dir', type=Path, default=ROOT / 'results/targets')
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'results/utility')
    parser.add_argument('--origins', nargs='+', type=int, default=[2015, 2016, 2017])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--wait-inputs', action='store_true')
    parser.add_argument('--historical-only', action='store_true')
    parser.add_argument('--hpo-only', action='store_true')
    args = parser.parse_args()
    # Finish historical work for every origin before waiting on target HPO.
    origins = list(dict.fromkeys([2015, *args.origins]))
    ridges = {year: fit_ridges(args, year) for year in origins}
    if args.historical_only:
        return
    selected = search(args, ridges[2015])
    if args.hpo_only:
        return
    for year in args.origins:
        for seed in args.seeds:
            final_evaluation(args, year, seed, ridges[year], selected)


if __name__ == '__main__':
    main()
