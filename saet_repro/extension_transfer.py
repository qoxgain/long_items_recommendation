"""C1/C2: frozen strong-target transfer on later music time blocks.

This module leaves the original experiment runners unchanged. Every reusable
unit binds its data, code, configuration, dependencies and output bytes. Query
opening is an explicit, irreversible audit event after selection preflight.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .data import DAY, load_data
from .evaluation import evaluate, evaluate_many, stratified_summaries
from .io import json_safe, sha256, write_json
from .target import Config, Model, train_target
from .utility import (FEATURE_NAMES, FEATURE_SUBSETS, RADII, TEMPERATURES, RIDGE_ALPHAS,
                      RidgeUtility, RouteContext, MultiPolicyScorer, forward_labels,
                      select_ridges)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'extension-transfer-v1'
BASE_CONFIGS = {
    'activity_mean': Config(method='activity_mean', lr=.01, rho=.0001, tau=.1),
    'activity_both': Config(method='activity_both', lr=.001, rho=0., tau=.1),
}
POLICIES = ('zero', 'uniform', 'semantic', 'full')
ORIGINS = (2018, 2019, 2020)
FINAL_SEEDS = (11, 22, 33, 44, 55)
OFFSETS = (270, 180, 90, 60)
EPOCHS = 6
SELECTION_SEED = 101


def object_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def hpo_configurations():
    configs = {'zero': dict(policy='zero', radius=0., temperature=1.)}
    for policy in POLICIES[1:]:
        for index, (radius, temperature) in enumerate(itertools.product(RADII, TEMPERATURES)):
            configs[f'{policy}_{index:02d}'] = dict(policy=policy, radius=radius, temperature=temperature)
    return configs


def choose_policies(records):
    """Tie resolution follows the frozen radius-major, temperature-minor grid."""
    expected = hpo_configurations()
    if [r['name'] for r in records] != list(expected):
        raise ValueError('Selection requires all 37 trials in frozen grid order')
    if any(r['configuration'] != expected[r['name']] for r in records):
        raise ValueError('Selection configuration differs from frozen grid')
    selected = {'zero': expected['zero']}
    for policy in POLICIES[1:]:
        candidates = [r for r in records if r['configuration']['policy'] == policy
                      and r['score'] is not None and np.isfinite(r['score'])]
        if not candidates:
            raise ValueError(f'No defined historical selection endpoint for {policy}')
        selected[policy] = max(candidates, key=lambda r: r['score'])['configuration']
    return selected


def routing_arrays(rows):
    fields = ('item_id', 'source_mass', 'zero', 'correction_norm', 'effective_donors', 'concentration')
    return {field: np.asarray([r[field] for r in rows], dtype=bool if field == 'zero'
                             else np.int64 if field in ('item_id', 'effective_donors') else float)
            for field in fields}


def runtime():
    return {'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
            'cpu': platform.processor() or platform.machine(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'threads': {key: os.environ.get(key) for key in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'feature_bank_cache': 'memory only; SAET_FEATURE_CACHE disabled during training'}


class CheckedScorer:
    """Check every scored null row and expose canonical rankings in the same pass."""
    def __init__(self, context, ridges, configurations, canonical=True):
        self.scorer = MultiPolicyScorer(context, ridges, configurations)
        self.context, self.configurations, self.canonical = context, configurations, canonical
        self.names = tuple(configurations) + (('canonical',) if canonical else ())
        self.digests = {name: hashlib.sha256() for name in configurations}
        self.scored_items = 0

    def __call__(self, ids):
        values = self.scorer(ids)
        for index, name in enumerate(self.configurations):
            self.digests[name].update(np.asarray(ids, dtype=np.int64).tobytes())
            self.digests[name].update(values[index].tobytes())
        self.scored_items += len(ids)
        if self.canonical:
            canonical = self.context.base.score_all(ids)
            zero = list(self.configurations).index('zero')
            np.testing.assert_array_equal(values[zero], canonical)
            return np.concatenate((values, canonical[None, ...]), axis=0)
        return values

    def hashes(self):
        return {name: digest.hexdigest() for name, digest in self.digests.items()}


class Runner:
    def __init__(self, data_dir, result_dir, protocol_path, wait_data=False):
        self.data_dir, self.result_dir = Path(data_dir), Path(result_dir)
        self.protocol_path = Path(protocol_path)
        self.wait_data = wait_data
        self.gate = self.result_dir / 'final_query_opened.json'
        self._frozen = None
        self._final_data = None

    def freeze(self):
        protocol = json.loads(self.protocol_path.read_text())
        section = protocol.get('stage_c_music')
        if not isinstance(section, dict):
            raise ValueError('Protocol requires a stage_c_music section')
        required = {'origins': list(ORIGINS), 'offset_days': [*OFFSETS, 30, 0],
                    'policies': list(POLICIES), 'radii': list(RADII),
                    'temperatures': list(TEMPERATURES), 'ridge_alphas': list(RIDGE_ALPHAS),
                    'final_policy_evaluations': 120, 'deterministic_activity_evaluations': 3,
                    'historical_target_recipe_matches_deployment': True,
                    'time_zone': 'UTC', 'origin_month_day': '01-01',
                    'fit_days': 365, 'query_days': 30}
        for key, value in required.items():
            if section.get(key) != value:
                raise ValueError(f'Protocol stage_c_music.{key} differs from executable matrix')
        for base, config in BASE_CONFIGS.items():
            entry = section.get('base_selection', {}).get(base, {})
            if entry.get('config') != asdict(config) or entry.get('epochs') != EPOCHS:
                raise ValueError(f'Protocol deployment configuration mismatch for {base}')
            if entry.get('selection_prefix') != 'cds_2015_d30':
                raise ValueError('C target configuration must use the old 2015 selection')
            old_selection = ROOT / 'results/targets/cds_2015_d30' / base / 'selected.json'
            if (sha256(old_selection) != entry.get('selection_sha256') or
                    json.loads(old_selection.read_text())['config'] != asdict(config)):
                raise ValueError(f'Old target selection checksum/configuration mismatch for {base}')
        if protocol.get('seeds') != list(FINAL_SEEDS):
            raise ValueError('Protocol final seeds differ from the executable matrix')
        code_files = ('saet_repro/extension_transfer.py', 'scripts/run_extension_transfer.py',
                      'saet_repro/target.py', 'saet_repro/utility.py', 'saet_repro/data.py',
                      'saet_repro/evaluation.py', 'saet_repro/io.py')
        for filename in code_files[2:]:
            if sha256(ROOT / filename) != protocol.get('baseline_sha256', {}).get(filename):
                raise ValueError(f'Frozen baseline implementation changed: {filename}')
        dependencies = {'python': platform.python_version()}
        for package in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib'):
            dependencies[package] = importlib.metadata.version(package)
        frozen = json_safe({'schema': SCHEMA, 'protocol_sha256': sha256(self.protocol_path),
            'protocol': protocol, 'code_sha256': {p: sha256(ROOT / p) for p in code_files},
            'dependencies': dependencies, 'base_configs': {k: asdict(v) for k, v in BASE_CONFIGS.items()},
            'epochs': EPOCHS, 'origins': ORIGINS, 'seeds': FINAL_SEEDS, 'selection_seed': SELECTION_SEED,
            'historical_offsets': OFFSETS, 'hpo_prefix': 'cds_2018_d30',
            'hpo_configurations': hpo_configurations(), 'radius_temperature_selection': 'first origin only; freeze for later origins',
            'selection_endpoint': 'tail80 item NDCG@20', 'final_policy_records': 120,
            'deterministic_activity_records': 3,
            'independence_status': 'New years were not used in available workspace evidence; no external independence claim',
            'historical_and_deployment_configuration_match': True,
            'source_target_user_spaces': 'separate; no cross-domain identity alignment',
            'data_dir': str(self.data_dir.resolve())})
        path = self.result_dir / 'protocol.frozen.json'
        if path.exists() and json.loads(path.read_text()) != frozen and self.gate.exists():
            raise RuntimeError('Protocol/code changed after final queries opened; use a new protocol version')
        write_json(path, frozen)
        write_json(self.result_dir / 'protocol.sha256.json', {'sha256': sha256(path),
                   'input_protocol_sha256': frozen['protocol_sha256']})
        self._frozen = frozen
        return frozen

    @property
    def identity(self):
        if self._frozen is None:
            raise RuntimeError('Freeze protocol before any data access')
        if sha256(self.protocol_path) != self._frozen['protocol_sha256']:
            raise RuntimeError('Protocol changed during run')
        if any(sha256(ROOT / p) != digest for p, digest in self._frozen['code_sha256'].items()):
            raise RuntimeError('Code changed during run')
        return {'schema': SCHEMA, 'protocol_sha256': self._frozen['protocol_sha256'],
                'frozen_protocol_sha256': object_hash(self._frozen),
                'code_sha256': self._frozen['code_sha256'], 'dependencies': self._frozen['dependencies']}

    def path(self, year, offset):
        if year not in ORIGINS or offset not in (*OFFSETS, 30, 0):
            raise ValueError('Prefix is outside the frozen C matrix')
        return self.data_dir / f'cds_{year}_d{offset}.npz'

    def wait_for(self, path):
        path = Path(path)
        while not path.exists():
            if not self.wait_data:
                raise FileNotFoundError(path)
            print(f'Waiting for {path}', flush=True)
            time.sleep(15)
        return path

    def load_prefix(self, year, offset):
        _ = self.identity
        if offset == 0:
            if not self.gate.exists():
                raise RuntimeError('Final queries require successful selection preflight and opening gate')
            record = json.loads(self.gate.read_text())
            if record['identity'] != self.identity:
                raise RuntimeError('Query gate identity mismatch')
            self.open_final_queries()
        path = self.wait_for(self.path(year, offset))
        manifest = self.prefix_manifest(year, offset)
        if sha256(path) != manifest['data_sha256']:
            raise RuntimeError(f'Prefix data differs from manifest: {path}')
        data = load_data(path)
        self.validate_data(data, year, offset)
        if offset == 0:
            self._final_data = (year, data)
        return data

    def prefix_manifest(self, year, offset):
        path = self.path(year, offset)
        manifest = json.loads(self.wait_for(path.with_suffix('.json')).read_text())
        end = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000) - offset * DAY
        expected = {'prefix': path.stem, 'origin_year': year, 'offset_days': offset,
                    'fit_start': end - 365 * DAY, 'fit_end': end, 'query_end': end + 30 * DAY,
                    'code_sha256': self._frozen['code_sha256']['saet_repro/data.py']}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f'Prefix temporal/build metadata mismatch: {path}')
        return manifest

    def validate_data(self, data, year, offset):
        """Audit time windows and each domain's local event indices independently."""
        manifest = self.prefix_manifest(year, offset)
        start, end, query_end = (manifest[k] for k in ('fit_start', 'fit_end', 'query_end'))
        for key in ('fit_start', 'fit_end', 'query_end'):
            if int(data[key]) != manifest[key]:
                raise RuntimeError(f'Prefix archive temporal mismatch: {key}')
        np.testing.assert_array_equal(data['cutoffs'], [end + d * DAY for d in (0, 30, 60, 90)])
        for domain in ('target', 'source'):
            def arr(name):
                return np.asarray(data[f'{domain}_{name}'])
            nusers, nitems = len(arr('user_ids')), len(arr('item_ids'))
            if len(np.unique(arr('user_ids'))) != nusers or len(np.unique(arr('item_ids'))) != nitems:
                raise RuntimeError(f'Duplicate local IDs: {domain}')
            users, items, times, ratings = (arr(k) for k in
                                           ('event_users', 'event_items', 'event_times', 'event_ratings'))
            if (np.any(users < 0) or np.any(users >= nusers) or np.any(items < 0) or
                    np.any(items >= nitems) or np.any(times >= query_end)):
                raise RuntimeError(f'Event bounds invalid: {domain}')
            stages = np.where(times < start, -1, np.where(times < end, 0, 1))
            expected = np.column_stack((users, items, stages))
            np.testing.assert_array_equal(arr('observed'), expected)
            np.testing.assert_array_equal(arr('positive'), expected[ratings >= 4])
            np.testing.assert_array_equal(arr('train'), expected[(ratings >= 4) & (stages == 0), :2])
            if np.any(arr('all_rating_times') < start) or np.any(arr('all_rating_times') >= end):
                raise RuntimeError(f'Activity history crosses fit window: {domain}')
            if (np.any(arr('all_rating_users') < 0) or np.any(arr('all_rating_users') >= nusers) or
                    np.any(arr('all_rating_items') < -1) or np.any(arr('all_rating_items') >= nitems)):
                raise RuntimeError(f'Activity local ID bounds invalid: {domain}')

    def final_data(self, year, data=None):
        if not self.gate.exists():
            raise RuntimeError('Final query gate is closed')
        if json.loads(self.gate.read_text())['identity'] != self.identity:
            raise RuntimeError('Query gate identity mismatch')
        if data is None:
            return self.load_prefix(year, 0)
        if self._final_data is None or self._final_data[0] != year or self._final_data[1] is not data:
            raise RuntimeError('Final data must come from the checked final-prefix loader')
        return data

    def provenance(self, path, **extra):
        return json_safe({**self.identity, 'data_path': str(path.resolve()),
                          'data_sha256': sha256(path),
                          'data_manifest_sha256': sha256(path.with_suffix('.json')), **extra})

    def cached(self, marker, frozen, artifacts, selection=True, require=False):
        """Quarantine stale bytes and recompute before opening; never reselect after opening."""
        marker = Path(marker)
        reason = 'missing completion marker'
        record = None
        if marker.exists():
            try:
                record = json.loads(marker.read_text())
                if record.get('frozen') != json_safe(frozen):
                    reason = 'identity mismatch'
                elif set(record.get('artifacts', {})) != set(artifacts):
                    reason = 'artifact set mismatch'
                elif all(Path(path).exists() and sha256(path) == record['artifacts'][name]
                         for name, path in artifacts.items()):
                    return record
                else:
                    reason = 'artifact missing or checksum mismatch'
            except (OSError, ValueError, KeyError):
                reason = 'invalid completion marker'
        if require or (selection and self.gate.exists()):
            raise RuntimeError(f'Selection dependency invalid: {marker}: {reason}')
        if marker.exists():
            stale = marker.parent / 'stale' / f'{marker.stem}_{time.time_ns()}'
            stale.mkdir(parents=True)
            for path in dict.fromkeys([marker, *map(Path, artifacts.values())]):
                if path.exists():
                    destination = stale / path.relative_to(marker.parent)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, destination)
            write_json(stale / 'reason.json', {'reason': reason, 'expected_identity': frozen})
        return None

    def finish(self, marker, frozen, artifacts, **extra):
        write_json(marker, {'frozen': frozen, 'artifacts': {k: sha256(v) for k, v in artifacts.items()},
                            **extra})

    def training_identity(self, path, base, seed, selection=True):
        return self.provenance(path, base=base, config=asdict(BASE_CONFIGS[base]), seed=seed, epochs=EPOCHS,
                               role='historical_or_selection' if selection else 'final')

    def source_identity(self, path, directory, base, seed):
        return self.provenance(path, config=asdict(BASE_CONFIGS[base]), seed=seed,
            base_model_sha256=sha256(directory / 'model.npz'), source_epochs=6,
            source_positive_cap=16, target_support_cap=5, negative_cap=5,
            source_beta_initial='zero', source_beta_optimizer='SGD at frozen base learning rate',
            theta_and_alpha_frozen=True, separate_user_id_spaces=True)

    def label_identity(self, path, directory, base, offset):
        return self.provenance(path, base=base, config=asdict(BASE_CONFIGS[base]),
            seed=SELECTION_SEED, epochs=EPOCHS, label_radius=.1, label_eta=.1,
            role='ridge_validation' if offset == 60 else 'ridge_training',
            base_model_sha256=sha256(directory / 'model.npz'),
            source_activity_sha256=sha256(directory / 'source_activity.npz'))

    def fit_base(self, data, path, directory, base, seed, selection=True):
        directory.mkdir(parents=True, exist_ok=True)
        config = BASE_CONFIGS[base]
        frozen = self.training_identity(path, base, seed, selection)
        artifacts = {'model': directory / 'model.npz', 'history': directory / 'training.json'}
        if self.cached(directory / 'training.complete.json', frozen, artifacts, selection=selection):
            model = Model.load(artifacts['model'], data)
            if model.config != config:
                raise RuntimeError('Saved target configuration differs from canonical deployment')
            return model
        write_json(directory / 'training.frozen.json', frozen)
        old_cache = os.environ.pop('SAET_FEATURE_CACHE', None)
        started = time.monotonic()
        try:
            def progress(epoch, _model):
                print(f'C {path.stem}/{base}/{seed}: epoch {epoch}/{EPOCHS}', flush=True)
            model, history = train_target(data, config, seed, epochs=EPOCHS, validation_callback=progress)
        finally:
            if old_cache is not None:
                os.environ['SAET_FEATURE_CACHE'] = old_cache
        model.save(artifacts['model'])
        write_json(artifacts['history'], history)
        self.finish(directory / 'training.complete.json', frozen, artifacts,
                    elapsed_seconds=time.monotonic() - started, runtime=runtime())
        return model

    def context_for(self, data, path, directory, model, seed, selection=True):
        if model.config != BASE_CONFIGS.get(model.config.method):
            raise RuntimeError('Source context requires the canonical target configuration')
        frozen = self.source_identity(path, directory, model.config.method, seed)
        artifact = directory / 'source_activity.npz'
        marker = directory / 'source_activity.json'
        if self.cached(marker, frozen, {'source_activity': artifact}, selection=selection):
            with np.load(artifact, allow_pickle=False) as saved:
                return RouteContext(data, model, seed=seed, source_beta=saved['beta'])
        context = RouteContext(data, model, seed=seed)
        save_npz(artifact, beta=context.source.beta)
        self.finish(marker, frozen, {'source_activity': artifact}, **context.source_summary(), runtime=runtime())
        return context

    def historical_directory(self, base, year, offset):
        return self.result_dir / base / 'historical' / f'cds_{year}' / f'd{offset}'

    def historical_fold(self, base, year, offset):
        if offset not in OFFSETS:
            raise ValueError('Historical folds cannot load deployment or final queries')
        data, path = self.load_prefix(year, offset), self.path(year, offset)
        directory = self.historical_directory(base, year, offset)
        started = time.monotonic()
        model = self.fit_base(data, path, directory, base, SELECTION_SEED)
        context = self.context_for(data, path, directory, model, SELECTION_SEED)
        frozen = self.label_identity(path, directory, base, offset)
        artifact = directory / 'labels.npz'
        if self.cached(directory / 'result.json', frozen, {'labels': artifact}):
            with np.load(artifact, allow_pickle=False) as saved:
                return {k: saved[k] for k in saved.files}
        write_json(directory / 'labels.frozen.json', frozen)
        labels = forward_labels(data, context, radius=.1)
        save_npz(artifact, **labels)
        self.finish(directory / 'result.json', frozen, {'labels': artifact},
                    n_labels=len(labels['labels']), n_query_items=len(labels['query_item_summary']),
                    label_positive_fraction=float(np.mean(labels['labels'] > 0)) if len(labels['labels']) else None,
                    elapsed_seconds=time.monotonic() - started, runtime=runtime())
        print(f'C historical {base}/{year}/d{offset}: {len(labels["labels"])} labels', flush=True)
        return labels

    def ridge_directory(self, base, year):
        return self.result_dir / base / 'historical' / f'cds_{year}' / 'ridges'

    def ridge_identity(self, base, year):
        return json_safe({**self.identity, 'base': base, 'config': asdict(BASE_CONFIGS[base]),
            'year': year, 'feature_names': FEATURE_NAMES, 'feature_subsets': FEATURE_SUBSETS,
            'alphas': RIDGE_ALPHAS, 'train_offsets': OFFSETS[:3], 'validation_offset': 60,
            'label_scaling': 'positive training RMS, no mean subtraction',
            'fold_hashes': {str(d): sha256(self.historical_directory(base, year, d) / 'labels.npz') for d in OFFSETS}})

    def fit_ridges(self, base, year):
        labels = {offset: self.historical_fold(base, year, offset) for offset in OFFSETS}
        directory = self.ridge_directory(base, year)
        frozen = self.ridge_identity(base, year)
        artifacts = {name: directory / f'{name}.npz' for name in FEATURE_SUBSETS}
        if self.cached(directory / 'selected.json', frozen, artifacts):
            return {name: RidgeUtility.load(path) for name, path in artifacts.items()}
        tx = np.concatenate([labels[d]['features'] for d in OFFSETS[:3]])
        ty = np.concatenate([labels[d]['labels'] for d in OFFSETS[:3]])
        vx, vy = labels[60]['features'], labels[60]['labels']
        if not len(ty) or not len(vy):
            raise ValueError('Historical ridge training and validation require observed labels')
        write_json(directory / 'frozen.json', frozen)
        ridges, diagnostics = select_ridges(tx, ty, vx, vy)
        for name, model in ridges.items():
            model.save(artifacts[name])
        self.finish(directory / 'selected.json', frozen, artifacts, base=base, year=year,
                    diagnostics=diagnostics, n_train_labels=len(ty), n_validation_labels=len(vy),
                    refit_offsets=OFFSETS, deployment_models_refit_all_four_folds=True)
        return ridges

    def hpo_directory(self, base):
        return self.result_dir / base / 'hpo' / 'cds_2018_d30'

    def search_identity(self, base):
        path = self.path(2018, 30)
        directory = self.hpo_directory(base)
        return self.provenance(path, base=base, config=asdict(BASE_CONFIGS[base]), seed=SELECTION_SEED,
            base_model_sha256=sha256(directory / 'model.npz'),
            source_activity_sha256=sha256(directory / 'source_activity.npz'),
            ridge_hashes={name: sha256(self.ridge_directory(base, 2018) / f'{name}.npz') for name in FEATURE_SUBSETS},
            configurations=hpo_configurations(), selection_metric='tail80 item NDCG@20',
            selection_rule='first origin once; subsequent origins reuse selected policies',
            tie_break='first trial in radius then temperature order')

    def search_artifacts(self, base):
        directory = self.hpo_directory(base)
        return {f'{name}_{ext}': directory / 'search' / f'{name}.{ext}'
                for name in hpo_configurations() for ext in ('json', 'npz', 'routing.npz')}

    def verify_ridges(self, base, year, ridges):
        canonical = {s: RidgeUtility.load(self.ridge_directory(base, year) / f'{s}.npz')
                     for s in FEATURE_SUBSETS}
        if set(ridges) != set(canonical) or any(
                object_hash(asdict(ridges[s])) != object_hash(asdict(canonical[s])) for s in canonical):
            raise RuntimeError('Deployment ridges differ from the verified selection artifacts')

    def search(self, base, ridges):
        self.verify_ridges(base, 2018, ridges)
        data, path = self.load_prefix(2018, 30), self.path(2018, 30)
        directory = self.hpo_directory(base)
        model = self.fit_base(data, path, directory, base, SELECTION_SEED)
        context = self.context_for(data, path, directory, model, SELECTION_SEED)
        frozen, artifacts = self.search_identity(base), self.search_artifacts(base)
        marker = directory / 'selected.json'
        previous = self.cached(marker, frozen, artifacts)
        if previous:
            return previous['policies']
        write_json(directory / 'search.frozen.json', frozen)
        configs = hpo_configurations()
        scorer = CheckedScorer(context, ridges, configs, canonical=False)
        started = time.monotonic()
        results = evaluate_many(data, 1, scorer, configs, batch_size=1)
        hashes = scorer.hashes()
        records = []
        for name, config in configs.items():
            summary = stratified_summaries(data, results[name])
            record = {'name': name, 'configuration': config, 'summary': {'1': summary},
                      'score': summary['tail80']['ndcg20'], 'score_sha256': hashes[name],
                      'routing': scorer.scorer.summary(name), 'frozen': frozen}
            save_npz(directory / 'search' / f'{name}.npz', **results[name])
            save_npz(directory / 'search' / f'{name}.routing.npz', **routing_arrays(scorer.scorer.diagnostics[name]))
            write_json(directory / 'search' / f'{name}.json', record)
            records.append(record)
        selected = choose_policies(records)
        self.finish(marker, frozen, artifacts, policies=selected, trials=records,
                    nominal_configurations=len(configs), distinct_score_configurations=len(set(hashes.values())),
                    per_policy_distinct_scores={p: len({hashes[n] for n, c in configs.items() if c['policy'] == p}) for p in POLICIES},
                    elapsed_seconds=time.monotonic() - started, runtime=runtime())
        print(f'C selected {base}: {selected}', flush=True)
        return selected

    def _verify_marker(self, marker, expected_identity=None):
        marker = Path(marker)
        if not marker.exists():
            raise RuntimeError(f'Selection dependency missing: {marker}')
        record = json.loads(marker.read_text())
        frozen = record['frozen']
        if any(frozen.get(k) != v for k, v in self.identity.items()):
            raise RuntimeError(f'Stale selection identity: {marker}')
        if expected_identity is not None and frozen != json_safe(expected_identity):
            raise RuntimeError(f'Selection dependency mismatch: {marker}')
        if 'data_path' in frozen and sha256(frozen['data_path']) != frozen['data_sha256']:
            raise RuntimeError(f'Selection data changed: {marker}')
        if 'data_path' in frozen and sha256(Path(frozen['data_path']).with_suffix('.json')) != frozen['data_manifest_sha256']:
            raise RuntimeError(f'Selection data manifest changed: {marker}')
        return record

    def preflight_selections(self):
        """Read only historical data bytes/manifests, never a d0 archive."""
        selected, ridges, dependencies = {}, {}, {}
        # Fail before loading any dataset if selection has not even started.
        for base in BASE_CONFIGS:
            for year in ORIGINS:
                for offset in OFFSETS:
                    marker = self.historical_directory(base, year, offset) / 'training.complete.json'
                    if not marker.exists():
                        raise RuntimeError(f'Selection dependency missing: {marker}')
        for year in ORIGINS:
            for offset in OFFSETS:
                self.load_prefix(year, offset)
        self.load_prefix(2018, 30)
        for base in BASE_CONFIGS:
            for year in ORIGINS:
                for offset in OFFSETS:
                    directory = self.historical_directory(base, year, offset)
                    path = self.path(year, offset)
                    for marker_name, names in (
                        ('training.complete.json', {'model': 'model.npz', 'history': 'training.json'}),
                        ('source_activity.json', {'source_activity': 'source_activity.npz'}),
                        ('result.json', {'labels': 'labels.npz'})):
                        marker = directory / marker_name
                        record = self._verify_marker(marker)
                        expected = (self.training_identity(path, base, SELECTION_SEED)
                                    if marker_name == 'training.complete.json' else
                                    self.source_identity(path, directory, base, SELECTION_SEED)
                                    if marker_name == 'source_activity.json' else
                                    self.label_identity(path, directory, base, offset))
                        self._verify_marker(marker, expected)
                        self.cached(marker, expected, {k: directory / v for k, v in names.items()}, require=True)
                        dependencies[str(marker.relative_to(self.result_dir))] = sha256(marker)
                    if json.loads((directory / 'result.json').read_text())['frozen']['base_model_sha256'] != sha256(directory / 'model.npz'):
                        raise RuntimeError('Historical labels do not bind current target model')
                    if json.loads((directory / 'result.json').read_text())['frozen']['source_activity_sha256'] != sha256(directory / 'source_activity.npz'):
                        raise RuntimeError('Historical labels do not bind current source activity')
                    if json.loads((directory / 'source_activity.json').read_text())['frozen']['base_model_sha256'] != sha256(directory / 'model.npz'):
                        raise RuntimeError('Historical source fit does not bind current target model')
                rd = self.ridge_directory(base, year)
                marker = rd / 'selected.json'
                frozen = self.ridge_identity(base, year)
                self._verify_marker(marker, frozen)
                self.cached(marker, frozen, {s: rd / f'{s}.npz' for s in FEATURE_SUBSETS}, require=True)
                dependencies[str(marker.relative_to(self.result_dir))] = sha256(marker)
                ridges[base, year] = {s: RidgeUtility.load(rd / f'{s}.npz') for s in FEATURE_SUBSETS}
            hd = self.hpo_directory(base)
            for marker_name, names in (
                ('training.complete.json', {'model': 'model.npz', 'history': 'training.json'}),
                ('source_activity.json', {'source_activity': 'source_activity.npz'})):
                marker = hd / marker_name
                record = self._verify_marker(marker)
                expected = (self.training_identity(self.path(2018, 30), base, SELECTION_SEED)
                            if marker_name == 'training.complete.json' else
                            self.source_identity(self.path(2018, 30), hd, base, SELECTION_SEED))
                self._verify_marker(marker, expected)
                if marker_name == 'source_activity.json' and record['frozen']['base_model_sha256'] != sha256(hd / 'model.npz'):
                    raise RuntimeError('HPO source activity binds a different base model')
                self.cached(marker, expected, {k: hd / v for k, v in names.items()}, require=True)
                dependencies[str(marker.relative_to(self.result_dir))] = sha256(marker)
            marker = hd / 'selected.json'
            frozen = self.search_identity(base)
            self._verify_marker(marker, frozen)
            record = self.cached(marker, frozen, self.search_artifacts(base), require=True)
            if any(trial != json.loads((hd / 'search' / f'{trial["name"]}.json').read_text())
                   for trial in record['trials']):
                raise RuntimeError('Selection trial records differ from verified trial artifacts')
            if record['policies'] != choose_policies(record['trials']):
                raise RuntimeError('Selected policies differ from frozen trial tie rule')
            selected[base] = record['policies']
            dependencies[str(marker.relative_to(self.result_dir))] = sha256(marker)
        return ridges, selected, dependencies

    def open_final_queries(self):
        ridges, selected, dependencies = self.preflight_selections()
        expected = {'identity': self.identity, 'protocol_sha256': self.identity['protocol_sha256'],
                    'selection_dependencies': dependencies, 'selected_policies': selected,
                    'origins': list(ORIGINS), 'seeds': list(FINAL_SEEDS)}
        if self.gate.exists():
            previous = json.loads(self.gate.read_text())
            if any(previous.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Selection dependencies changed after final query opening')
        else:
            write_json(self.gate, {**expected, 'opened_at_utc': datetime.now(timezone.utc).isoformat(),
                                  'event': 'All selection dependencies verified; immediately before first d0 load'})
        return ridges, selected

    def final_evaluation(self, base, year, seed, ridges, selected, data=None):
        if seed not in FINAL_SEEDS or set(selected) != set(POLICIES):
            raise ValueError('Final cell is outside the frozen matrix')
        if not self.gate.exists():
            raise RuntimeError('Final query gate is closed')
        gate = json.loads(self.gate.read_text())
        if gate['identity'] != self.identity:
            raise RuntimeError('Query gate identity mismatch')
        if selected != gate['selected_policies'][base]:
            raise RuntimeError('Deployment policies must match opening gate')
        data = self.final_data(year, data)
        self.verify_ridges(base, year, ridges)
        path = self.path(year, 0)
        directory = self.result_dir / base / 'final' / path.stem / f'seed_{seed}'
        started = time.monotonic()
        model = self.fit_base(data, path, directory, base, seed, selection=False)
        context = self.context_for(data, path, directory, model, seed, selection=False)
        before = model.theta.copy(), model.beta.copy(), model.alpha
        model_hash = sha256(directory / 'model.npz')
        frozen = self.provenance(path, base=base, config=asdict(BASE_CONFIGS[base]), seed=seed, epochs=EPOCHS,
            base_model_sha256=model_hash, source_activity_sha256=sha256(directory / 'source_activity.npz'),
            selected_policy_sha256=sha256(self.hpo_directory(base) / 'selected.json'),
            ridge_hashes={s: sha256(self.ridge_directory(base, year) / f'{s}.npz') for s in FEATURE_SUBSETS},
            query_gate_sha256=sha256(self.gate), configurations=selected, all_target_parameters_frozen=True)
        artifacts = {f'{p}_{name}': directory / p / name for p in POLICIES
                     for name in ('result.json', 'test.npz', 'routing.npz')}
        artifacts['canonical_test'] = directory / 'canonical_test.npz'
        if self.cached(directory / 'complete.json', frozen, artifacts, selection=False):
            return
        write_json(directory / 'evaluation.frozen.json', frozen)
        scorer = CheckedScorer(context, ridges, selected)
        results = evaluate_many(data, 1, scorer, scorer.names, batch_size=1)
        for key in results['canonical']:
            np.testing.assert_array_equal(results['zero'][key], results['canonical'][key])
        np.testing.assert_array_equal(model.theta, before[0])
        np.testing.assert_array_equal(model.beta, before[1])
        assert model.alpha == before[2] and sha256(directory / 'model.npz') == model_hash
        save_npz(artifacts['canonical_test'], **results['canonical'])
        null_check = {'scores_exact': True, 'all_rank_and_metric_arrays_exact': True,
                      'scored_items': scorer.scored_items, 'base_parameters_unchanged': True,
                      'base_model_sha256_before': model_hash, 'base_model_sha256_after': sha256(directory / 'model.npz')}
        for policy in POLICIES:
            output = directory / policy
            save_npz(output / 'test.npz', **results[policy])
            save_npz(output / 'routing.npz', **routing_arrays(scorer.scorer.diagnostics[policy]))
            write_json(output / 'result.json', {'base': base, 'method': policy, 'year': year, 'seed': seed,
                'configuration': selected[policy], 'summary': {'1': stratified_summaries(data, results[policy])},
                'routing': scorer.scorer.summary(policy), 'frozen': frozen, 'null_check': null_check,
                'score_sha256': scorer.hashes()[policy], 'runtime': runtime()})
        self.finish(directory / 'complete.json', frozen, artifacts, policies=list(POLICIES),
                    null_check=null_check, elapsed_seconds=time.monotonic() - started, runtime=runtime())
        print(f'C final {base}/{year}/{seed}: four policies complete', flush=True)

    def activity_evaluation(self, year, data=None):
        if not self.gate.exists():
            raise RuntimeError('Final query gate is closed')
        data = self.final_data(year, data)
        path = self.path(year, 0)
        directory = self.result_dir / 'activity' / 'final' / path.stem / 'seed_0'
        frozen = self.provenance(path, method='activity', seed=0,
            formula='target_activity_raw[:, 2]: log1p 365-day all-rating count',
            query_gate_sha256=sha256(self.gate), deterministic_seed_broadcast=True)
        artifacts = {'test': directory / 'test.npz', 'result': directory / 'result.json'}
        if self.cached(directory / 'complete.json', frozen, artifacts, selection=False):
            return
        values = np.asarray(data['target_activity_raw'])[:, 2]
        result = evaluate(data, 1, lambda ids: np.broadcast_to(values, (len(ids), len(values))))
        save_npz(artifacts['test'], **result)
        write_json(artifacts['result'], {'method': 'activity', 'year': year, 'seed': 0,
                   'summary': {'1': stratified_summaries(data, result)}, 'frozen': frozen})
        self.finish(directory / 'complete.json', frozen, artifacts)
