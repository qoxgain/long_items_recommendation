"""Forward-label source utility learning and bounded preference-only transfer.

Historical labels are actual pair losses after a source step, not gradient dot
products.  A RouteContext contains only fit-prefix information.  Query labels
are handled separately by forward_labels and are never read when routing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time

import numpy as np


FEATURE_NAMES = (
    'intercept', 'text_cosine', 'profile_cosine', 'gradient_cosine',
    'log_target_support', 'log_source_support', 'target_dispersion',
    'source_dispersion', 'target_gradient_norm', 'source_gradient_norm',
)
FEATURE_SUBSETS = {
    'semantic': (0, 1),
    'global': (0, 5, 7, 9),
    'behavior': (0, 1, 2, 4, 5, 6, 7),
    'full': tuple(range(10)),
}
RIDGE_ALPHAS = (1e-4, 1e-3, .01, .1, 1., 10.)
POLICIES = ('zero', 'uniform', 'semantic', 'global', 'behavior', 'full',
            'predicted_single')
RADII = (0., .01, .05, .1)
TEMPERATURES = (.1, 1., 10.)


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros_like(vector)


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.


def clip_step(gradient, eta=.1, radius=.1):
    """Return -eta*g projected onto the closed Euclidean radius ball."""
    if not np.isfinite(radius) or not np.isfinite(eta) or radius < 0 or eta < 0:
        raise ValueError('radius and eta must be nonnegative')
    gradient = np.asarray(gradient, dtype=np.float64)
    if radius == 0:
        return np.zeros_like(gradient)
    result = -float(eta) * gradient
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    return result * np.minimum(1., radius / np.maximum(norm, np.finfo(float).tiny))


def project_nonnegative_l1(values, mass=1.):
    """Exact projection onto {w >= 0, sum(w) <= mass}, including exact null."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(mass) or mass < 0 or not np.all(np.isfinite(values)):
        raise ValueError('finite values and nonnegative mass required')
    if values.ndim != 1:
        raise ValueError('values must be a vector')
    positive = np.maximum(values, 0.)
    if mass == 0 or positive.size == 0 or not np.any(positive):
        return np.zeros_like(positive)
    if positive.sum() <= mass:
        return positive
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending)
    active = np.flatnonzero(descending > (cumulative - mass) / np.arange(1, len(values) + 1))
    index = active[-1]
    threshold = (cumulative[index] - mass) / (index + 1)
    return np.maximum(positive - threshold, 0.)


def utility_weights(predicted_scaled_gain, temperature=1.):
    # argmax w'y - temperature*||w||^2/2 on the nonnegative L1 ball.
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return project_nonnegative_l1(np.asarray(predicted_scaled_gain) / temperature)


def corrected_scores(base_scores, phi, correction):
    """Skip even addition for the zero branch, preserving exact base values."""
    if not np.any(correction):
        return base_scores
    return np.asarray(base_scores) + np.asarray(phi) @ np.asarray(correction)


def pair_loss_from_scores(positive, negative, tau):
    differences = (np.asarray(positive)[:, None] - np.asarray(negative)[None, :]) / tau
    return float(np.logaddexp(0., -differences).mean())


def batch_loss(model, batch, correction=None):
    pos = model.score_features(batch['pos_anchor'], batch['pos_phi'], batch['pos_activity'])
    neg = model.score_features(batch['neg_anchor'], batch['neg_phi'], batch['neg_activity'])
    if correction is not None and np.any(correction):
        pos = pos + batch['pos_phi'] @ correction
        neg = neg + batch['neg_phi'] @ correction
    return pair_loss_from_scores(pos, neg, model.config.tau)


@dataclass
class RidgeUtility:
    subset: str
    alpha: float
    coefficient: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    label_rms: float

    @classmethod
    def fit(cls, features, labels, subset, alpha):
        columns = FEATURE_SUBSETS[subset]
        x = np.asarray(features, dtype=np.float64)[:, columns]
        y = np.asarray(labels, dtype=np.float64)
        if not len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('ridge requires nonempty finite data')
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-8)
        # Keep the intercept literal. Labels are divided by positive RMS,
        # never centered; zero keeps its meaning of no actual loss improvement.
        mean[0], scale[0] = 0., 1.
        z = (x - mean) / scale
        rms = max(float(np.sqrt(np.mean(y * y))), 1e-12)
        regularizer = np.eye(z.shape[1]) * alpha
        regularizer[0, 0] = 0.
        coefficient = np.linalg.solve(z.T @ z + regularizer, z.T @ (y / rms))
        return cls(subset, float(alpha), coefficient, mean, scale, rms)

    def predict(self, features, scaled=True):
        x = np.asarray(features, dtype=np.float64)[:, FEATURE_SUBSETS[self.subset]]
        prediction = ((x - self.mean) / self.scale) @ self.coefficient
        return prediction if scaled else prediction * self.label_rms

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, subset=self.subset, alpha=self.alpha,
                            coefficient=self.coefficient, mean=self.mean,
                            scale=self.scale, label_rms=self.label_rms)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(str(data['subset']), float(data['alpha']), data['coefficient'],
                       data['mean'], data['scale'], float(data['label_rms']))


def prediction_diagnostics(model, features, labels):
    y = np.asarray(labels)
    pred = model.predict(features, scaled=False)
    positive = pred > 0
    return {'n': len(y), 'mse': float(np.mean((y - pred) ** 2)),
            'zero_mse': float(np.mean(y ** 2)),
            'scaled_mse': float(np.mean(((y - pred) / model.label_rms) ** 2)),
            'sign_accuracy': float(np.mean((pred > 0) == (y > 0))),
            'positive_precision': float(np.mean(y[positive] > 0)) if positive.any() else None,
            'predicted_positive_fraction': float(positive.mean()),
            'label_positive_fraction': float(np.mean(y > 0)),
            'label_rms': model.label_rms}


def select_ridges(train_features, train_labels, validation_features, validation_labels):
    models, diagnostics = {}, {}
    all_features = np.concatenate((train_features, validation_features))
    all_labels = np.concatenate((train_labels, validation_labels))
    for subset in FEATURE_SUBSETS:
        candidates = [RidgeUtility.fit(train_features, train_labels, subset, a) for a in RIDGE_ALPHAS]
        reports = [prediction_diagnostics(m, validation_features, validation_labels) for m in candidates]
        best_index = min(range(len(candidates)), key=lambda i: (reports[i]['mse'], i))
        selected = candidates[best_index]
        diagnostics[subset] = {'selected_alpha': selected.alpha, 'validation': reports[best_index],
                               'trials': [dict(alpha=a, **r) for a, r in zip(RIDGE_ALPHAS, reports)]}
        models[subset] = RidgeUtility.fit(all_features, all_labels, subset, selected.alpha)
    return models, diagnostics


class RouteContext:
    """Fit-only donor batches and features for one independently fitted base."""

    def __init__(self, data, base, seed=101, donors=16, source_cap=16,
                 target_cap=5, negatives=5, eta=.1, source_epochs=6, source_beta=None):
        from .target import FeatureStore, Model
        self.data, self.base, self.seed = data, base, int(seed)
        self.donors, self.source_cap = int(donors), int(source_cap)
        self.target_cap, self.negatives, self.eta = int(target_cap), int(negatives), float(eta)
        self.target_store = base.store
        self.source_store = FeatureStore(data, base.config, domain='source')
        self.source = Model(base.config, self.source_store, theta=base.theta.copy(),
                            beta=np.zeros(5), alpha=base.alpha)
        self._source_epochs = source_epochs
        self._source_stats = {}
        self._target_stats = {}
        self._routes = {}
        self.source_fit_log = []
        self.source_fit_reused = source_beta is not None
        if source_beta is None:
            self._fit_source_activity()
        else:
            beta = np.asarray(source_beta, dtype=np.float64)
            if beta.shape != (5,) or not np.all(np.isfinite(beta)):
                raise ValueError('source_beta must contain five finite coefficients')
            self.source.beta[:] = beta
            self.source_fit_seconds = 0.

    def _fit_source_activity(self):
        from .target import pair_loss_and_gradient
        initial_theta, initial_alpha = self.source.theta.copy(), self.source.alpha
        rng = np.random.default_rng(stable_seed(self.seed, 'source-activity'))
        start = time.monotonic()
        nitems = len(self.data['source_item_ids'])
        for epoch in range(self._source_epochs):
            losses, pairs = [], 0
            for item in rng.permutation(nitems):
                pos, neg = self.source_store.sample_users(int(item), rng,
                    max_positive=self.source_cap, max_negative=self.negatives)
                if not len(pos) or not len(neg):
                    continue
                batch = self.source_store.pair_batch(int(item), pos, neg)
                loss, gradient = pair_loss_and_gradient(self.source, batch, regularize=False)
                self.source.beta -= self.source.config.lr * gradient['beta']
                losses.append(loss)
                pairs += len(pos) * len(neg)
            self.source_fit_log.append({'epoch': epoch + 1, 'items': len(losses),
                'pairs': pairs, 'mean_loss': float(np.mean(losses)) if losses else None})
        assert np.array_equal(self.source.theta, initial_theta)
        assert self.source.alpha == initial_alpha
        self.source_fit_seconds = time.monotonic() - start

    def _stats(self, item, source):
        from .target import pair_loss_and_gradient
        cache = self._source_stats if source else self._target_stats
        if item in cache:
            return cache[item]
        store = self.source_store if source else self.target_store
        model = self.source if source else self.base
        domain = 'source' if source else 'target'
        rng = np.random.default_rng(stable_seed(self.seed, domain, item, 'support'))
        pos, neg = store.sample_users(item, rng,
            max_positive=self.source_cap if source else self.target_cap,
            max_negative=self.negatives)
        if len(pos) and len(neg):
            batch = store.pair_batch(item, pos, neg)
            _, grads = pair_loss_and_gradient(model, batch, regularize=False)
            gradient = np.asarray(grads['theta'], dtype=np.float64)
        else:
            gradient = np.zeros_like(self.base.theta, dtype=np.float64)
        history = store.history(pos, item, exclude_focal=True) if len(pos) else np.empty((0, 64))
        if len(history):
            profile = history.mean(axis=0)
            dispersion = float(np.mean(np.sum((history - profile) ** 2, axis=1)))
        else:
            profile, dispersion = np.zeros(64), 0.
        result = {'gradient': gradient, 'profile': unit(profile), 'dispersion': dispersion,
                  'support': len(pos), 'gradient_norm': float(np.linalg.norm(gradient)),
                  'positive_users': np.asarray(pos), 'negative_users': np.asarray(neg)}
        cache[item] = result
        return result

    def donor_items(self, item):
        similarities = np.asarray(self.data['source_x']) @ np.asarray(self.data['target_x'][item])
        raw_target = str(self.data['target_item_ids'][item])
        eligible = np.flatnonzero(np.asarray(self.data['source_item_ids']) != raw_target)
        # Source IDs are lexically sorted, so stable sorting explicitly fixes ties.
        order = np.argsort(-similarities[eligible], kind='stable')[:self.donors]
        return eligible[order]

    def route_inputs(self, item):
        """Returns donor IDs, historical feature rows, and source gradients."""
        item = int(item)
        if item in self._routes:
            return self._routes[item]
        donors = self.donor_items(item)
        target = self._stats(item, False)
        rows, gradients = [], []
        for donor in donors:
            source = self._stats(int(donor), True)
            rows.append((1., cosine(self.data['target_x'][item], self.data['source_x'][donor]),
                cosine(target['profile'], source['profile']),
                cosine(target['gradient'], source['gradient']),
                np.log1p(target['support']), np.log1p(source['support']),
                target['dispersion'], source['dispersion'],
                target['gradient_norm'], source['gradient_norm']))
            gradients.append(source['gradient'])
        result = (donors, np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
                  np.asarray(gradients, dtype=np.float64).reshape(-1, len(self.base.theta)))
        self._routes[item] = result
        return result

    def source_summary(self):
        return {'activity_fit_epochs': self._source_epochs, 'activity_fit_log': self.source_fit_log,
                'activity_fit_seconds': self.source_fit_seconds,
                'activity_fit_reused': self.source_fit_reused,
                'source_beta': self.source.beta.tolist(),
                'theta_frozen': True, 'alpha_frozen': True, 'no_user_id_copy': True,
                'source_cap': self.source_cap, 'target_cap': self.target_cap,
                'negative_cap': self.negatives, 'donor_cap': self.donors}


def query_batches(data, context, stage=1):
    """Label-only sampling; support/features never inspect these query events."""
    if stage != 1:
        raise ValueError('forward labels are defined only for the next-30-day stage 1')
    positive = np.asarray(data['target_positive'])
    events = positive[positive[:, 2] == stage]
    users_by_item = {}
    for user, item, _ in events:
        users_by_item.setdefault(int(item), set()).add(int(user))
    nusers = len(data['target_user_ids'])
    for item in sorted(users_by_item):
        prior = set(map(int, context.target_store.observed_users[item]))
        positives = np.asarray(sorted(users_by_item[item] - prior), dtype=np.int64)
        if not len(positives):
            continue
        eligible_neg = np.asarray([u for u in range(nusers)
                                   if u not in prior and u not in users_by_item[item]], dtype=np.int64)
        if not len(eligible_neg):
            continue
        rng = np.random.default_rng(stable_seed(context.seed, item, 'query-label'))
        selected_pos = rng.choice(positives, min(context.target_cap, len(positives)), replace=False)
        selected_neg = rng.choice(eligible_neg, min(context.negatives, len(eligible_neg)), replace=False)
        batch = context.target_store.pair_batch(item, selected_pos, selected_neg)
        yield item, selected_pos, selected_neg, batch


def forward_labels(data, context, radius=.1):
    features, labels, items, donors, query_pos, query_neg, query_ptr = [], [], [], [], [], [], [0]
    negative_ptr = [0]
    mixture_records = []
    for item, positive, negative, batch in query_batches(data, context):
        source_ids, rows, gradients = context.route_inputs(item)
        base_loss = batch_loss(context.base, batch)
        for donor, row, gradient in zip(source_ids, rows, gradients):
            step = clip_step(gradient, context.eta, radius)
            gain = base_loss - batch_loss(context.base, batch, step)
            features.append(row)
            labels.append(gain)
            items.append(item)
            donors.append(donor)
        query_pos.extend(map(int, positive))
        query_neg.extend(map(int, negative))
        query_ptr.append(len(query_pos))
        negative_ptr.append(len(query_neg))
        mixture_records.append((item, base_loss, len(positive), len(negative)))
    return {'features': np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
            'labels': np.asarray(labels), 'item_id': np.asarray(items, dtype=np.int64),
            'donor_id': np.asarray(donors, dtype=np.int64),
            'query_positive_users': np.asarray(query_pos, dtype=np.int64),
            'query_negative_users': np.asarray(query_neg, dtype=np.int64),
            'query_positive_ptr': np.asarray(query_ptr, dtype=np.int64),
            'query_negative_ptr': np.asarray(negative_ptr, dtype=np.int64),
            'query_item_summary': np.asarray(mixture_records, dtype=np.float64),
            'label_radius': np.asarray(radius), 'label_eta': np.asarray(context.eta),
            'feature_names': np.asarray(FEATURE_NAMES)}


def policy_weights(policy, features, ridges, temperature=1., force_mass=False):
    n = len(features)
    if not n or policy == 'zero':
        return np.zeros(n)
    if policy == 'uniform':
        return np.full(n, 1. / n)
    regression = 'full' if policy == 'predicted_single' else policy
    predicted = ridges[regression].predict(features)
    if policy == 'predicted_single':
        weights = np.zeros(n)
        if predicted.max() > 0:
            weights[int(np.argmax(predicted))] = 1.
        return weights
    weights = utility_weights(predicted, temperature)
    if force_mass:
        total = weights.sum()
        if total > 0:
            weights /= total
    return weights


def policy_correction(policy, features, gradients, ridges, radius, temperature=1.,
                      eta=.1, mechanism=None):
    if policy == 'zero' or (radius == 0 and mechanism is None):
        return np.zeros(gradients.shape[1]), np.zeros(len(gradients))
    if mechanism not in (None, 'mass1', 'norm01'):
        raise ValueError('unknown mechanism control')
    controlled = mechanism is not None
    # Both controls preserve the native exact-null action. Only the mass
    # control normalizes weights; the norm control retains native source mass.
    weights = policy_weights(policy, features, ridges, temperature,
                             force_mass=mechanism == 'mass1')
    if not np.any(weights):
        return np.zeros(gradients.shape[1]), weights
    step_radius = .1 if controlled else radius
    steps = clip_step(gradients, eta, step_radius)
    correction = weights @ steps
    norm = float(np.linalg.norm(correction))
    if mechanism == 'norm01' and norm > 0:
        correction = correction * (.1 / norm)
    return correction, weights


class MultiPolicyScorer:
    """Extract all user features once per item; score all requested policies."""
    def __init__(self, context, ridges, configurations):
        self.context, self.ridges, self.configurations = context, ridges, configurations
        self.diagnostics = {name: [] for name in configurations}

    def __call__(self, item_ids):
        model = self.context.base
        output = np.empty((len(self.configurations), len(item_ids), len(self.context.data['target_user_ids'])))
        for column, item in enumerate(item_ids):
            anchor, phi, activity = self.context.target_store.all_features(int(item))
            base_scores = model.score_features(anchor, phi, activity)
            _, features, gradients = self.context.route_inputs(int(item))
            corrections = []
            for name, config in self.configurations.items():
                correction, weights = policy_correction(features=features, gradients=gradients,
                    ridges=self.ridges, eta=self.context.eta, **config)
                corrections.append(correction)
                conditional = weights / weights.sum() if weights.sum() else weights
                self.diagnostics[name].append({'item_id': int(item),
                    'source_mass': float(weights.sum()), 'zero': not bool(np.any(correction)),
                    'correction_norm': float(np.linalg.norm(correction)),
                    'effective_donors': int(np.count_nonzero(weights)),
                    'concentration': float(conditional @ conditional)})
            corrections = np.asarray(corrections)
            nonzero = np.any(corrections, axis=1)
            output[:, column, :] = base_scores[None, :]
            if np.any(nonzero):
                # Equivalent HPO candidates (all R=0, uniform temperatures,
                # inactive nulls) share actual matrix products.
                unique, inverse = np.unique(corrections[nonzero], axis=0, return_inverse=True)
                output[nonzero, column, :] += (phi @ unique.T).T[inverse]
        return output

    def summary(self, name):
        rows = self.diagnostics[name]
        if not rows:
            return {'n_items': 0}
        result = {'n_items': len(rows)}
        for field in ('source_mass', 'zero', 'correction_norm', 'effective_donors', 'concentration'):
            result['mean_' + field] = float(np.mean([r[field] for r in rows]))
        result['max_correction_norm'] = max(r['correction_norm'] for r in rows)
        return result
