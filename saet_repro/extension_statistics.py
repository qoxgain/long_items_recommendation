"""Additional descriptive statistics; original estimands remain unchanged."""
from __future__ import annotations

import hashlib
import numpy as np


def unique_item_cluster_interval(items, n_bootstrap=2000, seed=2026, confidence=.95):
    """Resample raw item identities jointly across origins, after seed averaging.

    Every sampled identity carries all its observed origin records. Each draw
    averages within origins before giving each origin equal weight. Draws with
    an empty origin are rejected, explicitly counted, and replaced. This is a
    descriptive sensitivity, not a remedy for shared-user dependence.
    """
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    identities = sorted({row['item_id'] for row in items})
    origins = sorted({row['origin'] for row in items})
    if not identities:
        return {'ci': [None, None], 'n_unique_items': 0, 'n_resamples': 0,
                'rejected_empty_origin_draws': 0}
    item_index = {item: index for index, item in enumerate(identities)}
    origin_index = {origin: index for index, origin in enumerate(origins)}
    values = np.zeros((len(origins), len(identities)))
    present = np.zeros_like(values)
    for row in items:
        index = origin_index[row['origin']], item_index[row['item_id']]
        if present[index]:
            raise ValueError('Duplicate item-origin row')
        if not np.isfinite(row['difference']):
            raise ValueError('Undefined item difference')
        values[index], present[index] = row['difference'], 1
    rng = np.random.default_rng(seed)
    draws, rejected = [], 0
    while len(draws) < n_bootstrap:
        sampled = rng.integers(len(identities), size=len(identities))
        weights = np.bincount(sampled, minlength=len(identities))
        denominators = present @ weights
        if np.any(denominators == 0):
            rejected += 1
            if rejected > 100 * n_bootstrap:
                raise ValueError('Too few clusters to preserve every origin')
            continue
        draws.append(float(np.mean((values @ weights) / denominators)))
    alpha = (1 - confidence) / 2
    return {'ci': np.quantile(draws, [alpha, 1 - alpha]).tolist(),
            'n_unique_items': len(identities), 'n_resamples': n_bootstrap,
            'seed': seed, 'confidence': confidence,
            'rejected_empty_origin_draws': rejected,
            'unit': 'unique item with all origin records after seed averaging',
            'origin_weighting': 'equal in every draw', 'interpretation': 'descriptive sensitivity'}


def decision_digest(corrections):
    """Exact final corrections define a decision, independently of grid labels.

    Canonical positive zero makes semantically identical signed zero byte
    representations identical. No decimal rounding merges near decisions.
    """
    values = np.ascontiguousarray(corrections, dtype='<f8').copy()
    if not np.all(np.isfinite(values)):
        raise ValueError('Decisions must be finite')
    values[values == 0] = 0.
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def routing_summary(routing, mask=None):
    """Conditional means exclude exact zero corrections, not zero predictions."""
    size = len(routing['zero'])
    mask = np.ones(size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (size,):
        raise ValueError('Routing mask shape mismatch')
    zero = np.asarray(routing['zero'], dtype=bool)[mask]
    active = ~zero
    result = {'n_routed_items': int(mask.sum()), 'n_nonzero_items': int(active.sum()),
              'zero_action_rate': float(zero.mean()) if len(zero) else None}
    for field in ('source_mass', 'correction_norm', 'effective_donors', 'concentration'):
        values = np.asarray(routing[field])[mask]
        result['mean_' + field] = float(values.mean()) if len(values) else None
        result['nonzero_mean_' + field] = float(values[active].mean()) if active.any() else None
        result['nonzero_sum_' + field] = float(values[active].sum())
    return result


def convex_mixture_check(records, tolerance=1e-12):
    """Check actual convex loss Jensen bound via recorded actual loss gains."""
    if not records:
        raise ValueError('Actual mixture records are required')
    actual = np.asarray([r['actual_mixture_gain'] for r in records], dtype=float)
    weighted = np.asarray([r['weighted_actual_single_gain'] for r in records], dtype=float)
    mass = np.asarray([r['source_mass'] for r in records], dtype=float)
    if not np.all(np.isfinite(np.concatenate([actual, weighted, mass]))):
        raise ValueError('Actual mixture records must be finite')
    violations = weighted - actual
    max_violation = float(max(0., violations.max()))
    return {'n_records': len(records), 'tolerance': tolerance,
            'max_violation': max_violation, 'n_violations_above_tolerance': int((violations > tolerance).sum()),
            'source_mass_min': float(mass.min()), 'source_mass_max': float(mass.max()),
            'passed': bool(max_violation <= tolerance and mass.min() >= -tolerance and mass.max() <= 1 + tolerance),
            'inequality': 'actual_mixture_gain >= weighted_actual_single_gain',
            'scope': 'actual d60 selection pair losses; no future ranking guarantee'}
