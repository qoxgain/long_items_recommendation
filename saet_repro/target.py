"""Independent reconstruction of the five R3 target scorers.

The supplied documents do not define phi. We use the unit-normalized
concatenation [h, x*h, abs(x-h)] (192 dimensions), an explicit reconstruction
choice. See docs/target_protocol.md for loss, sampling, and temporal rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

DAY = 86_400_000
METHODS = ("unbounded_mean", "activity_mean", "activity_decay",
           "activity_conditional", "activity_both")


@dataclass(frozen=True)
class Config:
    method: str = "activity_mean"
    lr: float = .001
    rho: float = .0001
    tau: float = .1
    history_cap: int = 50
    decay_days: float = 90.
    history_temperature: float = .2
    batch_size: int = 32
    max_positive: int = 5
    max_negative: int = 5

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method}")
        if min(self.tau, self.history_temperature, self.decay_days,
               self.history_cap, self.batch_size, self.max_positive,
               self.max_negative) <= 0:
            raise ValueError("Temperatures, caps, and batch size must be positive")


def _normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _raw_activity(times, cutoff):
    """All five quantities are log1p transformed; recency uses fractional days."""
    ages = (cutoff - np.asarray(times, dtype=np.int64)) / DAY
    recency = float(ages.min()) if len(ages) else 365.
    weeks = len(np.unique((cutoff - 1 - np.asarray(times, dtype=np.int64)) // (7 * DAY)))
    return np.log1p([np.count_nonzero(ages <= 30),
                     np.count_nonzero(ages <= 90),
                     np.count_nonzero(ages <= 365), recency, weeks])


class FeatureStore:
    """Fit-only history and activity features, with leave-focal-out support."""

    def __init__(self, data, config: Config, domain="target"):
        self.data, self.config, self.domain = data, config, domain
        self.x = np.asarray(data[f"{domain}_x"], dtype=np.float32)
        self.n_items = len(self.x)
        self.n_users = len(data[f"{domain}_user_ids"])
        self.fit_start, self.fit_end = int(data["fit_start"]), int(data["fit_end"])
        self.activity = np.asarray(data[f"{domain}_activity"], dtype=np.float32)
        self.activity_mean = np.asarray(data["activity_mean"])
        self.activity_scale = np.asarray(data["activity_scale"])
        self.train = np.asarray(data[f"{domain}_train"], dtype=np.int64).reshape(-1, 2)
        self.positive_users = [np.empty(0, dtype=np.int64) for _ in range(self.n_items)]
        self.observed_users = [np.empty(0, dtype=np.int64) for _ in range(self.n_items)]
        if len(self.train):
            order = np.argsort(self.train[:, 1], kind="stable")
            for group in np.split(order, np.flatnonzero(np.diff(self.train[order, 1])) + 1):
                self.positive_users[int(self.train[group[0], 1])] = np.unique(self.train[group, 0])
        obs = np.asarray(data[f"{domain}_observed"], dtype=np.int64).reshape(-1, 3)
        obs = obs[obs[:, 2] < 1]
        if len(obs):
            order = np.argsort(obs[:, 1], kind="stable")
            for group in np.split(order, np.flatnonzero(np.diff(obs[order, 1])) + 1):
                self.observed_users[int(obs[group[0], 1])] = np.unique(obs[group, 0])
        eu = np.asarray(data[f"{domain}_event_users"], dtype=np.int64)
        ei = np.asarray(data[f"{domain}_event_items"], dtype=np.int64)
        et = np.asarray(data[f"{domain}_event_times"], dtype=np.int64)
        er = np.asarray(data[f"{domain}_event_ratings"])
        keep = (et >= self.fit_start) & (et < self.fit_end) & (er >= 4)
        eu, ei, et = eu[keep], ei[keep], et[keep]
        # Reserve enough entries to remove every occurrence of one focal item
        # BEFORE retaining the latest cap histories, including repeated events.
        max_repeats = 1
        if len(eu):
            _, counts = np.unique(eu * self.n_items + ei, return_counts=True)
            max_repeats = int(counts.max())
        self.slots = config.history_cap + max_repeats
        self.history_items = np.full((self.n_users, self.slots), -1, dtype=np.int64)
        self.history_times = np.zeros((self.n_users, self.slots), dtype=np.int64)
        if len(eu):
            # Most recent first; item ID resolves equal timestamp histories.
            order = np.lexsort((ei, -et, eu))
            sorted_users = eu[order]
            for group in np.split(order, np.flatnonzero(np.diff(sorted_users)) + 1):
                take = group[:self.slots]
                user = eu[take[0]]
                self.history_items[user, :len(take)] = ei[take]
                self.history_times[user, :len(take)] = et[take]
        self._history_x = self.x[np.maximum(self.history_items, 0)]
        self._history_x[self.history_items < 0] = 0
        self._mean_history = self._history_batch(np.arange(self.n_users),
                                                 np.zeros(self.n_users, dtype=int), False)
        au = np.asarray(data[f"{domain}_all_rating_users"], dtype=np.int64)
        ai = np.asarray(data[f"{domain}_all_rating_items"], dtype=np.int64)
        at = np.asarray(data[f"{domain}_all_rating_times"], dtype=np.int64)
        self._rating_times = [np.empty(0, dtype=np.int64) for _ in range(self.n_users)]
        self._rating_items = [np.empty(0, dtype=np.int64) for _ in range(self.n_users)]
        if len(au):
            keep = (at >= self.fit_start) & (at < self.fit_end)
            au, ai, at = au[keep], ai[keep], at[keep]
            order = np.argsort(au, kind="stable")
            for group in np.split(order, np.flatnonzero(np.diff(au[order])) + 1):
                self._rating_times[au[group[0]]] = at[group]
                self._rating_items[au[group[0]]] = ai[group]
        self._support_activity = {}
        self._support_features = {}
        self._negative_users = {}

    def _history_batch(self, users, items, exclude_focal):
        users, items = np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        ids = self.history_items[users]
        valid = ids >= 0
        if exclude_focal:
            valid &= ids != items[:, None]
        valid &= np.cumsum(valid, axis=1) <= self.config.history_cap
        logits = np.zeros(ids.shape, dtype=np.float32)
        if self.config.method in ("activity_conditional", "activity_both"):
            logits += np.einsum("nd,nkd->nk", self.x[items], self._history_x[users]) / self.config.history_temperature
        if self.config.method in ("activity_decay", "activity_both"):
            ages = (self.fit_end - self.history_times[users]) / DAY
            logits -= np.log(2.) * ages / self.config.decay_days
        logits = np.where(valid, logits, -np.inf)
        maximum = np.max(logits, axis=1, keepdims=True)
        maximum[~np.isfinite(maximum)] = 0
        weights = np.exp(logits - maximum)
        summed = np.einsum("nk,nkd->nd", weights, self._history_x[users])
        return _normalize(summed).astype(np.float32)

    def history(self, users, item, exclude_focal=False):
        users = np.atleast_1d(users).astype(np.int64)
        items = np.broadcast_to(np.asarray(item, dtype=np.int64), users.shape)
        if not exclude_focal and self.config.method in ("unbounded_mean", "activity_mean", "activity_decay"):
            return self._mean_history[users]
        return self._history_batch(users, items, exclude_focal)

    def _activity_batch(self, users, items, exclude_focal):
        if not exclude_focal:
            return self.activity[users]
        rows = []
        for user, item in zip(users, items):
            key = int(user), int(item)
            if key not in self._support_activity:
                times = self._rating_times[user][self._rating_items[user] != item]
                raw = _raw_activity(times, self.fit_end)
                self._support_activity[key] = ((raw - self.activity_mean) / self.activity_scale).astype(np.float32)
            rows.append(self._support_activity[key])
        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    def features(self, users, items, exclude_focal=False):
        users = np.atleast_1d(users).astype(np.int64)
        items = np.broadcast_to(np.asarray(items, dtype=np.int64), users.shape)
        h = self.history(users, items, exclude_focal=exclude_focal)
        x = self.x[items]
        anchor = np.einsum("nd,nd->n", x, h)
        phi = _normalize(np.concatenate((h, x * h, np.abs(x - h)), axis=1)).astype(np.float32)
        activity = self._activity_batch(users, items, exclude_focal)
        return anchor, phi, activity

    def all_features(self, item):
        return self.features(np.arange(self.n_users), item)

    def pair_batch(self, item, pos_users, neg_users):
        result = {}
        for prefix, users, exclude in (("pos", pos_users, True), ("neg", neg_users, False)):
            for name, value in zip(("anchor", "phi", "activity"), self.features(users, item, exclude)):
                result[f"{prefix}_{name}"] = value
        return result

    def sample_users(self, item, rng, max_positive=None, max_negative=None):
        max_positive = max_positive or self.config.max_positive
        max_negative = max_negative or self.config.max_negative
        positive = self.positive_users[item]
        if item not in self._negative_users:
            self._negative_users[item] = np.setdiff1d(np.arange(self.n_users), self.observed_users[item], assume_unique=True)
        negative = self._negative_users[item]
        return (rng.choice(positive, min(max_positive, len(positive)), replace=False),
                rng.choice(negative, min(max_negative, len(negative)), replace=False))


class Model:
    def __init__(self, config, store, theta=None, beta=None, alpha=1.):
        self.config, self.store = config, store
        self.theta = np.zeros(3 * store.x.shape[1]) if theta is None else np.asarray(theta, dtype=np.float64).copy()
        self.beta = np.zeros(5) if beta is None else np.asarray(beta, dtype=np.float64).copy()
        self.alpha = float(alpha)

    def score_features(self, anchor, phi, activity):
        # Explicitly promote the anchor before multiplication: NumPy otherwise
        # rounds the alpha contribution to float32 and breaks exact gradients.
        score = self.alpha * np.asarray(anchor, dtype=np.float64) + np.asarray(phi) @ self.theta
        if self.config.method != "unbounded_mean":
            score = score + np.asarray(activity) @ self.beta
        return score

    def score_all(self, item_ids):
        return np.asarray([self.score_features(*self.store.all_features(int(i))) for i in item_ids])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, theta=self.theta, beta=self.beta, alpha=self.alpha,
                                config_json=json.dumps(asdict(self.config), sort_keys=True),
                                domain=self.store.domain, feature_formula="unit([h,x*h,abs(x-h)])")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, data, domain="target"):
        with np.load(path, allow_pickle=False) as saved:
            config = Config(**json.loads(str(saved["config_json"])))
            return cls(config, FeatureStore(data, config, domain),
                       theta=saved["theta"], beta=saved["beta"], alpha=float(saved["alpha"]))


def pair_loss_and_gradient(model, batch, regularize=True):
    """Mean within-item Cartesian pair loss, then mean over items.

    batch has pos/neg_anchor [P]/[N], phi [P,D]/[N,D], activity
    [P,5]/[N,5], or the corresponding arrays with leading item dimension B.
    Optional pos_mask/neg_mask mark padded rows. Return (loss, gradient dict).
    Loss=mean softplus((negative-positive)/tau)+rho/2*||theta||².
    Set regularize=False for empirical source gradients and query utilities.
    """
    batch = {k: np.asarray(v) for k, v in batch.items()}
    if batch["pos_anchor"].ndim == 1:
        batch = {k: v[None] for k, v in batch.items()}
    pa, na = batch["pos_anchor"], batch["neg_anchor"]
    pp, np_ = batch["pos_phi"], batch["neg_phi"]
    pac, nac = batch["pos_activity"], batch["neg_activity"]
    ps, ns = model.score_features(pa, pp, pac), model.score_features(na, np_, nac)
    mask = (batch.get("pos_mask", np.ones(pa.shape, bool))[:, :, None]
            & batch.get("neg_mask", np.ones(na.shape, bool))[:, None, :])
    counts = mask.sum(axis=(1, 2))
    # A warm item without an eligible pair contributes zero, while retaining
    # its uniform item weight in this batch.
    denominator = np.maximum(counts, 1)[:, None, None] * max(len(counts), 1)
    z = (ns[:, None, :] - ps[:, :, None]) / model.config.tau
    weights = mask / denominator
    loss = float(np.sum(np.logaddexp(0., z) * weights))
    sigmoid = np.exp(-np.logaddexp(0., -z))
    derivative = sigmoid * weights / model.config.tau
    positive_weight, negative_weight = -derivative.sum(axis=2), derivative.sum(axis=1)
    gradient = {
        "theta": np.einsum("bp,bpd->d", positive_weight, pp) + np.einsum("bn,bnd->d", negative_weight, np_),
        "beta": np.einsum("bp,bpd->d", positive_weight, pac) + np.einsum("bn,bnd->d", negative_weight, nac),
        "alpha": float(np.sum(positive_weight * pa) + np.sum(negative_weight * na)),
    }
    if model.config.method == "unbounded_mean":
        gradient["beta"][:] = 0
    if regularize:
        loss += .5 * model.config.rho * float(model.theta @ model.theta)
        gradient["theta"] += model.config.rho * model.theta
    return loss, gradient


def _bank_key(store, seed, epochs):
    digest = hashlib.sha256()
    for key in ("fit_start", "fit_end", f"{store.domain}_x", f"{store.domain}_train",
                f"{store.domain}_observed", f"{store.domain}_user_ids", f"{store.domain}_activity",
                f"{store.domain}_event_users", f"{store.domain}_event_items",
                f"{store.domain}_event_times", f"{store.domain}_event_ratings",
                f"{store.domain}_all_rating_users", f"{store.domain}_all_rating_items",
                f"{store.domain}_all_rating_times", "activity_mean", "activity_scale"):
        digest.update(np.ascontiguousarray(store.data[key]).tobytes())
    digest.update(json.dumps({k: v for k, v in asdict(store.config).items()
                              if k not in ("lr", "rho", "tau", "batch_size")}, sort_keys=True).encode())
    digest.update(f"v4|{store.domain}|{seed}|{epochs}".encode())
    return digest.hexdigest()[:24]


def _make_epoch_bank(store, rng):
    """Every warm item exactly once; independent uniform subsets each epoch."""
    n, p, q = store.n_items, store.config.max_positive, store.config.max_negative
    order = rng.permutation(n)
    users = np.zeros((n, p + q), dtype=np.int64)
    masks = np.zeros((n, p + q), dtype=bool)
    for row, item in enumerate(order):
        positive, negative = store.sample_users(int(item), rng)
        users[row, :len(positive)] = positive
        users[row, p:p + len(negative)] = negative
        masks[row, :len(positive)] = True
        masks[row, p:p + len(negative)] = True
    bank = np.zeros((n, p + q, 1 + 3 * store.x.shape[1] + 5), dtype=np.float32)
    # Build in vectorized blocks; do not allocate the full item-user history cube.
    for begin in range(0, n, 128):
        end = min(begin + 128, n)
        for start, stop, exclude in ((0, p, True), (p, p + q, False)):
            flat_users = users[begin:end, start:stop].reshape(-1)
            flat_items = np.repeat(order[begin:end], stop - start)
            anchor, phi, activity = store.features(flat_users, flat_items, exclude)
            values = np.concatenate((anchor[:, None], phi, activity), axis=1)
            bank[begin:end, start:stop] = values.reshape(end - begin, stop - start, -1)
    return bank, masks, order


def train_target(data, config, seed, epochs=6, validation_callback=None):
    """Exact NumPy gradients and Adam, six fixed epochs by default.

    Callback(epoch, model) is reporting-only; its return value is ignored.
    SAET_FEATURE_CACHE enables resumable on-disk banks shared across HPO trials.
    """
    started = time.monotonic()
    store = FeatureStore(data, config)
    model = Model(config, store)
    rng = np.random.default_rng(seed)
    size = len(model.theta) + 6
    m, v = np.zeros(size), np.zeros(size)
    step, history = 0, []
    cache = os.environ.get("SAET_FEATURE_CACHE")
    bank_dir = Path(cache) / _bank_key(store, seed, epochs) if cache else None
    if bank_dir:
        bank_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        feature_path = bank_dir / f"epoch_{epoch}.npy" if bank_dir else None
        meta_path = bank_dir / f"epoch_{epoch}_meta.npz" if bank_dir else None
        if feature_path and feature_path.exists() and meta_path.exists():
            bank = np.load(feature_path, mmap_mode="r")
            with np.load(meta_path) as meta:
                masks, order = meta["masks"], meta["order"]
                rng.bit_generator.state = json.loads(str(meta["rng_state"]))
        else:
            bank, masks, order = _make_epoch_bank(store, rng)
            if feature_path:
                temp = feature_path.with_name(feature_path.name + ".tmp")
                with open(temp, "wb") as output:
                    np.save(output, bank)
                os.replace(temp, feature_path)
                np.savez_compressed(meta_path, masks=masks, order=order,
                                    rng_state=json.dumps(rng.bit_generator.state))
        losses, pairs, updates = [], 0, 0
        p, d = config.max_positive, len(model.theta)
        for begin in range(0, len(order), config.batch_size):
            block = bank[begin:begin + config.batch_size]
            mask = masks[begin:begin + config.batch_size]
            batch = {"pos_anchor": block[:, :p, 0], "neg_anchor": block[:, p:, 0],
                     "pos_phi": block[:, :p, 1:d + 1], "neg_phi": block[:, p:, 1:d + 1],
                     "pos_activity": block[:, :p, d + 1:], "neg_activity": block[:, p:, d + 1:],
                     "pos_mask": mask[:, :p], "neg_mask": mask[:, p:]}
            loss, gradient = pair_loss_and_gradient(model, batch)
            vector = np.concatenate((gradient["theta"], gradient["beta"], [gradient["alpha"]]))
            step += 1
            m = .9 * m + .1 * vector
            v = .999 * v + .001 * vector * vector
            parameters = np.concatenate((model.theta, model.beta, [model.alpha]))
            parameters -= config.lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
            model.theta, model.beta, model.alpha = parameters[:d], parameters[d:d + 5], float(parameters[-1])
            losses.append(loss)
            pairs += int(np.sum(mask[:, :p].sum(axis=1) * mask[:, p:].sum(axis=1)))
            updates += 1
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else None,
                  "visited_items": len(order), "pair_count": pairs, "updates": updates,
                  "elapsed_seconds": time.monotonic() - started,
                  "theta_norm": float(np.linalg.norm(model.theta)), "alpha": model.alpha,
                  "beta": model.beta.tolist()}
        history.append(record)
        if validation_callback is not None:
            validation_callback(epoch + 1, model)
    return model, history
