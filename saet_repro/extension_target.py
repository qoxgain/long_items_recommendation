"""Resumable B1 target convergence; the original six-epoch implementation is frozen.

One transient feature bank is reused by all compatible configurations of a seed.
All models share fit features during historical evaluation; a bounded RAM cache
reuses their parameter-independent features across epochs. No evaluation bank is
persisted. Adam's operation order is identical to ``target.train_target``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import itertools
import json
import os
import platform
import resource
import time

import numpy as np

from .evaluation import evaluate_many, stratified_summaries
from .io import sha256, write_json
from .target import Config, FeatureStore, Model, _bank_key, _make_epoch_bank, pair_loss_and_gradient

METHODS = ("activity_mean", "activity_both")
SELECTION_SEEDS = (101, 202, 303)
FINAL_SEEDS = (11, 22, 33, 44, 55)
CHECKPOINTS = (6, 20, 50, 100)
SCHEMA = "extension-target-v1"
HISTORICAL_CACHE_MAX_BYTES = 2_800_000_000


def grid(method):
    return [Config(method=method, lr=lr, rho=rho, tau=tau)
            for lr, rho, tau in itertools.product((.001, .003, .01), (0., .0001), (.1, .3))]


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = ("saet_repro/target.py", "saet_repro/evaluation.py", "saet_repro/io.py", "saet_repro/data.py",
             "saet_repro/extension_target.py", "scripts/run_extension_targets.py")
    return {"schema": SCHEMA, "code_sha256": {name: sha256(root / name) for name in files},
            "dependencies": {"python": platform.python_version(), "numpy": np.__version__}}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_identity(data):
    """Hash fit and evaluation arrays, including their names, shapes and dtypes."""
    digest = hashlib.sha256()
    for key in sorted(data):
        value = np.asarray(data[key])
        if value.dtype.hasobject:
            raise ValueError(f"Object array cannot be fingerprinted safely: {key}")
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def peak_memory_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


@dataclass(frozen=True)
class Trajectory:
    config: Config
    seed: int
    directory: Path | None = None


class TrainingState:
    def __init__(self, spec, store, identity):
        self.spec = spec
        self.model = Model(spec.config, store)
        self.m = np.zeros(len(self.model.theta) + 6)
        self.v = np.zeros_like(self.m)
        self.rng = np.random.default_rng(spec.seed)
        self.step = self.epoch = 0
        self.history = []
        self.cumulative_update_norm = 0.
        self.identity = identity
        self.identity_sha256 = _json_hash(identity)

    @property
    def directory(self):
        return Path(self.spec.directory) if self.spec.directory is not None else None

    def save(self, checkpoint=False):
        directory = self.directory
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "last_state.npz"
        metadata = {"identity": self.identity, "identity_sha256": self.identity_sha256,
                    "rng_state": self.rng.bit_generator.state, "epoch": self.epoch,
                    "step": self.step, "history": self.history,
                    "cumulative_update_norm": self.cumulative_update_norm}
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, theta=self.model.theta, beta=self.model.beta,
                                alpha=self.model.alpha, m=self.m, v=self.v,
                                metadata_json=json.dumps(metadata, sort_keys=True))
        os.replace(temporary, path)
        write_json(directory / "last_state.sha256.json", {"sha256": sha256(path),
                   "identity_sha256": self.identity_sha256, "epoch": self.epoch})
        write_json(directory / "history.json", self.history)
        write_json(directory / "frozen.json", self.identity)
        if checkpoint:
            self.model.save(directory / f"checkpoint_{self.epoch:03d}.npz")
            write_json(directory / f"checkpoint_{self.epoch:03d}.json",
                       {"epoch": self.epoch, "identity_sha256": self.identity_sha256,
                        "model_sha256": sha256(directory / f"checkpoint_{self.epoch:03d}.npz"),
                        "record": self.history[-1]})

    def restore(self, checkpoints=CHECKPOINTS):
        """Reuse only verified state; quarantine stale or damaged artifacts."""
        directory = self.directory
        if directory is None or not directory.exists():
            return False
        path = directory / "last_state.npz"
        if not path.exists():
            return False
        try:
            manifest = json.loads((directory / "last_state.sha256.json").read_text())
            if manifest["sha256"] != sha256(path):
                raise ValueError("state checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
                if (metadata["identity"] != self.identity
                        or metadata["identity_sha256"] != self.identity_sha256
                        or manifest["identity_sha256"] != self.identity_sha256):
                    raise ValueError("state identity mismatch")
                theta, beta = saved["theta"].copy(), saved["beta"].copy()
                alpha = float(saved["alpha"])
                m, v = saved["m"].copy(), saved["v"].copy()
                if (theta.shape != self.model.theta.shape or beta.shape != (5,)
                        or m.shape != self.m.shape or v.shape != self.v.shape
                        or not all(np.all(np.isfinite(a)) for a in (theta, beta, alpha, m, v))):
                    raise ValueError("invalid state parameters")
            self.model.theta, self.model.beta, self.model.alpha = theta, beta, alpha
            self.m, self.v = m, v
            self.rng.bit_generator.state = metadata["rng_state"]
            self.step, self.epoch = int(metadata["step"]), int(metadata["epoch"])
            self.history = metadata["history"]
            self.cumulative_update_norm = float(metadata["cumulative_update_norm"])
            if len(self.history) != self.epoch or manifest["epoch"] != self.epoch:
                raise ValueError("state/history epoch mismatch")
            if [row["epoch"] for row in self.history] != list(range(1, self.epoch + 1)):
                raise ValueError("noncontiguous history")
            if self.step != sum(row["updates"] for row in self.history):
                raise ValueError("optimizer/history update count mismatch")
            for epoch in checkpoints:
                if epoch <= self.epoch:
                    checkpoint = directory / f"checkpoint_{epoch:03d}.npz"
                    note = json.loads(checkpoint.with_suffix(".json").read_text())
                    if (note["identity_sha256"] != self.identity_sha256
                            or note["model_sha256"] != sha256(checkpoint)):
                        raise ValueError("checkpoint checksum/identity mismatch")
            return True
        except (OSError, ValueError, KeyError, TypeError) as error:
            stale = directory.with_name(directory.name + f".stale_{time.time_ns()}")
            os.replace(directory, stale)
            write_json(stale / "stale_reason.json", {"reason": str(error),
                       "expected_identity_sha256": self.identity_sha256})
            # Clear any partially assigned values before deterministic recomputation.
            self.__init__(self.spec, self.model.store, self.identity)
            return False


def _train_epoch(state, bank, masks, order):
    model, config = state.model, state.spec.config
    started = time.monotonic()
    losses, gradient_norms, update_norms = [], [], []
    pairs, positives, updates = 0, 0, 0
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
        if not np.isfinite(loss) or not np.all(np.isfinite(vector)):
            raise FloatingPointError("Nonfinite loss or gradient")
        state.step += 1
        state.m = .9 * state.m + .1 * vector
        state.v = .999 * state.v + .001 * vector * vector
        parameters = np.concatenate((model.theta, model.beta, [model.alpha]))
        update = config.lr * (state.m / (1 - .9 ** state.step)) / (np.sqrt(state.v / (1 - .999 ** state.step)) + 1e-8)
        parameters -= update
        model.theta, model.beta, model.alpha = parameters[:d], parameters[d:d + 5], float(parameters[-1])
        losses.append(loss)
        gradient_norms.append(float(np.linalg.norm(vector)))
        update_norms.append(float(np.linalg.norm(update)))
        pairs += int(np.sum(mask[:, :p].sum(axis=1) * mask[:, p:].sum(axis=1)))
        positives += int(mask[:, :p].sum())
        updates += 1
    state.epoch += 1
    state.cumulative_update_norm += sum(update_norms)
    record = {"epoch": state.epoch, "loss": float(np.mean(losses)) if losses else None,
              "visited_items": len(order), "positive_count": positives,
              "pair_count": pairs, "updates": updates, "cumulative_updates": state.step,
              "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.,
              "gradient_norm_max": max(gradient_norms, default=0.),
              "update_norm_mean": float(np.mean(update_norms)) if update_norms else 0.,
              "update_norm_sum": sum(update_norms),
              "cumulative_update_norm": state.cumulative_update_norm,
              "training_seconds": time.monotonic() - started,
              "theta_norm": float(np.linalg.norm(model.theta)), "alpha": model.alpha,
              "beta": model.beta.tolist(), "peak_rss_mib": peak_memory_mib()}
    state.history.append(record)


class HistoricalFeatureCache:
    """Bounded, invocation-local cache of unchanged FeatureStore outputs.

    Admission is deterministic first-fit without eviction, so a query set larger
    than the budget still reuses its admitted items on every epoch. Activity is
    item-independent for ``all_features`` and stored once. Only returned arrays
    are retained; scores, labels, model parameters and training features are not.
    The source FeatureStore must remain fixed for this cache's lifetime.
    """

    def __init__(self, store, max_bytes=HISTORICAL_CACHE_MAX_BYTES):
        if int(max_bytes) != max_bytes or max_bytes < 0:
            raise ValueError("Feature cache budget must be a nonnegative integer")
        self.store, self.max_bytes = store, int(max_bytes)
        self.nbytes = 0
        self._features = {}
        self._activity = None

    def __len__(self):
        return len(self._features)

    def get(self, item):
        item = int(item)
        if item in self._features:
            return self._features[item]
        features = self.store.all_features(item)
        anchor, phi, activity = features
        added = anchor.nbytes + phi.nbytes
        if self._activity is None:
            added += activity.nbytes
        if self.nbytes + added <= self.max_bytes:
            if self._activity is None:
                self._activity = activity
                self._activity.setflags(write=False)
            anchor.setflags(write=False)
            phi.setflags(write=False)
            features = (anchor, phi, self._activity)
            self._features[item] = features
            self.nbytes += added
        return features


def evaluate_models(data, models, ks=(10, 20, 50), save_topk=True, feature_cache=None):
    """Exact full-candidate evaluation with one feature construction per item."""
    if not models:
        return []
    store = models[0].store
    if any(model.store is not store for model in models):
        raise ValueError("Models must share their FeatureStore")
    if feature_cache is not None and feature_cache.store is not store:
        raise ValueError("Feature cache must share the models' FeatureStore")
    names = tuple(str(index) for index in range(len(models)))

    def score_many(indices):
        # One item at a time bounds the conditional-history temporary arrays.
        scores = np.empty((len(models), len(indices), store.n_users), dtype=np.float64)
        for row, item in enumerate(indices):
            features = (store.all_features(int(item)) if feature_cache is None
                        else feature_cache.get(int(item)))
            for index, model in enumerate(models):
                scores[index, row] = model.score_features(*features)
        return scores

    results = evaluate_many(data, 1, score_many, names, ks=ks,
                            save_topk=save_topk, batch_size=1)
    return [results[name] for name in names]


def train_trajectories(data, trajectories, epochs, identity=None, evaluate_each_epoch=False,
                       checkpoints=CHECKPOINTS, callback=None, resume=True,
                       historical_cache_max_bytes=HISTORICAL_CACHE_MAX_BYTES):
    """Train compatible trajectories, restoring exact Adam/RNG continuation.

    ``trajectories`` is a list of Trajectory(config, seed, optional directory).
    Historical queries are used only when explicitly requested by the caller.
    Return TrainingState objects in the original trajectory order.
    """
    if not trajectories:
        return []
    configs = [spec.config for spec in trajectories]
    representation = lambda c: {k: v for k, v in asdict(c).items()
                                if k not in ("lr", "rho", "tau", "batch_size")}
    if any(representation(c) != representation(configs[0]) for c in configs):
        raise ValueError("Cannot share features across different representations")
    store = FeatureStore(data, configs[0])
    feature_cache = (HistoricalFeatureCache(store, historical_cache_max_bytes)
                     if evaluate_each_epoch else None)
    shared_identity = {**code_identity(), "run_identity": identity or {},
                       "data_content_sha256": data_identity(data),
                       "feature_data_sha256": _bank_key(store, 0, 0),
                       "evaluate_each_epoch": evaluate_each_epoch,
                       "historical_cache_max_bytes": (feature_cache.max_bytes
                                                       if feature_cache is not None else 0)}
    states = []
    for spec in trajectories:
        frozen = {**shared_identity, "config": asdict(spec.config), "seed": int(spec.seed)}
        state = TrainingState(spec, store, frozen)
        if resume:
            state.restore(checkpoints=checkpoints)
        states.append(state)
    started = time.monotonic()
    while any(state.epoch < epochs for state in states):
        next_epoch = min(state.epoch for state in states if state.epoch < epochs) + 1
        active = [state for state in states if state.epoch == next_epoch - 1]
        for seed in dict.fromkeys(state.spec.seed for state in active):
            group = [state for state in active if state.spec.seed == seed]
            rng_states = [json.dumps(state.rng.bit_generator.state, sort_keys=True) for state in group]
            if len(set(rng_states)) != 1:
                raise ValueError("Compatible seed trajectories have different sampling states")
            begin = time.monotonic()
            bank, masks, order = _make_epoch_bank(store, group[0].rng)
            bank_seconds = time.monotonic() - begin
            after_rng = group[0].rng.bit_generator.state
            for state in group:
                state.rng.bit_generator.state = after_rng
                _train_epoch(state, bank, masks, order)
                state.history[-1]["feature_bank_seconds_shared"] = bank_seconds
                state.history[-1]["feature_bank_trajectory_count"] = len(group)
            del bank, masks, order
        evaluation_seconds = 0.
        if evaluate_each_epoch:
            begin = time.monotonic()
            results = evaluate_models(data, [state.model for state in active], ks=(20,), save_topk=False,
                                      feature_cache=feature_cache)
            evaluation_seconds = time.monotonic() - begin
            for state, result in zip(active, results):
                summaries = stratified_summaries(data, result)
                state.history[-1]["historical_tail80_ndcg20"] = summaries["tail80"]["ndcg20"]
                state.history[-1]["historical_summary"] = summaries
                state.history[-1]["historical_feature_cache_bytes"] = feature_cache.nbytes
                state.history[-1]["historical_feature_cache_items"] = len(feature_cache)
        for state in active:
            state.history[-1].update({"evaluation_seconds_shared": evaluation_seconds,
                                     "evaluation_trajectory_count": len(active),
                                     "invocation_elapsed_seconds": time.monotonic() - started,
                                     "peak_rss_mib": peak_memory_mib()})
            state.save(checkpoint=state.epoch in checkpoints)
        if callback is not None:
            callback(next_epoch, active)
    return states


def selection_records(states, checkpoints=(6, 20, 50), seeds=SELECTION_SEEDS):
    """Only predeclared checkpoint means are eligible for selection."""
    configs = []
    for state in states:
        if state.spec.config not in configs:
            configs.append(state.spec.config)
    records = []
    for trial, config in enumerate(configs):
        by_seed = {state.spec.seed: state for state in states if state.spec.config == config}
        if set(by_seed) != set(seeds):
            raise ValueError("Selection requires all declared seeds for every configuration")
        for epoch in checkpoints:
            values = []
            for seed in seeds:
                rows = [row for row in by_seed[seed].history if row["epoch"] == epoch]
                if len(rows) != 1:
                    raise ValueError(f"Missing trial {trial}, seed {seed}, checkpoint {epoch}")
                value = rows[0].get("historical_tail80_ndcg20")
                if value is None or not np.isfinite(value):
                    raise ValueError("Undefined historical selection endpoint")
                values.append(float(value))
            records.append({"trial": trial, "config": asdict(config), "epochs": epoch,
                            "seed_scores": dict(zip(map(str, seeds), values)), "score": float(np.mean(values))})
    return records


def select_checkpoint(records):
    if not records:
        raise ValueError("No selectable checkpoints")
    return dict(max(records, key=lambda row: (row["score"], -row["epochs"], -row["trial"])))


def extension_decision(records, near_zero=1e-8):
    winner = select_checkpoint(records)
    earlier = next(row for row in records if row["trial"] == winner["trial"] and row["epochs"] == 20)
    difference = winner["score"] - earlier["score"]
    near = abs(earlier["score"]) < near_zero
    threshold = 1e-4 if near else .01 * abs(earlier["score"])
    return {"extend_to_100": winner["epochs"] == 50 and difference >= threshold,
            "winner_at_50_stage": winner, "same_config_epoch20_score": earlier["score"],
            "difference": difference, "threshold": threshold,
            "near_zero_denominator": near, "near_zero_cutoff": near_zero}
