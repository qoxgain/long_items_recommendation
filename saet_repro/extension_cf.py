"""B1/B2 continuing CF trajectories, isolated from the frozen six-epoch code.

Native architectures and objectives come directly from ``collaborative``. The
only additional objective fixes an item and contrasts a positive/negative user.
All model, Adam, sampling and RNG state is persisted without Python pickles.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import time
from zipfile import BadZipFile

import numpy as np
import torch
from torch.nn import functional as F

from .collaborative import (CFConfig, CollaborativeModel, UniformNegativeSampler,
                            fit_edges, past_edges, search_space)
from .evaluation import evaluate, stratified_summaries
from .io import json_safe, sha256, write_json

METHODS = ("mf_bpr", "lightgcn", "simplex", "lightgcn_user")
SELECTION_SEEDS = (101, 202, 303)
FINAL_SEEDS = (11, 22, 33, 44, 55)
CHECKPOINTS = (6, 20, 50, 100)
SCHEMA = "extension-cf-v1"


def grid(method):
    if method not in METHODS:
        raise ValueError(method)
    return search_space("lightgcn" if method == "lightgcn_user" else method)


def json_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = ("saet_repro/collaborative.py", "saet_repro/evaluation.py", "saet_repro/io.py",
             "saet_repro/data.py", "saet_repro/extension_cf.py",
             "scripts/run_extension_cf.py", "docs/extensions_cf.md", "requirements.txt")
    return {"schema": SCHEMA, "code_sha256": {name: sha256(root / name) for name in files},
            "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__},
            "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True}


def peak_memory_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


class UniformNegativeUserSampler:
    """Uniform eligible users excluding every pre-query rating for an item.

    Cohort user IDs already encode prefix eligibility. Rejection sampling is
    independent and exactly uniform over the remaining users, with replacement.
    Future positives are neither read nor excluded.
    """
    def __init__(self, data):
        self.n_users = len(data["target_user_ids"])
        self.n_items = len(data["target_item_ids"])
        edges = past_edges(data)
        self.forbidden = np.unique(edges[:, 1] * self.n_users + edges[:, 0])
        self.counts = np.bincount(edges[:, 1], minlength=self.n_items)
        if not self.n_users:
            raise ValueError("Empty eligible user cohort")

    def sample(self, items, count, rng):
        items = np.asarray(items, dtype=np.int64)
        if np.any(self.counts[items] >= self.n_users):
            raise ValueError("A training item has no unseen negative user")
        out = rng.integers(0, self.n_users, size=(len(items), count), dtype=np.int64)
        codes = items[:, None] * self.n_users
        while len(self.forbidden):
            positions = np.searchsorted(self.forbidden, codes + out)
            safe = np.minimum(positions, len(self.forbidden) - 1)
            bad = (positions < len(self.forbidden)) & (self.forbidden[safe] == codes + out)
            if not bad.any():
                break
            out[bad] = rng.integers(0, self.n_users, size=int(bad.sum()), dtype=np.int64)
        return out


def user_bpr_objective(model, users, items, negative_users):
    """Fixed-item BPR with exactly the native LightGCN ego-L2 convention."""
    if model.config.method != "lightgcn":
        raise ValueError("User-retrieval objective is defined only for LightGCN")
    all_users, all_items = model.propagated()
    item = all_items[items]
    positive_score = (all_users[users] * item).sum(dim=1)
    negative_score = (all_users[negative_users] * item[:, None, :]).sum(dim=2)
    native = -F.logsigmoid(positive_score[:, None] - negative_score).mean()
    penalty = (model.item_embedding(items).square().sum(dim=1).mean()
               + model.user_embedding(users).square().sum(dim=1).mean()
               + model.user_embedding(negative_users).square().sum(dim=2).mean()) / 2
    return native + model.config.l2 * penalty


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


class TrainingState:
    def __init__(self, data, config, method, seed, identity, directory=None):
        if method not in METHODS or config.method != ("lightgcn" if method == "lightgcn_user" else method):
            raise ValueError("Family and model configuration disagree")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.data, self.config, self.method, self.seed = data, config, method, int(seed)
        self.directory = Path(directory) if directory is not None else None
        self.identity = json_safe({**identity, "config": asdict(config), "method": method,
                                   "seed": self.seed})
        self.model = CollaborativeModel(data, config)
        self.identity.update({"cohort_fit_sha256": self.model.fit_sha256,
                              "user_ids_sha256": json_hash(self.model.user_ids),
                              "item_ids_sha256": json_hash(self.model.item_ids)})
        self.identity_sha256 = json_hash(self.identity)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.rng = np.random.default_rng(seed)
        self.sampler = (UniformNegativeUserSampler(data) if method == "lightgcn_user"
                        else UniformNegativeSampler(data))
        self.edges = fit_edges(data)
        if not len(self.edges):
            raise ValueError("No fit-positive edges")
        self.epoch = self.step = 0
        self.history = []
        self.cumulative_update_norm = 0.
        self.checkpoint_hashes = {}

    def save(self, checkpoint=False):
        directory = self.directory
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            checkpoint_path = directory / f"checkpoint_{self.epoch:03d}.npz"
            self.model.save(checkpoint_path)
            self.checkpoint_hashes[checkpoint_path.name] = sha256(checkpoint_path)
        arrays = {f"parameter::{key}": value.detach().cpu().numpy()
                  for key, value in self.model.state_dict().items()}
        optimizer = self.optimizer.state_dict()
        scalar_state = {}
        for index, state in optimizer["state"].items():
            scalar_state[str(index)] = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    arrays[f"optimizer::{index}::{key}"] = value.detach().cpu().numpy()
                else:
                    scalar_state[str(index)][key] = value
        numpy_rng = np.random.get_state()
        arrays["global_numpy_rng_keys"] = numpy_rng[1]
        arrays["torch_rng"] = torch.get_rng_state().numpy()
        metadata = {"identity": self.identity, "identity_sha256": self.identity_sha256,
                    "epoch": self.epoch, "step": self.step, "history": self.history,
                    "rng_state": self.rng.bit_generator.state,
                    "global_python_rng": random.getstate(),
                    "global_numpy_rng": [numpy_rng[0], *numpy_rng[2:]],
                    "optimizer_groups": optimizer["param_groups"],
                    "optimizer_scalars": scalar_state,
                    "checkpoint_sha256": self.checkpoint_hashes,
                    "cumulative_update_norm": self.cumulative_update_norm}
        arrays["metadata_json"] = np.asarray(json.dumps(json_safe(metadata), sort_keys=True))
        path = directory / "last_state.npz"
        temporary = path.with_suffix(".npz.tmp")
        # The one replaceable Adam state is uncompressed: avoid compressing tens
        # of MB every epoch. Named milestone model checkpoints stay compressed.
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
        write_json(directory / "history.json", self.history)
        write_json(directory / "frozen.json", self.identity)
        # Commit the receipt last. A crash between replacements is detected and
        # quarantined instead of silently accepting a mixed-epoch artifact set.
        write_json(directory / "last_state.sha256.json", {"sha256": sha256(path),
                   "identity_sha256": self.identity_sha256, "epoch": self.epoch,
                   "artifact_sha256": {name: sha256(directory / name)
                                       for name in ("history.json", "frozen.json")}})

    def restore(self):
        """Verify current provenance and content; quarantine stale state."""
        directory = self.directory
        if directory is None or not directory.exists():
            return False
        path = directory / "last_state.npz"
        if not path.exists():
            return False
        try:
            manifest = json.loads((directory / "last_state.sha256.json").read_text())
            if manifest["sha256"] != sha256(path):
                raise ValueError("State checksum mismatch")
            if set(manifest["artifact_sha256"]) != {"history.json", "frozen.json"}:
                raise ValueError("State artifact receipt is incomplete")
            for name, digest in manifest["artifact_sha256"].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f"State artifact checksum mismatch: {name}")
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
                if (metadata["identity"] != self.identity
                        or metadata["identity_sha256"] != self.identity_sha256
                        or manifest["identity_sha256"] != self.identity_sha256):
                    raise ValueError("State identity mismatch")
                if manifest["epoch"] != metadata["epoch"] or len(metadata["history"]) != metadata["epoch"]:
                    raise ValueError("State/history epoch mismatch")
                if (json.loads((directory / "history.json").read_text()) != metadata["history"]
                        or json.loads((directory / "frozen.json").read_text()) != self.identity):
                    raise ValueError("State/history/frozen content mismatch")
                for name, digest in metadata["checkpoint_sha256"].items():
                    if sha256(directory / name) != digest:
                        raise ValueError(f"Checkpoint checksum mismatch: {name}")
                parameters = {key.removeprefix("parameter::"): torch.from_numpy(saved[key].copy())
                              for key in saved.files if key.startswith("parameter::")}
                if any(not torch.isfinite(tensor).all() for tensor in parameters.values()):
                    raise ValueError("Nonfinite restored parameters")
                self.model.load_state_dict(parameters, strict=True)
                optimizer = {"state": {int(index): state for index, state
                                       in metadata["optimizer_scalars"].items()},
                             "param_groups": metadata["optimizer_groups"]}
                for group in optimizer["param_groups"]:
                    group["betas"] = tuple(group["betas"])
                for name in saved.files:
                    if name.startswith("optimizer::"):
                        _, index, key = name.split("::")
                        tensor = torch.from_numpy(saved[name].copy())
                        if not torch.isfinite(tensor).all():
                            raise ValueError("Nonfinite restored optimizer state")
                        optimizer["state"].setdefault(int(index), {})[key] = tensor
                self.optimizer.load_state_dict(optimizer)
                self.rng.bit_generator.state = metadata["rng_state"]
                random.setstate(_tuples(metadata["global_python_rng"]))
                numpy_rng = metadata["global_numpy_rng"]
                np.random.set_state((numpy_rng[0], saved["global_numpy_rng_keys"].copy(), *numpy_rng[1:]))
                torch.set_rng_state(torch.from_numpy(saved["torch_rng"].copy()))
            self.epoch, self.step = int(metadata["epoch"]), int(metadata["step"])
            self.history = metadata["history"]
            self.checkpoint_hashes = metadata["checkpoint_sha256"]
            self.cumulative_update_norm = float(metadata["cumulative_update_norm"])
            self.model.invalidate_cache()
            return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, BadZipFile) as error:
            stale = directory.with_name(directory.name + f".stale_{time.time_ns()}")
            os.replace(directory, stale)
            write_json(stale / "stale_reason.json", {"reason": str(error),
                       "expected_identity_sha256": self.identity_sha256})
            self.__init__(self.data, self.config, self.method, self.seed, self.identity, directory)
            return False

    def train_epoch(self):
        started = time.monotonic()
        self.model.train()
        self.model.invalidate_cache()
        order = self.rng.permutation(len(self.edges))
        total_loss, seen, pairs, updates = 0., 0, 0, 0
        gradients, update_norms = [], []
        parameters = list(self.model.parameters())
        for start in range(0, len(order), self.config.batch_size):
            batch = self.edges[order[start:start + self.config.batch_size]]
            count = self.config.negatives if self.config.method == "simplex" else 1
            negatives = self.sampler.sample(batch[:, 1 if self.method == "lightgcn_user" else 0],
                                            count, self.rng)
            users, items = torch.from_numpy(batch[:, 0].copy()), torch.from_numpy(batch[:, 1].copy())
            negative_tensor = torch.from_numpy(negatives)
            self.optimizer.zero_grad(set_to_none=True)
            loss = (user_bpr_objective(self.model, users, items, negative_tensor)
                    if self.method == "lightgcn_user" else self.model.objective(users, items, negative_tensor))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CF loss")
            loss.backward()
            gradient_norm = float(torch.sqrt(sum(torch.sum(parameter.grad.detach().square())
                                                for parameter in parameters if parameter.grad is not None)))
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite CF gradient")
            before = [parameter.detach().clone() for parameter in parameters]
            self.optimizer.step()
            with torch.no_grad():
                update_norm = float(torch.sqrt(sum(torch.sum((parameter - old).square())
                                                   for parameter, old in zip(parameters, before))))
            if not np.isfinite(update_norm):
                raise FloatingPointError("Nonfinite CF parameter update")
            gradients.append(gradient_norm)
            update_norms.append(update_norm)
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
            pairs += len(batch) * count
            updates += 1
            self.step += 1
        self.epoch += 1
        self.cumulative_update_norm += sum(update_norms)
        self.model.invalidate_cache()
        self.model.eval()
        row = {"epoch": self.epoch, "loss": total_loss / seen,
               "positive_edges": seen, "positive_count": seen, "pair_count": pairs,
               "updates": updates, "cumulative_updates": self.step,
               "cumulative_positive_edges": self.epoch * len(self.edges),
               "cumulative_pairs": self.epoch * pairs,
               "gradient_norm_mean": float(np.mean(gradients)), "gradient_norm_max": max(gradients),
               "update_norm_mean": float(np.mean(update_norms)), "update_norm_sum": sum(update_norms),
               "cumulative_update_norm": self.cumulative_update_norm,
               "training_seconds": time.monotonic() - started, "peak_rss_mib": peak_memory_mib()}
        self.history.append(row)
        return row


def train_trajectory(data, config, method, seed, epochs, identity=None, directory=None,
                     evaluate_each_epoch=False, checkpoints=CHECKPOINTS, callback=None, resume=True):
    """Continue one trajectory without changing Adam or stochastic draws."""
    if epochs < 1:
        raise ValueError("Epoch budget must be positive")
    frozen = {**code_identity(), **(identity or {}), "evaluate_each_epoch": evaluate_each_epoch}
    state = TrainingState(data, config, method, seed, frozen, directory)
    if resume:
        state.restore()
    # This is present before any historical query scoring, including epoch one.
    if directory is not None:
        write_json(Path(directory) / "frozen.json", state.identity)
    started = time.monotonic()
    while state.epoch < epochs:
        row = state.train_epoch()
        if evaluate_each_epoch:
            begin = time.monotonic()
            result = evaluate(data, 1, state.model.score_all, ks=(20,), save_topk=False)
            summary = stratified_summaries(data, result)
            row["historical_tail80_ndcg20"] = summary["tail80"]["ndcg20"]
            row["historical_summary"] = summary
            row["evaluation_seconds"] = time.monotonic() - begin
        row["invocation_elapsed_seconds"] = time.monotonic() - started
        row["peak_rss_mib"] = peak_memory_mib()
        state.save(checkpoint=state.epoch in checkpoints)
        if callback is not None:
            callback(row)
    return state


def selection_records(trajectories, checkpoints=(6, 20, 50), seeds=SELECTION_SEEDS,
                      expected_trials=12):
    """Build only declared checkpoint means; incomplete grids cannot select."""
    by_key = {(row["trial"], row["seed"]): row for row in trajectories}
    expected = {(trial, seed) for trial in range(expected_trials) for seed in seeds}
    if len(by_key) != len(trajectories) or set(by_key) != expected:
        raise ValueError("Selection requires every declared trial and seed exactly once")
    records = []
    for trial in range(expected_trials):
        config = by_key[trial, seeds[0]]["config"]
        if any(by_key[trial, seed]["config"] != config for seed in seeds):
            raise ValueError("Configuration differs across selection seeds")
        for epoch in checkpoints:
            values = []
            for seed in seeds:
                rows = [row for row in by_key[trial, seed]["history"] if row["epoch"] == epoch]
                if len(rows) != 1 or rows[0].get("historical_tail80_ndcg20") is None:
                    raise ValueError(f"Missing trial {trial}, seed {seed}, epoch {epoch}")
                value = float(rows[0]["historical_tail80_ndcg20"])
                if not np.isfinite(value):
                    raise ValueError("Undefined selection endpoint")
                values.append(value)
            records.append({"trial": trial, "config": config, "epochs": epoch,
                            "seed_scores": dict(zip(map(str, seeds), values)),
                            "score": float(np.mean(values))})
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
