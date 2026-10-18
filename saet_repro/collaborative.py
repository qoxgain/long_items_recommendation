"""Native user--item CF objectives, transposed only for item--user retrieval.

These are documented task adaptations, not claims of original-paper reproduction.
All fitting reads fit/pre-fit records only; query records are evaluation-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import json
import hashlib
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CFConfig:
    method: str = "mf_bpr"
    embedding_dim: int = 64
    lr: float = 0.001
    l2: float = 0.00001
    layers: int = 1
    history_mix: float = 0.5
    negative_weight: float = 10.0
    negatives: int = 10
    margin: float = 0.9
    history_cap: int = 50
    batch_size: int = 1024

    def __post_init__(self):
        if self.method not in {"mf_bpr", "lightgcn", "simplex"}:
            raise ValueError(f"Unknown method {self.method}")
        if self.embedding_dim < 1 or self.batch_size < 1 or self.layers < 1:
            raise ValueError("Dimensions, batch size and layers must be positive")
        if self.negatives < 1 or self.history_cap < 1:
            raise ValueError("Negative count and history cap must be positive")


def search_space(method: str) -> list[CFConfig]:
    if method == "mf_bpr":
        return [CFConfig(method=method, embedding_dim=d, lr=lr, l2=l2)
                for d, lr, l2 in product((32, 64), (.001, .005, .01), (1e-5, 1e-3))]
    if method == "lightgcn":
        return [CFConfig(method=method, layers=k, lr=lr, l2=l2)
                for k, lr, l2 in product((1, 2), (.001, .005, .01), (1e-5, 1e-3))]
    if method == "simplex":
        return [CFConfig(method=method, lr=lr, history_mix=m, negative_weight=w)
                for lr, m, w in product((.001, .005), (.25, .5, .75), (10., 50.))]
    raise ValueError(method)


def _scalar(data, key):
    return int(np.asarray(data[key]).item())


def fit_edges(data) -> np.ndarray:
    """Canonical fixed-fit positive user,item edges, with no query inspection."""
    edges = np.asarray(data["target_train"], dtype=np.int64).reshape(-1, 2)
    return np.unique(edges, axis=0)


def past_edges(data) -> np.ndarray:
    """Any rating before fit_end, including pre-fit observations."""
    blocks = [fit_edges(data)]
    for prefix in ("target_event", "target_all_rating"):
        if f"{prefix}_times" in data:
            users = np.asarray(data[f"{prefix}_users"], dtype=np.int64)
            items = np.asarray(data[f"{prefix}_items"], dtype=np.int64)
            past = (np.asarray(data[f"{prefix}_times"]) < _scalar(data, "fit_end")) & (items >= 0)
            blocks.append(np.column_stack((users[past], items[past])))
    # observed also includes pre-fit events and is sufficient for tiny fixtures.
    if "target_observed" in data:
        obs = np.asarray(data["target_observed"], dtype=np.int64).reshape(-1, 3)
        blocks.append(obs[obs[:, 2] < 1, :2])
    return np.unique(np.concatenate(blocks), axis=0)


class UniformNegativeSampler:
    """Exactly uniform independent draws from each user's unseen item set."""
    def __init__(self, data):
        self.n_items = len(data["target_item_ids"])
        self.n_users = len(data["target_user_ids"])
        edges = past_edges(data)
        self.forbidden = np.unique(edges[:, 0] * self.n_items + edges[:, 1])
        self.counts = np.bincount(edges[:, 0], minlength=self.n_users)
        if self.n_items == 0:
            raise ValueError("Empty catalog")

    def _contains(self, codes):
        if not len(self.forbidden):
            return np.zeros(np.shape(codes), dtype=bool)
        pos = np.searchsorted(self.forbidden, codes)
        safe = np.minimum(pos, max(0, len(self.forbidden) - 1))
        return (pos < len(self.forbidden)) & (self.forbidden[safe] == codes)

    def sample(self, users, count, rng):
        users = np.asarray(users, dtype=np.int64)
        if np.any(self.counts[users] >= self.n_items):
            raise ValueError("A training user has no unseen negative item")
        out = rng.integers(0, self.n_items, size=(len(users), count), dtype=np.int64)
        user_codes = users[:, None] * self.n_items
        bad = self._contains(user_codes + out)
        while bad.any():
            out[bad] = rng.integers(0, self.n_items, size=int(bad.sum()), dtype=np.int64)
            bad = self._contains(user_codes + out)
        return out


def history_table(data, cap: int) -> np.ndarray:
    """Most recent cap+1 fit-positive items; extra slot allows focal removal."""
    edges = fit_edges(data)
    n_users = len(data["target_user_ids"])
    times = {(int(u), int(i)): 0 for u, i in edges}
    if "target_event_times" in data:
        for u, i, t, r in zip(data["target_event_users"], data["target_event_items"],
                              data["target_event_times"], data["target_event_ratings"]):
            key = (int(u), int(i))
            if key in times and _scalar(data, "fit_start") <= t < _scalar(data, "fit_end") and r >= 4:
                times[key] = max(times[key], int(t))
    histories = [[] for _ in range(n_users)]
    for u, i in edges:
        histories[u].append((times[(int(u), int(i))], int(i)))
    out = np.full((n_users, cap + 1), -1, dtype=np.int64)
    for u, hist in enumerate(histories):
        # Item IDs are lexically sorted by the shared data contract.
        ids = [i for _, i in sorted(hist, key=lambda z: (-z[0], z[1]))[:cap + 1]]
        out[u, :len(ids)] = ids
    return out


def normalized_graph(data) -> torch.Tensor:
    edges = fit_edges(data)
    n_users = len(data["target_user_ids"])
    n = n_users + len(data["target_item_ids"])
    a, b = edges[:, 0], edges[:, 1] + n_users
    rows = np.concatenate((a, b))
    cols = np.concatenate((b, a))
    degree = np.bincount(rows, minlength=n).astype(np.float64)
    values = 1. / np.sqrt(degree[rows] * degree[cols])
    return torch.sparse_coo_tensor(torch.as_tensor(np.vstack((rows, cols))),
                                  torch.as_tensor(values, dtype=torch.float32), (n, n)).coalesce()


class CollaborativeModel(nn.Module):
    def __init__(self, data, config: CFConfig):
        super().__init__()
        self.config = config
        self.n_users = len(data["target_user_ids"])
        self.n_items = len(data["target_item_ids"])
        self.user_ids = np.asarray(data["target_user_ids"]).astype(str).copy()
        self.item_ids = np.asarray(data["target_item_ids"]).astype(str).copy()
        # Checkpoints must not silently bind the learned rows to another cohort
        # or reconstruct a different LightGCN graph / SimpleX history.
        fit_hash = hashlib.sha256(fit_edges(data).tobytes())
        fit_hash.update(past_edges(data).tobytes())
        if config.method == "simplex":
            fit_hash.update(history_table(data, config.history_cap).tobytes())
        self.fit_sha256 = fit_hash.hexdigest()
        self.user_embedding = nn.Embedding(self.n_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, config.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=.1)
        nn.init.normal_(self.item_embedding.weight, std=.1)
        if config.method == "lightgcn":
            self.register_buffer("graph", normalized_graph(data), persistent=False)
        if config.method == "simplex":
            self.register_buffer("histories", torch.as_tensor(history_table(data, config.history_cap)), persistent=False)
            self.history_transform = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
            nn.init.xavier_uniform_(self.history_transform.weight)
        self._score_cache = None

    def propagated(self):
        u, i = self.user_embedding.weight, self.item_embedding.weight
        if self.config.method != "lightgcn":
            return u, i
        current = torch.cat((u, i), dim=0)
        total = current
        for _ in range(self.config.layers):
            current = torch.sparse.mm(self.graph, current)
            total = total + current
        total = total / (self.config.layers + 1)
        return total[:self.n_users], total[self.n_users:]

    def simplex_users(self, users, focal_items=None):
        history = self.histories[users]
        valid = history >= 0
        if focal_items is not None:
            valid = valid & (history != focal_items[:, None])
        valid = valid & (valid.cumsum(dim=1) <= self.config.history_cap)
        embedded = self.item_embedding(history.clamp(min=0))
        mean = (embedded * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp(min=1)[:, None]
        mix = self.config.history_mix
        combined = (1 - mix) * self.user_embedding(users) + mix * self.history_transform(mean)
        return F.normalize(combined, dim=1, eps=1e-12)

    def pair_scores(self, users, items, exclude_focal=False):
        """Native user,item scores; optionally remove focal training positives."""
        if self.config.method == "simplex":
            u = self.simplex_users(users, items if exclude_focal else None)
            i = F.normalize(self.item_embedding(items), dim=1, eps=1e-12)
        else:
            all_u, all_i = self.propagated()
            u, i = all_u[users], all_i[items]
        return (u * i).sum(dim=1)

    def objective(self, users, positive, negatives):
        c = self.config
        if c.method == "simplex":
            u = self.simplex_users(users, positive)
            pos = F.normalize(self.item_embedding(positive), dim=1, eps=1e-12)
            neg = F.normalize(self.item_embedding(negatives), dim=2, eps=1e-12)
            positive_score = (u * pos).sum(dim=1)
            negative_score = (u[:, None, :] * neg).sum(dim=2)
            native = ((1 - positive_score).square() + c.negative_weight *
                      F.relu(negative_score - c.margin).square().mean(dim=1)).mean()
        else:
            all_u, all_i = self.propagated()
            u = all_u[users]
            positive_score = (u * all_i[positive]).sum(dim=1)
            negative_score = (u[:, None, :] * all_i[negatives]).sum(dim=2)
            native = -F.logsigmoid(positive_score[:, None] - negative_score).mean()
        # Regularize ego embeddings, including for LightGCN; scale per example.
        penalty = (self.user_embedding(users).square().sum(dim=1).mean()
                   + self.item_embedding(positive).square().sum(dim=1).mean()
                   + self.item_embedding(negatives).square().sum(dim=2).mean()) / 2
        if c.method == "simplex":
            penalty = penalty + self.history_transform.weight.square().mean() / 2
        return native + c.l2 * penalty

    def invalidate_cache(self):
        self._score_cache = None

    @torch.no_grad()
    def score_all(self, item_ids):
        """Transpose native scores: [requested_items, all_lexical_users]."""
        self.eval()
        if self._score_cache is None:
            if self.config.method == "simplex":
                parts = [self.simplex_users(torch.arange(s, min(s + 512, self.n_users)))
                         for s in range(0, self.n_users, 512)]
                u = torch.cat(parts)
                i = F.normalize(self.item_embedding.weight, dim=1, eps=1e-12)
            else:
                u, i = self.propagated()
            self._score_cache = (u.detach(), i.detach())
        u, i = self._score_cache
        ids = torch.as_tensor(np.asarray(item_ids, dtype=np.int64))
        return (i[ids] @ u.T).cpu().numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"parameter::{k}": v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        arrays["config_json"] = np.asarray(json.dumps(asdict(self.config), sort_keys=True))
        arrays["format_version"] = np.asarray(2)
        arrays["user_ids"] = self.user_ids
        arrays["item_ids"] = self.item_ids
        arrays["fit_sha256"] = np.asarray(self.fit_sha256)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path, data):
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != 2:
                raise ValueError("Unsupported CF checkpoint format")
            config = CFConfig(**json.loads(str(z["config_json"].item())))
            model = cls(data, config)
            if (not np.array_equal(z["user_ids"], model.user_ids)
                    or not np.array_equal(z["item_ids"], model.item_ids)
                    or str(z["fit_sha256"].item()) != model.fit_sha256):
                raise ValueError("CF checkpoint cohort or fit history differs")
            state = {k.removeprefix("parameter::"): torch.from_numpy(z[k].copy())
                     for k in z.files if k.startswith("parameter::")}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model


def train_cf(data, config: CFConfig, seed: int, epochs: int = 6, progress=None):
    """Every fit edge once/epoch; no callbacks receive query metrics."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = CollaborativeModel(data, config)
    sampler = UniformNegativeSampler(data)
    edges = fit_edges(data)
    if not len(edges):
        raise ValueError("No fit-positive edges")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    for epoch in range(epochs):
        model.train()
        model.invalidate_cache()
        order = rng.permutation(len(edges))
        loss_total = 0.
        seen = 0
        for start in range(0, len(order), config.batch_size):
            batch = edges[order[start:start + config.batch_size]]
            neg = sampler.sample(batch[:, 0], config.negatives if config.method == "simplex" else 1, rng)
            users, pos = torch.from_numpy(batch[:, 0].copy()), torch.from_numpy(batch[:, 1].copy())
            optimizer.zero_grad(set_to_none=True)
            loss = model.objective(users, pos, torch.from_numpy(neg))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CF training loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(batch)
            seen += len(batch)
        row = {"epoch": epoch + 1, "loss": loss_total / seen, "positive_edges": seen}
        history.append(row)
        if progress is not None:
            progress(row)
    model.invalidate_cache()
    model.eval()
    return model, history
