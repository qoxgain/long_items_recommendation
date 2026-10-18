"""Independent temporal prefixes for the Amazon music reconstruction.

Only fit events determine cohorts, warm catalogs, title encoders and activity
statistics.  Every retained user/item pair is its earliest event globally.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DAY = 86_400_000
COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]


def load_data(path):
    """Read an NPZ without object deserialization or open-file ownership leaks."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_data(path, data):
    """Publish a complete NPZ atomically so waiting runners never see a partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def deduplicate_reviews(frame):
    """Keep the earliest global pair; source order resolves timestamp ties."""
    missing = set(COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")
    frame = frame.loc[:, COLUMNS].dropna().copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype(np.int64)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise").astype(np.float32)
    if not np.isfinite(frame["rating"].to_numpy()).all():
        raise ValueError("Review ratings must be finite")
    frame = frame.sort_values("timestamp", kind="stable")
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="first")
    return frame.reset_index(drop=True)


def read_reviews(path, before=None):
    """Load the projected CSV once; optional future truncation is pair-safe."""
    pieces = []
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=500_000,
                             dtype={"user_id": str, "parent_asin": str,
                                    "rating": np.float32, "timestamp": np.int64}):
        if before is not None:
            chunk = chunk.loc[chunk["timestamp"] < before]
        pieces.append(chunk)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=COLUMNS)
    del pieces
    # Category codes avoid retaining millions of repeated Python ID strings.
    for column in ("user_id", "parent_asin"):
        frame[column] = frame[column].astype("category")
    return deduplicate_reviews(frame)


def select_users(eligible, maximum=None, salt="cds:users"):
    users = sorted(str(user) for user in eligible)
    if maximum is not None and len(users) > maximum:
        users = sorted(users, key=lambda user: (
            hashlib.sha256(f"2026|{salt}|{user}".encode()).digest(), user))[:maximum]
    return np.asarray(sorted(users), dtype=str)


def raw_activity(users, times, n_users, cutoff):
    """Five log1p features, identical to target leave-focal-out recomputation."""
    users, times = np.asarray(users, np.int64), np.asarray(times, np.int64)
    ages = (cutoff - times) / DAY
    values = np.zeros((n_users, 5), dtype=np.float64)
    for column, days in enumerate((30, 90, 365)):
        values[:, column] = np.bincount(users[ages <= days], minlength=n_users)
    recency = np.full(n_users, np.inf)
    np.minimum.at(recency, users, ages)
    values[:, 3] = np.where(np.isfinite(recency), recency, 365.)
    if len(users):
        weeks = (cutoff - 1 - times) // (7 * DAY)
        unique = np.unique(np.column_stack((users, weeks)), axis=0)
        values[:, 4] = np.bincount(unique[:, 0], minlength=n_users)
    return np.log1p(values).astype(np.float32)


def build_domain(events, fit_start, fit_end, query_end, max_users=None, salt="cds:users"):
    """Build one independently selected domain from globally deduplicated rows."""
    fit_positive = events.loc[(events["timestamp"] >= fit_start) &
                              (events["timestamp"] < fit_end) & (events["rating"] >= 4)]
    counts = fit_positive.groupby("user_id", observed=True)["parent_asin"].nunique()
    user_ids = select_users(counts.index[counts >= 3], max_users, salt)
    fit_positive = fit_positive.loc[fit_positive["user_id"].isin(user_ids)]
    item_ids = np.asarray(sorted(str(item) for item in fit_positive["parent_asin"].unique()), dtype=str)
    selected = events.loc[(events["timestamp"] < query_end) & events["user_id"].isin(user_ids)]
    user_index, item_index = pd.Index(user_ids), pd.Index(item_ids)
    all_users = user_index.get_indexer(selected["user_id"]).astype(np.int64)
    all_items = item_index.get_indexer(selected["parent_asin"]).astype(np.int64)
    all_times = selected["timestamp"].to_numpy(dtype=np.int64)
    all_ratings = selected["rating"].to_numpy(dtype=np.float32)
    fit = (all_times >= fit_start) & (all_times < fit_end)
    warm = all_items >= 0
    eu, ei, et, er = all_users[warm], all_items[warm], all_times[warm], all_ratings[warm]
    order = np.lexsort((ei, eu, et))
    eu, ei, et, er = eu[order], ei[order], et[order], er[order]
    stage = np.where(et < fit_start, -1, np.where(et < fit_end, 0, 1))
    observed = np.column_stack((eu, ei, stage)).astype(np.int64)
    positive = observed[er >= 4]
    train = positive[positive[:, 2] == 0, :2]
    degree = np.bincount(train[:, 1], minlength=len(item_ids)).astype(np.int64)
    rank = np.lexsort((item_ids, degree))
    tail80, tail50 = np.zeros(len(item_ids), bool), np.zeros(len(item_ids), bool)
    tail80[rank[:int(np.ceil(.8 * len(item_ids)))]] = True
    tail50[rank[:int(np.ceil(.5 * len(item_ids)))]] = True
    activity_raw = raw_activity(all_users[fit], all_times[fit], len(user_ids), fit_end)
    return {"user_ids": user_ids, "item_ids": item_ids,
            "train": train, "observed": observed, "positive": positive,
            "event_users": eu, "event_items": ei, "event_times": et, "event_ratings": er,
            "all_rating_users": all_users[fit], "all_rating_items": all_items[fit],
            "all_rating_times": all_times[fit], "all_rating_ratings": all_ratings[fit],
            "degree": degree, "tail80": tail80, "tail50": tail50,
            "activity_raw": activity_raw}


def title_embeddings(target_titles, source_titles, dimensions=64):
    """Fit TF-IDF and SVD on target fit catalog only, then project source."""
    target_titles = [title if isinstance(title, str) else "" for title in target_titles]
    source_titles = [title if isinstance(title, str) else "" for title in source_titles]
    target = np.zeros((len(target_titles), dimensions), np.float32)
    source = np.zeros((len(source_titles), dimensions), np.float32)
    metadata = {"vocabulary_size": 0, "components": 0, "dimensions": dimensions,
                "seed": 2026, "n_iter": 7, "max_features": 30_000, "ngram_range": [1, 2]}
    if not target_titles:
        return target, source, metadata
    vectorizer = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), dtype=np.float32)
    try:
        matrix = vectorizer.fit_transform(target_titles)
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return target, source, metadata
    other = vectorizer.transform(source_titles) if source_titles else None
    metadata["vocabulary_size"] = int(matrix.shape[1])
    if matrix.shape[1] == 1:
        target[:, 0] = matrix.toarray()[:, 0]
        if other is not None:
            source[:, 0] = other.toarray()[:, 0]
        metadata["components"] = 1
    else:
        components = min(dimensions, matrix.shape[1] - 1, max(1, matrix.shape[0] - 1))
        reducer = TruncatedSVD(n_components=components, n_iter=7, random_state=2026)
        reduced = reducer.fit_transform(matrix)
        target[:, :reduced.shape[1]] = reduced
        if other is not None:
            source[:, :reduced.shape[1]] = reducer.transform(other)
        metadata["components"] = int(reduced.shape[1])
    for matrix in (target, source):
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return target, source, metadata


def build_prefix(target_events, source_events, target_titles, source_titles, fit_end,
                 max_target_users=6000):
    """Construct one NPZ dictionary; title mappings contain raw ID -> title."""
    fit_start, query_end = fit_end - 365 * DAY, fit_end + 30 * DAY
    target = build_domain(target_events, fit_start, fit_end, query_end, max_target_users)
    source = build_domain(source_events, fit_start, fit_end, query_end, None, "digital:users")
    tx, sx, encoder = title_embeddings([target_titles.get(i, "") for i in target["item_ids"]],
                                      [source_titles.get(i, "") for i in source["item_ids"]])
    target["x"], source["x"] = tx, sx
    if not len(target["user_ids"]):
        raise ValueError("No eligible target users in fit window")
    mean = target["activity_raw"].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(target["activity_raw"].std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    target["activity"] = ((target["activity_raw"] - mean) / scale).astype(np.float32)
    source["activity"] = ((source["activity_raw"] - mean) / scale).astype(np.float32)
    data = {f"{domain}_{key}": value for domain, arrays in (("target", target), ("source", source))
            for key, value in arrays.items()}
    data.update(fit_start=np.asarray(fit_start, np.int64), fit_end=np.asarray(fit_end, np.int64),
                query_end=np.asarray(query_end, np.int64),
                cutoffs=np.asarray([fit_end + offset * DAY for offset in (0, 30, 60, 90)], np.int64),
                activity_mean=mean, activity_scale=scale,
                tail80=target["tail80"], tail50=target["tail50"], degree=target["degree"],
                encoder_json=np.asarray(json.dumps(encoder, sort_keys=True)))
    return data


def cache_titles(metadata_path, cache_path):
    """Stream metadata once into SQLite; never retain full metadata JSON objects."""
    metadata_path, cache_path = Path(metadata_path), Path(cache_path)
    signature = json.dumps({"path": str(metadata_path.resolve()), "size": metadata_path.stat().st_size,
                            "mtime_ns": metadata_path.stat().st_mtime_ns}, sort_keys=True)
    if cache_path.exists():
        with sqlite3.connect(cache_path) as connection:
            try:
                stored = connection.execute("SELECT value FROM info WHERE key='source'").fetchone()
                if stored and stored[0] == signature:
                    return cache_path
            except sqlite3.DatabaseError:
                pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with sqlite3.connect(temporary) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE titles (item TEXT PRIMARY KEY, title TEXT NOT NULL)")
            connection.execute("CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            opener = gzip.open if metadata_path.suffix == ".gz" else open
            batch = []
            with opener(metadata_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    item = row.get("parent_asin")
                    if item is None:
                        continue
                    title = row.get("title")
                    batch.append((str(item), title if isinstance(title, str) else ""))
                    if len(batch) >= 10_000:
                        connection.executemany("INSERT OR IGNORE INTO titles VALUES (?,?)", batch)
                        batch.clear()
            if batch:
                connection.executemany("INSERT OR IGNORE INTO titles VALUES (?,?)", batch)
            connection.execute("INSERT INTO info VALUES ('source',?)", (signature,))
            connection.commit()
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cache_path


def load_titles(cache_path, item_ids=None):
    """Load only compact ID/title pairs, optionally a bounded catalog subset."""
    with sqlite3.connect(cache_path) as connection:
        if item_ids is None:
            return dict(connection.execute("SELECT item,title FROM titles"))
        titles = {}
        item_ids = list(item_ids)
        for start in range(0, len(item_ids), 500):
            batch = item_ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            titles.update(connection.execute(f"SELECT item,title FROM titles WHERE item IN ({placeholders})", batch))
        return titles
