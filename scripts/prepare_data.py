#!/usr/bin/env python3
"""Build all 18 independent fit365/query30 prefixes, HPO prefix first."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from saet_repro.data import DAY, build_prefix, cache_titles, load_data, load_titles, read_reviews, save_data
from saet_repro.io import sha256, write_json


def origin(year):
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def prefix_order(years, offsets):
    pairs = [(year, offset) for year in years for offset in offsets]
    priority = [(2015, 30)] + [(year, 0) for year in years]
    return [pair for pair in priority if pair in pairs] + [pair for pair in pairs if pair not in priority]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", "--data-dir", dest="output_dir", type=Path,
                        default=ROOT / "data/processed")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/titles")
    parser.add_argument("--years", "--origins", nargs="+", type=int, default=[2015, 2016, 2017])
    parser.add_argument("--offsets", nargs="+", type=int, default=[270, 180, 90, 60, 30, 0])
    parser.add_argument("--wait-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output_dir / "prepare.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another data preparer owns this output directory")
    paths = {"target": args.raw_dir / "CDs_and_Vinyl.csv",
             "source": args.raw_dir / "Digital_Music.csv",
             "target_meta": args.raw_dir / "meta_CDs_and_Vinyl.jsonl.gz",
             "source_meta": args.raw_dir / "meta_Digital_Music.jsonl.gz"}
    while missing := [str(path) for path in paths.values() if not path.exists()]:
        if not args.wait_raw:
            raise FileNotFoundError("Required raw inputs: " + ", ".join(missing))
        print("Waiting for raw inputs: " + ", ".join(missing), flush=True)
        time.sleep(15)
    print("Loading and globally deduplicating review projections once", flush=True)
    latest = max(origin(year) - offset * DAY + 30 * DAY for year in args.years for offset in args.offsets)
    target_events = read_reviews(paths["target"], before=latest)
    source_events = read_reviews(paths["source"], before=latest)
    print(f"Retained deduplicated rows: target={len(target_events):,} source={len(source_events):,}", flush=True)
    print("Building/reusing streamed title caches", flush=True)
    target_cache = cache_titles(paths["target_meta"], args.cache_dir / "CDs_and_Vinyl.sqlite")
    source_cache = cache_titles(paths["source_meta"], args.cache_dir / "Digital_Music.sqlite")
    target_titles = load_titles(target_cache, target_events["parent_asin"].unique())
    source_titles = load_titles(source_cache, source_events["parent_asin"].unique())
    code_hash = sha256(ROOT / "saet_repro/data.py")
    raw_hashes = {key: sha256(path) for key, path in paths.items()}
    for year, offset in prefix_order(args.years, args.offsets):
        name = f"cds_{year}_d{offset}"
        path = args.output_dir / f"{name}.npz"
        manifest_path = path.with_suffix(".json")
        if path.exists() and manifest_path.exists() and not args.force:
            import json
            previous = json.loads(manifest_path.read_text())
            if previous.get("code_sha256") == code_hash and previous.get("raw_sha256") == raw_hashes:
                print(f"Reusing {name}", flush=True)
                continue
        start = time.monotonic()
        print(f"Building {name}", flush=True)
        data = build_prefix(target_events, source_events, target_titles, source_titles,
                            origin(year) - offset * DAY)
        save_data(path, data)
        summary = {"prefix": name, "origin_year": year, "offset_days": offset,
                   "fit_start": int(data["fit_start"]), "fit_end": int(data["fit_end"]),
                   "query_end": int(data["query_end"]), "code_sha256": code_hash,
                   "raw_sha256": raw_hashes, "data_sha256": sha256(path),
                   "elapsed_seconds": time.monotonic() - start,
                   "protocol": "independent fit365/query30; earliest global user-item event; target-only title encoder",
                   "counts": {domain: {"users": len(data[f"{domain}_user_ids"]),
                                         "items": len(data[f"{domain}_item_ids"]),
                                         "fit_positive": len(data[f"{domain}_train"]),
                                         "query_positive": int((data[f"{domain}_positive"][:, 2] == 1).sum())}
                              for domain in ("target", "source")}}
        write_json(manifest_path, summary)
        print(f"Published {name}: {summary['counts']} ({summary['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
