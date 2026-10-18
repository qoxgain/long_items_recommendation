#!/usr/bin/env python3
"""Run and resume the frozen target HPO/final matrix, activity_mean first."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from saet_repro.target import Config, FeatureStore, Model, train_target, METHODS
from saet_repro.evaluation import evaluate, stratified_summaries
from saet_repro.io import write_json, sha256


def load_prefix(path, wait=False):
    while not path.exists():
        if not wait:
            raise FileNotFoundError(path)
        print(f"Waiting for {path}", flush=True)
        time.sleep(15)
    from saet_repro.data import load_data
    return load_data(path)


def provenance(path, config, seed):
    # Written before evaluation; no final query value participates in this hash.
    return {"data_sha256": sha256(path), "target_code_sha256": sha256(ROOT / "saet_repro/target.py"),
            "runner_code_sha256": sha256(Path(__file__)), "config": asdict(config),
            "evaluator_code_sha256": sha256(ROOT / "saet_repro/evaluation.py"),
            "config_sha256": hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
            "seed": seed, "epochs": 6, "selection": "fixed final epoch; no query early stopping",
            "feature_formula": "unit([history, item*history, abs(item-history)])",
            "status": "independent reconstruction; original phi/source unavailable"}


def fit_and_evaluate(data, path, config, seed, model_path, result_path, test_path, feature_cache=False):
    frozen = provenance(path, config, seed)
    if result_path.exists() and test_path.exists() and model_path.exists():
        completed = json.loads(result_path.read_text())
        if completed.get("frozen") == frozen:
            return completed
    frozen_path = result_path.with_suffix(".frozen.json")
    history_path = result_path.with_suffix(".history.json")
    previous_frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    started = time.monotonic()
    if model_path.exists() and history_path.exists() and previous_frozen == frozen:
        model = Model.load(model_path, data)
        history = json.loads(history_path.read_text())
    else:
        write_json(frozen_path, frozen)
        previous_cache = os.environ.pop("SAET_FEATURE_CACHE", None)
        if feature_cache:
            os.environ["SAET_FEATURE_CACHE"] = str(ROOT / "data/cache/target_features" / config.method)
        try:
            def report_epoch(epoch, model):
                print(f"Epoch {epoch}/6 {path.stem}/{config.method}/seed_{seed}", flush=True)
            model, history = train_target(data, config, seed=seed, epochs=6,
                                          validation_callback=report_epoch)
        finally:
            os.environ.pop("SAET_FEATURE_CACHE", None)
            if previous_cache is not None:
                os.environ["SAET_FEATURE_CACHE"] = previous_cache
        model.save(model_path)
        write_json(history_path, history)
    print(f"Fitted {config.method} seed={seed} {path.stem} in {time.monotonic()-started:.1f}s", flush=True)
    result = evaluate(data, 1, model.score_all)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(test_path, **result)
    document = {"method": config.method, "seed": seed, "config": asdict(config),
                "history": history, "summary": {"1": stratified_summaries(data, result)},
                "frozen": frozen, "model_sha256": sha256(model_path),
                "elapsed_seconds": time.monotonic() - started}
    write_json(result_path, document)
    print(f"Scored {path.stem}/{config.method}/{seed}: {document['summary']['1']['tail80'].get('ndcg20')}", flush=True)
    return document


def search(data_dir, result_dir, method, wait):
    path = data_dir / "cds_2015_d30.npz"
    data = load_prefix(path, wait)
    directory = result_dir / path.stem / method
    selected_path = directory / "selected.json"
    if selected_path.exists() and (directory / "selected_model.npz").exists():
        selected = json.loads(selected_path.read_text())
        selected_config = Config(**selected["config"])
        if selected.get("frozen") == provenance(path, selected_config, 101):
            return selected_config
    records = []
    for trial, (lr, rho, tau) in enumerate(itertools.product((.001, .003, .01), (0., .0001), (.1, .3))):
        config = Config(method=method, lr=lr, rho=rho, tau=tau)
        base = directory / "search" / f"trial_{trial:02d}"
        document = fit_and_evaluate(data, path, config, 101, base.with_suffix(".npz"),
                                    base.with_suffix(".json"), base.with_name(base.name + "_test.npz"),
                                    feature_cache=True)
        score = document["summary"]["1"]["tail80"].get("ndcg20")
        records.append({"trial": trial, "score": score, "config": asdict(config)})
    # Deterministic trial-index resolution; no final-origin query is consulted.
    selectable = [r for r in records if r["score"] is not None and np.isfinite(r["score"])]
    if not selectable:
        raise RuntimeError(f"No defined tail80 NDCG@20 for HPO prefix {path}")
    selected = dict(max(selectable, key=lambda r: (r["score"], -r["trial"])))
    selected.update({"selection_prefix": "cds_2015_d30", "selection_metric": "tail80 macro item NDCG@20",
                     "trials": records.copy(), "seed": 101, "epochs": 6,
                     "frozen": provenance(path, Config(**selected["config"]), 101)})
    selected["trials"] = [{k: r[k] for k in ("trial", "score", "config")} for r in records]
    source = directory / "search" / f"trial_{selected['trial']:02d}.npz"
    temporary = directory / "selected_model.npz.tmp"
    shutil.copyfile(source, temporary)
    os.replace(temporary, directory / "selected_model.npz")
    write_json(selected_path, selected)
    cache = ROOT / "data/cache/target_features" / method
    if cache.exists():
        shutil.rmtree(cache)
    return Config(**selected["config"])


def deterministic_baselines(data_dir, result_dir, origins, wait):
    for year in origins:
        path = data_dir / f"cds_{year}_d0.npz"
        data = load_prefix(path, wait)
        store = FeatureStore(data, Config())
        for method in ("cosine", "activity"):
            directory = result_dir / path.stem / method / "seed_0"
            if (directory / "result.json").exists() and (directory / "test.npz").exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "frozen.json", {"method": method, "data_sha256": sha256(path),
                       "target_code_sha256": sha256(ROOT / "saet_repro/target.py")})
            if method == "cosine":
                scorer = lambda ids: np.asarray([store.all_features(int(i))[0] for i in ids])
            else:
                # Raw (unstandardized) log1p 365-day all-rating count. Monotone
                # transform preserves the count ranking, including stable ties.
                values = np.asarray(data["target_activity_raw"])[:, 2]
                scorer = lambda ids: np.broadcast_to(values, (len(ids), len(values)))
            result = evaluate(data, 1, scorer)
            np.savez_compressed(directory / "test.npz", **result)
            write_json(directory / "result.json", {"method": method, "seed": 0,
                       "summary": {"1": stratified_summaries(data, result)}})
            print(f"Scored baseline {year}/{method}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/targets")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["activity_mean", "unbounded_mean", "activity_decay", "activity_conditional", "activity_both"])
    parser.add_argument("--origins", nargs="+", type=int, default=[2015, 2016, 2017])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--wait-data", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    # A second accidental launch must not race resumable model/artifact writes.
    import fcntl
    args.result_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.result_dir / "runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another target runner owns this result directory")
    lock.write(str(os.getpid()))
    lock.flush()
    for method in args.methods:
        config = search(args.data_dir, args.result_dir, method, args.wait_data)
        if args.hpo_only:
            continue
        for year in args.origins:
            path = args.data_dir / f"cds_{year}_d0.npz"
            data = load_prefix(path, args.wait_data)
            for seed in args.seeds:
                directory = args.result_dir / path.stem / method / f"seed_{seed}"
                fit_and_evaluate(data, path, config, seed, directory / "model.npz",
                                 directory / "result.json", directory / "test.npz")
    if not args.hpo_only and not args.skip_baselines:
        deterministic_baselines(args.data_dir, args.result_dir, args.origins, args.wait_data)


if __name__ == "__main__":
    main()
