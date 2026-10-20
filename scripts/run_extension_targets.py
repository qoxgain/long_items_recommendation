#!/usr/bin/env python3
"""Frozen, resumable B1 convergence matrix for the two direct target families."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = int(os.environ.get("SAET_EXTENSION_THREADS", "1"))
if THREADS != 1:
    raise SystemExit("extensions_v1 freezes one numerical thread per worker")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(THREADS)

import numpy as np
from saet_repro.data import load_data
from saet_repro.evaluation import stratified_summaries
from saet_repro.extension_target import (METHODS, SELECTION_SEEDS, FINAL_SEEDS, CHECKPOINTS,
    HISTORICAL_CACHE_MAX_BYTES, Trajectory, code_identity, evaluate_models, extension_decision, grid, peak_memory_mib,
    select_checkpoint, selection_records, train_trajectories)
from saet_repro.io import sha256, write_json
from saet_repro.target import Config

ORIGINS = (2015, 2016, 2017)
HISTORICAL_PREFIX = "cds_2015_d30"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cgroup_resource_limits():
    result = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    cpu = result["cpu.max"]
    result["cpu_quota_cores"] = (float(cpu.split()[0]) / float(cpu.split()[1])
                                  if cpu and cpu.split()[0] != "max" else None)
    memory = result["memory.max"]
    result["memory_limit_gib"] = (int(memory) / 1024 ** 3 if memory and memory != "max" else None)
    return result


def freeze_protocol(args):
    """Publish and verify protocol/code identities before opening any data NPZ."""
    global_protocol = json.loads(args.protocol.read_text())
    stage = global_protocol["stage_b"]
    expected = {"hpo_seeds": list(SELECTION_SEEDS), "checkpoint_epochs": [6, 20, 50],
                "hpo_prefix": HISTORICAL_PREFIX, "final_origins": list(ORIGINS)}
    for key, value in expected.items():
        if stage[key] != value:
            raise ValueError(f"Global protocol disagrees with target implementation: {key}")
    rule = stage["extension_rule"]
    if (rule["near_zero_denominator"] != 1e-8 or rule["relative_threshold"] != .01
            or rule["absolute_threshold_when_near_zero"] != 1e-4
            or rule["initial_max_epoch"] != 50 or rule["extended_max_epoch"] != 100
            or not rule["extend_all_family_candidates"]):
        raise ValueError("Global extension rule disagrees with target implementation")
    if global_protocol["seeds"] != list(FINAL_SEEDS):
        raise ValueError("Global final seeds disagree with target implementation")
    baseline = global_protocol["baseline_sha256"]
    for name in ("saet_repro/target.py", "saet_repro/evaluation.py", "saet_repro/data.py", "saet_repro/io.py"):
        if sha256(ROOT / name) != baseline[name]:
            raise ValueError(f"Frozen original implementation changed: {name}")
    prefixes = [HISTORICAL_PREFIX] + [f"cds_{year}_d0" for year in ORIGINS]
    protocol = {"schema": "extension-target-protocol-v1", "global_protocol_sha256": sha256(args.protocol),
        "identity": code_identity(), "stage_b": stage, "methods": list(METHODS),
        "grids": {method: [asdict(config) for config in grid(method)] for method in METHODS},
        "final_seeds": list(FINAL_SEEDS), "checkpoints": list(CHECKPOINTS),
        "data_sha256": {prefix: sha256(args.data_dir / f"{prefix}.npz") for prefix in prefixes},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
        "threads_per_worker": THREADS,
        "cache_policy": {"training": "one transient epoch bank per family/seed",
            "historical_evaluation": "deterministic first-fit RAM cache of unchanged fit-only anchor/phi; shared activity; per-item fallback without eviction",
            "historical_max_bytes": HISTORICAL_CACHE_MAX_BYTES,
            "final_evaluation": "one uncached pass shared by all final seeds", "disk_cache_bytes": 0},
        "sampling": "unchanged target.py sampling, Cartesian within-item pairs and Adam update order",
        "history_policy": "every historical epoch evaluated; only 6/20/50 and conditionally100 selectable; final queries evaluated after fixed training",
        "tie_break": "maximum three-seed mean, fewer epochs, earlier frozen grid index",
        "visibility": "exploratory development analysis; original three origins already used for development"}
    path = args.result_dir / "target_protocol.json"
    sidecar = path.with_suffix(".sha256.json")
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Target protocol/code/data changed: use a new versioned output root; existing results remain intact")
        if not sidecar.exists() or json.loads(sidecar.read_text())["sha256"] != sha256(path):
            raise ValueError("Target protocol checksum missing or invalid")
    else:
        write_json(path, protocol)
        write_json(sidecar, {"sha256": sha256(path), "frozen_utc": utc_now()})
    return protocol, sha256(path)


def run_identity(protocol, protocol_hash, prefix, phase, selected_hash=None):
    identity = {"target_protocol_sha256": protocol_hash,
        "global_protocol_sha256": protocol["global_protocol_sha256"],
        "data_file_sha256": protocol["data_sha256"][prefix], "prefix": prefix, "phase": phase}
    if selected_hash is not None:
        identity["selection_sha256"] = selected_hash
    return identity


def verified_data(args, protocol, prefix):
    path = args.data_dir / f"{prefix}.npz"
    if sha256(path) != protocol["data_sha256"][prefix]:
        raise ValueError(f"Data changed after freeze: {prefix}")
    return load_data(path)


def artifacts_manifest(directory, names):
    return {name: sha256(directory / name) for name in names}


def verified_artifacts(directory, mapping):
    try:
        return bool(mapping) and all(sha256(directory / name) == value for name, value in mapping.items())
    except OSError:
        return False


def completed_selection(directory, protocol_hash):
    path, complete_path = directory / "selected.json", directory / "selected.complete.json"
    if not path.exists() or not complete_path.exists():
        return None
    try:
        document = json.loads(path.read_text())
        complete = json.loads(complete_path.read_text())
        if (document["target_protocol_sha256"] == protocol_hash
                and complete["selected_sha256"] == sha256(path)
                and verified_artifacts(directory, document["evidence_sha256"])):
            return document
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def report_epoch(method, phase):
    def callback(epoch, active):
        scores = [state.history[-1].get("historical_tail80_ndcg20") for state in active]
        finite = [score for score in scores if score is not None]
        print(json.dumps({"method": method, "phase": phase, "epoch": epoch,
            "active_trajectories": len(active), "historical_mean": float(np.mean(finite)) if finite else None,
            "peak_rss_mib": peak_memory_mib(), "utc": utc_now()}), flush=True)
    return callback


def search(args, protocol, protocol_hash, method):
    directory = args.result_dir / method
    existing = completed_selection(directory, protocol_hash)
    if existing is not None:
        return existing
    data = verified_data(args, protocol, HISTORICAL_PREFIX)
    specs = [Trajectory(config, seed, directory / "hpo" / f"trial_{trial:02d}" / f"seed_{seed}")
             for trial, config in enumerate(grid(method)) for seed in SELECTION_SEEDS]
    identity = run_identity(protocol, protocol_hash, HISTORICAL_PREFIX, "historical_selection")
    started = time.monotonic()
    states = train_trajectories(data, specs, 50, identity=identity, evaluate_each_epoch=True,
                                callback=report_epoch(method, "hpo"))
    records = selection_records(states)
    decision = extension_decision(records)
    write_json(directory / "extension_decision.json", {**decision, "target_protocol_sha256": protocol_hash})
    if decision["extend_to_100"]:
        states = train_trajectories(data, specs, 100, identity=identity, evaluate_each_epoch=True,
                                    callback=report_epoch(method, "hpo_extended"))
        records = selection_records(states, checkpoints=CHECKPOINTS)
    selected = select_checkpoint(records)
    evidence = {}
    for state in states:
        relative = state.directory.relative_to(directory)
        names = ["last_state.npz", "last_state.sha256.json", "history.json", "frozen.json"]
        for epoch in (CHECKPOINTS if decision["extend_to_100"] else CHECKPOINTS[:3]):
            names.extend([f"checkpoint_{epoch:03d}.npz", f"checkpoint_{epoch:03d}.json"])
        for name, digest in artifacts_manifest(state.directory, names).items():
            evidence[str(relative / name)] = digest
        write_json(state.directory / "complete.json", {"target_protocol_sha256": protocol_hash,
            "identity_sha256": state.identity_sha256, "epochs": state.epoch,
            "artifacts": artifacts_manifest(state.directory, names)})
    selected.update({"method": method, "target_protocol_sha256": protocol_hash,
        "global_protocol_sha256": protocol["global_protocol_sha256"],
        "selection_prefix": HISTORICAL_PREFIX, "selection_seeds": list(SELECTION_SEEDS),
        "selection_metric": "tail80 macro item NDCG@20", "selection_visibility": "exploratory development",
        "trials": records, "extension_decision": decision, "evidence_sha256": evidence,
        "invocation_elapsed_seconds": time.monotonic() - started, "peak_rss_mib": peak_memory_mib()})
    write_json(directory / "selected.json", selected)
    write_json(directory / "selected.complete.json", {"selected_sha256": sha256(directory / "selected.json"),
        "target_protocol_sha256": protocol_hash, "complete_utc": utc_now()})
    return selected


def final_complete(directory, identity):
    try:
        complete = json.loads((directory / "complete.json").read_text())
        return complete["identity"] == identity and verified_artifacts(directory, complete["artifacts"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_final(args, protocol, protocol_hash, method, selected):
    selected_hash = sha256(args.result_dir / method / "selected.json")
    config = Config(**selected["config"])
    epochs = int(selected["epochs"])
    for year in ORIGINS:
        prefix = f"cds_{year}_d0"
        identity = run_identity(protocol, protocol_hash, prefix, "final", selected_hash)
        specs = [Trajectory(config, seed, args.result_dir / method / "final" / prefix / f"seed_{seed}")
                 for seed in FINAL_SEEDS]
        pending = [spec for spec in specs if not final_complete(spec.directory, {**identity, "seed": spec.seed,
                    "config": asdict(config), "epochs": epochs})]
        if not pending:
            continue
        data = verified_data(args, protocol, prefix)
        started = time.monotonic()
        states = train_trajectories(data, pending, epochs, identity=identity, evaluate_each_epoch=False,
                                    callback=report_epoch(method, prefix))
        # The gate precedes the first final evaluation. No final metric selects epochs/configuration.
        write_json(args.result_dir / method / "final" / prefix / "final_query_opened.json",
                   {**identity, "opened_utc": utc_now(), "selected_epochs": epochs, "config": asdict(config)})
        begin = time.monotonic()
        results = evaluate_models(data, [state.model for state in states])
        evaluation_seconds = time.monotonic() - begin
        for state, result in zip(states, results):
            directory = state.directory
            state.model.save(directory / "model.npz")
            temporary = directory / "test.npz.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **result)
            os.replace(temporary, directory / "test.npz")
            document = {"method": method, "seed": state.spec.seed, "prefix": prefix, "epochs": epochs,
                "config": asdict(config), "history": state.history, "frozen": state.identity,
                "summary": {"1": stratified_summaries(data, result)},
                "model_sha256": sha256(directory / "model.npz"), "test_sha256": sha256(directory / "test.npz"),
                "elapsed_seconds_shared": time.monotonic() - started,
                "evaluation_seconds_shared": evaluation_seconds, "shared_trajectories": len(states),
                "peak_rss_mib": peak_memory_mib(), "visibility": "exploratory development analysis"}
            write_json(directory / "result.json", document)
            write_json(directory / "complete.json", {"identity": {**identity, "seed": state.spec.seed,
                "config": asdict(config), "epochs": epochs}, "complete_utc": utc_now(),
                "artifacts": artifacts_manifest(directory, ["model.npz", "test.npz", "result.json",
                    "last_state.npz", "last_state.sha256.json", "history.json", "frozen.json"])})
            print(json.dumps({"method": method, "prefix": prefix, "seed": state.spec.seed,
                "tail80_ndcg20": document["summary"]["1"]["tail80"]["ndcg20"]}), flush=True)


def pilot(args, protocol, protocol_hash, method):
    data = verified_data(args, protocol, HISTORICAL_PREFIX)
    directory = args.result_dir / "target_pilot" / method
    specs = [Trajectory(config, 101, directory / f"trial_{trial:02d}" / "seed_101")
             for trial, config in enumerate(grid(method))]
    started = time.monotonic()
    states = train_trajectories(data, specs, 2,
        identity=run_identity(protocol, protocol_hash, HISTORICAL_PREFIX, "pilot"),
        evaluate_each_epoch=True, checkpoints=(1, 2), callback=report_epoch(method, "pilot"))
    elapsed = time.monotonic() - started
    timings = []
    for epoch in range(2):
        rows = [state.history[epoch] for state in states]
        timings.append({"epoch": epoch + 1,
            "feature_bank_seconds": rows[0]["feature_bank_seconds_shared"],
            "training_seconds_all12": sum(row["training_seconds"] for row in rows),
            "historical_evaluation_seconds_all12": rows[0]["evaluation_seconds_shared"],
            "historical_feature_cache_bytes": rows[0]["historical_feature_cache_bytes"],
            "historical_feature_cache_items": rows[0]["historical_feature_cache_items"]})
    cold, warm = timings
    bank, train, evaluation = (warm[name] for name in ("feature_bank_seconds", "training_seconds_all12",
                                                      "historical_evaluation_seconds_all12"))
    # Conservative 3x twelve-model evaluation; actual HPO shares one cache/pass.
    cold_cost = sum(cold[name] for name in ("feature_bank_seconds", "training_seconds_all12",
                                          "historical_evaluation_seconds_all12"))
    estimate = {str(epoch): {"hpo_seconds": 3 * (cold_cost + (epoch - 1) * (bank + train + evaluation)),
        "final_seconds": 3 * (5 * (cold["feature_bank_seconds"] + (epoch - 1) * bank + epoch * train / 12)
                              + cold["historical_evaluation_seconds_all12"])}
        for epoch in CHECKPOINTS}
    document = {"method": method, "target_protocol_sha256": protocol_hash,
        "prefix": HISTORICAL_PREFIX, "trajectories": 12, "epochs": 2, "seed": 101,
        "invocation_seconds": elapsed, "epoch_timings": timings,
        "peak_rss_mib": peak_memory_mib(), "cpu_count_host": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)), "platform": platform.platform(),
        "cgroup_limits": cgroup_resource_limits(),
        "threads": THREADS, "estimated_family_budget": estimate,
        "estimate_limitations": "one cold and one warm historical epoch; final catalog size differs; HPO conservatively charges3x shared12 evaluation although36 models share one cache/pass; checkpoint IO excluded",
        "disk_cache_bytes": 0, "completed_utc": utc_now()}
    write_json(directory / "pilot.json", document)
    print(json.dumps(document), flush=True)


def task_list(methods):
    tasks = []
    for method in methods:
        for trial, config in enumerate(grid(method)):
            for seed in SELECTION_SEEDS:
                tasks.append({"phase": "hpo", "method": method, "trial": trial, "seed": seed,
                    "config": asdict(config), "checkpoints": [6, 20, 50], "conditional_checkpoint": 100,
                    "path": f"{method}/hpo/trial_{trial:02d}/seed_{seed}"})
        for year in ORIGINS:
            for seed in FINAL_SEEDS:
                tasks.append({"phase": "final", "method": method, "prefix": f"cds_{year}_d0", "seed": seed,
                    "configuration_and_epochs": "selected.json", "path": f"{method}/final/cds_{year}_d0/seed_{seed}"})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results_extensions/v1/stage_b")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions_v1.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--hpo-only", action="store_true")
    modes.add_argument("--final-only", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="retry previously recorded failed family phases; verified epoch state resumes")
    args = parser.parse_args()
    if args.list_tasks:
        tasks = task_list(args.methods)
        print(json.dumps({"tasks": tasks, "hpo": sum(t["phase"] == "hpo" for t in tasks),
            "final": sum(t["phase"] == "final" for t in tasks)}, indent=2))
        return 0
    args.result_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.result_dir / "target_runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another extension target runner owns this result directory")
    lock.write(str(os.getpid()))
    lock.flush()
    protocol, protocol_hash = freeze_protocol(args)
    failed = []
    for method in args.methods:
        phase = "pilot" if args.pilot else "final" if args.final_only else "hpo" if args.hpo_only else "full"
        failure_path = args.result_dir / method / f"failed_{phase}.json"
        if failure_path.exists() and not args.retry_failed:
            print(f"Unresolved failure {failure_path}; use --retry-failed", file=sys.stderr)
            failed.append(method)
            continue
        try:
            if args.pilot:
                pilot(args, protocol, protocol_hash, method)
            else:
                selected = (completed_selection(args.result_dir / method, protocol_hash) if args.final_only
                            else search(args, protocol, protocol_hash, method))
                if selected is None:
                    raise ValueError(f"No fully verified selection for {method}; complete HPO first")
                if not args.hpo_only:
                    run_final(args, protocol, protocol_hash, method, selected)
            if failure_path.exists():
                os.replace(failure_path, failure_path.with_name(f"recovered_{phase}_{time.time_ns()}.json"))
        except Exception as error:
            document = {"method": method, "phase": phase, "target_protocol_sha256": protocol_hash,
                "failed_utc": utc_now(), "error": repr(error), "traceback": traceback.format_exc()}
            write_json(failure_path, document)
            write_json(failure_path.with_name(f"failure_{phase}_{time.time_ns()}.json"), document)
            traceback.print_exc()
            failed.append(method)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
