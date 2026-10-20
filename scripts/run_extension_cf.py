#!/usr/bin/env python3
"""Run frozen B1/B2 CF convergence and task-direction experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from saet_repro.collaborative import CFConfig
from saet_repro.data import load_data
from saet_repro.evaluation import evaluate, stratified_summaries
from saet_repro.extension_cf import (CHECKPOINTS, FINAL_SEEDS, METHODS, SELECTION_SEEDS,
                                    code_identity, extension_decision, grid, json_hash,
                                    peak_memory_mib, select_checkpoint, selection_records,
                                    train_trajectory)
from saet_repro.io import json_safe, sha256, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_global_protocol(path):
    value = json.loads(Path(path).read_text())
    stage = value["stage_b"]
    expected = {"hpo_prefix": "cds_2015_d30", "hpo_seeds": list(SELECTION_SEEDS),
                "checkpoint_epochs": [6, 20, 50], "final_origins": [2015, 2016, 2017]}
    for key, expected_value in expected.items():
        if stage[key] != expected_value:
            raise ValueError(f"Unsupported frozen stage_b.{key}: {stage[key]!r}")
    rule = stage["extension_rule"]
    expected_rule = {"near_zero_denominator": 1e-8, "relative_threshold": .01,
                     "absolute_threshold_when_near_zero": 1e-4,
                     "initial_max_epoch": 50, "extended_max_epoch": 100,
                     "extend_all_family_candidates": True}
    if any(rule[key] != item for key, item in expected_rule.items()):
        raise ValueError("Global extension rule differs from implemented protocol")
    if value["seeds"] != list(FINAL_SEEDS) or not set(METHODS).issubset(stage["methods"]):
        raise ValueError("Global methods/final seeds differ from implemented protocol")
    if value["execution"]["threads_per_worker"] != 1:
        raise ValueError("This runner freezes one CPU thread per worker")
    return value


def freeze_family(root, method, protocol_path, protocol, data_path):
    """Freeze code, dependencies, complete grid and data bytes before evaluation."""
    content = {**code_identity(), "method": method,
               "global_protocol_path": str(Path(protocol_path).resolve()),
               "global_protocol_sha256": sha256(protocol_path),
               "grid": [asdict(config) for config in grid(method)],
               "hpo_prefix": "cds_2015_d30", "hpo_data_sha256": sha256(data_path),
               "hpo_seeds": list(SELECTION_SEEDS), "checkpoint_epochs": list(CHECKPOINTS),
               "initial_max_epoch": 50, "extension_rule": protocol["stage_b"]["extension_rule"],
               "near_zero_comparison": "abs(epoch20_score) < 1e-8",
               "selection": "maximum three-seed mean, ties fewer epochs then lower trial index",
               "final_origins": [2015, 2016, 2017], "final_seeds": list(FINAL_SEEDS),
               "query_stage": 1, "primary_endpoint": "tail80_ndcg20",
               "candidate_mode": "full_eligible_users_excluding_any_prior_rating",
               "visibility": "exploratory development analysis",
               "objective": "fixed_item_positive_vs_negative_user_bpr" if method == "lightgcn_user"
                            else "unchanged_native_user_to_item_objective",
               "resource_policy": {"device": "cpu", "threads_per_worker": 1,
                                   "cpu": platform.processor(), "machine": platform.machine(),
                                   "peak_rss_scope": "process high-water mark"},
               "checkpoint_policy": "milestone models at6/20/50/100; one atomic replaceable Adam/RNG state each epoch"}
    path = root / method / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if (old["protocol"] != content or old["protocol_sha256"] != json_hash(content)
                or json.loads(path.with_suffix(".sha256.json").read_text())["sha256"] != sha256(path)):
            raise ValueError(f"Family protocol changed: {path}. Use a new versioned output directory.")
    else:
        write_json(path, {"protocol": content, "protocol_sha256": json_hash(content),
                          "frozen_before_historical_query_utc": utc_now()})
        write_json(path.with_suffix(".sha256.json"), {"sha256": sha256(path)})
    return {"family_protocol_sha256": sha256(path),
            "global_protocol_sha256": sha256(protocol_path)}


def required_artifacts(identity, epochs):
    base = {"history.json", "last_state.npz", "last_state.sha256.json", "frozen.json"}
    if identity["role"] == "final":
        return base | {"model.npz", "test.npz", "result.json", "final_query_opened.json"}
    return base | {"runtime.json"} | {f"checkpoint_{epoch:03d}.npz"
                                     for epoch in CHECKPOINTS if epoch <= epochs}


def verify_complete(directory, identity, required_epochs):
    """A marker alone is insufficient: verify every registered artifact."""
    path = directory / "complete.json"
    if not path.exists():
        return None
    try:
        complete = json.loads(path.read_text())
        if (complete["run_identity"] != json_safe(identity) or complete["epochs"] < required_epochs
                or complete["status"] != "complete"
                or set(complete["artifact_sha256"]) != required_artifacts(identity, complete["epochs"])):
            return None
        for name, digest in complete["artifact_sha256"].items():
            if sha256(directory / name) != digest:
                return None
        return complete
    except (OSError, KeyError, ValueError, TypeError):
        return None


def mark_complete(directory, identity, epochs, names):
    result = {"run_identity": identity, "epochs": epochs,
              "artifact_sha256": {name: sha256(directory / name) for name in names},
              "completed_utc": utc_now(), "status": "complete"}
    write_json(directory / "complete.json", result)
    if (directory / "failed.json").exists():
        (directory / "failed.json").replace(directory / f"failure_resolved_{time.time_ns()}.json")
    return result


def run_historical(data, data_path, method, trial, seed, epochs, directory, protocol_identity,
                   retry_failed=False, pilot=False):
    config = grid(method)[trial]
    identity = {**protocol_identity, **code_identity(), "role": "pilot" if pilot else "hpo",
                "trial": trial, "method": method, "config": asdict(config), "seed": seed,
                "data_sha256": sha256(data_path), "prefix": data_path.stem,
                "fit_start": int(data["fit_start"]), "fit_end": int(data["fit_end"]),
                "query_end": int(data["query_end"]), "query_stage": 1}
    complete = verify_complete(directory, identity, epochs)
    if complete is not None:
        history = json.loads((directory / "history.json").read_text())
        print(f"resume complete {method} trial={trial:02d} seed={seed} epochs={complete['epochs']}", flush=True)
    else:
        if (directory / "failed.json").exists() and not retry_failed:
            raise RuntimeError(f"Previous failure retained at {directory}; pass --retry-failed to retry")
        started = time.monotonic()
        state = train_trajectory(data, config, method, seed, epochs, identity=identity,
                                 directory=directory, evaluate_each_epoch=True,
                                 callback=lambda row: print(
                                     f"{method} trial={trial:02d} seed={seed} epoch={row['epoch']} "
                                     f"loss={row['loss']:.6f} historical={row['historical_tail80_ndcg20']:.8f} "
                                     f"train_s={row['training_seconds']:.2f}", flush=True))
        history = state.history
        write_json(directory / "runtime.json", {"invocation_seconds": time.monotonic() - started,
                   "training_seconds": sum(row["training_seconds"] for row in history),
                   "evaluation_seconds": sum(row.get("evaluation_seconds", 0.) for row in history),
                   "peak_rss_mib": peak_memory_mib(), "threads": torch.get_num_threads(),
                   "epochs": state.epoch, "positive_edges": len(state.edges),
                   "cumulative_updates": state.step, "data_sha256": identity["data_sha256"]})
        names = ["history.json", "last_state.npz", "last_state.sha256.json", "frozen.json",
                 "runtime.json", *state.checkpoint_hashes]
        complete = mark_complete(directory, identity, state.epoch, names)
        del state
        gc.collect()
    return {"trial": trial, "seed": seed, "config": asdict(config), "history": history,
            "complete_sha256": sha256(directory / "complete.json")}


def preserve_failure(directory, error):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "failed.json").exists():
        (directory / "failed.json").replace(directory / f"failure_previous_{time.time_ns()}.json")
    write_json(directory / "failed.json", {"error": str(error), "type": type(error).__name__,
               "traceback": traceback.format_exc(), "failed_utc": utc_now()})


def run_hpo(data, data_path, root, method, protocol_identity, retry_failed=False):
    family = root / method

    def run_all(epochs):
        trajectories, failed = [], []
        for trial in range(12):
            for seed in SELECTION_SEEDS:
                directory = family / "hpo" / f"trial_{trial:02d}" / f"seed_{seed}"
                try:
                    trajectories.append(run_historical(data, data_path, method, trial, seed, epochs,
                                                        directory, protocol_identity, retry_failed))
                except Exception as error:
                    preserve_failure(directory, error)
                    failed.append({"trial": trial, "seed": seed, "error": str(error)})
                    print(f"FAILED {method} trial={trial} seed={seed}: {error}", flush=True)
        write_json(family / "hpo_status.json", {"required": 36, "complete": len(trajectories),
                   "epochs": epochs, "failed": failed, "complete_matrix": not failed})
        if failed:
            raise RuntimeError(f"{method}: {len(failed)} HPO trajectories failed; no selection produced")
        return trajectories

    trajectories = run_all(50)
    records = selection_records(trajectories)
    decision = extension_decision(records)
    write_json(family / "extension_decision.json", {**decision, **protocol_identity,
                                                  "initial_records": records})
    if decision["extend_to_100"]:
        trajectories = run_all(100)
        records = selection_records(trajectories, checkpoints=CHECKPOINTS)
    best = select_checkpoint(records)
    selected = {"method": method, **best, "criterion": "2015_d30_stage1_tail80_ndcg20_three_seed_mean",
                "criterion_value": best["score"], "trials": records,
                "hpo_seeds": list(SELECTION_SEEDS), "final_seeds": list(FINAL_SEEDS),
                "final_origins": [2015, 2016, 2017], "visibility": "exploratory development analysis",
                "extension_decision": decision, **protocol_identity, **code_identity(),
                "hpo_data_sha256": sha256(data_path), "no_final_validation_or_epoch_selection": True,
                "trajectory_complete_sha256": {f"trial_{row['trial']:02d}/seed_{row['seed']}": row["complete_sha256"]
                                                for row in trajectories}}
    selected_path = family / "selected.json"
    if selected_path.exists() and json.loads(selected_path.read_text()) != json_safe(selected):
        # Keep previous selection provenance; completed final caches key this hash.
        selected_path.replace(selected_path.with_name(f"selected.stale_{time.time_ns()}.json"))
    write_json(selected_path, selected)
    write_json(family / "selected.sha256.json", {"sha256": sha256(selected_path)})
    print(f"selected {method}: trial={best['trial']} epochs={best['epochs']} score={best['score']:.8f}", flush=True)
    return selected


def validate_selection(root, method, protocol_identity):
    family = root / method
    path = family / "selected.json"
    selected = json.loads(path.read_text())
    if json.loads((family / "selected.sha256.json").read_text())["sha256"] != sha256(path):
        raise ValueError("Selected configuration checksum mismatch")
    current = {**protocol_identity, **code_identity()}
    if any(selected.get(key) != value for key, value in current.items()):
        raise ValueError("Selected configuration identity is stale; rerun complete HPO")
    expected_names = {f"trial_{trial:02d}/seed_{seed}"
                      for trial in range(12) for seed in SELECTION_SEEDS}
    if set(selected["trajectory_complete_sha256"]) != expected_names:
        raise ValueError("Selection lacks complete three-seed HPO matrix")
    trajectories = []
    for name, digest in selected["trajectory_complete_sha256"].items():
        complete_path = family / "hpo" / name / "complete.json"
        if sha256(complete_path) != digest:
            raise ValueError(f"Historical completion checksum changed: {complete_path}")
        completion = json.loads(complete_path.read_text())
        trial_text, seed_text = name.split("/")
        trial, seed = int(trial_text.removeprefix("trial_")), int(seed_text.removeprefix("seed_"))
        identity = completion["run_identity"]
        expected = {**current, "method": method, "trial": trial, "seed": seed, "role": "hpo",
                    "config": asdict(grid(method)[trial]), "data_sha256": selected["hpo_data_sha256"]}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Historical completion identity differs: {complete_path}")
        budget = 100 if selected["extension_decision"]["extend_to_100"] else 50
        if verify_complete(complete_path.parent, identity, budget) is None:
            raise ValueError(f"Historical artifact receipt incomplete or changed: {complete_path}")
        trajectories.append({"trial": trial, "seed": seed, "config": identity["config"],
                             "history": json.loads((complete_path.parent / "history.json").read_text())})
    initial_records = selection_records(trajectories)
    decision = extension_decision(initial_records)
    records = selection_records(trajectories, checkpoints=CHECKPOINTS) if decision["extend_to_100"] else initial_records
    best = select_checkpoint(records)
    if (selected["extension_decision"] != decision or selected["trials"] != records
            or any(selected.get(key) != value for key, value in best.items())
            or selected["criterion_value"] != best["score"]):
        raise ValueError("Selection does not reproduce the complete historical grid")
    return selected


def run_final(data, data_path, root, method, seed, selected, protocol_identity, retry_failed=False):
    directory = root / method / "final" / data_path.stem / f"seed_{seed}"
    config = CFConfig(**selected["config"])
    identity = {**protocol_identity, **code_identity(), "role": "final",
                "method": method, "config": asdict(config), "seed": seed,
                "epochs": selected["epochs"], "prefix": data_path.stem,
                "data_sha256": sha256(data_path),
                "selected_sha256": sha256(root / method / "selected.json"),
                "fit_start": int(data["fit_start"]), "fit_end": int(data["fit_end"]),
                "query_end": int(data["query_end"]), "query_stage": 1,
                "validation": "none", "visibility": "exploratory development analysis"}
    if verify_complete(directory, identity, selected["epochs"]) is not None:
        print(f"resume complete {method} {data_path.stem} seed={seed}", flush=True)
        return json.loads((directory / "result.json").read_text())
    if (directory / "failed.json").exists() and not retry_failed:
        raise RuntimeError(f"Previous failure retained at {directory}; pass --retry-failed to retry")
    started = time.monotonic()
    state = train_trajectory(data, config, method, seed, selected["epochs"], identity=identity,
                             directory=directory, evaluate_each_epoch=False, checkpoints=(),
                             callback=lambda row: print(f"final {method} {data_path.stem} seed={seed} "
                                                        f"epoch={row['epoch']} loss={row['loss']:.6f}", flush=True))
    state.model.save(directory / "model.npz")
    # This immutable-choice receipt precedes opening this run's final scores.
    write_json(directory / "final_query_opened.json", {"identity": identity,
               "model_sha256": sha256(directory / "model.npz"), "opened_utc": utc_now()})
    evaluated = evaluate(data, 1, state.model.score_all, ks=(10, 20, 50), save_topk=True)
    temporary = directory / "test.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **evaluated)
    temporary.replace(directory / "test.npz")
    report = {"method": method, "config": asdict(config), "seed": seed,
              "epochs": selected["epochs"], "prefix": data_path.stem,
              "summary": {"1": stratified_summaries(data, evaluated)},
              "training": state.history, "seconds": time.monotonic() - started,
              "peak_rss_mib": peak_memory_mib(), "identity": identity,
              "model_sha256": sha256(directory / "model.npz"),
              "evaluation_sha256": sha256(directory / "test.npz"),
              "cumulative_updates": state.step,
              "visibility": "exploratory development analysis"}
    write_json(directory / "result.json", report)
    mark_complete(directory, identity, selected["epochs"], ["model.npz", "test.npz", "result.json",
                  "history.json", "frozen.json", "last_state.npz", "last_state.sha256.json",
                  "final_query_opened.json"])
    del state
    gc.collect()
    return json_safe(report)


def task_manifest(methods):
    return {"hpo": [{"method": method, "trial": trial, "seed": seed, "initial_epochs": 50,
                      "conditional_max_epochs": 100}
                     for method in methods for trial in range(12) for seed in SELECTION_SEEDS],
            "final": [{"method": method, "prefix": f"cds_{year}_d0", "seed": seed}
                       for method in methods for year in (2015, 2016, 2017) for seed in FINAL_SEEDS],
            "selection_checkpoints": list(CHECKPOINTS), "visibility": "exploratory development analysis"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--results-dir", "--output-root", type=Path, default=ROOT / "results_extensions/v1/stage_b")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions_v1.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--list-tasks", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--hpo-only", action="store_true")
    mode.add_argument("--final-only", action="store_true")
    parser.add_argument("--pilot-epochs", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.list_tasks:
        print(json.dumps(task_manifest(args.methods), indent=2))
        return 0
    if args.pilot_epochs < 1:
        parser.error("--pilot-epochs must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = validate_global_protocol(args.protocol)
    historical_path = args.data_dir / "cds_2015_d30.npz"
    failures = []
    for method in args.methods:
        try:
            identity = freeze_family(args.results_dir, method, args.protocol, protocol, historical_path)
            if args.pilot:
                data = load_data(historical_path)
                directory = args.results_dir / method / "pilot" / "seed_101"
                begun = time.monotonic()
                trajectory = run_historical(data, historical_path, method, 0, 101, args.pilot_epochs,
                                             directory, identity, args.retry_failed, pilot=True)
                history = trajectory["history"]
                estimate = {"method": method, "pilot_epochs": args.pilot_epochs,
                            "seconds": time.monotonic() - begun, "peak_rss_mib": peak_memory_mib(),
                            "mean_training_seconds_per_epoch": float(np.mean([r["training_seconds"] for r in history])),
                            "mean_historical_evaluation_seconds_per_epoch": float(np.mean([r["evaluation_seconds"] for r in history])),
                            "initial_hpo_trajectories": 36, "initial_hpo_epochs": 1800,
                            "note": "One first-grid pilot; extrapolation does not capture dimension/layer effects or contention"}
                write_json(directory / "pilot_estimate.json", estimate)
                print(json.dumps(estimate, sort_keys=True), flush=True)
                continue
            if not args.final_only:
                data = load_data(historical_path)
                run_hpo(data, historical_path, args.results_dir, method, identity, args.retry_failed)
                del data
                gc.collect()
            if args.hpo_only:
                continue
            selected = validate_selection(args.results_dir, method, identity)
            for year in (2015, 2016, 2017):
                path = args.data_dir / f"cds_{year}_d0.npz"
                # The choice and all HPO artifacts were checked before this load.
                data = load_data(path)
                for seed in FINAL_SEEDS:
                    directory = args.results_dir / method / "final" / path.stem / f"seed_{seed}"
                    try:
                        run_final(data, path, args.results_dir, method, seed, selected, identity, args.retry_failed)
                    except Exception as error:
                        preserve_failure(directory, error)
                        failures.append({"method": method, "prefix": path.stem, "seed": seed,
                                         "error": str(error)})
                del data
                gc.collect()
        except Exception as error:
            failures.append({"method": method, "error": str(error)})
            write_json(args.results_dir / method / "runner_failure.json", {"error": str(error),
                       "traceback": traceback.format_exc(), "failed_utc": utc_now()})
            print(f"FAILED family {method}: {error}", flush=True)
    write_json(args.results_dir / ("cf_pilot_status.json" if args.pilot else "cf_run_status.json"),
               {"methods": args.methods, "mode": "pilot" if args.pilot else "hpo" if args.hpo_only
                else "final" if args.final_only else "full", "failures": failures,
                "invocation_success": not failures, "global_protocol_sha256": sha256(args.protocol)})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
