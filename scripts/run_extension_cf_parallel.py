#!/usr/bin/env python3
"""Schedule one unchanged frozen CF family in two isolated CPU processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from scripts import run_extension_cf as original
from saet_repro.io import sha256, write_json

_CONTEXT = None
_DATA = None
_DATA_PATH = None


def legacy_writers(root=ROOT, proc_root=Path("/proc")):
    """Conservatively refuse a live original CLI in this checkout."""
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            names = [Path(os.fsdecode(value)).name for value in args if value]
            if "run_extension_cf.py" not in names:
                continue
            if "State:\tZ" in (process / "status").read_text():
                continue
            if (process / "cwd").resolve() == Path(root).resolve():
                found.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(found)


def historical_jobs(method, epochs):
    return [(method, trial, seed, epochs)
            for trial in range(12) for seed in original.SELECTION_SEEDS]


def final_jobs(method):
    return [(method, year, seed)
            for year in (2015, 2016, 2017) for seed in original.FINAL_SEEDS]


def _initialize(context):
    global _CONTEXT
    original.torch.set_num_threads(1)
    original.torch.set_num_interop_threads(1)
    _CONTEXT = context


def _load(path):
    global _DATA, _DATA_PATH
    if sha256(path) != _CONTEXT["data_sha256"][path.name]:
        raise ValueError(f"Input changed after scheduling freeze: {path}")
    if _DATA_PATH != path:
        _DATA = None
        gc.collect()
        _DATA = original.load_data(path)
        _DATA_PATH = path
    return _DATA


def _historical(job):
    method, trial, seed, epochs = job
    context = _CONTEXT
    path = Path(context["data_dir"]) / "cds_2015_d30.npz"
    directory = Path(context["results_dir"]) / method / "hpo" / f"trial_{trial:02d}" / f"seed_{seed}"
    try:
        return original.run_historical(
            _load(path), path, method, trial, seed, epochs, directory,
            context["identity"], retry_failed=context["retry_failed"])
    except Exception as error:
        original.preserve_failure(directory, error)
        raise


def _final(job):
    method, year, seed = job
    context = _CONTEXT
    root = Path(context["results_dir"])
    directory = root / method / "final" / f"cds_{year}_d0" / f"seed_{seed}"
    try:
        # The canonical full historical receipt is verified before opening d0.
        selected = original.validate_selection(root, method, context["identity"])
        if sha256(root / method / "selected.json") != context["selected_sha256"]:
            raise ValueError("Selection changed after the scheduler's final phase gate")
        path = Path(context["data_dir"]) / f"cds_{year}_d0.npz"
        report = original.run_final(
            _load(path), path, root, method, seed, selected, context["identity"],
            retry_failed=context["retry_failed"])
        # Large histories remain in the original run receipt, not pool messages.
        return {"method": method, "prefix": path.stem, "seed": seed,
                "complete_sha256": sha256(directory / "complete.json")}
    except Exception as error:
        original.preserve_failure(directory, error)
        raise


def run_family(args, identity, receipt):
    context = {"data_dir": str(args.data_dir.resolve()),
               "results_dir": str(args.results_dir.resolve()),
               "identity": identity, "retry_failed": args.retry_failed,
               "data_sha256": json.loads((receipt / "frozen.json").read_text())["data_sha256"]}
    spawn = multiprocessing.get_context("spawn")
    if not args.final_only:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn,
                                 initializer=_initialize, initargs=(context,)) as pool:
            trajectories = list(pool.map(_historical, historical_jobs(args.method, 50), chunksize=1))
            decision = original.extension_decision(original.selection_records(trajectories))
            write_json(receipt / "historical_schedule.json",
                       {"initial_trajectories": 36, "extension_decision": decision,
                        "rule_function": "run_extension_cf.extension_decision"})
            if decision["extend_to_100"]:
                # Extend every candidate and seed, never only the apparent winner.
                list(pool.map(_historical, historical_jobs(args.method, 100), chunksize=1))
        path = args.data_dir / "cds_2015_d30.npz"
        data = original.load_data(path)
        # Canonical original selection rechecks completed trajectories and writes
        # exactly the same selection/receipt schema as the serial entry point.
        original.run_hpo(data, path, args.results_dir, args.method, identity, args.retry_failed)
        del data
        gc.collect()
    selected = original.validate_selection(args.results_dir, args.method, identity)
    if args.hpo_only:
        return {"historical_complete": 36, "final_complete": 0, "epochs": selected["epochs"]}
    context["selected_sha256"] = sha256(args.results_dir / args.method / "selected.json")
    write_json(receipt / "final_phase.json",
               {"selected_sha256": context["selected_sha256"],
                "epochs": selected["epochs"], "final_tasks": final_jobs(args.method),
                "opened_after_selection_validation_utc": original.utc_now()})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn,
                             initializer=_initialize, initargs=(context,)) as pool:
        completed = list(pool.map(_final, final_jobs(args.method), chunksize=1))
    return {"historical_complete": 36, "final_complete": len(completed),
            "epochs": selected["epochs"], "final_receipts": completed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=original.METHODS, default="lightgcn_user")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results_extensions/v1/stage_b")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions_v1.json")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hpo-only", action="store_true")
    mode.add_argument("--final-only", action="store_true")
    args = parser.parse_args(argv)
    if args.list_tasks:
        print(json.dumps(original.task_manifest([args.method]), indent=2))
        return 0
    competitors = legacy_writers()
    if competitors:
        raise SystemExit(f"Stop the original CF CLI before scheduler handoff; live PIDs: {competitors}")
    original.torch.set_num_threads(1)
    original.torch.set_num_interop_threads(1)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / ".cf_parallel.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another parallel CF scheduler owns this output root")
        # Catch an already-starting serial CLI after taking our own lock. The
        # legacy CLI has no shared lock: operators must not launch it meanwhile.
        if legacy_writers():
            raise SystemExit("Original CF CLI became active during scheduler handoff")
        protocol = original.validate_global_protocol(args.protocol)
        path = args.data_dir / "cds_2015_d30.npz"
        identity = original.freeze_family(args.results_dir, args.method, args.protocol, protocol, path)
        receipt = args.results_dir / args.method / "parallel_execution" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}")
        receipt.mkdir(parents=True, exist_ok=False)
        write_json(receipt / "frozen.json", {
            "schema": "extension-cf-scheduler-v1", "identity": identity,
            "scheduler_sha256": sha256(Path(__file__)), "method": args.method,
            "workers": args.workers, "multiprocessing_start_method": "spawn",
            "threads_per_worker": 1, "selection_and_training": "unchanged original functions",
            "data_sha256": {f"cds_{year}_d{offset}.npz": sha256(
                args.data_dir / f"cds_{year}_d{offset}.npz")
                for year, offset in ((2015, 30), (2015, 0), (2016, 0), (2017, 0))},
            "started_utc": original.utc_now(), "pid": os.getpid(),
            "hpo_only": args.hpo_only, "final_only": args.final_only,
            "resource_note": "Independent trajectories share two CPU workers; recorded process RSS is not additive model memory."})
        started = time.monotonic()
        try:
            result = run_family(args, identity, receipt)
            write_json(receipt / "complete.json", {
                "complete": True, **result, "invocation_seconds": time.monotonic() - started,
                "frozen_sha256": sha256(receipt / "frozen.json"),
                "completed_utc": original.utc_now()})
            print(json.dumps({"scheduler_complete": True, **result}), flush=True)
            return 0
        except Exception as error:
            write_json(receipt / "failed.json", {
                "complete": False, "error": str(error), "traceback": traceback.format_exc(),
                "invocation_seconds": time.monotonic() - started,
                "failed_utc": original.utc_now()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
