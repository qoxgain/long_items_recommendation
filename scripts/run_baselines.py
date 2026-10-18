#!/usr/bin/env python3
"""Fit 12 tuning trials/family, freeze, then fit 3 origins x 5 seeds."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch
from saet_repro.collaborative import CFConfig, CollaborativeModel, search_space, train_cf
from saet_repro.evaluation import evaluate, stratified_summaries


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(jsonable(content), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    files = ["saet_repro/collaborative.py", "saet_repro/data.py", "saet_repro/evaluation.py",
             "scripts/run_baselines.py", "docs/cf_protocol.md", "docs/IMPLEMENTATION_CONTRACT.md"]
    return {p: sha256(ROOT / p) for p in files}


def run_one(data, data_path, config, seed, model_path, result_path, evaluation_path,
            frozen_path, selection, code_hashes, data_hash, epochs=6):
    identity = {"config": asdict(config), "seed": seed, "epochs": epochs,
                "code_sha256": code_hashes, "data_path": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                "data_sha256": data_hash, "selection": selection,
                "fit_start": int(data["fit_start"]), "fit_end": int(data["fit_end"]),
                "query_end": int(data["query_end"]), "query_stage": 1,
                "candidate_mode": "full_catalog_users_excluding_any_prior_rating",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "torch_threads": 1}}
    result_path, model_path = Path(result_path), Path(model_path)
    if result_path.exists():
        old = json.loads(result_path.read_text())
        frozen = json.loads(Path(frozen_path).read_text())
        if frozen["identity"] != identity:
            raise ValueError(f"Completed run provenance differs: {result_path}")
        if sha256(model_path) != frozen["model_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {model_path}")
        if sha256(evaluation_path) != old["evaluation_sha256"]:
            raise ValueError(f"Evaluation hash mismatch: {evaluation_path}")
        print(f"resume {result_path}", flush=True)
        return old
    started = time.monotonic()
    print(f"fit {config.method} seed={seed} split={data_path.stem} config={asdict(config)}", flush=True)
    model, history = train_cf(data, config, seed, epochs=epochs,
                             progress=lambda row: print(f"  epoch={row['epoch']} loss={row['loss']:.6f} edges={row['positive_edges']}", flush=True))
    model.save(model_path)
    frozen = {"identity": identity, "model_sha256": sha256(model_path),
              "frozen_before_query_scoring_utc": datetime.now(timezone.utc).isoformat(),
              "fixed_final_epoch": epochs, "training": history}
    # Must exist on disk before the first access to query scores/metrics.
    write_json(frozen_path, frozen)
    result = evaluate(data, 1, model.score_all, ks=(10, 20, 50), save_topk=True, batch_size=8)
    np.savez_compressed(evaluation_path, **result)
    summaries = stratified_summaries(data, result)
    report = {"method": config.method, "config": asdict(config), "seed": seed,
              "epochs": epochs, "training": history, "summary": {"1": summaries},
              "seconds": time.monotonic() - started, "frozen_path": str(frozen_path),
              "model_sha256": frozen["model_sha256"], "evaluation_sha256": sha256(evaluation_path),
              "independent_reconstruction": True}
    report = jsonable(report)
    write_json(result_path, report)
    print(f"done {config.method} {data_path.stem} seed={seed} seconds={report['seconds']:.1f} tail80_ndcg20={summaries['tail80']['ndcg20']}", flush=True)
    return report


def wait_for_data(path, wait):
    announced = False
    while not path.exists():
        if not wait:
            raise FileNotFoundError(path)
        if not announced:
            print(f"waiting for {path}", flush=True)
            announced = True
        time.sleep(30)
    from saet_repro.data import load_data
    return load_data(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--methods", nargs="+", choices=("mf_bpr", "lightgcn", "simplex"), default=["mf_bpr", "lightgcn", "simplex"])
    parser.add_argument("--wait-for-data", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    tuning_path = args.data_dir / "cds_2015_d30.npz"
    tuning_data = wait_for_data(tuning_path, args.wait_for_data)
    code_hashes = source_hashes()
    tuning_hash = sha256(tuning_path)
    for method in args.methods:
        search_dir = args.results_dir / "cds_2015_d30" / method / "search"
        search_dir.mkdir(parents=True, exist_ok=True)
        trials = []
        for index, config in enumerate(search_space(method)):
            stem = search_dir / f"trial_{index:02d}"
            report = run_one(tuning_data, tuning_path, config, 101, stem.with_suffix(".npz"),
                             stem.with_suffix(".json"), stem.with_suffix(".test.npz"),
                             stem.with_suffix(".frozen.json"),
                             {"role": "hpo", "criterion": "2015_d30_stage1_tail80_ndcg20", "trial": index},
                             code_hashes, tuning_hash)
            value = report["summary"]["1"]["tail80"]["ndcg20"]
            if value is None or not np.isfinite(value):
                raise ValueError(f"Undefined HPO criterion at {stem}")
            trials.append({"trial": index, "config": asdict(config), "tail80_ndcg20": value,
                           "result_sha256": sha256(stem.with_suffix(".json"))})
        # Exact ties choose the earliest prespecified grid point.
        best = max(trials, key=lambda row: (row["tail80_ndcg20"], -row["trial"]))
        selected = {"method": method, "config": best["config"], "trial": best["trial"],
                    "criterion": "2015_d30_stage1_tail80_ndcg20", "criterion_value": best["tail80_ndcg20"],
                    "hpo_seed": 101, "epochs": 6, "trials": trials, "code_sha256": code_hashes,
                    "tuning_data_sha256": tuning_hash,
                    "final_years": [2015, 2016, 2017], "final_seeds": [11, 22, 33, 44, 55],
                    "no_final_validation_or_epoch_selection": True}
        selected_path = search_dir.parent / "selected.json"
        if selected_path.exists() and json.loads(selected_path.read_text()) != selected:
            raise ValueError(f"Frozen selected config changed: {selected_path}")
        write_json(selected_path, selected)
        print(f"selected {method}: trial={best['trial']} tail80_ndcg20={best['tail80_ndcg20']}", flush=True)
        if args.hpo_only:
            continue
        config = CFConfig(**best["config"])
        for year in (2015, 2016, 2017):
            data_path = args.data_dir / f"cds_{year}_d0.npz"
            data = wait_for_data(data_path, args.wait_for_data)
            data_hash = sha256(data_path)
            for seed in (11, 22, 33, 44, 55):
                out = args.results_dir / data_path.stem / method / f"seed_{seed}"
                out.mkdir(parents=True, exist_ok=True)
                run_one(data, data_path, config, seed, out / "model.npz", out / "result.json",
                        out / "test.npz", out / "frozen.json",
                        {"role": "final", "selected_sha256": sha256(selected_path),
                         "selected_trial": best["trial"], "validation": "none"}, code_hashes, data_hash)


if __name__ == "__main__":
    main()
