#!/usr/bin/env python3
"""Strict, read-only audit and public reporting of the frozen v1 extensions.

Only the report directory is written. Partial mode is a progress snapshot: it
never imputes a missing seed or averages an incomplete set of origins.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from saet_repro.evaluation import _prepare, stratified_summaries
from saet_repro.extension_statistics import unique_item_cluster_interval, routing_summary
from saet_repro.io import sha256, json_safe, write_json
from saet_repro.statistics import paired_effects
from scripts.summarize_results import write_csv, markdown_table, number

STRATA = ("all", "tail50", "tail80")
METRICS = tuple(f"{m}{k}" for k in (10, 20, 50) for m in ("ndcg", "recall", "precision", "coverage")) + ("conditional_hit_rate20",)
COUNTS = ("items_catalog", "items_eligible", "items_requested", "items_evaluated", "items_with_positive", "items_without_positive", "items_hit20", "n_positive")
POPULATION = ("item_id", "n_candidates", "n_positive", "eligible_catalog", "requested", "evaluated")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_unique(rows, keys):
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in seen:
            raise ValueError(f"Duplicate record for {keys}: {key}")
        seen.add(key)


def validate_dependencies(record, root=ROOT):
    """Check frozen code bytes and installed versions, including nested records."""
    if isinstance(record, list):
        for value in record:
            validate_dependencies(value, root)
    elif isinstance(record, dict):
        for field in ("code_sha256", "analysis_code_sha256"):
            for name, digest in record.get(field, {}).items() if isinstance(record.get(field), dict) else ():
                if sha256(Path(root) / name) != digest:
                    raise ValueError(f"Frozen code SHA256 mismatch: {name}")
        for name, version in record.get("dependencies", {}).items():
            actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
            if str(actual) != str(version):
                raise ValueError(f"Frozen dependency version mismatch: {name}: {version} != {actual}")
        for key, value in record.items():
            if key not in ("code_sha256", "analysis_code_sha256", "dependencies"):
                validate_dependencies(value, root)


def validate_population(result, catalog, reference=None):
    items = np.asarray(catalog["target_item_ids"])
    if len(np.unique(items)) != len(items) or not np.array_equal(result["item_id"], items):
        raise ValueError("Evaluation item catalog contains duplicate IDs or differs from its prefix")
    if len(np.unique(result["item_id"])) != len(items):
        raise ValueError("Duplicate evaluation item IDs")
    if "stage" in result and int(result["stage"]) != 1:
        raise ValueError("Only stage 1 evaluation is allowed")
    if "candidate_mode" in result and str(result["candidate_mode"]) != "full_eligible_users":
        raise ValueError("Only full eligible-user candidates are allowed")
    eligible = np.asarray(catalog["target_degree"]) > 0
    for field in ("n_candidates", "n_positive"):
        values = np.asarray(result[field])
        if values.shape != eligible.shape or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"Invalid candidate population: {field}")
        expected = catalog.get("expected_" + field)
        if expected is not None and not np.array_equal(values, expected):
            raise ValueError(f"Candidate population differs from prefix events: {field}")
    if np.any(result["n_positive"] > result["n_candidates"]):
        raise ValueError("Positive count exceeds candidate population")
    expected = eligible & (result["n_positive"] > 0)
    for field, value in (("eligible_catalog", eligible), ("requested", eligible),
                         ("evaluated", expected), ("scored", expected), ("ranked", expected)):
        if field in result and (np.asarray(result[field]).dtype.kind != "b" or not np.array_equal(result[field], value)):
            raise ValueError(f"Incomplete evaluation mask: {field}")
    if reference is not None:
        for field in POPULATION:
            if not np.array_equal(result[field], reference[field]):
                raise ValueError(f"Paired candidate/evaluation population differs: {field}")
    for k in (10, 20, 50):
        for metric in ("ndcg", "recall", "precision", "hits"):
            field = f"{metric}{k}"
            if field not in result:
                continue
            value = np.asarray(result[field])
            if value.shape != expected.shape or np.any(~np.isfinite(value[expected])) or np.any(np.isfinite(value[~expected])):
                raise ValueError(f"Invalid evaluated/undefined metric mask: {field}")


def aggregate_rows(rows, origins, seeds):
    ensure_unique(rows, ("method", "stratum", "origin", "seed"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], row["stratum"], row["origin"]].append(row)
    by_origin = []
    for (method, stratum, origin), values in sorted(grouped.items()):
        expected = {None} if method.split("/")[-1] in ("activity", "cosine") else set(seeds)
        if {r["seed"] for r in values} != expected:
            continue
        result = {"method": method, "stratum": stratum, "origin": origin, "n_seeds": len(values), "complete": True}
        for metric in METRICS:
            numbers = np.asarray([r[metric] for r in values], dtype=float)
            result[metric] = float(numbers.mean())
            result[metric + "_seed_sd"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else None
        for field in COUNTS:
            if field == "items_hit20":
                # Hits depend on the fitted ranking; the candidate population does not.
                result[field] = float(np.mean([r[field] for r in values]))
                continue
            counts = {r[field] for r in values}
            if len(counts) != 1:
                raise ValueError(f"Seed-specific evaluation population: {method}/{origin}/{stratum}/{field}")
            result[field] = counts.pop()
        by_origin.append(result)
    overall = []
    for method, stratum in sorted({(r["method"], r["stratum"]) for r in by_origin}):
        values = [r for r in by_origin if r["method"] == method and r["stratum"] == stratum]
        if {r["origin"] for r in values} != set(origins):
            continue
        result = {"method": method, "stratum": stratum, "n_origins": len(origins),
                  "n_runs": sum(r["n_seeds"] for r in values), "origin_weighting": "equal", "complete": True}
        result.update({metric: float(np.mean([r[metric] for r in values])) for metric in METRICS})
        result.update({field + "_sum_over_origins": sum(r[field] for r in values) for field in COUNTS})
        overall.append(result)
    return by_origin, overall


def ridge_diagnostic_rows(document, base, year, subsets, alphas):
    """Expose every saved validation trial after checking the frozen selection."""
    diagnostics = document["diagnostics"]
    if set(diagnostics) != set(subsets):
        raise ValueError("Ridge diagnostic subsets differ from frozen matrix")
    rows = []
    for subset in subsets:
        selected = diagnostics[subset]
        trials = selected["trials"]
        if [r["alpha"] for r in trials] != list(alphas):
            raise ValueError("Ridge diagnostic trials differ from frozen alpha grid")
        for trial in trials:
            if trial["n"] != document["n_validation_labels"] or trial["n"] <= 0:
                raise ValueError("Ridge diagnostic validation population differs")
            for name in ("mse", "zero_mse", "scaled_mse", "label_rms"):
                if not np.isfinite(trial[name]) or trial[name] < 0:
                    raise ValueError("Invalid ridge diagnostic loss or scale")
            for name in ("sign_accuracy", "predicted_positive_fraction", "label_positive_fraction"):
                if not np.isfinite(trial[name]) or not 0 <= trial[name] <= 1:
                    raise ValueError("Invalid ridge diagnostic fraction")
            precision = trial["positive_precision"]
            if precision is None:
                if trial["predicted_positive_fraction"] != 0:
                    raise ValueError("Missing precision despite predicted positives")
            elif not np.isfinite(precision) or not 0 <= precision <= 1:
                raise ValueError("Invalid ridge positive precision")
        best = min(trials, key=lambda r: r["mse"])
        if (selected["selected_alpha"] != best["alpha"] or
                selected["validation"] != {k: v for k, v in best.items() if k != "alpha"}):
            raise ValueError("Ridge selected diagnostics violate frozen minimum-MSE tie rule")
        rows.extend({"stage": "c", "component": "ridge_validation", "row_kind": "ridge_validation",
                     "base": base, "origin": year, "offset": 60, "subset": subset,
                     "selected_alpha": selected["selected_alpha"],
                     "selected": trial["alpha"] == selected["selected_alpha"],
                     **trial} for trial in trials)
    return rows


class Audit:
    def __init__(self, args):
        self.args = args
        self.protocol = read_json(args.protocol)
        self.protocol_hash = sha256(args.protocol)
        sidecar = args.protocol.with_suffix(".sha256")
        if not sidecar.exists() or sidecar.read_text().split()[0] != self.protocol_hash:
            raise ValueError("Global protocol checksum is missing or differs")
        for name, digest in self.protocol["baseline_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Immutable baseline changed: {name}")
        self.status, self.runs, self.items, self.curves, self.budget = [], [], [], [], []
        self.selections, self.hpo, self.routes, self.diagnostics = [], [], [], []
        self.hashes, self.catalogs, self.references = {}, {}, {}
        self.stage_a = None
        self.stage_a_pairs = []

    def digest(self, path):
        path = Path(path)
        if path not in self.hashes:
            self.hashes[path] = sha256(path)
        return self.hashes[path]

    def check_hash(self, path, digest):
        if not digest or self.digest(path) != digest:
            raise ValueError(f"Artifact SHA256 mismatch: {path}")

    def artifacts(self, directory, mapping, required=()):
        if not set(required).issubset(mapping):
            raise ValueError(f"Incomplete registered artifact set at {directory}")
        for name, digest in mapping.items():
            self.check_hash(Path(directory) / name, digest)

    def task(self, stage, group, name, directory, callback, reused=False):
        row = {"stage": stage, "group": group, "cell": name,
               "state": "complete", "evidence": "validated_old_reuse" if reused else "new_extension"}
        # Roll back derived values when a late check rejects a cell.
        lists = (self.runs, self.items, self.curves, self.budget, self.selections, self.hpo, self.routes, self.diagnostics)
        sizes = [len(values) for values in lists]
        try:
            result = callback()
        except (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError) as error:
            for values, size in zip(lists, sizes):
                del values[size:]
            row["state"] = "failed" if (Path(directory) / "failed.json").exists() else ("missing" if isinstance(error, FileNotFoundError) else "invalid")
            row["error"] = str(error)
            result = None
        self.status.append(row)
        return result

    def catalog(self, path):
        path = Path(path)
        if path not in self.catalogs:
            with np.load(path, allow_pickle=False) as saved:
                data = {name: saved[name] for name in ("target_item_ids", "target_user_ids", "target_degree", "target_tail50", "target_tail80", "target_positive", "target_observed")}
            prepared = _prepare(data, 1, (10, 20, 50), None)
            catalog = {key: data[key] for key in ("target_item_ids", "target_degree", "target_tail50", "target_tail80")}
            catalog.update(expected_n_candidates=prepared["n_candidates"], expected_n_positive=prepared["n_positive"])
            self.catalogs[path] = catalog
        return self.catalogs[path]

    def evaluation(self, path, data_path, stored, stage, method, origin, seed, final=True):
        with np.load(path, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files if key != "top_users"}
        if "stage" not in result or "candidate_mode" not in result:
            raise ValueError("Evaluation lacks stage/candidate identity")
        catalog = self.catalog(data_path)
        refkey = str(data_path)
        validate_population(result, catalog, self.references.get(refkey))
        summaries = stratified_summaries(catalog, result)
        for stratum, values in summaries.items():
            for key in (*METRICS, *COUNTS):
                actual, recorded = values[key], stored["1"][stratum][key]
                if recorded is None:
                    valid = actual is None or not np.isfinite(actual)
                else:
                    valid = np.isclose(actual, recorded, rtol=1e-9, atol=1e-12, equal_nan=True)
                if not valid:
                    raise ValueError(f"Saved NPZ/JSON summary mismatch: {method}/{stratum}/{key}")
        self.references[refkey] = {key: result[key] for key in POPULATION}
        if final:
            for stratum, values in summaries.items():
                self.runs.append({"stage": stage, "method": method, "origin": origin, "seed": seed,
                                  "stratum": stratum, **{k: values[k] for k in (*METRICS, *COUNTS)},
                                  "conditional_denominator": values["items_evaluated"], "coverage_denominator": values["items_catalog"]})
            mask = result["evaluated"] & catalog["target_tail80"]
            self.items.extend({"stage": stage, "method": method, "origin": origin, "seed": seed,
                               "item_id": str(item), "ndcg20": float(value)}
                              for item, value in zip(result["item_id"][mask], result["ndcg20"][mask]))
        return summaries, result

    def stage_a_audit(self):
        directory = self.args.stage_a_dir
        def check():
            value = read_json(directory / "analysis_provenance.json")
            if not value["complete"] or value["protocol_sha256"] != self.protocol_hash or value["actual_final_evaluations"] != 411:
                raise ValueError("Incomplete Stage A or wrong global protocol")
            validate_dependencies(value)
            self.artifacts(directory, value["outputs_sha256"], ("analysis_target_pairs.csv", "analysis_selection_surface.csv", "analysis_secondary_metrics.csv", "analysis_mechanism_pairs.csv"))
            for row in value["input_artifacts"]:
                self.check_hash(ROOT / row["path"], row["sha256"])
            if not value["original_reports_unchanged"] or not value["convex_mixture"]["passed"]:
                raise ValueError("Stage A original/convex checks failed")
            with (directory / "analysis_target_pairs.csv").open(newline="", encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            ensure_unique(pairs, ("method", "baseline"))
            if len(pairs) != value["n_target_pairs"]:
                raise ValueError("Stage A target comparison count differs")
            self.stage_a_pairs = pairs
            self.stage_a = value
            return value
        self.task("a", "analysis_bundle", "read_only_reanalysis", directory, check, reused=True)

    def b_protocol(self, method):
        base = self.args.stage_b_dir
        if method.startswith("activity_"):
            path = base / "target_protocol.json"
            doc = read_json(path)
            self.check_hash(path, read_json(path.with_suffix(".sha256.json"))["sha256"])
            validate_dependencies(doc)
            if doc["global_protocol_sha256"] != self.protocol_hash or doc["stage_b"] != self.protocol["stage_b"]:
                raise ValueError("Target protocol differs from global protocol")
            from saet_repro.extension_target import grid
            if doc["grids"][method] != [asdict(config) for config in grid(method)]:
                raise ValueError("Target protocol grid differs from the prescribed twelve configurations")
            for prefix, digest in doc["data_sha256"].items():
                self.check_hash(self.args.data_dir / f"{prefix}.npz", digest)
            return doc["grids"][method], {"target_protocol_sha256": self.digest(path), "global_protocol_sha256": self.protocol_hash}, doc
        path = base / method / "protocol.json"
        outer = read_json(path)
        self.check_hash(path, read_json(path.with_suffix(".sha256.json"))["sha256"])
        from saet_repro.extension_cf import grid, json_hash
        doc = outer["protocol"]
        if json_hash(doc) != outer["protocol_sha256"] or doc["global_protocol_sha256"] != self.protocol_hash:
            raise ValueError("CF family protocol differs")
        validate_dependencies(doc)
        expected = {"method": method, "hpo_prefix": self.protocol["stage_b"]["hpo_prefix"],
                    "hpo_seeds": self.protocol["stage_b"]["hpo_seeds"],
                    "final_origins": self.protocol["stage_b"]["final_origins"],
                    "final_seeds": self.protocol["seeds"],
                    "extension_rule": self.protocol["stage_b"]["extension_rule"],
                    "grid": [asdict(config) for config in grid(method)], "query_stage": 1}
        if any(doc.get(key) != value for key, value in expected.items()):
            raise ValueError("CF family configuration matrix differs from the prescribed protocol")
        self.check_hash(self.args.data_dir / f"{doc['hpo_prefix']}.npz", doc["hpo_data_sha256"])
        return doc["grid"], {"family_protocol_sha256": self.digest(path), "global_protocol_sha256": self.protocol_hash}, doc

    def b_history(self, directory, method, seed, config, identity, trial=None, final=False, epochs=None):
        complete = read_json(directory / "complete.json")
        frozen = read_json(directory / "frozen.json")
        validate_dependencies(frozen)
        target = method.startswith("activity_")
        run = frozen["run_identity"] if target else frozen
        if any(run.get(k) != v for k, v in identity.items()) or frozen["seed"] != seed or frozen["config"] != config:
            raise ValueError("Training identity differs from protocol/config/seed")
        prefix = run["prefix"]
        expected_prefix = f"cds_{directory.parent.name.split('_')[1]}_d0" if final else self.protocol["stage_b"]["hpo_prefix"]
        if prefix != expected_prefix:
            raise ValueError("Trajectory prefix differs from prescribed cell")
        if frozen.get("evaluate_each_epoch") != (not final):
            raise ValueError("Trajectory historical/final evaluation phase differs")
        if target:
            if run.get("phase") != ("final" if final else "historical_selection"):
                raise ValueError("Target trajectory phase differs")
        elif (run.get("role") != ("final" if final else "hpo") or
              run.get("method") != method or (not final and run.get("trial") != trial)):
            raise ValueError("CF trajectory method/phase/trial differs")
        self.check_hash(self.args.data_dir / f"{prefix}.npz", run["data_file_sha256" if target else "data_sha256"])
        mapping = complete["artifacts" if target else "artifact_sha256"]
        self.artifacts(directory, mapping, ("history.json", "frozen.json", "last_state.npz", "last_state.sha256.json"))
        history = read_json(directory / "history.json")
        ensure_unique(history, ("epoch",))
        if not history or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
            raise ValueError("Trajectory history is not a contiguous epoch sequence")
        if epochs is not None and len(history) != epochs:
            raise ValueError("Trajectory has wrong selected/required epoch budget")
        if target:
            from saet_repro.extension_target import _json_hash
            if final:
                expected_identity = {**run, "seed": seed, "config": config, "epochs": len(history)}
                if complete["identity"] != expected_identity:
                    raise ValueError("Target completion identity differs")
            elif (complete["identity_sha256"] != _json_hash(frozen) or
                  complete["epochs"] != len(history) or
                  complete["target_protocol_sha256"] != identity["target_protocol_sha256"]):
                raise ValueError("Target completion identity/budget differs")
            required = {"history.json", "last_state.npz", "last_state.sha256.json", "frozen.json"}
            if final:
                required.update(("model.npz", "test.npz", "result.json"))
            else:
                required.update(f"checkpoint_{epoch:03d}.{suffix}"
                                for epoch in (6, 20, 50, 100) if epoch <= len(history)
                                for suffix in ("npz", "json"))
        else:
            from scripts.run_extension_cf import required_artifacts
            required = required_artifacts(run, len(history))
            if (complete["epochs"] != len(history) or complete["status"] != "complete" or
                    any(run.get(k) != v for k, v in complete["run_identity"].items()) or
                    any(complete["run_identity"].get(k) != v for k, v in identity.items())):
                raise ValueError("CF completion identity/budget differs")
        if set(mapping) != required:
            raise ValueError("Trajectory completion omits prescribed artifacts/checkpoints")
        for row in history:
            # Do not publish learned parameter vectors (beta) as trajectory data.
            self.curves.append({"stage": "b", "phase": "final" if final else "hpo", "method": method,
                                "prefix": prefix, "trial": trial, "seed": seed,
                                **{k: v for k, v in row.items() if isinstance(v, (int, float, str)) or v is None}})
        self.budget.append({"stage": "b", "phase": "final" if final else "hpo", "method": method,
                            "prefix": prefix, "trial": trial, "seed": seed, "epochs": len(history),
                            "updates": sum(r.get("updates", 0) for r in history),
                            "pairs": sum(r.get("pair_count", r.get("pairs", 0)) for r in history),
                            "positives": sum(r.get("positive_count", r.get("positive_edges", 0)) for r in history),
                            "training_seconds": sum(r.get("training_seconds", 0) for r in history),
                            "historical_evaluation_seconds": sum(
                                r.get("evaluation_seconds_shared", 0) / r.get("evaluation_trajectory_count", 1)
                                if target else r.get("evaluation_seconds", 0) for r in history),
                            "feature_bank_seconds_allocated": sum(
                                r.get("feature_bank_seconds_shared", 0) / r.get("feature_bank_trajectory_count", 1)
                                for r in history),
                            "peak_rss_mib": max((r.get("peak_rss_mib", 0) for r in history), default=0),
                            "independent_samples": False})
        return {"trial": trial, "seed": seed, "config": config, "history": history}

    def stage_b_audit(self):
        from saet_repro.extension_cf import selection_records, select_checkpoint, extension_decision
        section = self.protocol["stage_b"]
        for method in section["methods"]:
            directory = self.args.stage_b_dir / method
            proto = self.task("b", "protocols", method, directory, lambda m=method: self.b_protocol(m))
            trajectories = []
            for trial in range(12):
                for seed in section["hpo_seeds"]:
                    child = directory / "hpo" / f"trial_{trial:02d}" / f"seed_{seed}"
                    def check_hpo(child=child, seed=seed, trial=trial):
                        if proto is None:
                            raise FileNotFoundError("Valid family protocol not available")
                        result = self.b_history(child, method, seed, proto[0][trial], proto[1], trial)
                        if len(result["history"]) not in (50, 100):
                            raise ValueError("Incomplete initial/extended HPO trajectory")
                        return result
                    value = self.task("b", "hpo_trajectories", f"{method}/{trial}/{seed}", child, check_hpo)
                    if value:
                        trajectories.append(value)
            def check_selection():
                selected = read_json(directory / "selected.json")
                if proto is None or len(trajectories) != 36:
                    raise ValueError("Selection requires all 36 verified trajectories")
                records50 = selection_records(trajectories)
                decision = extension_decision(records50)
                epochs = 100 if decision["extend_to_100"] else 50
                if any(len(row["history"]) != epochs for row in trajectories):
                    raise ValueError("Conditional extension did not cover all family trajectories")
                records = selection_records(trajectories, checkpoints=(6, 20, 50, 100) if epochs == 100 else (6, 20, 50))
                best = select_checkpoint(records)
                for key, value in best.items():
                    if selected[key] != value:
                        raise ValueError(f"Selected checkpoint does not match historical winner: {key}")
                if selected["trials"] != records or selected["extension_decision"] != decision:
                    raise ValueError("Recorded HPO surface/extension trigger differs from histories")
                if any(selected.get(k) != v for k, v in proto[1].items()) or selected["method"] != method:
                    raise ValueError("Selected identity differs")
                if method.startswith("activity_"):
                    self.check_hash(directory / "selected.json", read_json(directory / "selected.complete.json")["selected_sha256"])
                    self.artifacts(directory, selected["evidence_sha256"])
                else:
                    validate_dependencies(selected)
                    self.check_hash(directory / "selected.json", read_json(directory / "selected.sha256.json")["sha256"])
                    for name, digest in selected["trajectory_complete_sha256"].items():
                        self.check_hash(directory / "hpo" / name / "complete.json", digest)
                self.selections.append({"stage": "b", "method": method, "trial": best["trial"], "epochs": best["epochs"], "historical_score": best["score"],
                                        "extended_to_100": decision["extend_to_100"], "configuration": json.dumps(best["config"], sort_keys=True)})
                self.hpo.extend({"stage": "b", "method": method, **r, "config": json.dumps(r["config"], sort_keys=True),
                                 "seed_scores": json.dumps(r["seed_scores"], sort_keys=True)} for r in records)
                return selected
            selected = self.task("b", "selected_families", method, directory, check_selection)
            for year in section["final_origins"]:
                prefix = f"cds_{year}_d0"
                for seed in self.protocol["seeds"]:
                    child = directory / "final" / prefix / f"seed_{seed}"
                    def check_final(child=child, seed=seed, year=year, prefix=prefix):
                        if selected is None:
                            raise FileNotFoundError("Verified complete selection not available")
                        target = method.startswith("activity_")
                        identity = {**proto[1], "selection_sha256" if target else "selected_sha256": self.digest(directory / "selected.json")}
                        self.b_history(child, method, seed, selected["config"], identity, final=True, epochs=selected["epochs"])
                        doc = read_json(child / "result.json")
                        if (doc["method"], doc["seed"], doc["prefix"], doc["epochs"], doc["config"]) != (method, seed, prefix, selected["epochs"], selected["config"]):
                            raise ValueError("Final result identity differs")
                        marker = read_json(child / "complete.json")
                        self.artifacts(child, marker["artifacts" if target else "artifact_sha256"], ("model.npz", "test.npz", "result.json"))
                        gate = read_json((child.parent if target else child) / "final_query_opened.json")
                        gate_identity = gate if target else gate["identity"]
                        if any(gate_identity.get(k) != v for k, v in identity.items()):
                            raise ValueError("Final query gate does not bind selected model")
                        self.evaluation(child / "test.npz", self.args.data_dir / f"{prefix}.npz", doc["summary"], "b", method, year, seed)
                    self.task("b", "final_models", f"{method}/{year}/{seed}", child, check_final)
        self.old_comparators()

    def old_comparators(self):
        manifest = read_json(ROOT / "reports/artifact_manifest.json")
        known = {str((ROOT / r["path"]).resolve()): r["sha256"] for r in manifest["artifacts"]}
        for method in ("mf_bpr", "lightgcn", "simplex", "activity_mean", "activity_both", "activity"):
            for year in self.protocol["stage_b"]["final_origins"]:
                for seed in ([None] if method == "activity" else self.protocol["seeds"]):
                    prefix = f"cds_{year}_d0"
                    family = "baselines" if method in ("mf_bpr", "lightgcn", "simplex") else "targets"
                    directory = self.args.old_results_dir / family / prefix / method / f"seed_{seed or 0}"
                    def check(directory=directory, seed=seed, year=year, method=method, prefix=prefix):
                        for name in ("test.npz", "result.json"):
                            path = directory / name
                            self.check_hash(path, known[str(path.resolve())])
                        doc = read_json(directory / "result.json")
                        if doc["method"] != method or doc["seed"] != (seed or 0):
                            raise ValueError("Old comparator method/seed differs")
                        self.evaluation(directory / "test.npz", self.args.data_dir / f"{prefix}.npz", doc["summary"], "b", "old/" + method, year, seed)
                    self.task("b", "old_six_epoch_comparators", f"{method}/{year}/{seed}", directory, check, reused=True)

    def stage_c_audit(self):
        from saet_repro.extension_transfer import Runner, BASE_CONFIGS, OFFSETS, FEATURE_SUBSETS, RIDGE_ALPHAS, hpo_configurations, choose_policies
        root, section = self.args.stage_c_dir, self.protocol["stage_c_music"]
        runner = Runner(self.args.music_data_dir, root, self.args.protocol)
        def check_protocol():
            path = root / "protocol.frozen.json"
            frozen = read_json(path)
            self.check_hash(path, read_json(root / "protocol.sha256.json")["sha256"])
            validate_dependencies(frozen)
            if frozen["protocol"] != self.protocol or frozen["protocol_sha256"] != self.protocol_hash:
                raise ValueError("C protocol differs from global freeze")
            runner._frozen = frozen  # Read-only restoration; never call freeze/open_final_queries.
            return frozen
        frozen = self.task("c", "protocol", "music", root, check_protocol)
        def marker(directory, name, files):
            if frozen is None:
                raise FileNotFoundError("Verified C protocol missing")
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Missing C artifact marker: {directory / name}")
            doc = runner._verify_marker(directory / name)
            if set(doc["artifacts"]) != set(files):
                raise ValueError("C marker artifact set differs")
            for key, file in files.items():
                self.check_hash(directory / file, doc["artifacts"][key])
            return doc
        for base in BASE_CONFIGS:
            for year in section["origins"]:
                for offset in OFFSETS:
                    directory = runner.historical_directory(base, year, offset)
                    def fold(directory=directory, base=base, year=year, offset=offset):
                        train = marker(directory, "training.complete.json", {"model": "model.npz", "history": "training.json"})
                        marker(directory, "source_activity.json", {"source_activity": "source_activity.npz"})
                        doc = marker(directory, "result.json", {"labels": "labels.npz"})
                        identity = doc["frozen"]
                        if identity["base"] != base or identity["config"] != asdict(BASE_CONFIGS[base]) or identity["seed"] != 101 or identity["epochs"] != 6:
                            raise ValueError("Historical label recipe differs from deployment")
                        self.check_hash(runner.path(year, offset), identity["data_sha256"])
                        self.check_hash(directory / "model.npz", identity["base_model_sha256"])
                        self.check_hash(directory / "source_activity.npz", identity["source_activity_sha256"])
                        self.diagnostics.append({"stage": "c", "component": "historical_labels", "row_kind": "fold_label",
                                                 "base": base, "origin": year,
                                                 "offset": offset, **{k: doc[k] for k in ("n_labels", "n_query_items", "label_positive_fraction")}})
                        self.budget.append({"stage": "c", "phase": "historical_target", "method": base, "prefix": f"cds_{year}_d{offset}",
                                            "seed": 101, "epochs": 6, "training_seconds": train.get("elapsed_seconds"), "independent_samples": False})
                    self.task("c", "historical_folds", f"{base}/{year}/d{offset}", directory, fold)
                directory = runner.ridge_directory(base, year)
                def ridge(directory=directory, base=base, year=year):
                    doc = marker(directory, "selected.json", {s: s + ".npz" for s in FEATURE_SUBSETS})
                    if doc["base"] != base or doc["year"] != year:
                        raise ValueError("Ridge diagnostic base or origin differs")
                    self.digest(directory / "selected.json")
                    self.diagnostics.extend(ridge_diagnostic_rows(doc, base, year, FEATURE_SUBSETS, RIDGE_ALPHAS))
                    return doc
                self.task("c", "ridge_fits", f"{base}/{year}", directory, ridge)
            directory = runner.hpo_directory(base)
            def search(directory=directory, base=base):
                doc = marker(directory, "selected.json", {k: str(v.relative_to(directory)) for k, v in runner.search_artifacts(base).items()})
                configs = hpo_configurations()
                ensure_unique(doc["trials"], ("name",))
                if {r["name"] for r in doc["trials"]} != set(configs):
                    raise ValueError("C HPO does not contain every frozen configuration")
                rows = []
                for record in doc["trials"]:
                    name = record["name"]
                    actual = read_json(directory / "search" / f"{name}.json")
                    if actual != record or record["configuration"] != configs[name]:
                        raise ValueError("C selection trials/configuration differ from artifacts")
                    summary, _ = self.evaluation(directory / "search" / f"{name}.npz", runner.path(2018, 30), record["summary"], "c", base + "/" + name, 2018, 101, final=False)
                    if summary["tail80"]["ndcg20"] != record["score"]:
                        raise ValueError("C historical selection score differs")
                    rows.append(record)
                    self.hpo.append({"stage": "c", "method": base, "name": name, "score": record["score"],
                                     "configuration": json.dumps(configs[name], sort_keys=True), "score_sha256": record["score_sha256"]})
                if choose_policies(rows) != doc["policies"]:
                    raise ValueError("C selected policies violate frozen tie rule")
                for policy, config in doc["policies"].items():
                    self.selections.append({"stage": "c", "method": base + "/" + policy, "configuration": json.dumps(config, sort_keys=True),
                                            "epochs": 6,
                                            "historical_score": next(r["score"] for r in rows if r["configuration"] == config),
                                            "nominal_configurations": doc["nominal_configurations"], "distinct_score_configurations": doc["distinct_score_configurations"]})
                return doc
            self.task("c", "hpo_searches", base, directory, search)
        def gate_check():
            if frozen is None:
                raise FileNotFoundError("C protocol unavailable")
            gate = read_json(runner.gate)
            # All cached calls here pass require=True; no cache quarantine or writes.
            _, selected, dependencies = runner.preflight_selections()
            expected = {"identity": runner.identity, "protocol_sha256": self.protocol_hash,
                        "selection_dependencies": dependencies, "selected_policies": selected,
                        "origins": section["origins"], "seeds": self.protocol["seeds"]}
            if any(gate.get(k) != v for k, v in expected.items()):
                raise ValueError("C final gate differs from verified selection dependencies")
            return gate
        gate = self.task("c", "final_query_gate", "music", root, gate_check)
        for year in section["origins"]:
            prefix = f"cds_{year}_d0"
            directory = root / "activity" / "final" / prefix / "seed_0"
            def activity(directory=directory, year=year):
                if gate is None:
                    raise FileNotFoundError("Verified final query gate unavailable")
                doc = marker(directory, "complete.json", {"test": "test.npz", "result": "result.json"})
                result = read_json(directory / "result.json")
                if result["method"] != "activity" or result["year"] != year or result["seed"] != 0:
                    raise ValueError("C activity identity differs")
                self.check_hash(runner.gate, doc["frozen"]["query_gate_sha256"])
                self.check_hash(runner.path(year, 0), doc["frozen"]["data_sha256"])
                self.evaluation(directory / "test.npz", runner.path(year, 0), result["summary"], "c", "activity", year, None)
            self.task("c", "activity_evaluations", str(year), directory, activity)
            for base in BASE_CONFIGS:
                for seed in self.protocol["seeds"]:
                    directory = root / base / "final" / prefix / f"seed_{seed}"
                    for policy in section["policies"]:
                        def final(directory=directory, base=base, seed=seed, year=year, policy=policy):
                            if gate is None:
                                raise FileNotFoundError("Verified final query gate unavailable")
                            files = {f"{p}_{n}": f"{p}/{n}" for p in section["policies"] for n in ("result.json", "test.npz", "routing.npz")}
                            files["canonical_test"] = "canonical_test.npz"
                            complete = marker(directory, "complete.json", files)
                            identity = complete["frozen"]
                            selected = gate["selected_policies"][base]
                            if (identity["base"], identity["seed"], identity["config"], identity["configurations"], identity["epochs"]) != (base, seed, asdict(BASE_CONFIGS[base]), selected, 6):
                                raise ValueError("C final recipe differs from frozen deployment")
                            self.check_hash(runner.path(year, 0), identity["data_sha256"])
                            self.check_hash(runner.gate, identity["query_gate_sha256"])
                            self.check_hash(runner.hpo_directory(base) / "selected.json", identity["selected_policy_sha256"])
                            self.check_hash(directory / "model.npz", identity["base_model_sha256"])
                            self.check_hash(directory / "source_activity.npz", identity["source_activity_sha256"])
                            for name, digest in identity["ridge_hashes"].items():
                                self.check_hash(runner.ridge_directory(base, year) / f"{name}.npz", digest)
                            result = read_json(directory / policy / "result.json")
                            if (result["base"], result["method"], result["year"], result["seed"], result["configuration"], result["frozen"]) != (base, policy, year, seed, selected[policy], identity):
                                raise ValueError("C policy result identity differs")
                            if not all(result["null_check"].get(k) for k in ("scores_exact", "all_rank_and_metric_arrays_exact", "base_parameters_unchanged")):
                                raise ValueError("C canonical null check failed")
                            if policy == "zero":
                                with np.load(directory / "canonical_test.npz", allow_pickle=False) as left, np.load(directory / policy / "test.npz", allow_pickle=False) as right:
                                    if set(left.files) != set(right.files) or any(not np.array_equal(left[k], right[k], equal_nan=left[k].dtype.kind in "fc") for k in left.files):
                                        raise ValueError("Zero policy differs from canonical base arrays")
                                train = marker(directory, "training.complete.json", {"model": "model.npz", "history": "training.json"})
                                self.budget.append({"stage": "c", "phase": "final_shared_four_policies", "method": base, "prefix": f"cds_{year}_d0", "seed": seed,
                                                    "epochs": 6, "training_seconds": train.get("elapsed_seconds"), "evaluation_seconds_shared": complete.get("elapsed_seconds"), "independent_samples": False})
                            _, evaluated = self.evaluation(directory / policy / "test.npz", runner.path(year, 0), result["summary"], "c", base + "/" + policy, year, seed)
                            with np.load(directory / policy / "routing.npz", allow_pickle=False) as saved:
                                routing = {k: saved[k] for k in saved.files}
                            if not np.array_equal(routing["item_id"], np.flatnonzero(evaluated["evaluated"])):
                                raise ValueError("Routing evaluation population differs")
                            mask = self.catalog(runner.path(year, 0))["target_tail80"][routing["item_id"]]
                            self.routes.append({"stage": "c", "method": base + "/" + policy, "origin": year, "seed": seed, "stratum": "tail80", **routing_summary(routing, mask)})
                        self.task("c", "final_policy_evaluations", f"{base}/{policy}/{year}/{seed}", directory, final)


def comparisons(audit, overall):
    results = []
    for stage in ("b", "c"):
        complete = {r["method"] for r in overall if r["stage"] == stage and r["stratum"] == "tail80"}
        if stage == "b":
            pairs = [(m, "old/" + m) for m in audit.protocol["stage_b"]["methods"] if m != "lightgcn_user"]
            pairs += [("lightgcn_user", "lightgcn"), ("activity_both", "activity_mean"), ("activity_both", "old/activity")]
        else:
            section = audit.protocol["stage_c_music"]
            pairs = [section["primary_transfer_pair"], section["primary_target_pair"], *section["secondary_pairs"], ("activity_mean/full", "activity_mean/zero")]
        for method, baseline in pairs:
            if method not in complete or baseline not in complete:
                continue
            records = [r for r in audit.items if r["stage"] == stage and r["method"] in (method, baseline)]
            value = paired_effects(records, method, baseline, n_bootstrap=audit.protocol["bootstrap_resamples"], seed=audit.protocol["bootstrap_seed"])
            if value["n_origins"] != 3 or any(value[k] for k in ("n_unmatched_item_origins", "n_undefined_item_origins", "n_undefined_seed_pairs")):
                raise ValueError("Incomplete paired population")
            value["cluster_sensitivity"] = unique_item_cluster_interval(value.pop("items"), n_bootstrap=audit.protocol["bootstrap_resamples"], seed=audit.protocol["bootstrap_seed"])
            value.update(stage=stage, stratum="tail80", relative_difference=None if value["baseline_mean"] == 0 else value["mean_difference"] / value["baseline_mean"])
            value["origin_consistency"] = dict(Counter("positive" if r["mean_difference"] > 0 else "negative" if r["mean_difference"] < 0 else "zero" for r in value["origins"]))
            results.append(value)
    return results


def render_report(summary, patch=False):
    title = "Manuscript Result Patch — 补充实验实测结果" if patch else "补充实验 v1 中文报告"
    lines = ["# " + title, "", "生成时间：" + summary["generated_utc"], "",
             "**状态：" + ("请求阶段的规定矩阵全部通过核验。" if summary["complete"] else "进度快照；请求阶段尚未全部完成，不是最终结论。") + "**", "",
             "请求阶段：" + ", ".join(summary["requested_stages"]) + "。未请求的阶段不表示已完成。", "",
             "2015–2017 的 A/B 结果是已查看开发数据上的探索性分析；C 的 2018–2020 音乐结果是本流程中新时间验证，"
             "不是独立领域确认。Movies and TV → Books 因资源条件未排程，未产生领域验证结论。", "",
             "主终点固定为 Tail80 宏平均合格商品 NDCG@20。先平均配对训练种子，再等权平均三个起点。"
             "商品自助法与跨起点唯一商品簇区间均为描述性；共享用户依赖仍未消除。训练轨迹、检查点、策略记录与独立数据集数量不能互换。", "",
             markdown_table(["阶段/组件", "完成", "缺失", "失败", "无效", "规定"],
                            [(key, row["complete"], row["missing"], row["failed"], row["invalid"], row["expected"]) for key, row in summary["status"].items()]), ""]
    if summary["stage_a"]:
        value = summary["stage_a"]
        link = summary["stage_a_relative_path"]
        lines += ["## A：旧材料重新核验与分析", "", f"复用并重新核验 {value['actual_final_evaluations']} 个旧正式评价；"
                  f"新增/保留目标配对共 {value['n_target_pairs']} 项，固定机制配对 {value['n_mechanism_pairs']} 项，"
                  f"历史名义选择配置 {value['n_historical_selection_records']} 项，不同决策 {value['n_distinct_historical_decisions']} 项。"
                  "这些是已有材料的新分析，不新增训练或独立样本。", "",
                  "完整区间和二级终点见 " + f"[目标配对]({link}/analysis_target_pairs.csv)、"
                  f"[选择曲面]({link}/analysis_selection_surface.csv)、[全指标]({link}/analysis_secondary_metrics.csv)、"
                  f"[机制配对]({link}/analysis_mechanism_pairs.csv)。", "",
                  markdown_table(["方法 − 参照", "均值差", "相对点差(%)", "商品 95% 区间", "商品簇 95% 区间", "伤害率"],
                                 [(r["method"] + " − " + r["baseline"], number(float(r["mean_difference"])),
                                   number(float(r["relative_percent_difference"])),
                                   "[" + ", ".join(number(float(r[k])) for k in ("ci_low", "ci_high")) + "]",
                                   "[" + ", ".join(number(float(r[k])) for k in ("unique_item_cluster_ci_low", "unique_item_cluster_ci_high")) + "]",
                                   number(float(r["harm_rate"]))) for r in summary["stage_a_pairs"]]), "",
                  "以上五项比较的两种区间均跨零，不能据此声称稳定优势。全部使用相同的 "
                  f"{summary['stage_a_pairs'][0]['n_item_origins']} 个商品—起点对、"
                  f"{summary['stage_a_pairs'][0]['n_unique_items']} 个唯一商品。"
                  f"完整选择与机制解读见 [A 阶段报告]({link}/ANALYSIS_REPORT_ZH.md)。", ""]
    for stage, label in (("b", "B：预算选择和任务方向"), ("c", "C：同领域新时间验证")):
        lines += ["## " + label, ""]
        selected = [r for r in summary["selections"] if r["stage"] == stage]
        if selected:
            lines += [markdown_table(["方法", "选定轮数", "历史终点", "配置"], [(r["method"], r.get("epochs", 6), number(r.get("historical_score")), r["configuration"]) for r in selected]), ""]
            if stage == "b":
                lines += ["B 相对旧六轮结果同时改变了候选训练预算与三个历史种子的选参过程，最终配置也可能变化。"
                          "这些比较检验更充分选择后的基线表现，不能把差值仅归因于训练轮数；"
                          "即使最终仍选六轮，也可能选择不同配置。", ""]
        if stage == "c":
            diagnostic = [r for r in summary.get("historical_diagnostics", [])
                          if r.get("row_kind") == "ridge_validation" and r.get("selected") and r["subset"] == "full"]
            if diagnostic:
                lines += ["历史 d60 完整收益回归器的选定配置诊断如下；这些不是正式部署指标。"
                          "全部特征组与 alpha 试验见 historical_diagnostics.csv。", "",
                          markdown_table(["底座", "起点", "alpha", "MSE", "零预测 MSE", "符号准确率", "正预测精度", "预测正比例"],
                                         [(r["base"], r["origin"], number(r["alpha"]),
                                           *[number(r[k]) for k in ("mse", "zero_mse", "sign_accuracy", "positive_precision",
                                                                   "predicted_positive_fraction")]) for r in diagnostic]), ""]
        rows = [r for r in summary["overall"] if r["stage"] == stage and r["stratum"] == "tail80"]
        if rows:
            lines += [markdown_table(["方法", "NDCG@20", "Recall@20", "Precision@20", "条件命中率", "全目录 HC@20", "实际记录"],
                                     [(r["method"], *[number(r[k]) for k in ("ndcg20", "recall20", "precision20", "conditional_hit_rate20", "coverage20")], r["n_runs"]) for r in rows]), ""]
        else:
            lines += ["尚无满足全部起点与种子矩阵的方法；不计算跨起点均值。", ""]
        pairs = [r for r in summary["comparisons"] if r["stage"] == stage]
        if pairs:
            lines += [markdown_table(["配对", "差值", "商品 95% 区间", "商品簇 95% 区间", "伤害率"],
                                     [(r["method"] + " − " + r["baseline"], number(r["mean_difference"]),
                                       "[" + ", ".join(number(v) for v in r["ci"]) + "]",
                                       "[" + ", ".join(number(v) for v in r["cluster_sensitivity"]["ci"]) + "]", number(r["harm"]["harm_rate"])) for r in pairs]), ""]
            for row in pairs:
                origin_text = "；".join(f"{r['origin']}: {number(r['mean_difference'])}" for r in row["origins"])
                leave_text = "；".join(f"去掉 {r['omitted_origin']}: {number(r['mean_difference'])}" for r in row["leave_one_origin_out"])
                lines += [f"{row['method']} − {row['baseline']}：逐起点差 {origin_text}。留一起点：{leave_text}。", ""]
    if patch:
        lines += ["## 正文表格与段落更新定位", "",
                  "以下仅使用已核验完整矩阵的实测结果。未完成阶段保留原表，不把空缺改写为零，不将进度快照写作正式结果。", "",
                  "- 表 1（队列和条件长尾评价人口）：保留原数据身份和分母；新增时间块按 metrics_by_origin.csv 的目录/可评价数单列，不合并成独立领域。",
                  "- 表 2（目标与协同模型比较）：保留原六轮结果；补充 B 已核验完整方法的所选预算结果与 LightGCN 用户检索目标对照。不同 epoch 不保证相等计算量。",
                  "- 表 3（配对 NDCG@20 差）：补充 A 的五项目标配对及商品簇敏感性；B 的旧六轮配对和 C 的同底座迁移配对分别单列，明确各自参照。",
                  "- 表 4（固定机制控制）：保留 12 个固定机制全体商品结果，补充伤害率和动作质量/范数；条件子集不得替代全体结果。",
                  "- 表 5（历史收益预测诊断）：保留原历史验证块诊断；C 按底座与起点另表报告 historical_diagnostics.csv，不能把预测损失改善当作部署 NDCG 保证。",
                  "- 表 6（指标与长尾分层敏感性）：按 metrics_overall.csv 和 metrics_by_origin.csv 补充 All/Tail50/Tail80 及 @10/20/50；区分条件命中率和全目录 HC。",
                  "- 表 7（历史选定配置）：保留原目标配置；B 的配置、轮数和预定义延长决定及 C 的固定底座/源修正配置另列，均以 hpo_selected.csv 为依据。",
                  "- 附录补充历史搜索完整曲面、零动作及名义与不同决策配置数；选零不等于所有收益预测非正。收敛曲线及计算预算分别见 historical_curves.csv 和 compute_budget.csv。", "",
                  "### 新增 C 小节可用结果", ""]
        primary = next((r for r in summary["comparisons"] if r["stage"] == "c" and r["method"] == "activity_both/full" and r["baseline"] == "activity_both/zero"), None)
        if primary:
            lines += [f"在预先固定的 2018、2019、2020 年音乐时间块上，组合历史完整收益策略相对相同底座零策略的"
                      f"Tail80 NDCG@20 等起点均值差为 {number(primary['mean_difference'])}，描述性商品区间为 "
                      f"[{number(primary['ci'][0])}, {number(primary['ci'][1])}]，唯一商品簇区间为 "
                      f"[{number(primary['cluster_sensitivity']['ci'][0])}, {number(primary['cluster_sensitivity']['ci'][1])}]。", ""]
            lines += [("该设置未观察到平均部署增量。" if primary["mean_difference"] <= 0 else "该设置观察到正的平均点差；区间和逐起点方向仍需同时报告。") +
                      "这一同领域新时间结果不能推广为独立领域的普遍结论。", ""]
            if any(interval[0] <= 0 <= interval[1] for interval in (primary["ci"], primary["cluster_sensitivity"]["ci"])):
                lines += ["主迁移比较的描述性区间包含零，尚未建立可靠的迁移收益证据。"
                          "次要的完整收益策略与均匀策略比较不能替代同底座零策略的主比较。", ""]
            negative_origins = [str(r["origin"]) for r in primary["origins"] if r["mean_difference"] < 0]
            negative_omissions = [str(r["omitted_origin"]) for r in primary["leave_one_origin_out"] if r["mean_difference"] < 0]
            if negative_origins:
                lines += ["主迁移差在 " + "、".join(negative_origins) + " 年为负。", ""]
            if negative_omissions:
                lines += ["分别去掉 " + "、".join(negative_omissions) + " 年后，剩余起点的平均主迁移差为负；平均正点差依赖起点构成。", ""]
        else:
            lines += ["C 主要迁移配对尚未通过完整矩阵校验，暂不提供可写入正文的新时间数值结论。", ""]
    lines += ["## 工件与解释范围", "", "metrics_runs.csv、metrics_by_origin.csv、metrics_overall.csv 保留 All/Tail50/Tail80 与各截止值。"
              "historical_curves.csv 保留每轮实测历史终点；compute_budget.csv 以实际轨迹记录更新/配对及计时，"
              "共享评估只计一次，进程峰值 RSS 不可相加解释为独立模型内存。", "",
              "status_cells.csv 分开记录 complete/missing/failed/invalid 和旧材料复用。artifact_manifest.json 保留本次读取核验工件的哈希。"
              "公共输出没有用户 ID、逐商品 ID 或排名；逐商品配对只在内存中进行。", ""]
    return "\n".join(lines)


def build_report(args):
    audit = Audit(args)
    for stage in args.stages:
        {"a": audit.stage_a_audit, "b": audit.stage_b_audit, "c": audit.stage_c_audit}[stage]()
    status = {}
    for key in sorted({r["stage"] + "/" + r["group"] for r in audit.status}):
        rows = [r for r in audit.status if r["stage"] + "/" + r["group"] == key]
        counts = Counter(r["state"] for r in rows)
        status[key] = {"expected": len(rows), **{state: counts[state] for state in ("complete", "missing", "failed", "invalid")}}
    complete = all(r["state"] == "complete" for r in audit.status)
    if not complete and not args.allow_partial:
        errors = [f"{r['stage']}/{r['cell']}: {r.get('error', r['state'])}" for r in audit.status if r["state"] != "complete"]
        raise RuntimeError("Strict summary refused incomplete/invalid requested stages; existing reports unchanged.\n" + "\n".join(errors[:12]))
    origins, overall = [], []
    for stage in ("b", "c"):
        years = audit.protocol["stage_b"]["final_origins"] if stage == "b" else audit.protocol["stage_c_music"]["origins"]
        by_origin, all_origins = aggregate_rows([r for r in audit.runs if r["stage"] == stage], years, audit.protocol["seeds"])
        origins.extend(dict(stage=stage, **r) for r in by_origin)
        overall.extend(dict(stage=stage, **r) for r in all_origins)
    pairs = comparisons(audit, overall)
    summary = {"schema": "extensions-v1-public-summary", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "complete": complete, "requested_stages": args.stages, "allow_partial": args.allow_partial,
               "protocol_sha256": audit.protocol_hash, "primary_endpoint": audit.protocol["primary_endpoint"],
               "status": status, "stage_a": audit.stage_a, "stage_a_pairs": audit.stage_a_pairs,
               "stage_a_relative_path": os.path.relpath(args.stage_a_dir, args.output_dir),
               "overall": overall, "comparisons": pairs, "selections": audit.selections,
               "historical_diagnostics": audit.diagnostics,
               "bootstrap_resamples": audit.protocol["bootstrap_resamples"], "bootstrap_seed": audit.protocol["bootstrap_seed"],
               "new_domain": "resource-gated; not evaluated", "aggregation": audit.protocol["aggregation"]}
    # Stage A provenance includes only artifact paths/hash aggregates, never raw IDs.
    output = args.output_dir
    if output.resolve() == (ROOT / "reports").resolve() or any(output.resolve().is_relative_to(p.resolve()) for p in (args.stage_a_dir, args.stage_b_dir, args.stage_c_dir, args.old_results_dir)):
        raise ValueError("Reporting requires a separate extension report directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows, columns in (
        ("status_cells", audit.status, ("stage", "group", "cell", "state", "evidence")),
        ("metrics_runs", audit.runs, ("stage", "method", "origin", "seed", "stratum")),
        ("metrics_by_origin", origins, ("stage", "method", "origin", "stratum")),
        ("metrics_overall", overall, ("stage", "method", "stratum")),
        ("historical_curves", audit.curves, ("stage", "phase", "method", "prefix", "trial", "seed", "epoch")),
        ("compute_budget", audit.budget, ("stage", "phase", "method", "prefix", "seed")),
        ("hpo_records", audit.hpo, ("stage", "method")), ("hpo_selected", audit.selections, ("stage", "method")),
        ("routing_runs", audit.routes, ("stage", "method", "origin", "seed", "stratum")),
        ("historical_diagnostics", audit.diagnostics, ("stage", "component", "base", "origin")),
    ):
        write_csv(output / f"{name}.csv", rows, columns)
    pair_rows = []
    for row in pairs:
        flat = {k: row[k] for k in ("stage", "method", "baseline", "mean_difference", "relative_difference", "n_origins", "n_item_origins", "n_unique_items", "n_seed_pairs")}
        flat.update(ci_low=row["ci"][0], ci_high=row["ci"][1], cluster_ci_low=row["cluster_sensitivity"]["ci"][0], cluster_ci_high=row["cluster_sensitivity"]["ci"][1],
                    harm_rate=row["harm"]["harm_rate"], improvement_rate=row["harm"]["improvement_rate"], tie_rate=row["harm"]["unchanged_rate"])
        for origin in row["origins"]:
            flat[f"origin_{origin['origin']}_difference"] = origin["mean_difference"]
        for omitted in row["leave_one_origin_out"]:
            flat[f"leave_out_{omitted['omitted_origin']}_difference"] = omitted["mean_difference"]
        pair_rows.append(flat)
    write_csv(output / "paired_effects.csv", pair_rows, ("stage", "method", "baseline"))
    write_json(output / "paired_effects.json", pairs)
    write_json(output / "summary.json", summary)
    artifact_records = [{"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "sha256": digest} for p, digest in sorted(audit.hashes.items())]
    write_json(output / "artifact_manifest.json", {"protocol_sha256": audit.protocol_hash, "reporting_code_sha256": sha256(Path(__file__)),
               "dependencies": {"python": platform.python_version(), "numpy": np.__version__}, "artifacts": artifact_records,
               "complete": complete, "requested_stages": args.stages, "no_raw_user_or_item_ids": True})
    (output / "REPORT_ZH.md").write_text(render_report(summary), encoding="utf-8")
    (output / "MANUSCRIPT_RESULT_PATCH.md").write_text(render_report(summary, patch=True), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions_v1.json")
    parser.add_argument("--stages", nargs="+", choices=("a", "b", "c"), default=["a", "b", "c"])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--music-data-dir", type=Path, default=ROOT / "data_extensions/v1/music")
    parser.add_argument("--old-results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--stage-a-dir", type=Path, default=ROOT / "reports/extensions_v1/stage_a")
    parser.add_argument("--stage-b-dir", type=Path, default=ROOT / "results_extensions/v1/stage_b")
    parser.add_argument("--stage-c-dir", type=Path, default=ROOT / "results_extensions/v1/stage_c_music")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/extensions_v1")
    args = parser.parse_args(argv)
    if len(set(args.stages)) != len(args.stages):
        parser.error("Stages must not be duplicated")
    try:
        summary = build_report(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"complete": summary["complete"], "stages": args.stages, "status": summary["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
