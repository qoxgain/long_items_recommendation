#!/usr/bin/env python3
"""Draw publication figures from complete, strictly audited extension reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "saet-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "saet-plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mf_bpr", "lightgcn", "simplex", "lightgcn_user", "activity_mean", "activity_both")
LABELS = {
    "mf_bpr": "MF-BPR", "lightgcn": "LightGCN", "simplex": "SimpleX",
    "lightgcn_user": "LightGCN\nuser objective", "activity_mean": "Activity + mean",
    "activity_both": "Activity + both", "activity_decay": "Activity + decay",
    "activity": "Activity",
}
COLORS = ("#2166ac", "#b2182b", "#218c74", "#8c6bb1", "#555555")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def label(method):
    if "/" in method:
        base, policy = method.split("/", 1)
        return f"{LABELS.get(base, base)} / {policy}"
    return LABELS.get(method, method)


def source_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name, rows, caption):
    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    path = output / f"{name}.csv"
    source_csv(path, rows)
    paths.append(path)
    path = output / f"{name}.txt"
    path.write_text(caption + "\n")
    paths.append(path)
    return paths


def forest(rows, output, name, title, cluster_keys):
    if not rows:
        raise ValueError("No complete paired comparisons")
    fig, ax = plt.subplots(figsize=(9.0, max(3.4, .6 * len(rows) + 1.1)))
    y = np.arange(len(rows))
    for j, (lo, hi, color, marker, legend) in enumerate((
        ("ci_low", "ci_high", COLORS[0], "o", "Item bootstrap"),
        (*cluster_keys, COLORS[1], "s", "Unique-item cluster bootstrap"),
    )):
        point = np.array([number(row, "mean_difference") for row in rows])
        lower = np.array([number(row, lo) for row in rows])
        upper = np.array([number(row, hi) for row in rows])
        if np.any(lower > upper):
            raise ValueError("Reversed interval")
        pos = y + (-.11 if j == 0 else .11)
        ax.hlines(pos, lower, upper, color=color, linewidth=1.7)
        ax.scatter(point, pos, color=color, marker=marker, s=28, label=legend, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{label(r['method'])} − {label(r['baseline'])}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Difference in Tail80 I-NDCG@20")
    ax.set_title(title, loc="left", pad=12)
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    fig.tight_layout()
    return save_figure(fig, output, name, rows,
        "Paired seed means within origin, then equal origin weighting. Bars show 95% "
        "descriptive bootstrap intervals (2,000 draws, seed 2026). Shared-user and "
        "time dependence are not fully removed. Intervals crossing zero do not establish an advantage.")


def stage_a(summary, output):
    rows = summary["stage_a_pairs"]
    if len(rows) != 5:
        raise ValueError("Stage A requires the five prespecified paired comparisons")
    return forest(rows, output, "a_target_paired_effects",
                  "A · Previously viewed development origins (2015–2017)",
                  ("unique_item_cluster_ci_low", "unique_item_cluster_ci_high"))


def stage_b(report, summary, output):
    selected = {r["method"]: r for r in read_csv(report / "hpo_selected.csv") if r["stage"] == "b"}
    if set(selected) != set(METHODS):
        raise ValueError("B requires all six verified selections")
    curves = [r for r in read_csv(report / "historical_curves.csv")
              if r["stage"] == "b" and r["phase"] == "hpo"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    plotted = []
    for method, ax in zip(METHODS, axes.flat):
        chosen = selected[method]
        maximum = 100 if chosen["extended_to_100"] == "True" else 50
        selected_trial = int(chosen["trial"])
        rows = [r for r in curves if r["method"] == method]
        expected = {(trial, seed, epoch) for trial in range(12) for seed in (101, 202, 303)
                    for epoch in range(1, maximum + 1)}
        lookup = {(int(r["trial"]), int(r["seed"]), int(r["epoch"])): r for r in rows}
        if set(lookup) != expected or len(rows) != len(expected):
            raise ValueError(f"Incomplete/duplicated historical curve: {method}")
        x = np.arange(1, maximum + 1)
        for trial in range(12):
            values = np.array([[number(lookup[trial, seed, e], "historical_tail80_ndcg20")
                                for e in x] for seed in (101, 202, 303)])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            chosen_trial = trial == selected_trial
            ax.plot(x, mean, color=COLORS[0] if chosen_trial else "#999999",
                    alpha=1 if chosen_trial else .35, linewidth=1.8 if chosen_trial else .7,
                    zorder=3 if chosen_trial else 1)
            if chosen_trial:
                ax.fill_between(x, mean - sd, mean + sd, color=COLORS[0], alpha=.12)
                epoch = int(chosen["epochs"])
                if not np.isclose(mean[epoch - 1], number(chosen, "historical_score"),
                                  rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Selected score differs from curve: {method}")
                ax.scatter([epoch], [mean[epoch - 1]], s=38, marker="D", color=COLORS[1], zorder=5)
                ax.axvline(epoch, color=COLORS[1], linestyle=":", linewidth=.8)
            plotted.extend({"method": method, "trial": trial, "epoch": int(epoch), "n_seeds": 3,
                            "mean_historical_ndcg20": float(m), "seed_sd": float(s),
                            "selected_trial": chosen_trial, "selected_epoch": int(chosen["epochs"])}
                           for epoch, m, s in zip(x, mean, sd))
        ax.set_title(label(method).replace("\n", " "), fontsize=10, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Historical I-NDCG@20")
        ax.grid(alpha=.18)
    fig.suptitle("B · Historical selection curves; complete three-seed means", x=.07, ha="left")
    fig.tight_layout()
    paths = save_figure(fig, output, "b_historical_convergence", plotted,
        "Grey curves: all 12 configurations. Blue: historically selected configuration; "
        "shading: ±1 optimization-seed SD, not a confidence interval. Diamond: selected "
        "checkpoint. All curves use seeds 101/202/303. The same prespecified rule chooses "
        "6/20/50 epochs, extending all family trajectories to 100 only when triggered. "
        "No final-query metric selects a curve or checkpoint.")
    overall = [r for r in read_csv(report / "metrics_overall.csv")
               if r["stage"] == "b" and r["stratum"] == "tail80"]
    lookup = {r["method"]: r for r in overall}
    required = set(METHODS) | {f"old/{m}" for m in METHODS if m != "lightgcn_user"}
    if not required.issubset(lookup):
        raise ValueError("B final comparison matrix incomplete")
    chosen_rows = [lookup[m] for m in METHODS]
    old_rows = [lookup[f"old/{m}"] for m in METHODS if m != "lightgcn_user"]
    for row in chosen_rows + old_rows:
        if row["complete"] != "True" or int(row["n_origins"]) != 3 or int(row["n_runs"]) != 15:
            raise ValueError("B plotted model lacks the complete three-origin/five-seed matrix")
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(METHODS))
    old_positions = [i for i, m in enumerate(METHODS) if m != "lightgcn_user"]
    ax.bar(np.array(old_positions) - .18, [number(r, "ndcg20") for r in old_rows],
           width=.34, color="#92979c", label="Original six-epoch experiment")
    ax.bar(x + .18, [number(r, "ndcg20") for r in chosen_rows],
           width=.34, color=COLORS[0], label="Historical budget/configuration selection")
    ax.set_xticks(x, [label(m) for m in METHODS])
    ax.set_ylabel("Tail80 I-NDCG@20")
    ax.set_title("B · Development-origin outcomes (2015–2017)", loc="left")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    paths += save_figure(fig, output, "b_final_comparison", chosen_rows + old_rows,
        "Five optimization seeds averaged within each origin, then three origins weighted "
        "equally. Bars are point estimates, not uncertainty intervals. The original run "
        "has no user-objective LightGCN comparator, so its grey bar is absent. These are "
        "previously viewed development origins, not independent confirmation.")
    return paths


def stage_c(report, summary, output):
    rows = [r for r in read_csv(report / "metrics_by_origin.csv")
            if r["stage"] == "c" and r["stratum"] == "tail80"]
    methods = {"activity"} | {f"{base}/{policy}" for base in ("activity_mean", "activity_both")
                              for policy in ("zero", "uniform", "semantic", "full")}
    lookup = {(r["method"], int(r["origin"])): r for r in rows}
    expected = {(method, year) for method in methods for year in (2018, 2019, 2020)}
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError("C requires all policies and all three origins")
    for (method, _), row in lookup.items():
        if row["complete"] != "True" or int(row["n_seeds"]) != (1 if method == "activity" else 5):
            raise ValueError("C plotted origin has incomplete optimization seeds")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    years = (2018, 2019, 2020)
    for base, ax in zip(("activity_mean", "activity_both"), axes):
        for k, policy in enumerate(("zero", "uniform", "semantic", "full")):
            values = [number(lookup[base + "/" + policy, y], "ndcg20") for y in years]
            ax.plot(years, values, marker=("o", "s", "^", "D")[k], color=COLORS[k],
                    linestyle=("-", "--", ":", "-.")[k], label=policy, markersize=5)
        ax.plot(years, [number(lookup["activity", y], "ndcg20") for y in years],
                marker="x", color=COLORS[4], linestyle="--", label="activity")
        ax.set_title(label(base), loc="left")
        ax.set_xticks(years)
        ax.set_xlabel("Final-query origin")
        ax.grid(alpha=.18)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Tail80 I-NDCG@20")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("C · New-time validation within the same domain", x=.07, ha="left")
    fig.tight_layout()
    paths = save_figure(fig, output, "c_metrics_by_origin", rows,
        "Each point is the mean of five optimization seeds (activity is deterministic). "
        "Lines join prespecified origins for visibility, not interpolation. Overlapping "
        "curves are retained, including exact zero-radius ties. These are same-domain "
        "new-time results, not independent-domain validation. See paired intervals "
        "instead of inferring significance from point-estimate separation.")
    pairs = [r for r in read_csv(report / "paired_effects.csv") if r["stage"] == "c"]
    if len(pairs) != 6:
        raise ValueError("C requires all six prespecified paired comparisons")
    return paths + forest(pairs, output, "c_paired_effects",
                          "C · Paired effects across the three new-time origins",
                          ("cluster_ci_low", "cluster_ci_high"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/extensions_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/extensions_v1/figures")
    parser.add_argument("--stages", nargs="+", choices=("a", "b", "c"), default=("a", "b", "c"))
    args = parser.parse_args()
    report = args.report_dir
    summary = json.loads((report / "summary.json").read_text())
    audit = json.loads((report / "artifact_manifest.json").read_text())
    if (summary.get("schema") != "extensions-v1-public-summary"
            or not summary.get("complete") or summary.get("allow_partial")
            or not audit.get("complete")
            or summary.get("protocol_sha256") != audit.get("protocol_sha256")
            or not set(args.stages).issubset(summary.get("requested_stages", []))):
        raise SystemExit("Figures require a complete strict report containing every requested stage")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "white"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in dict.fromkeys(args.stages):
        paths += stage_a(summary, args.output_dir) if stage == "a" else (
            stage_b(report, summary, args.output_dir) if stage == "b" else stage_c(report, summary, args.output_dir))
    inputs = ["summary.json", "artifact_manifest.json"]
    if "b" in args.stages:
        inputs += ["historical_curves.csv", "hpo_selected.csv", "metrics_overall.csv"]
    if "c" in args.stages:
        inputs += ["metrics_by_origin.csv", "paired_effects.csv"]
    provenance = {
        "schema": "extensions-v1-figures", "stages": args.stages,
        "protocol_sha256": summary["protocol_sha256"],
        "strict_report_generated_utc": summary["generated_utc"],
        "inputs": {name: digest(report / name) for name in inputs},
        "plotting_code_sha256": digest(__file__),
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                         "matplotlib": matplotlib.__version__},
        "outputs": {p.name: digest(p) for p in paths},
        "raw_user_or_item_ids": False,
    }
    name = "figure_manifest_" + "".join(dict.fromkeys(args.stages)) + ".json"
    (args.output_dir / name).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "stages": args.stages, "files": len(paths),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
