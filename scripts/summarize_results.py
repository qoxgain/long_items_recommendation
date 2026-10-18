#!/usr/bin/env python3
"""Generate public aggregate evidence from completed independent reconstruction runs.

The default is deliberately strict: every prespecified cell must exist and pass
validation. --allow-partial writes an explicitly incomplete progress report, but
never averages over a missing seed or origin. Raw IDs and rankings stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from saet_repro.evaluation import stratified_summaries
from saet_repro.io import json_safe, sha256, write_json
from saet_repro.statistics import paired_effects

YEARS = (2015, 2016, 2017)
SEEDS = (11, 22, 33, 44, 55)
OFFSETS = (270, 180, 90, 60, 30, 0)
TARGETS = ("unbounded_mean", "activity_mean", "activity_decay",
           "activity_conditional", "activity_both")
COLLABORATIVE = ("mf_bpr", "lightgcn", "simplex")
DETERMINISTIC = ("cosine", "activity")
POLICIES = ("zero", "uniform", "semantic", "global", "behavior", "full", "predicted_single")
STRATA = ("all", "tail80", "tail50")
METRICS = tuple(f"{metric}{k}" for k in (10, 20, 50)
                for metric in ("ndcg", "recall", "precision", "coverage"))
ROUTING_FIELDS = ("source_mass", "zero", "correction_norm", "effective_donors", "concentration")


@dataclass(frozen=True)
class Cell:
    group: str
    name: str
    required: tuple[Path, ...]
    origin: int | None = None
    method: str | None = None
    seed: int | None = None
    result: Path | None = None
    evaluation: Path | None = None
    frozen: Path | None = None
    prefix: str | None = None


def expected_cells(data_dir, results_dir):
    """Enumerate the frozen matrix; directory glob counts cannot satisfy it."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    cells = []
    for year in YEARS:
        for offset in OFFSETS:
            prefix = f"cds_{year}_d{offset}"
            path = data_dir / prefix
            cells.append(Cell("data_prefixes", prefix, (path.with_suffix(".npz"), path.with_suffix(".json")),
                              year, prefix=prefix))
    for family, methods in (("targets", TARGETS), ("baselines", COLLABORATIVE)):
        for method in methods:
            directory = results_dir / family / "cds_2015_d30" / method
            selection = (directory / "selected.json",)
            if family == "targets":
                selection += (directory / "selected_model.npz",)
            cells.append(Cell(f"{family}_selected", method, selection, method=method,
                              result=directory / "selected.json", origin=2015, prefix="cds_2015_d30"))
            for trial in range(12):
                stem = directory / "search" / f"trial_{trial:02d}"
                result = stem.with_suffix(".json")
                evaluation = stem.with_name(stem.name + ("_test.npz" if family == "targets" else ".test.npz"))
                frozen = stem.with_suffix(".frozen.json")
                cells.append(Cell(f"{family}_hpo", f"{method}/trial_{trial:02d}",
                                  (result, evaluation, stem.with_suffix(".npz"), frozen),
                                  2015, method, 101, result, evaluation, frozen, "cds_2015_d30"))
            for year in YEARS:
                for seed in SEEDS:
                    prefix = f"cds_{year}_d0"
                    directory = results_dir / family / prefix / method / f"seed_{seed}"
                    result, evaluation = directory / "result.json", directory / "test.npz"
                    frozen = directory / ("result.frozen.json" if family == "targets" else "frozen.json")
                    cells.append(Cell(f"{family}_final", f"{prefix}/{method}/seed_{seed}",
                                      (result, evaluation, directory / "model.npz", frozen),
                                      year, method, seed, result, evaluation, frozen, prefix))
    for year in YEARS:
        prefix = f"cds_{year}_d0"
        for method in DETERMINISTIC:
            directory = results_dir / "targets" / prefix / method / "seed_0"
            result, evaluation, frozen = (directory / name for name in ("result.json", "test.npz", "frozen.json"))
            cells.append(Cell("deterministic_final", f"{prefix}/{method}", (result, evaluation, frozen),
                              year, method, None, result, evaluation, frozen, prefix))
        for offset in OFFSETS[:4]:
            directory = results_dir / "utility" / f"cds_{year}" / "historical" / f"d{offset}"
            required = tuple(directory / name for name in ("labels.npz", "model.npz", "result.json", "labels.frozen.json"))
            cells.append(Cell("utility_historical", f"cds_{year}/d{offset}", required,
                              year, result=directory / "result.json", frozen=directory / "labels.frozen.json",
                              prefix=f"cds_{year}_d{offset}"))
        directory = results_dir / "utility" / f"cds_{year}" / "ridges"
        required = tuple(directory / name for name in ("selected.json", "frozen.json", "validation_mixtures.json",
                                                       "semantic.npz", "global.npz", "behavior.npz", "full.npz"))
        cells.append(Cell("utility_ridges", f"cds_{year}", required, year, result=directory / "selected.json",
                          frozen=directory / "frozen.json"))
        for seed in SEEDS:
            shared = results_dir / "utility" / prefix / f"seed_{seed}"
            for mechanism, policies in (("main", POLICIES), ("mass1", POLICIES[1:]), ("norm01", POLICIES[1:])):
                for policy in policies:
                    name = f"{mechanism}_{policy}"
                    directory = shared / name
                    result, evaluation = directory / "result.json", directory / "test.npz"
                    frozen = shared / "evaluation.frozen.json"
                    required = (result, evaluation, directory / "routing.npz", frozen, shared / "complete.json")
                    cells.append(Cell("utility_primary" if mechanism == "main" else "utility_mechanism",
                                      f"{prefix}/seed_{seed}/{name}", required, year,
                                      name, seed, result, evaluation, frozen, prefix))
    directory = results_dir / "utility" / "cds_2015_d30"
    cells.append(Cell("utility_selected", "policies", (directory / "selected.json", directory / "search.frozen.json"),
                      result=directory / "selected.json", frozen=directory / "search.frozen.json", prefix="cds_2015_d30"))
    for policy in POLICIES:
        for trial in range(1 if policy == "zero" else 12):
            name = "zero" if policy == "zero" else f"{policy}_{trial:02d}"
            result, evaluation = directory / "search" / f"{name}.json", directory / "search" / f"{name}.npz"
            frozen = directory / "search.frozen.json"
            cells.append(Cell("utility_hpo", name, (result, evaluation, frozen),
                              2015, policy, 101, result, evaluation, frozen, "cds_2015_d30"))
    return cells


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_catalog(path):
    # Deliberately do not read user IDs, event tables or the saved top-user ranks.
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in ("target_item_ids", "target_degree", "target_tail80", "target_tail50")}


def validate_hpo_selection(directory, document, family, method, verify_hash):
    """Tie selected configurations to every recorded trial and its actual score."""
    trials = document["trials"]
    if len(trials) != 12 or {row["trial"] for row in trials} != set(range(12)):
        raise ValueError("Selected model must contain each of the 12 HPO trials exactly once")
    score_key = "score" if family == "targets" else "tail80_ndcg20"
    for row in trials:
        trial_path = directory / "search" / f"trial_{row['trial']:02d}.json"
        trial = read_json(trial_path)
        score = trial["summary"]["1"]["tail80"]["ndcg20"]
        if (trial["method"] != method or trial["seed"] != 101 or
                trial["config"] != row["config"] or row[score_key] != score):
            raise ValueError("HPO selection history differs from actual trial results")
        if family == "baselines":
            verify_hash(trial_path, row["result_sha256"])
    selectable = [row for row in trials if row[score_key] is not None and np.isfinite(row[score_key])]
    if not selectable:
        raise ValueError("No defined HPO selection metric")
    best = max(selectable, key=lambda row: (row[score_key], -row["trial"]))
    recorded_score = document["score" if family == "targets" else "criterion_value"]
    if (document["trial"] != best["trial"] or document["config"] != best["config"] or
            recorded_score != best[score_key]):
        raise ValueError("Selected model is not the prespecified HPO winner")
    if family == "targets":
        winner = read_json(directory / "search" / f"trial_{best['trial']:02d}.json")
        if document["frozen"] != winner["frozen"]:
            raise ValueError("Selected target provenance differs from winning trial")
        verify_hash(directory / "selected_model.npz", winner["model_sha256"])


def validate_utility_selection(directory, document, frozen):
    """Reconstruct seven policy winners from the frozen 73-candidate search."""
    names = ["zero"] + [f"{policy}_{trial:02d}" for policy in POLICIES[1:] for trial in range(12)]
    trials = document["trials"]
    if (len(trials) != len(names) or {row["name"] for row in trials} != set(names) or
            set(document["policies"]) != set(POLICIES) or set(frozen["configurations"]) != set(names)):
        raise ValueError("Utility selection must contain each of the 73 candidates and seven policies")
    by_name = {row["name"]: row for row in trials}
    for name in names:
        row = by_name[name]
        trial = read_json(directory / "search" / f"{name}.json")
        if (trial["name"] != name or trial["configuration"] != frozen["configurations"][name] or
                row["configuration"] != trial["configuration"] or
                row["score"] != trial["summary"]["1"]["tail80"]["ndcg20"]):
            raise ValueError("Utility selection history differs from frozen trial evidence")
    if document["policies"]["zero"] != frozen["configurations"]["zero"]:
        raise ValueError("Selected zero policy differs from frozen exact-null configuration")
    for policy in POLICIES[1:]:
        candidates = [by_name[f"{policy}_{trial:02d}"] for trial in range(12)]
        candidates = [row for row in candidates if row["score"] is not None and np.isfinite(row["score"])]
        if not candidates:
            raise ValueError("No defined utility HPO selection metric")
        best = max(candidates, key=lambda row: row["score"])
        if document["policies"][policy] != best["configuration"]:
            raise ValueError("Selected utility policy is not the prespecified HPO winner")


def validate_evaluation(path, data, stored_summary=None):
    """Recompute every public number from a complete stage-1 result array."""
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files if key != "top_users"}
    if int(result["stage"]) != 1 or str(result["candidate_mode"].item()) != "full_eligible_users":
        raise ValueError("Only stage 1 full-eligible-user evaluations are allowed")
    if not np.array_equal(data["target_item_ids"], result["item_id"]):
        raise ValueError("Evaluation item catalog differs from its independent prefix")
    eligible = data["target_degree"] > 0
    if not np.array_equal(result["eligible_catalog"], eligible) or not np.array_equal(result["requested"], eligible):
        raise ValueError("Evaluation does not request the entire warm catalog")
    expected = eligible & (result["n_positive"] > 0)
    for name in ("evaluated", "scored", "ranked"):
        if not np.array_equal(result[name], expected):
            raise ValueError(f"Incomplete full-candidate {name} mask")
    for k in (10, 20, 50):
        for metric in ("ndcg", "recall", "precision", "hits"):
            value = result[f"{metric}{k}"]
            if value.shape != expected.shape or np.any(~np.isfinite(value[expected])):
                raise ValueError(f"Undefined evaluated {metric}{k} values")
            if np.any(np.isfinite(value[~expected])):
                raise ValueError("Items without eligible query positives must remain undefined")
    summaries = stratified_summaries(data, result)
    if stored_summary is not None:
        for stratum in STRATA:
            for metric in METRICS:
                actual = summaries[stratum][metric]
                recorded = stored_summary["1"][stratum][metric]
                if recorded is None:
                    if np.isfinite(actual):
                        raise ValueError(f"Recorded summary is undefined: {stratum}/{metric}")
                elif not np.isclose(actual, recorded, rtol=1e-9, atol=1e-12, equal_nan=True):
                    raise ValueError(f"Recorded summary differs from arrays: {stratum}/{metric}")
    mask = expected & data["target_tail80"]
    paired = {"item_id": result["item_id"][mask], "ndcg20": result["ndcg20"][mask]}
    return summaries, paired


def aggregate_runs(rows, years=YEARS, seeds=SEEDS):
    """Require all seeds within an origin and all origins for the macro mean.

    Missing/undefined means stay undefined. Optimization-seed dispersion is
    descriptive and is never used as an independent-sample confidence interval.
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["method"], row["stratum"], row["origin"])
        if row["seed"] in grouped[key]:
            raise ValueError(f"Duplicate run summary for {key}, seed={row['seed']}")
        grouped[key][row["seed"]] = row
    origins = []
    for (method, stratum, origin), values in sorted(grouped.items()):
        expected = {None} if method in DETERMINISTIC else set(seeds)
        if set(values) != expected:
            continue
        result = {"method": method, "stratum": stratum, "origin": origin,
                  "n_seeds": len(expected), "complete": True}
        for metric in METRICS:
            numbers = np.asarray([row[metric] for row in values.values()], dtype=float)
            result[metric] = float(numbers.mean())
            result[metric + "_seed_sd"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else None
        for name in ("items_catalog", "items_evaluated", "items_without_positive", "n_positive"):
            counts = {row[name] for row in values.values()}
            if len(counts) != 1:
                raise ValueError(f"Seed-specific evaluation population for {method}/{origin}/{stratum}")
            result[name] = counts.pop()
        origins.append(result)
    methods = defaultdict(list)
    for row in origins:
        methods[row["method"], row["stratum"]].append(row)
    overall = []
    for (method, stratum), values in sorted(methods.items()):
        if {row["origin"] for row in values} != set(years):
            continue
        result = {"method": method, "stratum": stratum, "n_origins": len(years),
                  "n_runs": sum(row["n_seeds"] for row in values),
                  "origin_weighting": "equal", "complete": True}
        result.update({metric: float(np.mean([row[metric] for row in values])) for metric in METRICS})
        overall.append(result)
    return origins, overall


def comparison_records(method, paired_runs):
    rows = []
    for (name, origin, seed), result in paired_runs.items():
        if name == method:
            rows.extend({"method": name, "origin": origin, "seed": seed,
                         "item_id": str(item), "ndcg20": float(value)}
                        for item, value in zip(result["item_id"], result["ndcg20"]))
    return rows


def make_comparisons(overall, paired_runs, n_bootstrap):
    complete = {row["method"] for row in overall if row["stratum"] == "tail80"}
    pairs = {(method, "activity_mean") for method in complete if method != "activity_mean"}
    for prefix in ("main", "mass1", "norm01"):
        pairs.update((f"{prefix}_full", f"{prefix}_{policy}")
                     for policy in POLICIES[1:] if policy != "full")
    comparisons = []
    for method, baseline in sorted(pairs):
        if method not in complete or baseline not in complete:
            continue
        rows = comparison_records(method, paired_runs) + comparison_records(baseline, paired_runs)
        result = paired_effects(rows, method, baseline, n_bootstrap=n_bootstrap)
        if result["n_origins"] != len(YEARS) or result["n_unmatched_item_origins"] or result["n_undefined_seed_pairs"]:
            raise ValueError(f"Incomplete paired evaluation population for {method} vs {baseline}")
        # Item IDs are needed for pairing locally, but never exported.
        result.pop("items")
        result["stratum"] = "tail80"
        differences = [row["mean_difference"] for row in result["origins"]]
        result["origin_consistency"] = {
            "positive": sum(value > 0 for value in differences),
            "negative": sum(value < 0 for value in differences),
            "zero": sum(value == 0 for value in differences)}
        comparisons.append(result)
    return comparisons


def write_csv(path, rows, default_fields):
    fields = list(default_fields)
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(json_safe(row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    temporary.replace(path)


def number(value, digits=6):
    return "未定义" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render_report(summary, selection_rows=()):
    complete = summary["complete"]
    lines = ["# 独立重建实验报告", "", f"生成时间：{summary['generated_utc']}。",
             "", "**状态：" + ("规定实验矩阵已完整运行并通过工件核验。**" if complete else
                 "实验尚未完成；本文件是进度快照，不是最终复现实验结论。**"), "",
             "本报告只统计本仓库新实现实际写出的完整候选评估工件。原始代码、模型检查点与逐商品预测已丢失，"
             "因此不能重放原第二轮 A 阶段，也不能声称重新完成原第二轮 B 阶段的覆盖审计。"
             "历史报告数值只保存在 [historical_reference.csv](historical_reference.csv)，不进入本报告的新结果计算。", "",
             "## 运行完成情况", "",
             markdown_table(["实验部分", "已核验", "规定数量", "异常"],
                            [(key, row["completed"], row["expected"], row["invalid"])
                             for key, row in summary["status"].items()]), "",
             "缺失单元不会被当作零分；一个起点必须完成全部五个种子才能形成起点均值，一个方法必须完成全部三个起点才能形成跨起点均值。"
             "确定性 cosine/activity 每起点只运行一次，配对统计使用 seed=None 广播。", "",
             "## 任务与重建差异", "",
             "Digital Music → CDs and Vinyl；2015、2016、2017 三个时间起点；每个起点六个独立前缀，共 18 个。"
             "每个前缀拟合前 365 天并查询其后 30 天。目标用户最多 6,000 人，完整候选排除任何历史评分；"
             "训练、尾部目录、标题编码与标准化均只依赖当时拟合数据。Tail80 按拟合正反馈度数定义。", "",
             "主指标为有合格查询正样本商品的宏平均 NDCG@20；没有查询正样本的商品指标保持未定义，仍进入全目录覆盖率分母。"
             "先对同起点五个优化种子取均值，再对三个起点等权平均，不按商品数量或种子数增加起点权重。"
             "NDCG/Recall/Precision/Coverage@10/20/50 及 All/Tail50 见聚合 CSV。", "",
             "原文未完整给出偏好特征映射，本次明确选择 `unit(concat(h, x*h, abs(x-h)))`（192 维）；"
             "CF 模型为独立实现的任务适配版本。当前 Amazon 2023 标题元数据不提供历史时点快照，标题当时的可用性无法核实。"
             "具体差异与冻结协议见 [范围说明](../docs/RECONSTRUCTION_SCOPE.md)、"
             "[数据协议](../docs/data_protocol.md)、[目标模型协议](../docs/target_protocol.md)、"
             "[CF 协议](../docs/cf_protocol.md) 和 [评估协议](../docs/evaluation_protocol.md)。", "",
             "HPO 在 2015 d30 使用 seed=101、六个固定训练轮次：五个目标模型各 12 次、三种 CF 各 12 次；"
             "迁移部署共 73 个候选（精确零迁移一次，其余六种策略各 12 个半径/温度组合）。"
             "历史 d270/d180/d90 学习收益回归，d60 选 ridge 后四折重拟合；最终 d0 不参与选参或选轮。", "",
             "## 新运行主结果", ""]
    main_rows = [row for row in summary["overall"] if row["stratum"] == "tail80"
                 and not row["method"].startswith(("mass1_", "norm01_"))]
    if main_rows:
        lines += [markdown_table(["方法", "Tail80 NDCG@20", "Recall@20", "Coverage@20", "实际运行数"],
                                 [(r["method"], number(r["ndcg20"]), number(r["recall20"]),
                                   number(r["coverage20"]), r["n_runs"]) for r in main_rows]), ""]
        target_rows = {row["method"]: row for row in main_rows if row["method"] in TARGETS}
        if set(target_rows) == set(TARGETS):
            strongest = max(target_rows.values(), key=lambda row: row["ndcg20"])
            lines += [f"本次五种目标模型中，{strongest['method']} 的 Tail80 NDCG@20 点估计最高"
                      f"（{number(strongest['ndcg20'])}）；迁移底座按预先声明的协议固定为 activity_mean"
                      f"（{number(target_rows['activity_mean']['ndcg20'])}）。本次实验没有检验在最强目标模型上加入迁移，"
                      "因此没有证据表明迁移能进一步改善最强目标模型。固定非零半径机制控制的点估计属于独立探索性结果，"
                      "应与经 HPO 选定的主策略结果分别解释，不能替代主策略的迁移增益结论。", ""]
    else:
        lines += ["尚无完成全部三个起点与规定种子的主方法，暂不报告跨起点性能均值。", ""]
    lines += ["## 源迁移增益、伤害与起点一致性", ""]
    focus = next((r for r in summary["comparisons"]
                  if r["method"] == "main_full" and r["baseline"] == "activity_mean"), None)
    if focus:
        lo, hi = focus["ci"]
        signs = focus["origin_consistency"]
        lines += [f"完整收益策略相对冻结 activity_mean 底座的等权 Tail80 NDCG@20 差为 "
                  f"**{number(focus['mean_difference'])}**，商品配对描述性 95% 区间 "
                  f"[{number(lo)}, {number(hi)}]。三个起点中正向 {signs['positive']}、"
                  f"负向 {signs['negative']}、零差 {signs['zero']}。", "",
                  markdown_table(["起点", "底座", "完整收益策略", "配对差", "伤害商品比例"],
                                 [(r["origin"], number(r["baseline_mean"]), number(r["method_mean"]),
                                   number(r["mean_difference"]), number(r["harm_rate"])) for r in focus["origins"]]), "",
                  f"等起点权重伤害比例为 {number(focus['harm']['harm_rate'])}；"
                  f"改善比例为 {number(focus['harm']['improvement_rate'])}。"
                  f"商品－起点对数量为 {focus['n_item_origins']}，种子先配对求差并平均，未作为独立数据集。", ""]
        if focus["mean_difference"] <= 0:
            lines += ["本次完整收益策略未提高该预设主指标的跨起点均值；这不能被表述为源迁移已获益。", ""]
        elif lo <= 0 <= hi or signs["negative"]:
            lines += ["跨起点均值虽为正，但区间或起点方向提示不稳定性；本次开发实验不足以支持稳定迁移获益结论。", ""]
        else:
            lines += ["本次三个音乐开发起点方向一致；此结果仍只是该重建与数据范围内的描述性证据，不能替代新类别验证。", ""]
    else:
        lines += ["完整收益策略与底座的规定矩阵尚不齐全，不计算缺失条件下的增益或置信区间。", ""]
    source_pairs = [r for r in summary["comparisons"] if r["method"].startswith("main_")
                    and r["baseline"] == "activity_mean"]
    if source_pairs:
        lines += [markdown_table(["策略 − activity_mean", "配对差", "描述性 95% 区间", "伤害率"],
                                 [(r["method"], number(r["mean_difference"]),
                                   f"[{number(r['ci'][0])}, {number(r['ci'][1])}]", number(r["harm"]["harm_rate"]))
                                  for r in source_pairs]), ""]
    selected_policies = {row["method"]: json.loads(row["configuration"]) for row in selection_rows
                         if row["group"] == "utility_selected"}
    if focus and all(policy in selected_policies and selected_policies[policy]["radius"] == 0
                     for policy in POLICIES[1:]):
        lines += ["本次主策略 HPO 为 uniform、semantic、global、behavior、full、predicted_single 六种迁移策略"
                  "均选择了 R=0（见 `hpo_selected.csv`），部署时均执行零修正。因此它们与底座相同的分数和配对区间 "
                  "[0, 0] 来自选定的零迁移动作；这不说明学到的收益预测始终非正，也不能把该退化区间解释为"
                  "对非零迁移的普遍无效性、无伤害性或跨数据集不确定性的保证。", ""]
    lines += ["区间按起点内部商品重采样 2,000 次（或命令行显式指定次数，见 JSON），每次起点等权平均。"
              "`paired_effects.json` 还给出整起点 bootstrap 与逐起点剔除敏感性。共享用户依赖、跨起点重复商品依赖"
              "和重叠时间窗依赖没有被商品 bootstrap 保留；只有三个开发起点，且比较属于探索性分析，"
              "不报告确认性显著性或把优化种子当作独立样本。", "",
              "## 机制与历史收益诊断", "",
              "`mass1_*` 统一非零源权重质量，`norm01_*` 统一非零修正范数为 0.1；两类独立控制均保留原生精确零动作，"
              "使用固定 R=0.1、T=1，与各主策略的 HPO 配置不同。全部控制的主指标见 `metrics_overall.csv`，"
              "源质量、零动作率、修正范数、有效 donor 数与集中度见 `routing_summary.csv`。"
              "历史回归验证集参与 ridge 选择，不能称为未触碰测试集；预测诊断和真实混合收益诊断见 `utility_diagnostics.csv`。", "",
              f"精确零迁移已逐项核对的完整运行：{summary['exact_zero_verified_runs']}/15。"
              "核对 main_zero 与相同起点、相同 seed 的 canonical activity_mean 指标和目录数组。", "",
              "## 工件与运行证据", "",
              "所有 CSV 仅含起点、方法、种子和聚合数值，不包含用户 ID、逐用户排序或逐商品原始记录。"
              "`artifact_manifest.json` 保存参与报告的实际工件 SHA256、当前代码 SHA256、冻结记录的代码匹配情况、"
              "依赖版本与生成环境；大型数据、模型、路由和逐商品数组留在本地 `data/` 与 `results/`。", "",
              "`runtime_summary.csv` 按组件记录 runner 实测秒数；多个组件可并行，不能相加解释为端到端墙钟时间。"
              "utility 的 19 策略共享一次评估计时，只计一次；未有独立计时的工作不虚构耗时。"
              "该报告不将历史测试数量或历史指标当作本工程新运行证据。", "",
              "生成最终报告：`.venv/bin/python scripts/summarize_results.py`；"
              "查看未完成进度：追加 `--allow-partial`。默认命令在任一规定单元缺失或核验失败时退出且不覆盖现有报告。", ""]
    if summary["invalid_cells"]:
        lines += ["### 当前核验异常", ""]
        lines += [f"- `{row['group']}/{row['name']}`：{row['error']}" for row in summary["invalid_cells"]]
        lines += [""]
    return "\n".join(lines)


def build_report(data_dir, results_dir, output_dir, allow_partial=False, n_bootstrap=2000):
    data_dir, results_dir, output_dir = map(Path, (data_dir, results_dir, output_dir))
    cells = expected_cells(data_dir, results_dir)
    counts = Counter(cell.group for cell in cells)
    status = {group: {"expected": count, "completed": 0, "missing": 0, "invalid": 0}
              for group, count in counts.items()}
    missing, invalid, artifact_paths, code_checks = [], [], set(), []
    catalogs, hashes, runs, paired_runs = {}, {}, [], {}
    data_rows, selection_rows, diagnostic_rows, routing_rows, runtime_rows = [], [], [], [], []
    zero_verified = 0
    verified_paths = set()

    def digest(path):
        path = Path(path)
        if path not in hashes:
            hashes[path] = sha256(path)
        return hashes[path]

    def relative(path):
        path = Path(path)
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def verify_hash(path, expected):
        if expected and digest(path) != expected:
            raise ValueError(f"SHA256 mismatch: {relative(path)}")
        path = Path(path)
        if expected and (path.is_relative_to(data_dir) or path.is_relative_to(results_dir)):
            verified_paths.add(path)

    def check_frozen(cell, doc):
        if cell.frozen is None:
            if cell.group == "targets_selected":
                frozen = doc["frozen"]
            elif cell.group == "baselines_selected":
                frozen = {"identity": doc}
            else:
                return
        else:
            frozen = read_json(cell.frozen)
        if cell.frozen is not None and "frozen" in doc and doc["frozen"] != frozen:
            raise ValueError("Result and pre-evaluation frozen provenance differ")
        identity = frozen.get("identity", frozen)
        if cell.prefix:
            verify_hash(data_dir / f"{cell.prefix}.npz", identity.get("data_sha256", identity.get("tuning_data_sha256")))
        if cell.group in ("targets_hpo", "targets_final", "baselines_hpo", "baselines_final"):
            if (doc["method"] != cell.method or doc["seed"] != cell.seed or
                    identity["seed"] != cell.seed or doc["config"] != identity["config"] or
                    identity["epochs"] != 6):
                raise ValueError("Result identity differs from its prespecified method, seed or configuration")
            if cell.group.endswith("_final"):
                family = cell.group.split("_")[0]
                selected_path = results_dir / family / "cds_2015_d30" / cell.method / "selected.json"
                selected = read_json(selected_path)
                if doc["config"] != selected["config"]:
                    raise ValueError("Final configuration differs from the HPO selection")
                if family == "baselines":
                    verify_hash(selected_path, identity["selection"]["selected_sha256"])
        checkpoint = next((p for p in cell.required if p.name == "model.npz" or
                           (p.suffix == ".npz" and p.stem.startswith("trial_") and p == cell.result.with_suffix(".npz"))), None)
        if checkpoint:
            verify_hash(checkpoint, frozen.get("model_sha256", doc.get("model_sha256")))
        if cell.evaluation:
            verify_hash(cell.evaluation, doc.get("evaluation_sha256"))
        code_hashes = dict(identity.get("code_sha256", {})) if isinstance(identity.get("code_sha256"), dict) else {}
        for field, path in (("target_code_sha256", "saet_repro/target.py"),
                            ("evaluator_code_sha256", "saet_repro/evaluation.py"),
                            ("runner_code_sha256", "scripts/run_targets.py")):
            if field in identity:
                code_hashes[path] = identity[field]
        for name, expected in code_hashes.items():
            path = ROOT / name
            actual = digest(path) if path.is_file() else None
            code_checks.append({"cell": f"{cell.group}/{cell.name}", "path": name,
                                "frozen_sha256": expected, "current_sha256": actual, "matches": actual == expected})
            if name.endswith(".py") and actual != expected:
                raise ValueError(f"Frozen implementation differs from current code: {name}")
        if cell.group.startswith("utility_"):
            if cell.group in ("utility_hpo", "utility_primary", "utility_mechanism"):
                name = cell.name if cell.group == "utility_hpo" else cell.method
                if (doc.get("name", doc.get("method")) != name or
                        doc["configuration"] != frozen["configurations"][name]):
                    raise ValueError("Utility result identity differs from its frozen configuration")
            if cell.group == "utility_historical":
                verify_hash(cell.result.parent / "labels.npz", doc["artifacts"]["labels"])
                base_path = cell.result.parent / "model.npz"
                source_directory = cell.result.parent
            elif cell.group == "utility_ridges":
                for name, value in doc["artifacts"].items():
                    path = cell.result.parent / ("validation_mixtures.json" if name == "validation_mixtures" else name + ".npz")
                    verify_hash(path, value)
                for offset, value in frozen["fold_hashes"].items():
                    verify_hash(results_dir / "utility" / f"cds_{cell.origin}" / "historical" / f"d{offset}" / "labels.npz", value)
                return
            elif cell.group in ("utility_primary", "utility_mechanism"):
                source_directory = cell.result.parent.parent
                marker = read_json(source_directory / "complete.json")
                expected_names = {f"main_{policy}" for policy in POLICIES} | {
                    f"{mechanism}_{policy}" for mechanism in ("mass1", "norm01") for policy in POLICIES[1:]}
                if (marker["frozen"] != frozen or len(marker["policies"]) != 19 or
                        set(marker["policies"]) != expected_names or set(frozen["configurations"]) != expected_names):
                    raise ValueError("Shared utility completion marker differs from frozen evaluation")
                for name in ("result.json", "test.npz", "routing.npz"):
                    verify_hash(cell.result.parent / name, marker["artifacts"][f"{cell.method}_{name}"])
                selected_path = results_dir / "utility" / "cds_2015_d30" / "selected.json"
                verify_hash(selected_path, frozen["selected_policy_sha256"])
                mechanism, policy = cell.method.split("_", 1)
                expected_config = (read_json(selected_path)["policies"][policy] if mechanism == "main" else
                                   dict(policy=policy, radius=.1, temperature=1., mechanism=mechanism))
                if (doc["configuration"] != expected_config or doc["seed"] != cell.seed or
                        frozen["seed"] != cell.seed or doc["year"] != cell.origin):
                    raise ValueError("Final utility configuration differs from its selected or prespecified policy")
                base_path = results_dir / "targets" / cell.prefix / "activity_mean" / f"seed_{cell.seed}" / "model.npz"
            else:
                source_directory = results_dir / "utility" / "cds_2015_d30"
                base_path = results_dir / "targets" / "cds_2015_d30" / "activity_mean" / "selected_model.npz"
                if cell.group == "utility_selected":
                    for name, value in doc["artifacts"].items():
                        stem, suffix = name.rsplit("_", 1)
                        verify_hash(source_directory / "search" / f"{stem}.{suffix}", value)
            verify_hash(base_path, frozen.get("base_model_sha256"))
            verify_hash(source_directory / "source_activity.npz", frozen.get("source_activity_sha256"))
            for name, value in frozen.get("ridge_hashes", {}).items():
                year = cell.origin if cell.group in ("utility_primary", "utility_mechanism") else 2015
                verify_hash(results_dir / "utility" / f"cds_{year}" / "ridges" / f"{name}.npz", value)

    for cell in cells:
        verified_paths.clear()
        absent = [relative(path) for path in cell.required if not path.is_file()]
        if absent:
            status[cell.group]["missing"] += 1
            missing.append({"group": cell.group, "name": cell.name, "missing_artifacts": absent})
            continue
        try:
            doc = read_json(cell.result) if cell.result else None
            if cell.group == "data_prefixes":
                manifest = read_json(cell.required[1])
                verify_hash(cell.required[0], manifest["data_sha256"])
                verify_hash(ROOT / "saet_repro/data.py", manifest["code_sha256"])
                catalogs[cell.prefix] = load_catalog(cell.required[0])
                row = {"prefix": cell.prefix, "origin": cell.origin, "offset_days": manifest["offset_days"],
                       **{key: manifest[key] for key in ("fit_start", "fit_end", "query_end")}}
                for domain, values in manifest["counts"].items():
                    row.update({f"{domain}_{name}": value for name, value in values.items()})
                data_rows.append(row)
                runtime_rows.append({"component": "data_prefix", "cell": cell.name, "seconds": manifest["elapsed_seconds"]})
            else:
                check_frozen(cell, doc)
                if cell.evaluation:
                    if cell.prefix not in catalogs:
                        raise ValueError(f"Independent data prefix not complete: {cell.prefix}")
                    summaries, paired = validate_evaluation(cell.evaluation, catalogs[cell.prefix], doc["summary"])
                    if cell.group.endswith("_final") or cell.group in ("utility_primary", "utility_mechanism"):
                        for stratum, values in summaries.items():
                            runs.append({"group": cell.group, "method": cell.method, "origin": cell.origin,
                                         "seed": cell.seed, "stratum": stratum,
                                         **{key: values[key] for key in (*METRICS, "items_catalog", "items_evaluated",
                                                                         "items_without_positive", "n_positive")}})
                        paired_runs[cell.method, cell.origin, cell.seed] = paired
                        if cell.group.startswith("utility_"):
                            with np.load(cell.result.parent / "routing.npz", allow_pickle=False) as route:
                                for stratum in STRATA:
                                    selected = (np.ones(len(route["item_id"]), bool) if stratum == "all" else
                                                catalogs[cell.prefix]["target_" + stratum][route["item_id"]])
                                    routing_rows.append({"method": cell.method, "origin": cell.origin, "seed": cell.seed,
                                                         "stratum": stratum, "n_evaluated_items": int(selected.sum()),
                                                         **{f"mean_{field}": float(np.mean(route[field][selected]))
                                                            if selected.any() else None for field in ROUTING_FIELDS}})
                            if cell.method == "main_zero":
                                base = results_dir / "targets" / cell.prefix / "activity_mean" / f"seed_{cell.seed}" / "test.npz"
                                with np.load(base, allow_pickle=False) as left, np.load(cell.evaluation, allow_pickle=False) as right:
                                    for key in left.files:
                                        if not np.array_equal(left[key], right[key], equal_nan=left[key].dtype.kind in "fc"):
                                            raise ValueError(f"Exact null differs from canonical target: {key}")
                                zero_verified += 1
                    if cell.group in ("targets_hpo", "targets_final", "baselines_hpo", "baselines_final"):
                        runtime_rows.append({"component": cell.group, "cell": cell.name,
                                             "seconds": doc.get("elapsed_seconds", doc.get("seconds"))})
                if cell.group in ("targets_selected", "baselines_selected"):
                    validate_hpo_selection(cell.result.parent, doc, cell.group.split("_")[0],
                                           cell.method, verify_hash)
                    selection_rows.append({"group": cell.group, "method": cell.method, "origin": 2015,
                                           "trial": doc["trial"], "configuration": json.dumps(doc["config"], sort_keys=True),
                                           "selection_tail80_ndcg20": doc.get("score", doc.get("criterion_value"))})
                elif cell.group == "utility_selected":
                    validate_utility_selection(cell.result.parent, doc, read_json(cell.frozen))
                    for method, config in doc["policies"].items():
                        selection_rows.append({"group": cell.group, "method": method, "origin": 2015,
                                               "configuration": json.dumps(config, sort_keys=True)})
                    runtime_rows.append({"component": "utility_hpo_shared", "cell": cell.name,
                                         "seconds": doc["elapsed_seconds"]})
                elif cell.group == "utility_historical":
                    diagnostic_rows.append({"component": "historical_labels", "origin": cell.origin, "prefix": cell.prefix,
                                            **{key: doc[key] for key in ("n_labels", "n_query_items", "label_positive_fraction", "label_rms")}})
                    runtime_rows.append({"component": cell.group, "cell": cell.name, "seconds": doc["elapsed_seconds"]})
                elif cell.group == "utility_ridges":
                    for method, values in doc["diagnostics"].items():
                        diagnostic_rows.append({"component": "ridge_d60_selection", "origin": cell.origin, "method": method,
                                                "selected_alpha": values["selected_alpha"], **values["validation"]})
                    mixtures = read_json(cell.result.parent / "validation_mixtures.json")
                    for policy in POLICIES:
                        entries = [row for row in mixtures if row["policy"] == policy]
                        if entries:
                            diagnostic_rows.append({"component": "d60_mixture_label_only", "origin": cell.origin,
                                "method": policy, "n": len(entries),
                                **{f"mean_{name}": float(np.mean([row[name] for row in entries])) for name in
                                   ("actual_mixture_gain", "weighted_actual_single_gain", "source_mass", "correction_norm",
                                    "oracle_single_gain", "oracle_with_null_gain")}})
            status[cell.group]["completed"] += 1
            artifact_paths.update(cell.required)
            artifact_paths.update(verified_paths)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            status[cell.group]["invalid"] += 1
            invalid.append({"group": cell.group, "name": cell.name, "error": str(error)})
            # Discard any summaries created before a later artifact check failed.
            runs = [row for row in runs if not (row["method"] == cell.method and row["origin"] == cell.origin
                                                and row["seed"] == cell.seed)]
            paired_runs.pop((cell.method, cell.origin, cell.seed), None)
            routing_rows = [row for row in routing_rows if not (row["method"] == cell.method and
                            row["origin"] == cell.origin and row["seed"] == cell.seed)]
    complete = not missing and not invalid
    if not complete and not allow_partial:
        problems = [f"{group}={row['completed']}/{row['expected']}" for group, row in status.items()
                    if row["completed"] != row["expected"]]
        details = "; ".join(row["error"] for row in invalid[:3])
        raise ValueError("Required experiment matrix is incomplete: " + ", ".join(problems) + ("; " + details if details else ""))
    origin_rows, overall = aggregate_runs(runs)
    comparisons = make_comparisons(overall, paired_runs, n_bootstrap)
    for year in YEARS:
        for seed in SEEDS:
            path = results_dir / "utility" / f"cds_{year}_d0" / f"seed_{seed}" / "complete.json"
            if path in artifact_paths:
                runtime_rows.append({"component": "utility_final_19_shared", "cell": f"{year}/seed_{seed}",
                                     "seconds": read_json(path)["elapsed_seconds"]})
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
               "status": status, "missing_cells": missing, "invalid_cells": invalid,
               "overall": overall, "origins": origin_rows, "comparisons": comparisons,
               "exact_zero_verified_runs": zero_verified,
               "aggregation": "macro conditional item metrics; seeds within origin, then equal three origins",
               "historical_values_used_in_new_results": False, "bootstrap_resamples": n_bootstrap}
    versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_paths = sorted(set(ROOT.glob("saet_repro/*.py")) | set(ROOT.glob("scripts/*.py")) |
                        set(ROOT.glob("docs/*protocol.md")) | {ROOT / "requirements.txt"})
    manifest = {"generated_utc": summary["generated_utc"], "complete": complete,
                "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
                "code_sha256": {relative(path): digest(path) for path in code_paths if path.is_file()},
                "artifacts": [{"path": relative(path), "sha256": digest(path), "bytes": path.stat().st_size}
                              for path in sorted(artifact_paths)],
                "frozen_code_checks": code_checks,
                "runtime_note": "Component times can overlap and include resume-dependent work; not end-to-end wall time."}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
        ("metrics_runs", runs, ("method", "origin", "seed", "stratum", *METRICS)),
        ("metrics_by_origin", origin_rows, ("method", "origin", "stratum", "n_seeds", *METRICS)),
        ("metrics_overall", overall, ("method", "stratum", "n_origins", "n_runs", *METRICS)),
        ("data_summary", data_rows, ("prefix", "origin", "offset_days")),
        ("hpo_selected", selection_rows, ("group", "method", "origin", "configuration")),
        ("utility_diagnostics", diagnostic_rows, ("component", "origin", "method")),
        ("routing_summary", routing_rows, ("method", "origin", "seed", "stratum", "n_evaluated_items")),
        ("runtime_summary", runtime_rows, ("component", "cell", "seconds")),
    ):
        write_csv(output_dir / f"{name}.csv", rows, fields)
    write_json(output_dir / "run_status.json", {key: summary[key] for key in
               ("generated_utc", "complete", "status", "missing_cells", "invalid_cells", "exact_zero_verified_runs")})
    write_json(output_dir / "paired_effects.json", comparisons)
    write_json(output_dir / "artifact_manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    path = output_dir / "REPRODUCTION_REPORT.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(render_report(summary, selection_rows), encoding="utf-8")
    temporary.replace(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    try:
        summary = build_report(args.data_dir, args.results_dir, args.output_dir,
                               args.allow_partial, args.bootstrap_resamples)
    except ValueError as error:
        parser.exit(1, f"Report not generated: {error}\n")
    print("Complete final report" if summary["complete"] else "Incomplete progress report; no missing-cell averages")
    for group, counts in summary["status"].items():
        print(f"{group}: {counts['completed']}/{counts['expected']} (invalid={counts['invalid']})")
    print(args.output_dir / "REPRODUCTION_REPORT.md")


if __name__ == "__main__":
    main()
