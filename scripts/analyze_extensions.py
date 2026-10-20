#!/usr/bin/env python3
"""Stage A: strict read-only analysis of completed reconstruction artifacts.

No model fitting or final-query selection is available from this entry point.
Original files stay untouched. A frozen analysis identity is checked on reruns.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from saet_repro.data import load_data
from saet_repro.evaluation import stratified_summaries, evaluate
from saet_repro.io import sha256, write_json
from saet_repro.statistics import paired_effects
from saet_repro.extension_statistics import (
    unique_item_cluster_interval, decision_digest, routing_summary, convex_mixture_check,
)
from saet_repro.target import Model
from saet_repro.utility import (
    RouteContext, RidgeUtility, FEATURE_SUBSETS, POLICIES, MultiPolicyScorer,
    policy_correction,
)
from scripts.summarize_results import expected_cells, load_catalog, write_csv

YEARS = (2015, 2016, 2017)
SEEDS = (11, 22, 33, 44, 55)
SECONDARY = ('ndcg10', 'ndcg20', 'ndcg50', 'recall20', 'precision20',
             'conditional_hit_rate20', 'coverage20')
COUNTS = ('items_catalog', 'items_evaluated', 'items_eligible', 'items_requested',
          'items_with_positive', 'items_without_positive', 'items_hit20', 'n_positive')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path, exclude=()):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files if key not in exclude}


def freeze(path, identity):
    if path.exists():
        if read_json(path) != identity:
            raise ValueError(f'Analysis identity mismatch: {path}; use a new output directory')
    else:
        write_json(path, identity)


def compare_published(reproduced, original):
    """Compare every original aggregate CSV field, including six-digit rounding."""
    checks = []
    for name in ('metrics_runs', 'metrics_by_origin', 'metrics_overall', 'data_summary',
                 'hpo_selected', 'utility_diagnostics', 'routing_summary', 'runtime_summary'):
        old, new = original / f'{name}.csv', reproduced / f'{name}.csv'
        with old.open(newline='') as handle:
            left = list(csv.DictReader(handle))
        with new.open(newline='') as handle:
            right = list(csv.DictReader(handle))
        if len(left) != len(right):
            raise ValueError(f'Published row count differs: {name}')
        max_difference, numeric, six_equal = 0., 0, True
        for a, b in zip(left, right):
            if set(a) != set(b):
                raise ValueError(f'Published columns differ: {name}')
            for key in a:
                if a[key] == b[key]:
                    continue
                try:
                    x, y = float(a[key]), float(b[key])
                except ValueError as error:
                    raise ValueError(f'Published field differs: {name}/{key}') from error
                numeric += 1
                max_difference = max(max_difference, abs(x - y))
                six_equal &= f'{x:.6f}' == f'{y:.6f}'
                if not np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
                    raise ValueError(f'Published number differs: {name}/{key}')
        if not six_equal:
            raise ValueError(f'Published six-decimal values differ: {name}')
        checks.append({'table': name, 'rows': len(left), 'all_fields_match': True,
                       'max_absolute_difference': max_difference, 'six_decimal_match': six_equal,
                       'original_sha256': sha256(old), 'reproduced_sha256': sha256(new)})
    return checks


def pair_row(records, method, baseline, n_bootstrap, seed):
    result = paired_effects(records, method, baseline, n_bootstrap=n_bootstrap, seed=seed)
    if (result['n_item_origins'] != 483 or result['n_unique_items'] != 480
            or result['n_origins'] != 3 or result['n_seed_pairs'] != 2415
            or any(result[k] for k in ('n_unmatched_item_origins', 'n_undefined_item_origins', 'n_undefined_seed_pairs'))):
        raise ValueError(f'Incomplete required pair population: {method}/{baseline}')
    cluster = unique_item_cluster_interval(result['items'], n_bootstrap=n_bootstrap, seed=seed)
    row = {key: result[key] for key in ('method', 'baseline', 'n_origins', 'n_item_origins',
                                      'n_unique_items', 'n_seed_pairs', 'method_mean', 'baseline_mean', 'mean_difference')}
    row.update(stratum='tail80', metric='ndcg20', interpretation='exploratory descriptive',
               relative_point_difference=(result['mean_difference'] / result['baseline_mean']
                                          if result['baseline_mean'] else None),
               relative_percent_difference=(100 * result['mean_difference'] / result['baseline_mean']
                                            if result['baseline_mean'] else None),
               ci_low=result['ci'][0], ci_high=result['ci'][1],
               unique_item_cluster_ci_low=cluster['ci'][0], unique_item_cluster_ci_high=cluster['ci'][1],
               cluster_rejected_draws=cluster['rejected_empty_origin_draws'],
               n_bootstrap=n_bootstrap, bootstrap_seed=seed,
               harm_rate=result['harm']['harm_rate'], improvement_rate=result['harm']['improvement_rate'],
               tie_rate=result['harm']['unchanged_rate'])
    for origin in result['origins']:
        for key in ('mean_difference', 'harm_rate', 'improvement_rate', 'unchanged_rate', 'n_items'):
            row[f"origin_{origin['origin']}_{key}"] = origin[key]
    for omitted in result['leave_one_origin_out']:
        row[f"leave_out_{omitted['omitted_origin']}_mean_difference"] = omitted['mean_difference']
    return row


def final_materials(data_dir, results_dir):
    records, secondary, routes = [], [], defaultdict(list)
    catalogs = {year: load_catalog(data_dir / f'cds_{year}_d0.npz') for year in YEARS}
    references = {}
    for cell in expected_cells(data_dir, results_dir):
        if not (cell.group.endswith('_final') or cell.group in ('utility_primary', 'utility_mechanism')):
            continue
        result = load_npz(cell.evaluation, exclude=('top_users',))
        data = catalogs[cell.origin]
        shared = {key: result[key] for key in ('item_id', 'n_candidates', 'n_positive', 'eligible_catalog', 'requested', 'evaluated')}
        if cell.origin in references:
            for key, value in shared.items():
                if not np.array_equal(value, references[cell.origin][key]):
                    raise ValueError(f'Candidate or evaluation population differs: {cell.name}/{key}')
        else:
            references[cell.origin] = shared
        mask = result['evaluated'] & data['target_tail80']
        records.extend({'origin': cell.origin, 'seed': cell.seed, 'method': cell.method,
                        'item_id': str(item), 'ndcg20': float(value)}
                       for item, value in zip(result['item_id'][mask], result['ndcg20'][mask]))
        for stratum, summary in stratified_summaries(data, result).items():
            secondary.append({'level': 'run', 'method': cell.method, 'origin': cell.origin,
                              'seed': cell.seed, 'stratum': stratum, 'n_seeds': 1, 'n_origins': 1,
                              **{key: summary[key] for key in (*SECONDARY, *COUNTS)},
                              'conditional_denominator': summary['items_evaluated'],
                              'coverage_denominator': summary['items_catalog'],
                              'precision_denominator': 'min(20,n_candidates) per evaluated item'})
        if cell.group.startswith('utility_'):
            route = load_npz(cell.result.parent / 'routing.npz')
            if not np.array_equal(route['item_id'], np.flatnonzero(result['evaluated'])):
                raise ValueError(f'Routing population differs: {cell.name}')
            mask = data['target_tail80'][route['item_id']]
            routes[cell.method].append({'origin': cell.origin, 'seed': cell.seed, **routing_summary(route, mask)})
    return records, secondary, routes


def secondary_aggregates(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[row['method'], row['stratum'], row['origin']].append(row)
    origins = []
    for (method, stratum, origin), rows in sorted(groups.items()):
        expected = {None} if method in ('activity', 'cosine') else set(SEEDS)
        if {row['seed'] for row in rows} != expected or len(rows) != len(expected):
            raise ValueError('Incomplete secondary seed matrix')
        row = {'level': 'origin', 'method': method, 'stratum': stratum, 'origin': origin,
               'seed': '', 'n_seeds': len(rows), 'n_origins': 1}
        row.update({key: float(np.mean([r[key] for r in rows])) for key in SECONDARY})
        for key in COUNTS:
            row[key] = float(np.mean([r[key] for r in rows]))
        row.update(conditional_denominator=rows[0]['items_evaluated'], coverage_denominator=rows[0]['items_catalog'],
                   precision_denominator='min(20,n_candidates) per evaluated item')
        origins.append(row)
    overall = []
    for method, stratum in sorted({(r['method'], r['stratum']) for r in origins}):
        rows = [r for r in origins if r['method'] == method and r['stratum'] == stratum]
        if {r['origin'] for r in rows} != set(YEARS):
            raise ValueError('Incomplete secondary origin matrix')
        row = {'level': 'overall', 'method': method, 'stratum': stratum, 'origin': 'equal_three_origins',
               'seed': '', 'n_origins': 3, 'n_seeds': sum(r['n_seeds'] for r in rows)}
        row.update({key: float(np.mean([r[key] for r in rows])) for key in SECONDARY})
        row.update({key: sum(r[key] for r in rows) for key in COUNTS})
        row.update(conditional_denominator='per-origin evaluated count; equal origin average',
                   coverage_denominator='per-origin full catalog count; equal origin average',
                   precision_denominator='min(20,n_candidates) per evaluated item')
        overall.append(row)
    return runs + origins + overall


def aggregate_routes(rows):
    if {(r['origin'], r['seed']) for r in rows} != {(y, s) for y in YEARS for s in SEEDS}:
        raise ValueError('Incomplete routing matrix')
    # Each run has the same number of items within its origin. Equal run means
    # therefore implement seed-first and equal-origin weighting exactly.
    result = {'routing_stratum': 'tail80', 'n_routed_item_seed_rows': sum(r['n_routed_items'] for r in rows),
              'n_nonzero_item_seed_rows': sum(r['n_nonzero_items'] for r in rows),
              'zero_action_rate': float(np.mean([r['zero_action_rate'] for r in rows]))}
    probability = 1 - result['zero_action_rate']
    for field in ('source_mass', 'correction_norm', 'effective_donors', 'concentration'):
        result['mean_' + field] = float(np.mean([r['mean_' + field] for r in rows]))
        weighted_sum = float(np.mean([r['nonzero_sum_' + field] / r['n_routed_items'] for r in rows]))
        result['nonzero_mean_' + field] = weighted_sum / probability if probability > 0 else None
    result['nonzero_conditioning'] = 'conditional on actual nonzero correction; equal origin/item/seed probability measure'
    return result


def selection_surface(data_dir, results_dir):
    directory = results_dir / 'utility/cds_2015_d30'
    selection = read_json(directory / 'selected.json')
    data = load_data(data_dir / 'cds_2015_d30.npz')
    base = Model.load(results_dir / 'targets/cds_2015_d30/activity_mean/selected_model.npz', data)
    source = load_npz(directory / 'source_activity.npz')['beta']
    context = RouteContext(data, base, seed=101, source_beta=source)
    ridges = {s: RidgeUtility.load(results_dir / 'utility/cds_2015/ridges' / f'{s}.npz') for s in FEATURE_SUBSETS}
    configs = read_json(directory / 'search.frozen.json')['configurations']
    historical = load_npz(directory / 'search/zero.npz')
    indices = np.flatnonzero(historical['evaluated'])
    inputs = [context.route_inputs(int(item))[1:] for item in indices]
    rows = []
    zero_score = read_json(directory / 'search/zero.json')['summary']['1']['tail80']['ndcg20']
    for name, config in configs.items():
        corrections, routing = [], defaultdict(list)
        native_zero, native_mass_zero = [], []
        for features, gradients in inputs:
            correction, weights = policy_correction(features=features, gradients=gradients, ridges=ridges, **config)
            corrections.append(correction)
            mass = float(weights.sum())
            conditional = weights / mass if mass else weights
            for field, value in {'zero': not np.any(correction), 'source_mass': mass,
                                 'correction_norm': np.linalg.norm(correction),
                                 'effective_donors': np.count_nonzero(weights), 'concentration': conditional @ conditional}.items():
                routing[field].append(value)
            native_config = dict(config, radius=.1)
            native, native_weights = policy_correction(features=features, gradients=gradients, ridges=ridges, **native_config)
            native_zero.append(not np.any(native))
            native_mass_zero.append(not np.any(native_weights))
        routing = {key: np.asarray(value) for key, value in routing.items()}
        stats = routing_summary(routing)
        saved = read_json(directory / 'search' / f'{name}.json')
        for field in ('source_mass', 'correction_norm', 'effective_donors', 'concentration'):
            np.testing.assert_allclose(stats['mean_' + field], saved['routing']['mean_' + field], atol=1e-12, rtol=1e-9)
        np.testing.assert_equal(stats['zero_action_rate'], saved['routing']['mean_zero'])
        score = saved['summary']['1']['tail80']['ndcg20']
        selected = config == selection['policies'][config['policy']]
        rows.append({'name': name, **config, 'selection_tail80_ndcg20': score,
                     'difference_from_zero': score - zero_score, 'zero_radius_score': zero_score,
                     'comparison_to_zero': 'better' if score > zero_score else 'worse' if score < zero_score else 'tie',
                     'selected': selected, 'tie_break': selection['tie_break'],
                     'decision_sha256': decision_digest(corrections),
                     'decision_population': 'all 593 historically evaluated items; fit-only route reconstruction',
                     'native_zero_action_rate_at_radius_0_1': float(np.mean(native_zero)),
                     'native_zero_weight_rate_at_radius_0_1': float(np.mean(native_mass_zero)),
                     'routing_reconstruction_matches_saved': True, **stats})
    global_count = len({r['decision_sha256'] for r in rows})
    for policy in POLICIES:
        entries = [r for r in rows if r['policy'] == policy]
        nonzero = [r['selection_tail80_ndcg20'] for r in entries if r['radius'] > 0]
        for row in entries:
            row.update(nominal_configurations_per_policy=len(entries),
                       distinct_decisions_per_policy=len({r['decision_sha256'] for r in entries}),
                       nominal_configurations_all_policies=len(rows), distinct_decisions_all_policies=global_count,
                       best_nonzero_score=max(nonzero) if nonzero else None,
                       best_nonzero_minus_zero=max(nonzero) - zero_score if nonzero else None,
                       n_strictly_better_than_zero=sum(r['comparison_to_zero'] == 'better' for r in entries),
                       n_strictly_worse_than_zero=sum(r['comparison_to_zero'] == 'worse' for r in entries),
                       n_tied_with_zero=sum(r['comparison_to_zero'] == 'tie' for r in entries))
    return rows


def exact_null_checks(data_dir, results_dir):
    checks = []
    for year in YEARS:
        data = load_data(data_dir / f'cds_{year}_d0.npz')
        for seed in SEEDS:
            directory = results_dir / 'utility' / f'cds_{year}_d0/seed_{seed}'
            base_dir = results_dir / 'targets' / f'cds_{year}_d0/activity_mean/seed_{seed}'
            model = Model.load(base_dir / 'model.npz', data)
            original_parameters = (model.theta.copy(), model.beta.copy(), model.alpha)
            context = RouteContext(data, model, seed=seed, source_beta=load_npz(directory / 'source_activity.npz')['beta'])
            scorer = MultiPolicyScorer(context, {}, {'zero': dict(policy='zero', radius=0., temperature=1.)})
            score_values, item_rows = 0, 0
            def checked_score(indices):
                nonlocal score_values, item_rows
                expected = model.score_all(indices)
                actual = scorer(indices)[0]
                if not np.array_equal(actual, expected):
                    raise ValueError(f'Exact-null score differs: {year}/{seed}')
                score_values += actual.size
                item_rows += len(indices)
                return actual
            replay = evaluate(data, 1, checked_score)
            canonical = load_npz(base_dir / 'test.npz')
            saved_zero = load_npz(directory / 'main_zero/test.npz')
            if set(replay) != set(canonical) or set(replay) != set(saved_zero):
                raise ValueError('Exact-null evaluation fields differ')
            for key in replay:
                for reference in (canonical, saved_zero):
                    if not np.array_equal(replay[key], reference[key], equal_nan=replay[key].dtype.kind in 'fc'):
                        raise ValueError(f'Exact-null replay differs: {year}/{seed}/{key}')
            if not (np.array_equal(model.theta, original_parameters[0]) and np.array_equal(model.beta, original_parameters[1])
                    and model.alpha == original_parameters[2]):
                raise ValueError('Frozen base parameters changed')
            checks.append({'origin': year, 'seed': seed, 'exact_score_equality': True,
                           'exact_rank_equality': True, 'exact_all_array_equality': True,
                           'frozen_parameters_unchanged': True, 'n_item_rows': item_rows,
                           'n_score_values': score_values, 'max_score_difference': 0.})
            print(f'Exact null {year}/{seed}: {item_rows} items, {score_values} scores', flush=True)
    return checks


def render_report(target, selection, mechanism, provenance):
    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                         ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])
    def num(value):
        return '未定义' if value is None else f'{value:.6f}'
    lines = ['# A 阶段补充分析（已有开发结果）', '',
             'A1–A5 已执行；仅复用并核验已有逐商品材料，未训练模型、未用最终查询重新选参。'
             '2015–2017 起点此前已用于开发，全部比较为探索性、描述性分析。', '',
             f"冻结分析身份：`analysis_frozen.json`；全局协议 SHA256：`{provenance['protocol_sha256']}`。", '',
             '## A1 材料与复现', '',
             '原版严格汇总在 `strict_original/` 独立输出，411 个正式评价单元完整，全部八张原聚合 CSV '
             '字段与六位小数一致。主比较为 483 个商品—起点对、480 个唯一商品，先在商品内配对平均五个训练种子。'
             '确定性活动计数仅三个实际运行，在配对时广播，不增记实验。原 `results/` 与已发表 `reports/` 未修改。', '',
             '差异与哈希：`published_table_comparison.json`、`analysis_provenance.json`；材料清单：`material_availability.csv`。', '',
             '## A2 目标比较', '',
             table(['方法 − 参照', '均值差', '相对点差(%)', '分层商品 95% 区间', '唯一商品簇 95% 区间', '伤害率'],
                   [(r['method'] + ' − ' + r['baseline'], num(r['mean_difference']), num(r['relative_percent_difference']),
                     f"[{num(r['ci_low'])}, {num(r['ci_high'])}]", f"[{num(r['unique_item_cluster_ci_low'])}, {num(r['unique_item_cluster_ci_high'])}]",
                     num(r['harm_rate'])) for r in target]), '',
             '全部区间使用 2,000 次重采样、随机种子 2026。原分层方法在各起点内部重采样商品，再等权平均起点。'
             '敏感性方法联合抽取 480 个唯一商品簇，同一商品保留全部起点记录及共同抽样次数，每次仍等权汇总起点。'
             '共享用户依赖仍未消除；不以 2,415 个商品—种子行或五个训练种子构造独立样本。'
             '逐起点、逐一留出起点、改善/伤害/平局比例全部在 `analysis_target_pairs.csv`。', '',
             '## A3 历史选择与零半径', '',
             table(['策略', '名义数', '不同决策数', '最优非零 − 零', '严格优/劣/并列', '选中 R/T'],
                   [(p, entries[0]['nominal_configurations_per_policy'], entries[0]['distinct_decisions_per_policy'],
                     num(entries[0]['best_nonzero_minus_zero']),
                     f"{entries[0]['n_strictly_better_than_zero']}/{entries[0]['n_strictly_worse_than_zero']}/{entries[0]['n_tied_with_zero']}",
                     ', '.join(f"{r['radius']}/{r['temperature']}" for r in entries if r['selected']))
                    for p in POLICIES for entries in [[r for r in selection if r['policy'] == p]]]), '',
             f"73 个名义记录对应 {selection[0]['distinct_decisions_all_policies']} 个不同实际决策（跨策略去重）。"
             '决策依据全部 593 个历史可评价商品的精确修正向量，不按指标相等去重。'
             '历史逐商品路由未保存，故用已有底座、源活动与 ridge 检查点重构纯历史路由；重构均值与保存记录一致。'
             '并列按半径再温度原顺序选择。R=0 强制输出零权重；CSV 另列 R=0.1、相同温度下的原生零动作率及零权重率，'
             '不能把选择零半径解释为每个商品收益预测都非正。全部半径、温度、实际路由量、条件非零质量/范数保存在 `analysis_selection_surface.csv`。', '',
             '## A4 次要指标与固定机制', '',
             '`analysis_secondary_metrics.csv` 包含全部 29 方法、All/Tail50/Tail80 的逐运行、逐起点、等起点汇总。'
             'I-NDCG@10/20/50、Recall@20、Precision@20 和条件至少一命中率仅在可评价商品上平均；'
             'HC@20 使用完整对应分层目录分母。目录数、可评价数、命中数及分母分别保存；跨起点总数不代替等权指标。'
             '主终点仍为 Tail80 I-NDCG@20。', '',
             table(['固定机制 − activity_mean', '均值差', '描述性 95% 区间', '伤害率', '零动作率', '非零质量', '非零范数'],
                   [(r['method'], num(r['mean_difference']), f"[{num(r['ci_low'])}, {num(r['ci_high'])}]", num(r['harm_rate']),
                     num(r['zero_action_rate']), num(r['nonzero_mean_source_mass']), num(r['nonzero_mean_correction_norm'])) for r in mechanism]), '',
             '两组各六种策略均固定 R=0.1、T=1，保留原生零动作。配对全体为同一 483 商品—起点集合；'
             '路由量先按各起点商品/种子均匀分配再等权起点，非零量按该测度条件化。'
             '不将条件机制统计当作无偏总体收益；全部起点差、留一及区间见 `analysis_mechanism_pairs.csv`。', '',
             '## A5 验收', '',
             f"15 个标准零路径从冻结检查点重评分，共 {sum(r['n_score_values'] for r in provenance['exact_null'])} 个分数逐值完全相等；"
             '重新评价的排序、全部指标与掩码和 canonical/保存零路径逐数组完全一致，底座参数未改变。', '',
             f"历史 d60 实际凸损失混合检查 {provenance['convex_mixture']['n_records']} 条，"
             f"最大正违约 {provenance['convex_mixture']['max_violation']:.17g}，容差 1e-12；"
             '检查的是实际混合收益不少于加权单 donor 实际收益（剩余质量分给零动作）。'
             '此界只涉及该历史实际配对损失，不覆盖 norm01 外推缩放，也不保证未来 NDCG。', '',
             '本阶段真实数值可用于补充论文区间、选择曲线与机制表；不能据此新增独立验证或充分收敛声明。'
             'B/C 阶段完成情况由各阶段报告单列。', '']
    return '\n'.join(lines)


def run(args):
    started = time.monotonic()
    protocol = read_json(args.protocol)
    stage = protocol['stage_a']
    if stage['origins'] != list(YEARS) or protocol['seeds'] != list(SEEDS) or not stage['no_training']:
        raise ValueError('Unsupported Stage A protocol')
    output = args.output_dir
    if output.resolve() == (ROOT / 'reports').resolve() or output.resolve().is_relative_to(args.results_dir.resolve()):
        raise ValueError('Stage A requires a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    original_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'reports').glob('*')) if p.is_file()}
    for path, expected in protocol['baseline_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Frozen baseline code/report changed: {path}')
    own_code = {path: sha256(ROOT / path) for path in ('scripts/analyze_extensions.py', 'saet_repro/extension_statistics.py')}
    identity = {'protocol_sha256': sha256(args.protocol), 'analysis_code_sha256': own_code,
                'original_reports_sha256': original_hashes, 'data_dir': str(args.data_dir.resolve()),
                'results_dir': str(args.results_dir.resolve()), 'n_bootstrap': protocol['bootstrap_resamples'],
                'bootstrap_seed': protocol['bootstrap_seed'], 'stage': 'A',
                'dependencies': {'numpy': np.__version__, 'python': platform.python_version()},
                'visibility': stage['visibility'], 'no_training': True}
    freeze(output / 'analysis_frozen.json', identity)
    (output / 'analysis_frozen.sha256').write_text(sha256(output / 'analysis_frozen.json') + '\n')
    cells = expected_cells(args.data_dir, args.results_dir)
    availability = [{'group': c.group, 'cell': c.name, 'required_files': len(c.required),
                     'present_files': sum(p.is_file() for p in c.required), 'complete': all(p.is_file() for p in c.required)} for c in cells]
    write_csv(output / 'material_availability.csv', availability, ('group', 'cell', 'required_files', 'present_files', 'complete'))
    if not all(r['complete'] for r in availability):
        raise ValueError('Missing baseline materials; see material_availability.csv')
    command = [sys.executable, str(ROOT / 'scripts/summarize_results.py'), '--data-dir', str(args.data_dir),
               '--results-dir', str(args.results_dir), '--output-dir', str(output / 'strict_original'),
               '--bootstrap-resamples', str(protocol['bootstrap_resamples'])]
    print('Running original strict summarizer in isolated output', flush=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    (output / 'strict_original.log').write_text(completed.stdout + completed.stderr)
    strict = read_json(output / 'strict_original/summary.json')
    final_count = sum(v['completed'] for k, v in strict['status'].items() if k.endswith('_final') or k in ('utility_primary', 'utility_mechanism'))
    if not strict['complete'] or final_count != 411 or strict['exact_zero_verified_runs'] != 15:
        raise ValueError('Strict original matrix failed')
    differences = compare_published(output / 'strict_original', ROOT / 'reports')
    write_json(output / 'published_table_comparison.json', differences)
    records, secondary, routes = final_materials(args.data_dir, args.results_dir)
    n_bootstrap, seed = protocol['bootstrap_resamples'], protocol['bootstrap_seed']
    target = [pair_row(records, m, b, n_bootstrap, seed) for m, b in stage['retained_pairs'] + stage['new_pairs']]
    write_csv(output / 'analysis_target_pairs.csv', target, ('method', 'baseline'))
    write_csv(output / 'analysis_secondary_metrics.csv', secondary_aggregates(secondary), ('level', 'method', 'origin', 'seed', 'stratum'))
    mechanism = []
    for name in sorted(routes):
        if name.startswith(('mass1_', 'norm01_')):
            mechanism.append(dict(pair_row(records, name, 'activity_mean', n_bootstrap, seed), **aggregate_routes(routes[name])))
    if len(mechanism) != 12:
        raise ValueError('Missing fixed mechanism policies')
    write_csv(output / 'analysis_mechanism_pairs.csv', mechanism, ('method', 'baseline'))
    print('Reconstructing historical routing from frozen checkpoints without fitting', flush=True)
    selection = selection_surface(args.data_dir, args.results_dir)
    write_csv(output / 'analysis_selection_surface.csv', selection, ('name', 'policy', 'radius', 'temperature'))
    mixtures = []
    for year in YEARS:
        mixtures.extend(read_json(args.results_dir / 'utility' / f'cds_{year}/ridges/validation_mixtures.json'))
    convex = convex_mixture_check(mixtures)
    if not convex['passed']:
        raise ValueError('Actual convex-mixture check failed')
    nulls = exact_null_checks(args.data_dir, args.results_dir)
    if original_hashes != {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'reports').glob('*')) if p.is_file()}:
        raise ValueError('Original report files changed')
    manifest = read_json(output / 'strict_original/artifact_manifest.json')
    # Rehash all inputs at completion to detect mutation during analysis.
    for record in manifest['artifacts']:
        path = Path(record['path'])
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != record['sha256']:
            raise ValueError(f'Input changed during analysis: {path}')
    provenance = dict(identity, complete=True, generated_utc=datetime.now(timezone.utc).isoformat(),
                      actual_final_evaluations=final_count, strict_command=command,
                      strict_manifest_sha256=sha256(output / 'strict_original/artifact_manifest.json'),
                      input_artifacts=manifest['artifacts'], published_comparisons=differences,
                      exact_null=nulls, convex_mixture=convex, original_reports_unchanged=True,
                      no_final_query_selection=True, n_target_pairs=len(target), n_mechanism_pairs=len(mechanism),
                      n_historical_selection_records=len(selection),
                      n_distinct_historical_decisions=selection[0]['distinct_decisions_all_policies'],
                      elapsed_seconds=time.monotonic() - started, threads=1,
                      limitations=['shared-user dependence remains', '2015–2017 development results',
                                   'convex loss bound does not guarantee NDCG'])
    (output / 'ANALYSIS_REPORT_ZH.md').write_text(render_report(target, selection, mechanism, provenance), encoding='utf-8')
    provenance['outputs_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'analysis_provenance.json'}
    write_json(output / 'analysis_provenance.json', provenance)
    print(json.dumps({'complete': True, 'final_evaluations': final_count, 'target_pairs': len(target),
                      'mechanism_pairs': len(mechanism), 'seconds': provenance['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/extensions_v1.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/extensions_v1/stage_a')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
