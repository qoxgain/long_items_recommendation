# 补充实验 v1 中文报告

生成时间：2026-10-20T12:00:07.857722+00:00

**状态：请求阶段的规定矩阵全部通过核验。**

请求阶段：c。未请求的阶段不表示已完成。

2015–2017 的 A/B 结果是已查看开发数据上的探索性分析；C 的 2018–2020 音乐结果是本流程中新时间验证，不是独立领域确认。Movies and TV → Books 因资源条件未排程，未产生领域验证结论。

主终点固定为 Tail80 宏平均合格商品 NDCG@20。先平均配对训练种子，再等权平均三个起点。商品自助法与跨起点唯一商品簇区间均为描述性；共享用户依赖仍未消除。训练轨迹、检查点、策略记录与独立数据集数量不能互换。

| 阶段/组件 | 完成 | 缺失 | 失败 | 无效 | 规定 |
| --- | --- | --- | --- | --- | --- |
| c/activity_evaluations | 3 | 0 | 0 | 0 | 3 |
| c/final_policy_evaluations | 120 | 0 | 0 | 0 | 120 |
| c/final_query_gate | 1 | 0 | 0 | 0 | 1 |
| c/historical_folds | 24 | 0 | 0 | 0 | 24 |
| c/hpo_searches | 2 | 0 | 0 | 0 | 2 |
| c/protocol | 1 | 0 | 0 | 0 | 1 |
| c/ridge_fits | 6 | 0 | 0 | 0 | 6 |

## B：预算选择和任务方向

尚无满足全部起点与种子矩阵的方法；不计算跨起点均值。

## C：同领域新时间验证

| 方法 | 选定轮数 | 历史终点 | 配置 |
| --- | --- | --- | --- |
| activity_mean/full | 6 | 0.030437 | {"policy": "full", "radius": 0.0, "temperature": 0.1} |
| activity_mean/semantic | 6 | 0.030437 | {"policy": "semantic", "radius": 0.0, "temperature": 0.1} |
| activity_mean/uniform | 6 | 0.030437 | {"policy": "uniform", "radius": 0.0, "temperature": 0.1} |
| activity_mean/zero | 6 | 0.030437 | {"policy": "zero", "radius": 0.0, "temperature": 1.0} |
| activity_both/full | 6 | 0.038668 | {"policy": "full", "radius": 0.1, "temperature": 0.1} |
| activity_both/semantic | 6 | 0.038436 | {"policy": "semantic", "radius": 0.0, "temperature": 0.1} |
| activity_both/uniform | 6 | 0.038507 | {"policy": "uniform", "radius": 0.05, "temperature": 0.1} |
| activity_both/zero | 6 | 0.038436 | {"policy": "zero", "radius": 0.0, "temperature": 1.0} |

历史 d60 完整收益回归器的选定配置诊断如下；这些不是正式部署指标。全部特征组与 alpha 试验见 historical_diagnostics.csv。

| 底座 | 起点 | alpha | MSE | 零预测 MSE | 符号准确率 | 正预测精度 | 预测正比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| activity_mean | 2018 | 10.000000 | 0.001536 | 0.001538 | 0.529337 | 0.533752 | 0.162500 |
| activity_mean | 2019 | 10.000000 | 0.001299 | 0.001299 | 0.543161 | 0.522189 | 0.378924 |
| activity_mean | 2020 | 10.000000 | 0.001797 | 0.001808 | 0.543571 | 0.534790 | 0.400357 |
| activity_both | 2018 | 0.000100 | 0.003788 | 0.003826 | 0.520918 | 0.477756 | 0.131888 |
| activity_both | 2019 | 10.000000 | 0.002493 | 0.002462 | 0.524383 | 0.537061 | 0.215527 |
| activity_both | 2020 | 10.000000 | 0.003779 | 0.003791 | 0.511071 | 0.459521 | 0.156607 |

| 方法 | NDCG@20 | Recall@20 | Precision@20 | 条件命中率 | 全目录 HC@20 | 实际记录 |
| --- | --- | --- | --- | --- | --- | --- |
| activity | 0.020207 | 0.052878 | 0.002707 | 0.054131 | 0.000374 | 3 |
| activity_both/full | 0.024223 | 0.062510 | 0.003246 | 0.064913 | 0.000458 | 15 |
| activity_both/semantic | 0.024021 | 0.062451 | 0.003243 | 0.064854 | 0.000458 | 15 |
| activity_both/uniform | 0.023782 | 0.061280 | 0.003184 | 0.063683 | 0.000448 | 15 |
| activity_both/zero | 0.024021 | 0.062451 | 0.003243 | 0.064854 | 0.000458 | 15 |
| activity_mean/full | 0.017820 | 0.049145 | 0.002457 | 0.049145 | 0.000333 | 15 |
| activity_mean/semantic | 0.017820 | 0.049145 | 0.002457 | 0.049145 | 0.000333 | 15 |
| activity_mean/uniform | 0.017820 | 0.049145 | 0.002457 | 0.049145 | 0.000333 | 15 |
| activity_mean/zero | 0.017820 | 0.049145 | 0.002457 | 0.049145 | 0.000333 | 15 |

| 配对 | 差值 | 商品 95% 区间 | 商品簇 95% 区间 | 伤害率 |
| --- | --- | --- | --- | --- |
| activity_both/full − activity_both/zero | 0.000202 | [-0.000212, 0.000647] | [-0.000192, 0.000653] | 0.022905 |
| activity_both/zero − activity | 0.003813 | [-0.007287, 0.013760] | [-0.007465, 0.013704] | 0.035997 |
| activity_both/full − activity_both/uniform | 0.000441 | [0.000035, 0.000988] | [0.000014, 0.000951] | 0.027635 |
| activity_both/full − activity_both/semantic | 0.000202 | [-0.000212, 0.000647] | [-0.000192, 0.000653] | 0.022905 |
| activity_both/full − activity_mean/full | 0.006402 | [-0.001133, 0.015263] | [-0.001237, 0.014905] | 0.056396 |
| activity_mean/full − activity_mean/zero | 0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | 0.000000 |

activity_both/full − activity_both/zero：逐起点差 2018: -0.000213；2019: 0.000687；2020: 0.000134。留一起点：去掉 2018: 0.000410；去掉 2019: -0.000040；去掉 2020: 0.000237。

activity_both/zero − activity：逐起点差 2018: -0.008591；2019: 0.013431；2020: 0.006600。留一起点：去掉 2018: 0.010016；去掉 2019: -0.000996；去掉 2020: 0.002420。

activity_both/full − activity_both/uniform：逐起点差 2018: 0.000022；2019: 0.000795；2020: 0.000504。留一起点：去掉 2018: 0.000650；去掉 2019: 0.000263；去掉 2020: 0.000409。

activity_both/full − activity_both/semantic：逐起点差 2018: -0.000213；2019: 0.000687；2020: 0.000134。留一起点：去掉 2018: 0.000410；去掉 2019: -0.000040；去掉 2020: 0.000237。

activity_both/full − activity_mean/full：逐起点差 2018: 0.002633；2019: 0.008306；2020: 0.008268。留一起点：去掉 2018: 0.008287；去掉 2019: 0.005451；去掉 2020: 0.005470。

activity_mean/full − activity_mean/zero：逐起点差 2018: 0.000000；2019: 0.000000；2020: 0.000000。留一起点：去掉 2018: 0.000000；去掉 2019: 0.000000；去掉 2020: 0.000000。

## 工件与解释范围

metrics_runs.csv、metrics_by_origin.csv、metrics_overall.csv 保留 All/Tail50/Tail80 与各截止值。historical_curves.csv 保留每轮实测历史终点；compute_budget.csv 以实际轨迹记录更新/配对及计时，共享评估只计一次，进程峰值 RSS 不可相加解释为独立模型内存。

status_cells.csv 分开记录 complete/missing/failed/invalid 和旧材料复用。artifact_manifest.json 保留本次读取核验工件的哈希。公共输出没有用户 ID、逐商品 ID 或排名；逐商品配对只在内存中进行。
