# C1/C2：固定底座上的新时间迁移验证

本阶段继续使用 Digital Music → CDs and Vinyl，预先固定 2018、2019、2020 年 1 月 1 日 UTC 三个起点。每个前缀独立使用过去 365 天拟合和紧接的 30 天查询；每个起点包含 d270/d180/d90/d60 历史标注、d30 选参及 d0 正式前缀。原 2015–2017 数据、训练器与输出保持原样。新年份属于本流程中新时间验证，不能称为独立领域验证；Movies and TV → Books 尚未运行。

全局协议为 [`configs/extensions_v1.json`](../configs/extensions_v1.json)，运行器为 [`scripts/run_extension_transfer.py`](../scripts/run_extension_transfer.py)，实现为 [`saet_repro/extension_transfer.py`](../saet_repro/extension_transfer.py)。默认数据目录为 `data_extensions/v1/music`，结果目录为 `results_extensions/v1/stage_c_music`。

## 冻结底座与历史选择

两个底座沿用旧 2015 年历史块选中的配置和 6 轮训练：`activity_mean` 使用 lr=0.01、rho=0.0001、tau=0.1；`activity_both` 使用 lr=0.001、rho=0、tau=0.1。完整配置保存在冻结协议中。C 不在事后采用 B 收敛实验的选择结果；“组合历史底座”不意味着已证明它是所有训练预算下的最优目标模型。

每个底座、每个起点分别拟合四个历史目标模型，均使用种子 101。历史标签衡量源修正是否降低后续目标损失。d270/d180/d90 标注拟合收益回归，d60 选择 ridge 系数，再用四折重拟合部署回归器。历史目标配置与正式部署配置一致。ridge 网格为 0.0001、0.001、0.01、0.1、1、10；共 24 个标注折和 6 组回归器。不同历史检查点不是独立数据集。

部署策略为 `zero`、`uniform`、`semantic` 和 `full`。源修正半径为 0、0.01、0.05、0.1，温度为 0.1、1、10。每个底座在 **2018 d30** 上评价 1 个零策略和每个非零策略族的 12 个配置，共 37 条记录；两底座合计 74 条。每个策略族按 Tail80 I-NDCG@20 选择，精确平局沿半径优先、温度次之的冻结网格顺序。选定半径和温度固定用于三个正式起点；各起点的收益回归仅用其自身历史块训练。

## 正式评价与证据边界

正式优化种子为 11、22、33、44、55。每个底座/起点/种子仅训练一个目标模型，四个策略共享该模型，合计 30 个目标模型和 120 个策略评价；另有三个确定性用户活动基线。源域和目标域用户空间分开，不要求跨域用户身份对齐。

主迁移比较为 `activity_both/full − activity_both/zero`，主目标比较为 `activity_both/zero − activity`。全量合格目标用户参与候选排名，排除已与该商品发生任意评分的用户。主终点对有未来正例的 Tail80 商品取宏平均；无正例商品的条件指标未定义，不补零。全目录覆盖率另用完整目录分母。

先在起点内平均五个优化种子，再等权平均三个起点。报告商品自助法区间、跨起点唯一商品簇区间、逐起点差和留一起点分析；这些描述性区间仍未消除共享用户依赖。次要比较的正区间不能替代预先指定的同底座零策略主比较。

## 冻结、恢复与零策略检查

`protocol.frozen.json` 绑定全局协议、原底座选择、实现和依赖身份。数据验证检查精确起止时间、阶段边界、数组索引以及 NPZ 与清单的一致性。历史标注和 HPO 不允许打开 d0。

首次正式数据访问前，运行器核验全部历史折、回归器、74 条搜索记录与选择规则，并写入 `final_query_opened.json`。打开后改变协议或选择依赖会拒绝运行，不能重新选择参数以适配正式结果。恢复时核验工件字节、模型配置和来源身份；正式查询打开前可隔离损坏缓存并重算，打开后不得静默改变历史选择。

每个正式模型评价时，零策略与冻结目标模型在所有实际评价商品上的分数逐位相同，完整排名和指标数组也逐位相同；同时核验目标参数和模型文件未改变。该精确零策略性质不是 NDCG 提升保证。半径为零导致的全体回退，也不能证明收益预测器正确识别了所有有害更新。

## 执行与报告

在完成原数据准备和原 2015 年目标配置选择后运行：

```bash
.venv/bin/python scripts/prepare_extensions.py
.venv/bin/python scripts/run_extension_transfer.py --hpo-only
.venv/bin/python scripts/run_extension_transfer.py --final-only
.venv/bin/python scripts/summarize_extensions.py \
  --stages c --output-dir reports/extensions_v1/stage_c_summary
```

不提供运行模式时，迁移入口依次完成历史标注、HPO、查询开放和正式评价。`--historical-only` 仅完成标注与回归器，`--freeze-only` 只冻结并核验协议。同一输出目录有运行器锁；中断后使用相同命令恢复。

严格汇总要求所有规定单元通过核验后才写报告。已验证的实测结果见 [`stage_c_summary/REPORT_ZH.md`](../reports/extensions_v1/stage_c_summary/REPORT_ZH.md)，全部配对见 [`paired_effects.csv`](../reports/extensions_v1/stage_c_summary/paired_effects.csv)。论文补丁中的表号对应最新稿件；最终整合版本须结合 B 完整结果后重新生成。公共报告保存聚合数值和工件哈希，逐商品数组、用户身份与原始排名保留在本地。
