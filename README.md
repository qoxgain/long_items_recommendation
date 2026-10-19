# long_tail_items_recommendation

SAET 第三轮商品检索用户实验的独立 Python 重建工程。研究方向为 Amazon Reviews 2023 的 **Digital Music → CDs and Vinyl**。

原实验代码、模型和逐商品输出已无法恢复。本工程根据三份用户提供的论文/报告及协议记录重新实现，不是原代码镜像。旧报告的数值仅记录于 [historical_reference.csv](reports/historical_reference.csv)，不能作为本次运行结果。具体重建假设见 [RECONSTRUCTION_SCOPE.md](docs/RECONSTRUCTION_SCOPE.md)。

本次真实数据实验已完成：18 个时间前缀、3 个正式起点、5 个优化种子，411 个正式评估单元，另有 169 个历史选参评估和 12 个收益标注模型。完整结果见 [重建实验报告](reports/REPRODUCTION_REPORT.md)，完成清单见 [run_status.json](reports/run_status.json)。58 项测试通过。

新运行 Tail80 I-NDCG@20 的跨起点均值：活动基线 0.033162，活动＋平均历史 0.032206，时间衰减＋商品条件化历史 0.036150，LightGCN 0.024061。历史选参为全部主迁移策略选中零修正，完整收益策略与冻结底座同为 0.032206；本轮未测得主迁移策略增益。固定非零强度的机制对照单独报告，不能代替主结果。

## 模型与协议

- 目标模型：无界线性偏好、活动＋平均历史、时间衰减历史、商品条件化历史、二者组合。
- 常见对照：用户活动计数、语义余弦、MF-BPR、LightGCN、SimpleX。
- 源修正：零迁移、均匀、语义/供体统计/行为/完整特征的历史收益回归、预测最佳单源；另做相同质量和相同范数机制对照。
- 3 个起点（2015/2016/2017-01-01）、5 个优化种子（11/22/33/44/55）。每个独立前缀使用过去 365 天拟合、紧接 30 天查询。
- 目标和 CF 每族 12 个候选，仅在首次起点之前的历史块选参，正式模型固定 6 轮，不用最终查询早停。
- 收益标签由每起点前 270/180/90/60 天的独立模型生成；前三块拟合、第四块选择 ridge，再四块重拟合。部署参数单独在 d30 块选择。
- 全量合格用户候选，排除该商品已发生过任意评分的用户；有未来正例的商品上取宏平均。无正例指标为未定义，不填零。

完整接口和预算见 [IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md)。原第二轮检查点审计无法重放；未运行的新类别、表示或损失扩展不会算作复现结果。

## 安装与测试

建议 Linux、Python 3.12、CPU；实际依赖固定在 `requirements.txt`。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
```

`torch` 使用 CPU 官方轮子索引。数据下载需要网络；其后训练和评价可以离线执行。不同依赖、平台、表示及采样实现可能改变数值，因此不能保证与历史报告逐位一致。

## 从公开数据完整运行

在仓库根目录、已激活上述虚拟环境后依次执行：

```bash
python scripts/download_data.py --workers 2
python scripts/prepare_data.py
python scripts/run_targets.py
python scripts/run_baselines.py
python scripts/run_utility.py
python scripts/summarize_results.py
```

默认运行全部 18 个时间前缀、3 个正式起点和 5 个种子。目标模型、CF 基线和迁移策略分别保存独立目录，可在数据准备完成后并行运行；迁移入口使用 `--wait-inputs` 时会等待目标检查点。脚本保留已完成工件，适合中断后重启；目标入口的复用检查未覆盖所有工件字节哈希，详见 [独立审计](docs/ARTIFACT_AUDIT.md)。正式汇总要求实验矩阵完整并核验来源、配置、检查点与指标，`--allow-partial` 只用于监测进度。

原始压缩评分和元数据约 1.3 GB；评分投影、前缀和临时特征缓存还需要额外空间。正式实验不依赖 GPU。耗时取决于 CPU 和磁盘，首次运行包含全部历史选参与机制对照。

## 代码与工件

`saet_repro/` 包含数据、目标模型、协同基线、源收益机制、完整候选评价与统计代码。`scripts/` 是数据处理和训练入口，`tests/` 保存数值与时间边界检查，`docs/` 保存各模块协议。命令行可运行 `python scripts/<入口>.py --help` 查看参数。

下载器从 [Amazon Reviews 2023 官方数据](https://amazon-reviews-2023.github.io/) 获取完整类别评分与标题元数据，保存来源、行数和 SHA256。前缀构建不截取下载流，不使用未来活动筛选用户。每个前缀独立建立词表/SVD，当前标题元数据的历史性限制在协议中说明。

数据、特征缓存、检查点及逐商品预测保存在本地 `data/`、`results/`，不作为普通 Git 文件提交。GitHub 保存可运行源代码、固定协议、测试和聚合结果。运行过程可以从完整工件恢复，未完成记录不能算作已完成实验。

## 结果解释

新结果必须来自本工程保存的训练日志和预测数组。汇总先平均优化种子，再等权平均起点；配对商品重采样区间仅为描述性区间，未完全保留共享用户与时间依赖。精确零迁移和修正范数约束可核验，但不保证 NDCG 改善。不得把历史数据上的多次开发当作新类别独立确认。
