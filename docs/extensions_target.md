# B1 直接目标收敛扩展

本实现对应补充执行清单的 B1，属于原 2015–2017 三个开发起点上的探索性分析。旧 `saet_repro/target.py`、`results/` 及 `reports/` 不变。新实现复用旧特征、抽样、成对损失和 Adam 运算次序；不据最终查询改变配置或训练轮数。

## 冻结与矩阵

入口为 `scripts/run_extension_targets.py`，默认输出为 `results_extensions/v1/stage_b/`。运行前核验 `configs/extensions_v1.json`、旧核心文件哈希，并生成 `target_protocol.json` 和 `target_protocol.sha256.json`。独立协议包含全局协议哈希、实现/运行器/评价器/数据加载器哈希、Python/NumPy 和相关依赖版本、四个输入 NPZ 哈希、完整配置与决策规则；冻结发生在打开 NPZ 数组之前。改变身份须使用新的版本输出根，不能覆盖既有冻结协议。

两个方法是 `activity_mean` 与 `activity_both`。每族沿用 12 配置，顺序为 `itertools.product((.001,.003,.01), (0.,.0001), (.1,.3))`，三个维度依次为 lr、rho、tau；其余 Config 默认值保持原样。历史前缀为 `cds_2015_d30`，选择种子为 101、202、303，共 72 条历史轨迹。每轮评价完整候选用户上的历史 Tail80 I-NDCG@20，并记录 All、Tail50、Tail80 摘要；选择仅允许第 6、20、50 轮。

先求每个配置/检查点的三个种子均值，再共同选配置与轮数；精确平局优先轮数较少，再优先配置网格顺序。若该族优胜者位于第 50 轮，且超过同配置第 20 轮至少 1%，将该族**全部 36 条轨迹**延长至第 100 轮，再从 6/20/50/100 联合选择。分母绝对值小于 `1e-8` 时改用绝对差 `1e-4` 阈值。

每族选定方案重新训练 2015、2016、2017 的 `d0` 前缀，每个前缀种子 11、22、33、44、55，共 30 条最终模型轨迹。最终查询只在固定训练结束后评价一次；保存原评价器的完整商品数组、候选/评价掩码、top users 和 @10/20/50 指标。最终五个种子不用于历史选择。每个最终前缀评价前写 `final_query_opened.json`，其中包含冻结协议、选参结果哈希及所选配置/轮数。

## 共享、恢复与记录

同族所有配置/种子共用 FeatureStore。每个种子/轮数只构造一次临时训练特征 bank，12 个配置顺序消费；不会持久化 100 轮特征缓存。历史评价通过 `evaluate_many` 共用每商品特征，在同一次评价中计算该族当前所有模型。仅由 fit 历史确定且不随 epoch 改变的 anchor/phi 使用有界只读 RAM 缓存；上限为 2,800,000,000 bytes，按确定性商品顺序首次填入，容量不足时逐商品回退为即时计算，不驱逐已缓存条目。活动特征只共享一份，缓存限本次进程调用，不写入磁盘。缓存不会存放查询标签，也不改变特征值、评分或梯度运算。final 五个模型共享一次不缓存的评价。BLAS、OMP、MKL、NumExpr 固定一线程；同机并行训练数量由顶层调度控制。

每个 epoch 保存训练损失、梯度范数均值/最大值、更新范数均值/总和、累计更新范数、实际正例数、Cartesian 配对数、更新次数/累计次数、训练时间、共享特征构造时间、共享评价时间、共享轨迹数和进程 RSS 峰值。共享时间不可直接按模型重复相加。RSS 是进程高水位，pilot 同时记录实际 cgroup CPU/内存限额。每轮历史指标仅作预先定义的曲线和检查点选择。

`last_state.npz` 含参数、Adam m/v、步数、精确 RNG 状态和完整历史。恢复时重新计算代码、依赖、输入文件以及全部数组的名字/shape/dtype/内容身份，核验状态文件哈希、身份、连续 epoch、更新计数和已有检查点哈希。损坏/陈旧轨迹移至 `.stale_<timestamp>`，保存原因后重算；不因文件存在而直接复用。每轮状态使用原子替换；文件组中途写入中断可能触发保守重算。最终 complete 清单校验 model、test、result、state、history 和身份文件。

目录形式：

```text
stage_b/target_protocol.json
stage_b/target_protocol.sha256.json
stage_b/<method>/hpo/trial_XX/seed_N/{last_state.npz,history.json,frozen.json,...}
stage_b/<method>/hpo/trial_XX/seed_N/checkpoint_006.npz
stage_b/<method>/hpo/trial_XX/seed_N/checkpoint_020.npz
stage_b/<method>/hpo/trial_XX/seed_N/checkpoint_050.npz
stage_b/<method>/selected.json
stage_b/<method>/selected.complete.json
stage_b/<method>/extension_decision.json
stage_b/<method>/final/cds_YEAR_d0/seed_N/{model.npz,test.npz,result.json,complete.json,...}
stage_b/target_pilot/<method>/pilot.json
```

`selected.json` 保留全部网格/种子/检查点分数、延长决定、选择身份和每条 HPO 轨迹证据哈希。第 100 轮检查点只在触发规则后出现。失败会写 `failed_<phase>.json` 及不可变的带时间戳失败记录，重试成功保留 recovered 记录。未完成矩阵不能据部分记录称全部完成。

## 执行和验证

```bash
.venv/bin/python scripts/run_extension_targets.py --list-tasks
.venv/bin/python scripts/run_extension_targets.py --pilot
.venv/bin/python scripts/run_extension_targets.py --hpo-only
.venv/bin/python scripts/run_extension_targets.py --final-only
.venv/bin/python scripts/run_extension_targets.py --retry-failed
.venv/bin/python -m pytest -q tests/test_extension_target.py tests/test_target.py
```

省略运行模式将顺序完成 HPO 与 final。`--methods activity_mean` 可选择一个族；协议仍声明完整两族网格，任务清单/顶层验收须报告实际完成矩阵。`--list-tasks` 只列任务，不打开查询。单个根目录有 target runner 文件锁，阻止意外双启动写同一轨迹。

pilot 在独立目录运行每族全部 12 配置、种子 101 的两个历史 epoch，包括实际查询评价；分别记录首次冷缓存与第二次暖缓存的时间、内存、资源限额和线性预算推算。最终前缀大小不同；外推仅供资源计划，不能作为保证耗时。pilot 结果不参与选择，正式 HPO 会从自身轨迹开始。此前尚未启用历史 RAM 缓存的单 epoch pilot 及其冻结协议保存在 `stage_b/legacy_pilot/pre_ram_cache/`；该归档时尚无两族目标模型的正式 HPO 或 final，新的缓存实现先重新冻结协议再启动训练。

2026-10-20 的缓存 pilot 已完成。环境实际限制为 2 CPU、8 GiB RAM；测量期间与 CF 和 C-final 两个单线程任务短时并发，存在 CPU 争用。两族均缓存 593 个历史评价商品，计 2,746,896,000 bytes。下表时间单位为秒，评价时间由该族 12 个模型共享，不能按模型重复累计；RSS 为同一进程的高水位。

| 方法 | 两轮总时间 | 冷轮评价 | 暖轮评价 | 暖轮特征 bank | 暖轮训练（12 模型合计） | 进程峰值 RSS / MiB |
|---|---:|---:|---:|---:|---:|---:|
| activity_mean | 33.82 | 13.37 | 9.11 | 2.03 | 3.27 | 4307.05 |
| activity_both | 79.77 | 49.28 | 11.94 | 2.61 | 5.27 | 4316.81 |

原始记录为 `stage_b/target_pilot/<method>/pilot.json`。这次重新冻结的目标实现协议 SHA256 为 `56c724e0a6a0b5b82cbe79eaaee68aa41ab09cdda0e0504a93b60748d3c1e9ef`。测量用于资源安排，不作为推荐效果或完整收敛证据。

新增测试核验全部 24 个方法/配置在小型 fixture 上前六轮参数、损失、计数与旧训练器逐轮精确相等；核验两族 Adam/RNG 分段恢复、陈旧/损坏身份重算、共享训练特征次数、共享/单独评价逐数组相等、完整种子要求、网格/轮数平局和预定义延长阈值。缓存测试另外核验两族缓存/无缓存训练参数和指标完全相等、容量回退和 FeatureStore 身份隔离。这些合成 fixture 只用于代码验证，不是实证结果。
