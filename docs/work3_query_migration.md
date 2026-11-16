# 开题方案代码迁移与整理：完整修改及审阅报告（更新至2026-11-16）

> 版本说明：本报告保留d82f258首轮实施及发布记录。独立审阅指出的F1、F2、F3与G1，以及本轮修正和新验证，见末尾“独立审阅后的修正记录”；该节优先于首轮能力描述。历史验证结果不自动计作本轮复验。

> 最新状态见末节“2026-11-16：R1容错修复、正式训练复验与性能取舍”：M0a为4dd394d，正式CPU两worker更新及必要评估通过；本轮不保留性能候选，不宣称整体训练加速。前面各节按日期保留历史范围。

> 后续目录整理从a710e34独立实施；本报告保留当时的文件名、输出路径和验证命令。当前可运行路径见[工程导航](README.md)、[维护记录](maintenance.md)及唯一迁移表，不将历史`outputs`命令解释成新的默认落点。

基线：`837194591b67c83e9cd2943d987de19e84a2c996`。本地分支：`work3-query-migration-20261115`。
原有未跟踪目录 `research_diagnostics/` 保留，不属于本次改动。

## 报告结论与审阅范围

本次已经把工作三的QUERY方法接入实际CPU多进程采样、Actor自回归评分、PPO重放、Lightning更新、模型保存加载和独立评估入口。结果是可运行、可检查的代码迁移，不是调度效果结论。默认配置仍为有限预算烟测，不能将该配置及其随机初始化后的权重称为已训练有效的正式模型。

研究内容一、二仅核查现有入口及组合配置，没有改写它们的环境、训练器或资源配置。工作三保留原有事件驱动环境；新的实际时间代价通过可选权重进入该环境，历史配置默认权重为0。LLM部分按用户后续要求缩减为离线与mock通路验证，真实API未配置、未调用。

审阅时应同时阅读源码及本报告的限制：部分接口状态只保留类型表达，并没有实现对应的新算法；首轮仅支持一种固定检索计划，本轮补为有限结构化计划，仍不代表通用语义理解；完整规模吞吐、长训练收益、Linux运行及全部历史测试一致性均未完成验收。

## 首轮完整文件清单与职责

以下20个文件构成首轮提交范围；不包含用户已有的研究诊断目录、数据、模型或临时输出。本轮另增加两个离线计划示例，见末节。

| 文件 | 类型 | 主要职责及审阅重点 |
| --- | --- | --- |
| `envs/work3/scheduling_queries.py` | 新增 | `ScheduleCondition`、`CompletionTable`、`SchedulingQueries`；固定团队窗口、完整团队枚举、保留前缀的补全查询 |
| `envs/work3/query_program.py` | 新增 | 白名单计划核验、运行时对象绑定、低频预编译及缓存；模型生成结果与语义核验结果明确分开 |
| `envs/work3/decision_snapshot.py` | 修改 | 原快照增加查询表、查询版本和程序指纹；QUERY生成事实状态并清空旧估计值 |
| `envs/work3/environment.py` | 修改 | 增加实际时间推进费用及账本字段；没有更换原有团队窗口、资格、事件或转站算法 |
| `models/work3/actor_critic.py` | 修改 | QUERY开关、分层摘要、14维评分输入、共享候选投影、各动作阶段缓存与重放、显式热启动过滤 |
| `models/work3/graph_builder.py` | 修改 | 可选生成7维工序分组事实，供飞机层汇总；原节点特征版本仍为graph_v6 |
| `models/work3/ppo_trainer.py` | 修改 | QUERY拒绝时间辅助头，保存及核验新输入语义和检查点版本 |
| `training/work3_vector_env.py` | 修改 | CPU worker生成查询快照；主进程传递不可变程序，环境进程不持有策略网络 |
| `training/work3_runtime_config.py` | 修改 | QUERY免除旧时间辅助配置必需项；保留历史配置校验 |
| `scripts/work3/experiment_protocol.py` | 修改 | 新增QUERY方法，旧B/C/D/E/F/G含义不变 |
| `scripts/work3/train_ppo_work3.py` | 修改 | 现有训练编排选择QUERY、关闭预测塑形和标签缓存、传递实际目标权重及worker种子，拒绝覆盖已有输出 |
| `scripts/work3/evaluate_query.py` | 新增 | QUERY专用评估；区分真实完成、预算截断和失败；独立可行性及费用核对 |
| `scripts/work3/evaluate_c_vs_d.py` | 修改 | 显式拒绝QUERY误入历史评估入口 |
| `utils/work3/objective_evaluator.py` | 修改 | 可选实际工期目标的独立计算、报告及非负有限权重校验 |
| `conf/work3/train_query.yaml` | 新增 | 两worker默认烟测、真实283输入、PPO、实际目标权重与独立输出位置 |
| `conf/work3/condition_program.json` | 新增 | 离线已核验的只读检索说明及调用计划 |
| `tests/work3/test_scheduling_queries.py` | 新增 | 查询、评分/重放、更新/存取、程序以及小批次完成的集中回归文件 |
| `tests/work3/test_work3_same_timestamp_events.py` | 修改 | 只修正一个测试的站位清空前提，保留原身份与扰动断言 |
| `README.md` | 修改 | 当前三项研究入口、最短运行命令、历史方法及限制 |
| `docs/work3_query_migration.md` | 新增 | 本报告，供独立审阅和复验 |

提交按“保持生产行为的夹具修正”“方法及调用链迁移”“完整报告”拆分，便于审阅者分别检查。

## 实施范围与顺序

1. 查询：`envs/work3/scheduling_queries.py` 复用环境的资格、工时、前序及连续资源窗口算法；扩展已有 `DecisionSnapshot`，不建立第二套现场状态。
2. 动作：`ActorCriticWork3.select_snapshot` 的工序、分支、逐人前缀及对齐读入查询结果；`evaluate_action_log_probs` 共享评分代码，只重算网络，不调用查询。
3. 入口：`experiment_protocol` 新增 QUERY，`work3_vector_env` 在 CPU worker 生成条件；现有 Lightning/PPO 训练链保留，QUERY 退出时间辅助与预测塑形。
4. 融合：核对 graph_v6，保留站位/飞机/整线与节点读出，仅补集合大小、标准工作量和技能需求摘要。
5. 语义：在同一查询实现之上核验、缓存白名单结构化程序；离线程序贯穿真实对象查询与评分，外部服务由配置提供。
6. 验证：先写缺口回归并确认失败，再实现；覆盖只读性、连续窗口、前缀、重放、真实更新/存取、程序生命周期、两 worker 和小规模完整生产。工作一、二只检查入口与配置。

## 调用链对照

`Work3VectorEnv.snapshot(s)` → worker `build_decision_snapshot` → `MultiAircraftGraphBuilder.build_graph` + 查询 → `select_snapshot` → 环境 `step/process_due_events` → 实际成本差分 → `PPOTransition` → Lightning → `PPOTrainerWork3.compute_ppo_minibatch_loss` → `evaluate_action_log_probs`。

固定团队调用 `duration_for_team`、`_predecessor_release_time`、`_find_team_earliest_slot`，后者继续调用工人日历和 `_next_station_conflict_end`。不修改资源、事件、不可撤销指派或转站模式。查询对象仅限已经揭示的状态。

## 主要变化与边界

| 原行为 | 新行为 | 原因 |
| --- | --- | --- |
| 时间紧迫度、残差监督及预测势函数进入主路径 | QUERY 使用独立的14维条件布局，覆盖工序、分支、每个人员前缀和完整团队对齐；不创建时间头、融合层或标签缓存 | 确定性事实复用物理算法，未定前序明确返回未知 |
| 飞机分组主要用均值 | 在原飞机投影中加入数量、标准工时及五类技能人数总量；站位投影加入本站人员节点摘要 | 保留集合规模和站内资源信息，不新增实体或平行决策网络 |
| 历史状态第26维使用估计完工时间 | QUERY 第26维记录已预约比例、第31维记录待投产比例；名义周期进度仍由已知事实给出 | 清除独立预测依赖，语义另行版本化 |
| 采样与重放重复拼接候选 | 共享 `_candidate_embeddings`，缓存原始阶段数值、候选顺序、人员前缀、掩码与程序指纹 | PPO 重放不访问现场、不执行物理搜索，投影和图骨干仍有梯度 |
| 旧检查点按图形状校验 | 增加 `work3_actor_query_v1` 及完整输入语义核验 | 不能将旧时间投影或优化器解释为新条件输入 |

`ScheduleCondition.EXACT` 仅表示**当前已知承诺不再改变**时固定团队/固定对齐的最早连续窗口；未来扰动、未定前序或未安排的下一周期不属于精确预测。部分团队通过完整合法团队集合筛选，保留全部已选成员，以最早起点、再以最早终点及人员ID稳定排序。没有近似截断；结果类型保留明确的下界和预算状态，但当前算法不主动产生近似或预算截断结果。

没有跨状态的可变查询缓存。每个快照重新读取承诺日历，因此同一物理时刻新指派也会改变条件。图节点仍为 graph_v6；新增分组事实由独立的 `hierarchy_version`、布局及检查点语义描述。

## 条件输入、动作与训练的具体契约

### 14维数值布局

| 下标 | 含义 | 有效性与归一化 |
| --- | --- | --- |
| 0–4 | 精确、下界、未知、无合法补全、预算未穷尽的状态独热 | 当前正式枚举路径主要返回精确、未知、无补全；下界/预算类型不代表已经实现对应查询算法 |
| 5 | 最早开工相对当前时刻的差 | `(start-now)/H0` |
| 6 | 对应完工相对当前时刻的差 | `(end-now)/H0` |
| 7 | 完整团队实际工时 | `duration/H0` |
| 8 | 时间数值有效标记 | 未知结果没有起止时刻，填充数值不能脱离此标记使用 |
| 9–13 | 物料、前序、人员、站位、对齐等待标记 | 分开编码，不与资格或工时相加 |

工序评分读取每个工序在当前站位的最早合法团队结果。留站分支读取该工序条件；后移分支沿用既有合法掩码，其未来执行时间编码为未知，不构造精确下一站预测。每一步选人都在已选人员前缀下筛选完整团队；对齐阶段读取最终固定团队的两个既有对齐结果。

因此，查询对后移的支持并不是搜索任意新站位，也不是预先替后移工序选择下一站团队。对象关系、资格、站位绑定和同步事实继续来自现有图与环境，没有单独实现一套通用对象/关系查询目录。

### 快照、重放与梯度

CPU worker先执行连续窗口搜索及合法团队枚举，再将查询表随 `DecisionSnapshot` 发送给主进程。采样端从表中筛选前缀、构造阶段数值并保存；这里仍有CPU张量构造和条件筛选，但不会在GPU minibatch中重新执行物理窗口搜索。

动作记录包含候选工序顺序、图节点索引、分支掩码、工人顺序、逐步资格掩码、逐步前缀、各阶段原始条件、查询版本与程序指纹。PPO重放用当前参数重新编码图和投影这些原始数值，没有冻结采样时旧隐向量。图、条件和动作记录由已有buffer复制到CPU。

强制推进的可学习概率为0；可选推进仅累计推进选择概率；后移在工序及分支选择后截断，不增加人员或对齐概率。原有worker/episode/segment分组GAE、bootstrap和异常收尾编排没有另写一套实现。

### 检查点与历史权重

QUERY的检查点标识为 `work3_actor_query_v1`，方法标识为 `hierarchical_condition_query_v1`，查询版本为 `work3_conditions_v1`。元数据还包含条件布局、事实状态版本、分层布局、程序指纹及既有图schema。

完整加载要求输入语义一致。旧模型如需使用，只能显式调用 `load_warm_start_weights`，并检查返回的加载、缺失、跳过列表；改变含义的状态首层、工序投影、时间融合及查询参数不会按旧权重静默恢复。该接口不是完整断点续训，新主CLI也没有自动续接旧优化器的行为。检查点仍标记 `resume_capability=non_exact`。

## 实际生产奖励

新 YAML 显式设置 `w_makespan=1`、`w_h=1` 和已有稳定性/后移权重，`gamma=1`。历史配置的 `w_makespan` 缺省为0，原目标保持不变。

令 `C` 为批次实际离线时刻，`H0` 为基准节拍，`Hq` 为第q次实际周期长度，目标为：

`J = w_makespan*C/H0 + w_h*Σ max(Hq-H0,0)/H0 + w_t*D_T + w_w*D_W + w_p*G + w_RT*R_T + w_RW*R_W`。

`D_T`、`D_W` 是相对基准的开工位置/团队偏离，`G` 是实际后移累计惩罚，`R_T`、`R_W` 是既有正式版本变更账本。没有改写这些既有定义。

每步只增加实际推进的 `w_makespan*(t_after-t_before)/H0`，其余项仍用原差分账本。由时间增量相消，`Σ Δt = C-0`，因此 `Σ reward = -J`；终局不再补扣工期，预测势函数为0。周期超期和工期是显式加权的两项目标：在 nominal_anchored_v1 且周期数固定时可能线性相关，不能把二者解释成两份独立收益；调权重时须报告此关系。独立目标计算与环境累计账本同时核对。

## 验证结果（实际执行）

环境：Windows，`D:\Conda\envs\rag_env\python.exe`，Python 3.11.15，PyTorch 2.12.0+cu132，RTX 4060 Laptop 8GB。没有停止已有GPU任务，没有连接远程训练服务器。

- 基线指定三文件：30通过、1失败，复现提示词中的转站夹具问题；仅补齐站位清空前提，保留飞机—工序身份和同刻扰动断言，未修改放行守卫。
- `python -m pytest tests/work3/test_scheduling_queries.py -q`：11通过。覆盖连续窗口、资格/忙碌、容量、效率、未知前序、前缀、两种对齐、后移/推进截断、只读重放、参数更新/存取、离线/mock程序及两条真实完成的小轨迹。
- 最终组合回归：67通过、46项未选入本次范围，未使用skip；实际命令如下（运行时另将pytest临时目录指定在仓库外）：

```powershell
python -m pytest tests/work3/test_scheduling_queries.py tests/work3/test_environment_actions.py tests/work3/test_work3_worker_mask_replay.py tests/work3/test_work3_same_timestamp_events.py tests/work3/test_work3_p2_1_objective_report.py tests/work3/test_work3_event_reward_accounting.py tests/work3/test_work3_training_runtime.py -k "not test_work3_training_runtime or runtime_config or runtime_seed or precision_resolver or training_cli_uses_yaml" -q -p no:cacheprovider --tb=short
```

- 两条完整轨迹复用仓库 M02 的有效五站/两架次夹具，每条10道真实工序；无扰动和受扰均完成全部工序及飞机离线，独立可行性和费用账本一致。
- 新 YAML 真实283入口：两worker、8步、一次Lightning更新，FP32采样/重放最大误差0；截断报告为decision_limit，未标成功。
- 同入口真实680：覆盖 `paths.baseline=data/work3/real_680_k10_baseline.json`、`paths.raw_data=data/680.csv`，两worker、2步、一次更新，FP32误差0。
- 同入口283 GPU：`runtime.device=cuda`、`runtime.amp_dtype=bf16`、两worker、2步、一次更新，误差0；PyTorch显存峰值980,127,744字节，主进程内存峰值2,131,886,080字节。两个CPU worker均未初始化CUDA。
- 新评估CLI用烟测权重执行2步：正常存取并报告 `success=false/decision_limit/makespan=null`，没有将截断包装为完成。
- 工作一、二经 `initialize_hydra_runtime(..., create_run_context=False)` 组合并校验：分别为 `homogeneous_graphsage_strict` / `hetero_gat`，均为工序—站位—人员联合决策；二者的重调度开关分别为false/true，未启动长训练。

更广的82项历史回归得到77通过、5失败。这5项随后在从8371945导出的**未修改源码副本**上重跑，同样失败：

1. `test_work3_m02_disturbed_pulse_trace.py` 中两个“可修改预约”断言、一个“扰动先于预约开工”断言、一个手工转站推进夹具，共4项，与保留的不可撤销/同刻先开工/放行条件冲突。
2. `test_graph_building_latency_performance`：本机历史构图阈值25ms未满足；当前约37.07ms，干净基准约45.91ms。这不是新方法吞吐测试，不能从一次耗时推断性能改进。

未删除、skip或放宽上述历史断言；这些失败仍需单独整理，不能称全库测试全绿。另发现的本次兼容问题（旧简化目标状态无时间字段、旧YAML无新增objective项）已修正并复验通过。

## LLM与清理

按用户追加要求，未配置或调用真实API，只做离线及mock可用性检查。`condition_program.json` 的文字说明绑定当前候选实例，经白名单 `team_completions` 读取最新事实，结果进入14维数值条件及实际评分。普通事件仅重执行同一程序。

可选的 `python -m envs.work3.query_program` 是低频预编译入口，服务模型、完整端点及密钥均从配置指定的环境变量读取；有限接口反馈重试，缓存按说明与接口版本命名。输出标记 `pending_semantic_review`，正式加载器拒绝直接使用。HTTP结构参照 [Chat Completions官方接口](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)；真实服务、文字理解质量及复杂说明生命周期未验证。

保留一份查询/重放回归及必要的历史测试；旧时间头、监督和势函数继续由历史profile使用，没有移动或删除研究数据、文献、有效模型、划分和旧实验。一次性基准源码副本自动回收，探针输出置于仓库外；记录结论后清理本轮GPU/680诊断模型、其报告及新增模块产生的字节码缓存。用户原有 `research_diagnostics/` 和既存缓存不批量删除。

首次交付时改动未提交，HEAD仍为8371945。随后按用户要求整理为独立审阅分支的提交；代码提交及GitHub审阅入口记录在本报告末尾。方法改动集中于work3目录及其入口；保持生产行为的夹具修正单独位于 `test_work3_same_timestamp_events.py` 的一个测试中，便于逐项审阅。

尚未证明收益或收敛；未长时间重训、未做跨种子实验、未在Linux服务器实跑。完整283工艺两架次的长轨迹探针因耗时偏高中止，本轮仅对真实283/680做短入口检查；完整规模后段的枚举开销仍需测量，不宣称已有高吞吐能力。

## 可复验运行命令

以下从仓库根目录执行，Python必须来自 `rag_env`。训练输出已存在时程序会拒绝覆盖，应换输出目录；不要删除已有有效模型来复用命令。

```powershell
conda activate rag_env

# 新方法：无扰动通路烟测，实际执行一次PPO更新。
python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml

# 使用上一步生成的权重评估；烟测权重仅用于接口核验。
python -m scripts.work3.evaluate_query --checkpoint outputs/work3/query/checkpoints/smoke_method_query_weights.pt --report outputs/work3/query/undisturbed.json
python -m scripts.work3.evaluate_query --checkpoint outputs/work3/query/checkpoints/smoke_method_query_weights.pt --report outputs/work3/query/disturbed.json --scenario-id EARLY_LOW_S0

# 受扰训练试点：同一Actor、固定训练划分、独立输出目录与明确预算。
python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml --set runtime.run_mode=pilot --set runtime.successful_batch_target=1 --set runtime.total_env_steps=10000 --set runtime.max_wall_seconds=3600 --set paths.checkpoint_dir=outputs/work3/query_pilot/checkpoints --set paths.report_dir=outputs/work3/query_pilot

# 680短入口检查；原始工艺文件与多机基准必须同时切换。
python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml --set paths.baseline=data/work3/real_680_k10_baseline.json --set paths.raw_data=data/680.csv --set runtime.total_env_steps=2 --set ppo.steps_per_iter=2 --set ppo.batch_size=2 --set paths.checkpoint_dir=outputs/work3/query_680/checkpoints --set paths.report_dir=outputs/work3/query_680

# 最小保留回归。
python -m pytest tests/work3/test_scheduling_queries.py -q
```

GPU检查在新训练命令后追加 `--set runtime.device=cuda --set runtime.amp_dtype=bf16`，并使用新的输出目录。默认评估受120秒预算限制，未完成时应看到截断及空终局工期，而不是成功指标。

历史方法可继续使用 `conf/work3/train_pilot.yaml` 配合 `--method C` 或 `--method D` 等原字母，评估仍走 `scripts/work3/evaluate_c_vs_d.py`。工作一、二使用README列出的既有Hydra入口，其有效基准模型和数据清单不随本次代码提交重新生成。

## 给其他AI助手的审阅指引

建议先读本报告的限制与完整文件表，再按以下顺序核查，不应以既有旧审计文档替代当前源码：

1. 对照 `SchedulingQueries.fixed_team` 与环境提交路径，验证未知前序、半开窗口、站位容量、共同连续时间及两种对齐的语义一致。
2. 核查 `CompletionTable.best/worker_features` 是否始终保留已选前缀，以及“完整枚举”在真实人员规模下的成本。不要把本轮短入口成功当作后段吞吐已验收。
3. 沿 `build_decision_snapshot → select_snapshot → evaluate_action_log_probs` 检查条件确实进入logits，并确认重放没有访问当前现场、重新搜索窗口或使用旧隐向量。
4. 审核14维状态有效标记、候选排序、工人ID/节点索引/站位编号转换以及后移、推进概率截断。
5. 核查分层信息是否重复或有缺口，特别是标准工时与技能人数的汇总含义，以及原站位资源绑定与新增人员摘要是否一致。
6. 重新计算目标差分账本；区分工期与周期超期的加权关系，检查截断轨迹和非零起始时刻是否需要额外约定。当前工期相消论证建立在本环境从时刻0重置的基础上。
7. 检查新旧检查点拒绝规则、热启动过滤及优化器边界，不以同维度推断同语义。
8. 将离线程序视为已实现的一种检索计划，而非通用语义约束系统；重点检查待核验计划不能直接使用，真实LLM能力留待另行验收。

建议将审阅意见分成“本次引入的问题”“基准既有问题”“尚未实现或未验证的能力”，每项给出具体文件、触发条件、影响及最小复现。不要要求删除生产守卫来满足旧夹具，不要默认更换转站模式、允许撤销预约或放松后移资格。

### 仍需后续处理的明确事项

- 完整规模查询及采样吞吐尚未验收，当前完整枚举没有预算截断或保证剪枝机制。
- 五项历史失败仍保留；完整测试集并未全绿，也没有运行全部仓库测试。
- CPU到GPU的阶段条件传递仍有开销，尚未做系统性性能剖析。
- 查询等待原因/冲突解释未覆盖所有资源链组合的独立逐项审计；现有回归不等于完整形式化证明。
- 下界和预算状态属于类型预留；没有额外的精确同步转站预测接口或通用关系查询平台。
- LLM仅验证一个合法离线计划、基本拒绝与mock缓存；真实服务、复杂否定/例外、多说明替换撤销等未完整验证。
- 未验证Linux服务器运行、跨种子稳定性、长训练收敛或优于历史PPO。

以上均应保留为审阅上下文，而不是在报告中表述为已完成能力。

## 首轮发布与复验记录

2026-11-15按用户要求，将本报告扩展为完整审阅报告，并将代码组织为三个独立目的的提交。提交范围共20个文件，未加入 `research_diagnostics/`，没有上传本轮临时模型、测试日志、密钥或API配置。

- 基准：`837194591b67c83e9cd2943d987de19e84a2c996`。
- 夹具修正提交：`014f49d66b2211e4b27796ac96ab1aaf0e340b7d`。
- 方法代码提交：`39fc3e0b45c04db5cf84c6a0548fc251a5a7fd91`。
- 本报告作为代码提交之后的独立文档提交保存；报告版本以其所在Git提交为准。
- 发布分支：`work3-query-migration-20261115`；不合并、不修改原基准分支，不使用强制推送。

本轮提交前在同一 `rag_env` 重新执行上面的组合回归，实际结果为 **67 passed, 46 deselected，211.61秒**；`git diff --check`通过。本轮没有重跑先前的GPU、680入口或82项扩展历史测试，它们仍按前一实施轮的已执行记录报告，不能算作本轮新增验证。没有配置或调用真实LLM服务。

GitHub审阅入口：

- [审阅分支](https://github.com/1ush3n/APAL-Dynamic-2/tree/work3-query-migration-20261115)
- [相对8371945的完整差异](https://github.com/1ush3n/APAL-Dynamic-2/compare/837194591b67c83e9cd2943d987de19e84a2c996...work3-query-migration-20261115)
- [固定方法代码提交](https://github.com/1ush3n/APAL-Dynamic-2/commit/39fc3e0b45c04db5cf84c6a0548fc251a5a7fd91)

其他AI助手在具有仓库访问权限的环境中，可按以下命令取得审阅版本；如本地已有未提交修改，应先保护现有改动，再切换分支：

```powershell
git fetch origin work3-query-migration-20261115
git switch --track origin/work3-query-migration-20261115
git diff 837194591b67c83e9cd2943d987de19e84a2c996...HEAD --stat
```

独立审阅应以固定代码提交与基准之间的差异为源码证据，结合报告提交阅读。没有创建或自动合并PR，其他AI需要自行完成代码审阅；本报告不是第三方审阅结论。

## 独立审阅后的修正记录（2026-11-15）

修正起点为 `d82f25826d3c3707688225e0870224a981fec5fe`。用户提供的独立审阅定位了F1、F2、F3及G1；本轮先把三个问题写入原查询回归文件，在Windows的rag_env中得到3项预期失败，再修复。没有修改图编码、分层网络参数尺寸、资格/后移规则、转站模式或奖励权重来消除这些问题。

### F2：等待原因与真正起作用的搜索跳转一致

`AirLineEnvWork3._find_team_earliest_slot` 和 `_next_station_conflict_end` 新增可选 `wait_causes` 输出容器，记录实际发生的人员跳转、容量跳转及已知冲突工序。查询直接使用共享搜索提供的记录，删除原来只检查初始区间的推断。默认物理调用不需要传入容器，最早窗口的扫描、端点、容差和迭代保护规则不变。

真实680反例中，三个占用均经正常 `env.step` 提交：人员先忙到10，站位再满到20，结果为start=20、duration=3，人员/站位等待标记均为1，冲突包含三道相关工序。再加入同队[21,30)的正常预约，验证“人员→站位→人员”继续推迟到30；容量升为3时，两例均可在10开工。没有通过直接伪造站位占用来代替此反例。

### F1：数值输入契约与程序来源分离

`ActorCriticWork3.input_semantics` 不再包含说明/程序指纹，而是保留方法、查询接口、状态/图/分层字段与布局，并显式记录 `relative_now_h0_v1` 归一化约定。程序指纹继续保存于快照、动作记录、训练配置与检查点的独立来源字段。

采样检查快照自己的接口版本和来源指纹完整性；PPO检查记录自己的方法、字段布局、归一化、张量形状、有限性和动作/前缀掩码，不再要求该样本来自Actor当前唯一的程序。旧样本的条件仍按采样原值重放，不追溯改写F2修正前的旧数值。

加载已发布v1格式时，显式移除旧元数据中错误绑定的 `query_program_hash`；只有已知v1接口允许补全原本隐含的归一化字段，其余schema必须匹配。回归中重建了d82f258的元数据格式进行兼容检查，没有覆盖或修改用户有效旧模型。错误字段、错误归一化、错误接口及非有限原始条件仍被拒绝；旧时间输入也未因此获得兼容资格。

测试覆盖同义说明修改后加载原权重、旧新来源混合重放，以及两种不同检索范围计划产生的样本混合重放。采样与重放在参数未更新时的log_prob保持一致。

### F3：显式继续给定现场，并核对初始费用

`evaluate_query_trajectory(..., reset=True)` 保留原来从零评估的便捷行为；`reset=False` 不清空现有任务、日历、事件、执行前缀或费用。续评估记录 `initial_time`、`initial_environment_steps`、`initial_cost`、`initial_objective` 和 `initial_commitments`，决策快照编号沿用环境已有step_count。

令起始累计费用为C0、最终累计费用为C1、调用时点注入扰动直接结算的费用为E、后续实际step奖励为r：

`增量费用 = C1-C0`；`-Σr+E = C1-C0`。

完整账本另要求C0等于起始状态独立计算的目标、C1等于最终状态独立目标。这样既不把历史费用再次计入续评估奖励，也不会把不一致的人工导入费用伪装成成功。已有历史step_rewards不是重置或丢弃的对象；后续核对从调用时的记录偏移开始。

回归先正常完成0_15并触发第一次脉动，再预约0_18，以 `reset=False` 注入当前时刻的延迟并继续。Actor首次看到的时间及已完工状态不回退，预约团队和对齐保持不变，仅被动顺延；最终两架次十道工序完成、离线，可行性、完整和增量账本均通过。

在调用方已经持有合法的 `env`、`actor` 时，可使用：

```python
from scripts.work3.evaluate_query import evaluate_query_trajectory

result = evaluate_query_trajectory(env, actor, reset=False, max_decisions=100)
```

这不是现场状态文件的通用导入器。CLI仍负责新批次评估；本轮未新增未约定的现场序列化格式。继续状态时必须沿用与其累计费用一致的目标权重。

### G1：有限、可区分的对象—关系—条件调用计划

程序执行结构独立使用 `work3_query_plan_v2`；数值接口仍为 `work3_conditions_v1`。旧默认JSON仍可加载，内部转换为“绑定当前候选→团队窗口查询”。新增白名单步骤如下：

| 步骤 | 参数与用途 |
| --- | --- |
| `bind_candidate` | 绑定本次全部当前候选 |
| `bind_tasks` | 非空非负整数 `aircraft_ids`、`task_ids`；按笛卡尔积绑定具体实例，并核对对象存在 |
| `related_tasks` | 同机直接 `predecessors` / `successors`，明确是否包含自身 |
| `where` | `material_wait` / `predecessor_known`；用 `negate` 表达否定，仅查询已揭示事实 |
| `team_completions` | 末步，对选中的当前候选查询原有完整合法团队窗口 |

一个计划2至6步，首步绑定、末步查询，中间只允许上述关系和条件；多次筛选为交集。不支持任意代码、循环、任意新API或任意新增生产约束。未覆盖的候选返回UNKNOWN及 `outside_program_scope` 原因；不删除候选、不改变资格/后移掩码，也不谎称无合法团队。精确窗口仍是完整枚举的原物理结果。

新增两个已核验、针对283数据的离线程序：

1. `conf/work3/program_predecessor_focus.json`：仅检索飞机0工序18及同机直接前序中，前序释放已知的当前候选。
2. `conf/work3/program_material_focus.json`：仅检索飞机1工序15及同机直接后继中，已揭示物料仍未就绪的当前候选。

在第一次脉动后的同一现场，揭示飞机1工序15的物料释放时刻为100。第一程序给飞机0工序18提供精确窗口，第二程序给飞机1工序15提供从100开始的窗口；另一对象的条件为未知。编号相同的不同飞机实例不会串用。两者候选与合法掩码相同，14维条件确实不同，并通过投影前向钩子验证这些原始数值进入 `task_proj`。直接前序/后继的非空关系、否定筛选和非法API/参数/对象也有回归。

人工计划和mock生成计划在同一现场执行结果相同；编译器缓存键包含计划结构版本、数值接口和说明文字，重复说明不再次生成。可选 `verified_examples` / `llm.example_program_paths` 最多传入三个已核验示例，有限反馈重试保留这些示例。生成结果仍为 `pending_semantic_review`，未经语义核验不能进入运行链。

训练、评估均已接入 `paths.query_program`；未设置时继续默认全候选计划。可用以下覆盖选择已核验程序，并为运行指定新的输出位置：

```powershell
python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml --set paths.query_program=conf/work3/program_predecessor_focus.json --set paths.checkpoint_dir=outputs/work3/review/checkpoints --set paths.report_dir=outputs/work3/review
python -m scripts.work3.evaluate_query --checkpoint outputs/work3/review/checkpoints/smoke_method_query_weights.pt --report outputs/work3/review/material_focus.json --set paths.query_program=conf/work3/program_material_focus.json --max-decisions 2
```

以上补齐的是可区分计划、对象/关系/条件执行及统一数值输入的有限能力。mock不证明LLM真正理解任意说明；复杂例外、任意对象对列表、一般逻辑表达式、多说明撤销替换及新限制仍不支持或未验收。模型不能自动把不支持的文字限制变成物理约束。真实API仍未配置、未调用。

### 本轮实际验证

环境保持Windows、rag_env、Python3.11.15、PyTorch2.12.0+cu132、RTX4060 8GB。没有停止原GPU任务、连接远程服务器或修改依赖声明。

- 初次三个反例：3失败，均为预期问题；修复后3通过。
- 查询集中回归在中途完整执行16项并通过；后增加两项非空关系/否定检查，不能把该次16项记录改写为18项全跑。
- 扩展组合回归：**93通过、46未选入，257.97秒**。命令为首轮组合命令再加 `tests/work3/test_work3_f04_physical_constraints.py`；未使用skip或放宽生产约束。
- 共享窗口的独立端点/容量检查：**14通过、7未选入，46.40秒**，命令见下。
- 最后补验状态编号、承诺保持和worker程序来源：**2通过、16未选入，55.83秒**。
- 最后补验契约损坏拒绝、不同计划及关系绑定：**4通过、14未选入，23.07秒**。这些与上面的组合回归有重叠，不相加成独立测试总数。
- 正常Windows进程通信下，两worker、4步、一次Lightning更新通过，且检查点来源确为配置注入的程序，两个CPU worker均未初始化CUDA。
- 配置注入程序的GPU BF16烟测：两worker、2步、一次更新，采样/重放最大误差0；显存峰值980,127,744字节，主进程内存峰值1,624,006,656字节；正常报告decision_limit，不标成功。
- 新评估CLI读取上述权重，并切换到另一来源的物料关注程序，正常加载和运行2步；完整及增量账本为true，`success=false`、`decision_limit`、`makespan=null`。这验证跨程序来源加载，不代表烟测模型已有调度收益。

组合回归的完整命令如下。Windows执行时设置 `OMP_NUM_THREADS=1`、`PYTHONDONTWRITEBYTECODE=1`，使用rag_env的绝对Python路径，并把 `--basetemp` 指向仓库外临时目录：

```powershell
python -m pytest tests/work3/test_scheduling_queries.py tests/work3/test_environment_actions.py tests/work3/test_work3_worker_mask_replay.py tests/work3/test_work3_same_timestamp_events.py tests/work3/test_work3_p2_1_objective_report.py tests/work3/test_work3_event_reward_accounting.py tests/work3/test_work3_training_runtime.py tests/work3/test_work3_f04_physical_constraints.py -k "not test_work3_training_runtime or runtime_config or runtime_seed or precision_resolver or training_cli_uses_yaml" -q -p no:cacheprovider --tb=short
```

GPU本轮使用的参数如下；实际 `--output`、`--report` 指向仓库外具名临时文件，记录结果后删除，因此未发布诊断权重：

```powershell
python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml --set paths.query_program=conf/work3/program_predecessor_focus.json --set runtime.device=cuda --set runtime.amp_dtype=bf16 --set runtime.total_env_steps=2 --set ppo.steps_per_iter=2 --set ppo.batch_size=2
```

```powershell
python -m pytest tests/work3/test_work3_f01_feasibility.py -k "station_capacity or team_search or station_conflict_advancement or common_gap or independent_endpoint_oracle or n04" -q -p no:cacheprovider --tb=short
```

用户独立审阅在Linux受限宿主得到的66通过/1项AF_UNIX权限失败，以及最终10项单进程检查通过，是另一个环境的记录；本轮不改通信实现来绕过该宿主限制，也不声称已在正常Linux环境完成复验。用户报告的96次真实前缀对照和三次受控梯度更新未在本轮重复，不能算作新增证据。

### 审阅边界与清理

F1、F2、F3已有直接回归；G1具备上述有限的真实计划差异与正式配置路径，仍应接受下一次独立审阅。首轮五项历史失败没有在本轮删除或修复；没有进行全库重构、长训练、完整工业批次吞吐验收或跨种子效果实验。有限回归不能证明“没有bug”或完成全部开题实验。

本轮临时模型、报告和pytest临时输出置于仓库外，记录结果后清理具名诊断产物。用户原有 `research_diagnostics/`、数据、已有检查点及历史测试继续保留。新增永久内容只有两个程序JSON，回归仍集中在原查询文件；未增加网络、数据库、执行框架或新的训练器。

本轮修正代码提交为 [`196acec12470231a6133fef6f2a3b94ddbba7283`](https://github.com/1ush3n/APAL-Dynamic-2/commit/196acec12470231a6133fef6f2a3b94ddbba7283)。[相对独立审阅提交d82f258的修正差异](https://github.com/1ush3n/APAL-Dynamic-2/compare/d82f25826d3c3707688225e0870224a981fec5fe...196acec12470231a6133fef6f2a3b94ddbba7283)可单独审阅，本报告随后以独立文档提交保存。发布仍使用 `work3-query-migration-20261115` 分支，不合并基准分支，不强制推送。

## 2026-11-16：R1容错修复、正式训练复验与性能取舍

本节是新一轮实施记录。结论是：**保留R1的两行生产修复及六项集中回归，不加入性能补丁**。Windows正式两worker训练、真实PPO更新和必要评估已复验；当前测得的环境微优化上限不足以支持改动。没有恢复时间预测，也没有重新设计网络、奖励或训练器。本节不将前面各轮的测试数量相加。

### 版本、工作区与环境

- 分支：`work3-query-migration-20261115`。
- 起点及开始时再次读取的远程HEAD：`9d824eacd28e6e6f297ddbb044e474a87fa34a6d`。
- R1独立提交及本轮性能对照M0a：[`4dd394dd2a5e365e92fe8234eab9650541b3b505`](https://github.com/1ush3n/APAL-Dynamic-2/commit/4dd394dd2a5e365e92fe8234eab9650541b3b505)。此后只有本报告的文档提交，没有M1或组合优化提交。
- 开始时无已跟踪文件修改；用户原有未跟踪目录`research_diagnostics/`保留。因此运行报告的`source_tree_dirty=true`，不能将整个工作区称为干净。未reset、切换用户工作区、合并基准或强制推送。
- Windows，`D:\Conda\envs\rag_env\python.exe`；Python3.11.15、PyTorch2.12.0+cu132、PyG2.7.0、Lightning2.6.5、OmegaConf2.3.0、NumPy2.4.6。没有安装或升级依赖。
- 本轮CPU、FP32、主线程1、环境线程1、两个spawn worker。保留正式AMP上下文及现有Lightning优化器生命周期。worker报告CUDA均未初始化；已有GPU活动未停止，本轮不使用GPU或连接远程服务器。
- 未取得Dot的Library原始归档，未核对原文件SHA或重跑其30项探索、480对计时。历史结论来自本轮用户提示词，不当作本轮运行证据。没有凭归档名称重新制造补丁或合入六项组合。

### 实际修改：原逻辑 → 新逻辑 → 原因

| 文件/函数 | 原逻辑 | 新逻辑与原因 |
|---|---|---|
| `envs/work3/query_program.py::compile_condition_program` | 对模型结果直接`.get`，`null`、数组、字符串先抛出不在反馈重试范围内的`AttributeError` | 仅在`.get`前增加`isinstance(plan, dict)`检查，失败抛`ValueError("查询计划必须为JSON对象")`，进入原有有限反馈重试 |
| `tests/work3/test_scheduling_queries.py` | 正常生成、缓存与pending审核有检查，缺非对象返回值 | 两组参数化、共六个场景覆盖`None/[]/"bad"`的第二次成功及两次耗尽；同时检查示例/反馈、准确调用次数、缓存、pending拒绝及失败不落盘 |
| 本报告 | 缺少本轮正式训练与热点证据 | 记录实际执行、精确比较边界、性能否决与剩余未验项；不新增生产入口或配置 |

未扩大异常捕获、吞掉任意`AttributeError`、替换坏结果为空字典或绕过语义审核。`attempts=2`仍是总计两次尝试。缓存键、白名单、原说明核对、示例上限和正常离线执行保持原样；成功编译仍是`pending_semantic_review`，正式loader拒绝未经核验的程序。没有请求真实LLM。

### 真实训练与A/A核验

固定使用`conf/work3/train_query.yaml`：QUERY、seed42、两worker、总8步、每段8步、batch8、epoch1、120秒预算、默认全候选查询。680仅同时覆盖baseline与raw_data，其他算法和资源设置相同。所有运行均未耗尽120秒预算。

| 运行身份 | 范围 | 报告耗时/秒 | 成功主优化器步/尝试/AMP跳过 | replay样本/最大误差 |
|---|---|---:|---|---|
| `aa_283_a1` | 正式`main()`，带精确比较探针 | 31.625 | 1 / 1 / 0 | 8 / 0 |
| `aa_283_b` | 同M0a再次运行，带同探针 | 30.641 | 1 / 1 / 0 | 8 / 0 |
| `profile_283` | 正式`main()`及两个worker，cProfile | 32.641 | 1 / 1 / 0 | 8 / 0 |
| `profile_680` | 正式`main()`及两个worker，cProfile | 44.282 | 1 / 1 / 0 | 8 / 0 |
| `cli_283` | 无探针的原`python -m`正式入口 | 36.078 | 1 / 1 / 0 | 8 / 0 |

各次独立完成实际rollout、record、分组GAE/returns、完整PPO loss、backward、Lightning主优化器step与checkpoint；各自`lightning_fit_calls=1`、`history[0].ppo_updates=1`。这里是五次独立烟测，不能称40步生产长训。全部以`decision_limit`截断，`terminated=false`、成功批次0、研究结论资格false；未将不完整工业轨迹判为终局可行。

检查的是`actor_critic_state`和`lightning_training_state.optimizer_state/optimizer_step_counts`，不是不存在的`training_state`或必然存在的顶层`optimizer_state`。五份权重均有限、与各自初始指纹不同，优化器状态非空，成功/尝试均1、跳过0。283 loss=`-0.019621893763542175`、裁剪前grad norm=`11.951152801513672`；680分别为`-0.022127298638224602`、`30.53557014465332`，均有限。QUERY没有时间头权重、时间标签或预测塑形依赖。

A/A两次有序比较共59条边界：37次分类logits/采样、8次完整动作及采样后Torch RNG、5次快照批、4次环境step批、2次close审计，以及GAE/returns、完整loss、裁剪后梯度各一次。梯度探针在原`on_before_optimizer_step`返回后读取；前文grad norm则是原函数报告的裁剪前范数。保存的是按容器顺序及浮点/张量字节计算的SHA，而非allclose；初始权重、所有59条摘要、最终actor、优化器、主进程Python/NumPy/Torch RNG和整个Lightning训练状态均精确相同。138个参数张量实际变化。无探针283正式CLI的最终actor指纹也与两次A/A相同。比较器能拒绝一ULP、候选顺序和等待原因的负控变化。

范围限于实际捕获的原始条件/掩码/候选顺序、图store张量和图根私有分层索引、动作记录、已返回事件与审计、更新张量；**没有证明Tensor别名/stride、PyG store父引用、未暴露的全部未来事件队列及每一步worker RNG完全等价**。本轮没有图、事件、worker或性能改动，不将此范围扩写成全库等价证明。探针额外复制/哈希会占用时间和内存，其耗时不能作为无探针性能基准。

首轮`aa_283_a`在外部比较器把PyG的dataclass类当实例时退出，尚未PPO更新；修正探针后用新名称`a1`重跑。原错误日志保留，既不归因生产代码，也不记作训练成功。无探针完整CLI从解释器启动至退出墙钟55.309秒；表中报告耗时不含最终保存/报告尾部，两个范围不能混用。主进程峰值工作集约5.34GB（283）、6.52GB（680），不是worker合计峰值，也不是GPU显存。

### 当前耗时占比与不实施性能候选的依据

profiling与正式性能计时分开。以下是cProfile定位数据，包含仪器开销，不是M1收益。主进程互斥顶层边界如下；列出的子阶段不得再次加到父阶段上。

| 主进程边界/秒 | 283 | 680 |
|---|---:|---:|
| 环境池构造/启动 | 14.521 | 16.206 |
| reset往返 | 1.543 | 2.356 |
| 8次Actor选动作 | 3.045 | 4.967 |
| 2次bootstrap编码 | 0.894 | 1.195 |
| snapshot往返 | 0.577 | 1.707 |
| step往返 | 0.018 | 0.042 |
| Lightning.fit | 9.690 | 15.324 |
| close | 1.215 | 1.335 |
| checkpoint保存 | 0.083 | 0.090 |
| 整个main墙钟 / 该进程CPU时间 | 33.101 / 14.422 | 44.839 / 22.297 |

fit内部完整PPO前向/loss为3.784/6.296秒，manual_update为5.785/8.871秒，后者主要为backward。GAE约毫秒以下。主进程响应等待16.744/20.434秒嵌套于启动/reset/snapshot/step，不能再加一遍；worker的命令等待主要是等待主进程，不是可优化环境计算。短烟测启动占比很高，不能外推稳定长训比例。worker查询、构图、快照复制、事件/费用、记录，及主进程buffer/minibatch/优化/保存的原始函数边界保留在外部profile JSON和汇总中。

另外通过正常规则动作准备已有生产进度，再实际执行8次QUERY决策：

| 现场起点 | now / cycle | 候选 | 完成 / 预约未开 / 运行 | 已入线飞机 | 8快照实际枚举团队总数 |
|---|---|---:|---|---:|---:|
| 283默认完整计划，step64 | 129.705337 / 5 | 25 | 61 / 2 / 1 | 5 | 2881 |
| 680默认完整计划，step64 | 104.475042 / 2 | 65 | 41 / 20 / 3 | 2 | 6557 |
| 283现有前序focus，0_18确为候选 | 14.227162 / 2 | 13 | 见原始facts | 2 | 48 |

283/680实际H0为292.5726004548256/449.18630954112496、25/58人。每个快照的资格人数、组合规模、状态、候选与动作保留在facts JSON；相同步数不等于同阶段。三个场景使用本轮诊断权重，不是历史优质权重，也不是纯规则终局。full与focus现场不同、信息范围不同，不能把其耗时差称作等价优化收益。

现场profile活动时间为5.005/8.601/4.593秒；首次graph builder构造另可定位为0.467/0.703/0.466秒，包含在snapshot上层时间中。外层wall另含暂停profile后的事实/指纹审计，不能当纯策略链时间。默认两规模查询累计0.283/0.981秒，Actor选动作3.062/4.883秒；没有只测便宜的focus来代替默认查询。

| 候选 | 当前证据与处理 |
|---|---|
| P07 `WorkerCalendar.is_available` | 团队窗口搜索实际调用`find_earliest_slot`；正式283/680两worker合计本函数仅约30.5/5.7微秒，不实施 |
| P08 容量扫描早退 | 正式训练累计0.000904/0.070621秒；即便两个并行worker时间全部串行相加且函数100%免费，整main的极乐观上限也只有约0.0027%/0.1575%。真实中段现场上限约0.60%/3.60%，早退仅省其中一部分；不实施 |
| P01 发布参照局部复用 | 当前构图每task只取一次`publication_reference`，后续已有局部复用；未发现所述重复，也未取得Dot原补丁。全部调用耗时免费对正式链上限也仅约0.17%/0.32%；不凭名称复刻 |
| P19 worker完工扫描 | 内联扫描没有独立计时边界，未获得热点证据；不把整个worker派发函数自耗时当它的耗时。不新增索引、改变暖机/重置/上报顺序或IPC |
| 其他方向 | Actor每次动作和每条重放各只编码一次图；不硬造重复编码。不改P12复制、所有权、buffer、浮点归约、事件、线程或精度 |

正式候选准入事先记录在外部`REGISTRATION.md`：精确差分先通过；若准备保留，283/680×名义/受扰、三个会话、AB/BA平衡、各条件各会话配对中位改善均正、合并至少约5%且超过A/A噪声，无明显可重复p95/内存退化。本轮占比筛选即停止，**没有M1，没有候选配对计时、IQR/p95收益或整体训练加速结论**。不为了产生优化提交而继续低占比实验，也不降低门槛。

### 必要测试、评估与可复制入口

以下命令均在仓库根目录，用rag_env执行；实际pytest临时目录在仓库外。R1原代码红灯6失败/18未选入，均为`.get`的`AttributeError`；修复后第一条7通过/17未选入、14.39秒。第二条3通过/21未选入、13.84秒：小双机名义/受扰真正完成且独立物理/费用检查通过；`reset=False`继承已完成前缀、预约团队/对齐并完成续调度。两条测试集合分别记账，无skip、删断言或放宽生产守卫。

```powershell
$python = 'D:\Conda\envs\rag_env\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python -m pytest tests/work3/test_scheduling_queries.py -k 'non_object_program or offline_program_binds_real_candidate' -q -p no:cacheprovider --tb=short
& $python -m pytest tests/work3/test_scheduling_queries.py -k 'query_network_completes_real_two_aircraft_batch or continuation_evaluation_preserves_progress_and_initial_cost' -q -p no:cacheprovider --tb=short
```

最短正式训练/评估命令如下。`$run`每次创建唯一目录，不依赖已经清理的诊断模型；这是一条可实际先训练、后加载的命令链。无扰动/预登记扰动使用同一检查点和Actor，继续给定中段现场使用已测试的函数接口`evaluate_query_trajectory(..., reset=False)`，CLI仍从零评估。

```powershell
$python = 'D:\Conda\envs\rag_env\python.exe'
$run = Join-Path $env:TEMP ('apal_query_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
& $python -m scripts.work3.train_ppo_work3 --config conf/work3/train_query.yaml --output (Join-Path $run 'query.pt') --report (Join-Path $run 'train.json')
& $python -m scripts.work3.evaluate_query --config conf/work3/train_query.yaml --checkpoint (Join-Path $run 'query.pt') --report (Join-Path $run 'nominal.json') --max-decisions 8
& $python -m scripts.work3.evaluate_query --config conf/work3/train_query.yaml --checkpoint (Join-Path $run 'query.pt') --report (Join-Path $run 'disturbed.json') --scenario-id EARLY_LOW_S0 --max-decisions 8
```

本轮实际执行上述正式模块命令，输出具名为`cli_283.*`、`eval_nominal.*`和`eval_disturbed.*`。两个评估都完成8步、3/2830工序、`success=false`、`decision_limit`、`makespan=null`，完整和增量账本均true。预登记场景`EARLY_LOW_S0`的tau=1465.2455，而短评估只到18.631264，**该CLI前缀尚未触发扰动**；真正受扰命中后完成的证据来自上述小双机回归，不混称工业受扰终局。检查点明确标记smoke、非正式效果评估资格；不据其短片段判断策略优劣。

本轮`git diff --check`通过。没有重跑全库或此前五项历史失败；它们仍按前轮报告记录，不能写成已被R1修好。没有修改共享物理路径，所以不启动昂贵的工业规则终局差分或工作一/二全回归。

### 外部证据、清理与剩余边界

运行时专用目录为`D:\DevTools\codex\home\visualizations\2026\11\15\01a1de09-7824-7311-92bb-ab8d69fb468c\work3_20261116_recheck`；R1原始红绿日志来自同级`apal_r1_20261116`。结束时只保留同级外部包`APAL_work3_R1_M0a_20261116_evidence.zip`：49个条目、737133字节，SHA256=`954bd85ea01aed2c25233b2c133513fe1dc22c9b78e3edaeead65395aca02615`，ZIP CRC及包内逐文件SHA复读通过。包内有单个可复用`recheck.py`、复现命令、预登记、输入/配置身份、原始JSON profile/facts/trace、必要控制台日志及SHA清单。profile是函数统计，不是大规模完整工业trace。

保留的默认基准SHA256：283=`46c0e4482b557640e9674ebb69f11dc3d7ffd648ef3ddf6ac46270ba6dafc46e`；其CSV=`4059f36706bef9539278bd035b8fa8056611bf5af1eefda0625c78b9c64244e1`。680及程序/配置身份见包内清单和facts；未用文件名或Library解析文本的SHA冒充原文件校验。

关键报告、原始统计/摘要和复现脚本先归档并复读，再清理本轮登记的5个诊断`.pt`、pytest临时目录及上述两个外部展开目录；不删除外部ZIP。模型可按包内命令重新生成，不把大权重放进仓库。用户原有`research_diagnostics/`、数据、有效checkpoint、旧研究记录和测试未动。历史B/C/D/E/F/G入口和含义保留，本轮没有新增old/new并列入口。

尚未验证：真实LLM服务、GPU、本轮正常Linux多进程、完整工业批次/长训练与跨种子收益。本轮Windows图Tensor IPC及正式两worker更新通过，不代替Linux宿主权限问题的复验；没有修改通信机制来绕过限制。精确A/A与有限更新证明本轮所记录的通路可运行，不证明所有平台无bug、PPO收敛或整体训练加速。
