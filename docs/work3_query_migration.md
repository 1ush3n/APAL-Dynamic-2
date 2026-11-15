# 开题方案代码迁移与整理：完整修改及审阅报告（2026-11-15）

基线：`837194591b67c83e9cd2943d987de19e84a2c996`。本地分支：`work3-query-migration-20261115`。
原有未跟踪目录 `research_diagnostics/` 保留，不属于本次改动。

## 报告结论与审阅范围

本次已经把工作三的QUERY方法接入实际CPU多进程采样、Actor自回归评分、PPO重放、Lightning更新、模型保存加载和独立评估入口。结果是可运行、可检查的代码迁移，不是调度效果结论。默认配置仍为有限预算烟测，不能将该配置及其随机初始化后的权重称为已训练有效的正式模型。

研究内容一、二仅核查现有入口及组合配置，没有改写它们的环境、训练器或资源配置。工作三保留原有事件驱动环境；新的实际时间代价通过可选权重进入该环境，历史配置默认权重为0。LLM部分按用户后续要求缩减为离线与mock通路验证，真实API未配置、未调用。

审阅时应同时阅读源码及本报告的限制：部分接口状态只保留类型表达，并没有实现对应的新算法；目前的查询程序只支持一种检索计划；完整规模吞吐、长训练收益、Linux运行及全部历史测试一致性均未完成验收。

## 完整文件清单与职责

以下20个文件构成本次提交范围；不包含用户已有的研究诊断目录、数据、模型或临时输出。

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

## 本轮发布与复验记录

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
