# 工作三训练运行时实施计划

> **执行约束：**本计划按任务顺序在当前会话内逐项执行。每项先写反例/回归测试并记录首次结果，再做最小实现、定向验证、更新主任务表，最后提交一个可运行版本；不得为编号隔离提交不可运行的中间代码。

**目标：** 在不改变工作三环境、条件分支PPO、奖励和C/D方法定义的前提下，完成YAML配置、CPU向量环境、AMP与工作三专用PyTorch Lightning生命周期，并通过分阶段运行门槛。

**架构：** 主进程仅保留一个Actor、Critic、时间头与唯一优化器；`spawn`子进程各自拥有一个CPU事件环境，只传递不可变CPU决策快照和动作。先验证单环境FP32语义，再验证双环境FP32预算/掩码/GAE/episode塑形版本，最后接入AMP和Lightning。

**技术栈：** Python、PyTorch、PyTorch Geometric、OmegaConf、PyTorch Lightning、pytest、`multiprocessing`标准库。

**设计依据：** `docs/工作三_训练运行时AMP向量化与Lightning设计稿_20260924.md`；任务记录：`docs/工作三_代码一致性问题与修复任务表.md`中W3-13。

## 全局约束

- 按用户确认的实施顺序：单环境FP32语义对照 → 双环境spawn/FP32 → AMP与Lightning → 配置和profile检查点验收 → 有限预算闭环；前一阶段未过，不进入后一阶段。
- 正式完整批次试点参数不在本计划中预先冻结；12000步/5400秒等历史值仍只是试点保护候选，完成运行时验收后另行核对并提交配置。
- 聚合交互预算只统计所有环境实际完成的`env.step()`调用总数，包括`ADVANCE_TO_NEXT_EVENT`；Lightning优化步数、向量轮数单独计数。
- 墙钟上限停止派发新动作；只接纳结清的完整响应，超时未返回的转移整体丢弃并标记基础设施中断，不从缺失观测bootstrap。
- 场景清单固定映射到`(worker_id, episode_index)`并分别报告计划、实际执行和命中；不同策略实际暴露不同不等于相同事件暴露。正式配对评测必须逐一运行完整固定测试清单，不以成功批次数停止或筛除失败。
- GAE按`(worker_id, episode_id, segment_id)`隔离；完成、失败、截断和在途中断分别处理，禁止跨episode连接。
- 预测器快照按episode绑定版本；旧版本保留至最后一个引用episode结束，只刷新新episode使用的版本。
- C profile关闭学习时间预测时允许无时间头检查点；D profile必须严格加载训练完成且图/特征版本兼容的在线时间头及预测器组件，缺失或随机初始化一律拒绝作为正式D。
- 新代码遵守仓库`AGENTS.md`约定：文件路径用`pathlib`；类型标注；张量/数据边界校验形状；维度变换注释写明形状；全局随机源和确定性设置统一记录；CUDA策略前向与重放启用同配置`torch.amp.autocast`；奖励、费用、GAE和重要性比率保持FP32。
- 不修改用户现有的`training/lightning_module.py`工作区差异；不引入Ray、DDP、第二个Actor、自动worker重启或默认梯度累积；不做完整批次训练或方法性能结论，直至全部运行时门槛通过。

## 文件职责与预期边界

| 文件 | 职责 |
|---|---|
| `conf/work3/train_pilot.yaml` | 工作三运行配置：方法profile、数据/事件清单、种子、环境数、交互/墙钟上限、精度、PPO参数和输出目录；不包含正式实验未确认参数。 |
| `training/work3_runtime_config.py` | OmegaConf加载、必填项/值域校验、resolved配置与SHA256、统一随机种子和worker派生种子。 |
| `envs/work3/decision_snapshot.py` | 不可变、CPU可序列化的决策快照与逐人团队补全输入/掩码计算；不持有环境或CUDA对象。 |
| `training/work3_vector_env.py` | 顶层spawn worker入口、命令/结果协议、同步快照与step协调、预算计数、超时结算和确定性关闭。 |
| `models/work3/actor_critic.py` | 保留现有条件分支动作头；增加消费快照的采样/重放路径，原环境兼容入口仅作薄适配。 |
| `models/work3/ppo_buffer.py` | 保存快照、精度一致的采样记录及worker/episode/segment标识；分段计算GAE；待补真实转站标签缓存跨rollout段存在。 |
| `models/work3/ppo_trainer.py` | 保留PPO数学目标，拆出纯损失/前向重放计算；不再创建优化器或调用`backward/step`。 |
| `training/work3_lightning.py` | 工作三专用`LightningDataModule`与`LightningModule`，唯一优化器、手动优化、AMP/检查点所有权；不复用旧通用Agent模块。 |
| `scripts/work3/train_ppo_work3.py` | 保留兼容旧调用所需的公共导出；新入口只负责配置、种子、组件构造、`Trainer.fit`和报告，不再承载训练大循环。 |
| `tests/work3/test_work3_training_runtime.py` | 配置、快照、单/双环境协调、预算、GAE、塑形版本、Lightning、AMP、检查点和端到端smoke的定向回归；若文件过大，仅按职责拆为相邻`test_work3_runtime_*.py`。 |

除上述已确认边界外，先查现有类和测试再决定是否新增文件；若一项已有共享实现可直接复用，不复制一套近似实现。任务实际改动文件在对应任务记录中列明。

---

### 任务1：YAML运行配置与确定性入口

**依赖：** 无。此任务只建立单一配置入口，不启动训练。

**文件：**

- 新增：`conf/work3/train_pilot.yaml`
- 新增：`training/work3_runtime_config.py`
- 测试：`tests/work3/test_work3_training_runtime.py`
- 修改：`docs/工作三_代码一致性问题与修复任务表.md`

**接口：**

- `load_work3_runtime_config(path: Path, overrides: Sequence[str] = ()) -> DictConfig`
- `validate_work3_runtime_config(config: DictConfig) -> None`
- `seed_work3_runtime(seed: int, *, deterministic: bool) -> None`
- `derive_worker_seed(seed: int, worker_id: int, episode_index: int) -> int`
- `resolved_config_fingerprint(config: DictConfig) -> tuple[str, str]`返回resolved YAML文本及SHA256。

- [ ] 写测试：默认配置含profile、`num_envs=1`、FP32、事件清单/散列字段、总步数/墙钟边界、线程限制、PPO参数和`pathlib`输出路径。
- [ ] 写测试：CLI式OmegaConf覆盖进入resolved YAML并改变配置哈希；缺字段、非法精度、非正预算、越界环境数均明确失败。
- [ ] 写测试：相同主种子和`(worker_id, episode_index)`派生相同环境种子，不同worker或episode不会复用同一派生种子；Python/NumPy/PyTorch种子初始化可重复。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -q`执行新测试，记录预期失败原因；若测试通过，检查是否已有配置实现或测试前提已变化，不人为制造失败。
- [ ] 使用OmegaConf现有依赖实现配置加载、值域校验、resolved配置/散列和统一种子函数；不安装第二套配置框架。
- [ ] 重跑该测试文件，并运行`tests/work3/test_work3_r08_training_reproducibility.py`与`tests/work3/test_train_ppo.py`。
- [ ] 在任务表记载配置字段、测试首轮结果/最终结果、实际修改文件和提交号；仅提交本任务文件。

### 任务2：CPU决策快照与逐人团队补全掩码

**依赖：** 任务1。

**文件：**

- 新增：`envs/work3/decision_snapshot.py`
- 修改：`envs/work3/environment.py`（只增加导出快照所需的只读原始状态接口；环境`step()`继续权威校验）
- 修改：`models/work3/actor_critic.py`
- 测试：`tests/work3/test_work3_training_runtime.py`及已有`tests/work3/test_work3_worker_mask_replay.py`

**接口：**

```python
@dataclass(frozen=True)
class DecisionSnapshot:
    worker_id: int
    episode_id: int
    episode_index: int
    state_features: torch.Tensor
    time_features: torch.Tensor
    graph_snapshot: HeteroData
    candidate_task_keys: tuple[str, ...]
    candidate_task_features: torch.Tensor
    branch_masks: tuple[tuple[bool, bool], ...]
    team_contexts: tuple[TeamCompletionContext, ...]

def worker_completion_mask(
    context: TeamCompletionContext,
    selected_worker_ids: tuple[int, ...],
    max_station_workers: int,
) -> tuple[bool, ...]: ...
```

`TeamCompletionContext`保存有序本站工人稳定ID、技能集合、效率、日历区间快照、任务技能/人数需求和团队规则所需字段。日历/资源时序特征不替代环境资源排程；最终团队可行性仍由环境复核。图与张量在生成时clone到CPU，不保留环境引用、可变状态别名或CUDA张量。

- [ ] 写缺陷测试：构造技能/人数受限的两名合法工人，逐指针断言只有能保留合法补全的成员为True；改变源环境技能、日历或工人绑定后，已发快照内容保持不变。
- [ ] 写等价测试：同一现场下逐任务比较环境的留站/后移分支掩码与快照掩码；对每个可能的已选工人前缀比较逐人掩码及候选顺序。
- [ ] 写重放测试：从快照采样留站动作并在不变快照上重放，log-prob误差不超过FP32当前基线容差；后移和强制推进仍不计算团队/对齐头概率。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py tests/work3/test_work3_worker_mask_replay.py -q`记录首轮结果。
- [ ] 提取纯CPU团队补全规则并让快照逐步掩码调用它；增加Actor快照采样/重放入口，保留现有环境入口为兼容适配，不能产生两套分支合法性规则。
- [ ] 环境`step()`收到完整团队时再次按当前状态校验；快照旧/伪造团队必须被拒绝。
- [ ] 重跑上述测试及`tests/work3/test_actor_critic.py`、`tests/work3/test_work3_f04_physical_constraints.py`。
- [ ] 更新任务表记录掩码等价矩阵、重放误差和提交号后提交。

### 任务3：单环境FP32快照运行路径与旧路径逐步对照

**依赖：** 任务2；未通过单环境对照不得开多环境。

**文件：**

- 新增：`training/work3_vector_env.py`（同一协议支持`num_envs=1`，worker入口为模块顶层函数）
- 修改：`scripts/work3/train_ppo_work3.py`（仅挂接新FP32单环境路径和固定动作测试入口）
- 测试：`tests/work3/test_work3_training_runtime.py`

**协议：** worker命令为reset、snapshot、step、close；结果包含快照或完整step返回（观测、原始奖励、terminated、truncated、info/费用分项）。动作由主进程提供，worker不加载Actor、不初始化CUDA。测试适配器可以直接执行冻结的动作序列，不调用策略采样。

- [ ] 写测试：从同一基线和同一场景分别构造旧直接环境路径与`num_envs=1`新路径，向双方逐步发送相同冻结动作序列。
- [ ] 每一步比较：当前时刻、事件类型/次序、动作接受结果、原始奖励及分项费用、实际开完工、同步转站、终止/截断和观测关键字段；结束后使用既有独立轨迹检查器比较完整性与可行性。
- [ ] 测试step计数把显式`ADVANCE_TO_NEXT_EVENT`和普通动作都计为一次，环境内部事件推进不虚增动作数。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k single_env -q`记录首轮差异；差异必须定位到定义的语义，不能通过放宽断言掩盖。
- [ ] 实现一个worker的spawn协议、消息校验和关闭；保持FP32，不接Lightning/AMP。
- [ ] 重跑固定动作逐步对照及`tests/work3/test_work3_same_timestamp_events.py`、`tests/work3/test_work3_actual_hit_counterexample.py`、`tests/work3/test_work3_f04_trajectory_checker.py`。
- [ ] 对照未解释差异为零、worker关闭无遗留后更新任务表并提交；否则留在本任务修复。

### 任务4：双环境spawn同步采样、预算与在途结算

**依赖：** 任务3通过。

**文件：**

- 修改：`training/work3_vector_env.py`
- 修改：`conf/work3/train_pilot.yaml`
- 测试：`tests/work3/test_work3_training_runtime.py`

- [ ] 写测试：`spawn`下两个CPU worker各持独立环境且主进程只有一个Actor；worker异常可见，`close()`后进程退出，不在子进程导入CUDA上下文。
- [ ] 写测试：总预算为奇数且两个worker活动时，最后一轮按固定worker顺序仅派发剩余额度；实际完成`step()`总数恰等于上限，不因强制推进漏计或超发。
- [ ] 写测试：固定事件计划按`(worker_id, episode_index)`映射；报告分别保存计划场景、实际启动/执行场景、事件触发、命中、未命中原因和在途场景。不得声称相同计划等于相同事件暴露。
- [ ] 写测试：达到墙钟预算后停止新派发；完整返回结清为转移，超过有界结算时间的请求整体丢弃并标记基础设施中断；无后继观测时不调用价值bootstrap。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k 'vector or budget or timeout or event_plan' -q`记录反例结果。
- [ ] 最小实现同步“取活动快照→单Actor批次决策→并发step→汇总完整响应”协议、严格step预算、清理和报告，不加自动重启/异步Actor。
- [ ] 重跑双worker测试、任务3单worker对照以及`tests/work3/test_work3_r08_training_reproducibility.py`。
- [ ] 更新任务表并提交；若平台spawn测试因Windows/Linux行为差异失败，先在当前Windows rag_env定位，不能跳过子进程测试进入下一阶段。

### 任务5：worker/episode/segment GAE、跨段标签与塑形版本固定

**依赖：** 任务4通过。

**文件：**

- 修改：`models/work3/ppo_buffer.py`
- 修改：`models/work3/potential_shaping.py`或新增紧邻的episode预测器版本注册器（仅在现有塑形器无法持有版本引用时新增）
- 修改：`training/work3_vector_env.py`
- 测试：`tests/work3/test_work3_training_runtime.py`

- [ ] 写测试：两个worker、两个episode及两个采样段交错加入奖励；GAE不得跨任一键串接。正常采样段用其合法后继value bootstrap；成功和真实失败终止不接到reset后的状态。
- [ ] 写测试：待补标签缓存跨至少两个rollout清理仍保留，直至其真实cycle transfer后才出现训练样本；任务截断/失败时未揭示周期保持无标签。
- [ ] 写测试：环境A结束并刷新latest预测器时，运行中的环境B仍使用episode启动时版本；只有新episode使用新版本，最后一个旧版本引用释放后才清理旧副本。
- [ ] 写测试：transition中的图/时间输入、掩码和标签是采样时CPU副本，不因后续环境变化/重置而变更。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k 'gae or label_cache or predictor_version' -q`先跑新测试并记录。
- [ ] 最小实现按复合键隔离rollout和待补标签；按episode绑定不可变塑形预测器版本；墙钟未结清请求不生成转移/标签/价值。
- [ ] 重跑`tests/work3/test_ppo_trainer.py`、`tests/work3/test_work3_f07_time_integration.py`、`tests/work3/test_work3_f11_termination_reporting.py`与新测试。
- [ ] 更新任务表记录关键边界及提交号。

### 任务6：工作三Lightning生命周期与PPO优化器所有权

**依赖：** 任务5通过；本任务先用FP32，不启用AMP。

**文件：**

- 新增：`training/work3_lightning.py`
- 修改：`models/work3/ppo_trainer.py`
- 修改：`scripts/work3/train_ppo_work3.py`
- 测试：`tests/work3/test_work3_training_runtime.py`
- 不修改：`training/lightning_module.py`

- [ ] 写测试：工作三Lightning模块由`Trainer.fit()`驱动至少一个PPO优化步，报告的Lightning optimization step与environment step分别计数。
- [ ] 写测试：`configure_optimizers()`是Actor、Critic、共享图编码器和时间头参数的唯一优化器所有者；参数去重，不允许共享图参数进入两个优化器。
- [ ] 写测试：PPO损失纯计算路径不创建optimizer、不调用`backward()`/`step()`；Lightning手动优化路径用Lightning管理的优化器/`manual_backward`完成更新，梯度裁剪和有限值检查有效。
- [ ] 写测试：DataModule的DataLoader `num_workers=0`，采样只由独立CPU环境worker池并行。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k lightning -q`记录红灯。
- [ ] 从`PPOTrainerWork3`提取PPO/时间监督损失与重放计算，保留原损失公式/条件熵语义；Lightning模块拥有唯一优化器和更新步骤，训练入口改为配置化`Trainer.fit()`编排。
- [ ] 用CPU FP32跑至少一次Lightning更新，重放概率与采样概率一致；运行原`test_ppo_trainer.py`和训练入口回归。
- [ ] 任务表记录未改旧Lightning文件、参数所有权检查、CPU更新结果和提交号。

### 任务7：AMP一致精度路径、配置/profile检查点

**依赖：** 任务6 FP32通过；双环境阶段此前已通过。

**文件：**

- 修改：`training/work3_lightning.py`
- 修改：`training/work3_runtime_config.py`
- 修改：`scripts/work3/train_ppo_work3.py`
- 修改/新增：工作三检查点加载器（优先复用`PPOTrainerWork3`现有检查点契约）
- 测试：`tests/work3/test_work3_training_runtime.py`

- [ ] 写测试：CUDA AMP可用时，采样与PPO重放用相同dtype/设备/autocast上下文，记录log-prob最大误差；不可用硬件明确拒绝所选AMP而不伪报AMP训练。
- [ ] 写测试：奖励、费用、GAE、优势、value目标和重要性比率显式保持FP32；AMP前向输出、辅助损失、梯度范数和报告数值均有限。
- [ ] 写测试：Actor/Critic共享参数只由一个Lightning optimizer更新；FP16时GradScaler仅归Lightning精度插件所有，BF16不创建第二个scaler。
- [ ] 写测试：C profile缺时间头仍可严格加载Actor/Critic/图版本/配置元数据；D profile缺时间头、随机未训练头、错误图版本或错误特征版本均明确失败；兼容D加载时间头和塑形副本。
- [ ] 写测试：检查点保存resolved YAML/哈希、代码提交、场景散列、图特征版本、C/D profile、Actor初始指纹、模型权重、Lightning优化器/精度状态和随机状态；不包含环境进度时标`non_exact`续训。
- [ ] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k 'amp or checkpoint or profile' -q`运行红灯用例。
- [ ] 用Lightning精度插件启用AMP；rollout前向显式使用`torch.amp.autocast`同一精度路径；统计/GAE等保持FP32。只在本机RTX 4060 8GB及本次固定`num_envs`/batch/rollout配置上记录峰值，不外推任意配置。
- [ ] 实现profile严格载入并重跑新测试、`tests/work3/test_work3_f09_experiment_groups.py`、`tests/work3/test_work3_r08_training_reproducibility.py`和Lightning回归。
- [ ] 更新任务表并提交；此时仍不启动完整批次试点。

### 任务8：端到端小预算运行门槛与试点协议冻结

**依赖：** 任务1—7全部通过；不作为正式性能实验。

**文件：**

- 修改：`conf/work3/train_pilot.yaml`
- 修改：`scripts/work3/train_ppo_work3.py`及必要的工作三runtime组件
- 测试：`tests/work3/test_work3_training_runtime.py`
- 记录：`docs/工作三_代码一致性问题与修复任务表.md`

- [ ] 写集成测试：同一冻结训练事件计划/种子下，C和D各运行有限交互预算；报告计划事件、实际启动/执行事件、触发/命中、真实转站标签、优化更新次数、实际step数、终止/截断/失败原因、独立可行性和原始奖励分项。
- [ ] 写C/D门禁测试：两组初始Actor指纹相同；D记录正负残差和辅助监督更新次数及episode冻结的塑形版本；正式D profile缺少训练时间头时不能生成正式检查点/结果。
- [ ] 写资源记录测试：CPU线程数、环境数、rollout/batch、AMP dtype、主机峰值内存、GPU峰值显存、墙钟和结算请求数有明确字段；研究结果资格在smoke/pilot保持false。
- [ ] 以配置指定的`rag_env`执行完整工作三测试集和新runtime测试；保留3个检查点依赖用例的skip实情，不计为通过。全仓既有失败依照任务表基线分类单列，不宣称全仓通过。
- [ ] 在RTX 4060上执行有明确小步数/墙钟保护的smoke，确认至少一次Lightning优化更新、C/D profile规则和实际hit/事件标签日志完整；如CUDA资源条件不满足，停止并记录原因，不退化成声称完成的CPU AMP运行。
- [ ] 单独运行`num_envs=2` FP32及配置所选AMP的集成验收；所有失败/截断保留，不能仅筛成功episode。
- [ ] 所有门槛通过后，只冻结下一步完整批次试点的配置、训练集事件清单及哈希、种子、总聚合step/墙钟限制、`num_envs`、精度、batch/rollout和C/D初始权重指纹；提交冻结记录。此任务不自动授权正式测试集评测或方法优劣结论。

## 任务提交和记录规则

1. 开始每项前检查`git status --short --branch`；不覆盖或暂存与当前任务无关的工作区变更。
2. 新缺陷测试首次执行结果原样写入任务表；已有回归测试允许首次即通过。若新测试首次通过，先核实既有修复和前提，不人为制造红灯。
3. 只有当前任务的反例、定向回归和必要兼容测试通过，才提交该任务；可把测试与修复同一提交。任务表同步写入修改方式、命令、结果、提交号和未通过/未运行范围。
4. 一项未通过时留在该项定位，不开始下一项；不因局部通过声称完整方法或全仓通过。
5. 全部runtime门槛通过前不推送GitHub、不启动完整批次训练或正式C/D配对实验；推送须在后续确认总体测试门槛后进行。

## 计划自审

- 设计稿第4节快照、预算、事件暴露：任务2、4、8覆盖。
- 第5节GAE、标签跨段、塑形副本episode版本：任务5覆盖。
- 第6节Lightning唯一优化器和工作三模块：任务6覆盖。
- 第7节AMP精度边界、线程/资源：任务7、8覆盖。
- 第8节配置、种子、profile检查点：任务1、7覆盖。
- 第10节阶段顺序与试点门槛：任务3→4→5→6→7→8严格覆盖。
- 非目标/用户工作区边界和逐任务提交：全局约束与提交规则覆盖。
