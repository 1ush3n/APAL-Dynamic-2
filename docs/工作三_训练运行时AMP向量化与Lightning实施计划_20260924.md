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

- [x] 写缺陷测试：技能/人数限制逐指针补全；环境技能、日历变更不污染已发快照；候选掩码、CPU张量/图复制和版本不匹配均有断言。
- [x] 写等价测试：同一现场逐任务比较候选顺序与留站/后移掩码；对每个候选工序的各个工人前缀比较纯规则与环境规则的候选顺序。
- [x] 写重放测试：固定留站分支从快照采样并重放；已有回归覆盖后移和强制推进不计算团队/对齐概率。
- [x] 红灯记录：`pytest tests/work3/test_work3_training_runtime.py -k snapshot -q`在实现前为`4 failed, 13 deselected`；新增Actor入口测试后再次以`1 failed, 17 deselected`复现缺少`make_decision_snapshot`。
- [x] 提取纯CPU团队补全规则并由环境校验和Actor逐人掩码共同调用；增加快照采样入口，保留旧环境入口作为快照适配，不维护第二套规则。
- [x] 环境step的权威团队校验保持有效；`test_work3_f04_physical_constraints.py`覆盖不合技能、站外、重复及人数不足团队的拒绝。
- [x] 定向验证：`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py tests/work3/test_work3_worker_mask_replay.py tests/work3/test_actor_critic.py tests/work3/test_work3_f04_physical_constraints.py -q`，`43 passed in 23.72s`；`git diff --check`通过。
- [x] 代码/测试提交`9d4cab0`；任务表回填另行提交。等价断言覆盖当次现场全部候选任务和技能/人数可行前缀；留站采样—PPO重放log-prob差在`1e-6`容差内。未运行训练或完整批次轨迹。

### 任务3：单环境FP32快照运行路径与旧路径逐步对照

**依赖：** 任务2；未通过单环境对照不得开多环境。

**文件：**

- 新增：`training/work3_vector_env.py`（同一协议支持`num_envs=1`，worker入口为模块顶层函数）
- 修改：`scripts/work3/train_ppo_work3.py`（仅挂接新FP32单环境路径和固定动作测试入口）
- 测试：`tests/work3/test_work3_training_runtime.py`

**协议：** worker命令为reset、snapshot、step、close；结果包含快照或完整step返回（观测、原始奖励、terminated、truncated、info/费用分项）。动作由主进程提供，worker不加载Actor、不初始化CUDA。测试适配器可以直接执行冻结的动作序列，不调用策略采样。

- [x] 写测试：从同一基线分别构造直接环境与`num_envs=1`spawn环境，冻结完整动作序列并逐步回放；明确覆盖普通动作与`ADVANCE_TO_NEXT_EVENT`。
- [x] 每步比较观测、原始奖励、费用信息、终止/截断、事件顺序、完工记录和step计数；最终独立复核完整任务数、真实开工位置及轨迹可行性。
- [x] 首轮反例记录：新worker缺失时单环境测试`1 failed, 19 deselected`；运行时团队候选逐次构造完整日历快照的性能边界测试`1 failed, 19 deselected`；另发现旧测试硬编码所选分支为留站，与现有合法分支掩码冲突，改为验证所选分支确实合法而非固定分支值。
- [x] 独立位置校验反例：首轮审计测试暴露真实开工位置字段缺失（`KeyError: aircraft_station_at_start`）；确认即时开工不经过`TASK_START`事件，改在统一的`_on_task_started`回调采集。worker和直接环境的开工位置逐项相等后，独立检查器据此验证飞机位置，不再将计划站位复制为实际站位。
- [x] 实现模块顶层spawn worker与reset/snapshot/step/close请求协议；子进程仅持CPU环境及图构造器，不创建Actor、不加载权重、不初始化CUDA。复用统一快照构造器，旧Actor入口适配同一实现。纯团队补全规则直接读技能/效率profile，避免每次候选扫描复制完整工人日历；日历快照仅在决策快照构造时生成。
- [x] 定向验证：`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py tests/work3/test_work3_same_timestamp_events.py tests/work3/test_work3_actual_hit_counterexample.py tests/work3/test_work3_f04_trajectory_checker.py tests/work3/test_work3_worker_mask_replay.py tests/work3/test_actor_critic.py tests/work3/test_work3_f04_physical_constraints.py -q`，`61 passed in 158.60s`；`git diff --check`通过。完整基准实例的2830/2830项任务完成，直接路径与worker逐步输出、费用账本、事件轨迹和最终独立可行性一致；测试确认worker退出。
- [x] 代码/测试提交`199ae8f`。本项仅完成单环境FP32语义对照；未启动双环境、AMP、Lightning、训练或正式性能试验。任务表记录另行提交。

### 任务4：双环境spawn同步采样、预算与在途结算

**依赖：** 任务3通过。

**文件：**

- 修改：`training/work3_vector_env.py`
- 配置核查：`conf/work3/train_pilot.yaml`（已有`runtime.num_envs`覆盖入口；保留默认单环境smoke配置，不改文件）
- 测试：`tests/work3/test_work3_training_runtime.py`

- [x] 写测试：`spawn`下两个CPU worker各持独立环境且主进程只有一个Actor；worker异常可见，`close()`后进程退出，不初始化CUDA上下文。
- [x] 写测试：总预算为奇数且两个worker活动时，最后一轮按固定worker顺序仅派发剩余额度；预算17按worker顺序分配为9/8，实际step总数恰等于上限。
- [x] 写测试：固定事件计划按`(worker_id, episode_index)`映射；reset/step结果分别保留计划场景、是否启动、事件触发、实际命中、未命中原因和在途状态。相同计划不等同于相同事件暴露。
- [x] 写测试：过期墙钟不派发新动作；在途请求只有完整响应才作为转移，超出有界结算时间则返回缺失结果并标记worker中断，不合成后继观测。
- [x] TDD红灯：初始双worker/场景计划定向组`3 failed`，原因是计划构造器缺失且运行时拒绝`num_envs=2`；单独双worker预算反例`1 failed`，复现旧运行时仅支持单worker。墙钟测试用例改为“已过期截止时间”的确定前提，并通过移除接口后观察到缺少`wall_clock_deadline`参数的预期失败，再实现恢复。
- [x] 最小实现同步“并行取快照→主进程策略→并发step→等待完整响应”协议、聚合预算、固定worker顺序、超时/异常清理和场景状态报告；不加自动重启、异步Actor或CUDA worker。
- [x] 配置复用：现有OmegaConf配置已接受`runtime.num_envs=2`覆盖；默认smoke保持1，故未改YAML，避免提前改变单环境对照入口。
- [x] 定向验证：五项双worker/计划/命中/错误/超时测试`5 passed, 20 deselected in 56.25s`；回归命令`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py tests/work3/test_work3_r08_training_reproducibility.py tests/work3/test_work3_same_timestamp_events.py tests/work3/test_work3_actual_hit_counterexample.py tests/work3/test_work3_f04_trajectory_checker.py tests/work3/test_work3_worker_mask_replay.py tests/work3/test_actor_critic.py tests/work3/test_work3_f04_physical_constraints.py -q`为`76 passed in 237.25s`；`py_compile`与`git diff --check`通过。组合回归包含任务3冻结动作序列对照及R08复现测试；没有重新运行2830工序轨迹或训练。
- [x] 代码/测试提交`79e48e3`；任务表回填另行提交。任务4仅完成双环境FP32协调器及边界验收；GAE、跨段标签、episode塑形版本仍属任务5，AMP/Lightning和完整批次试点尚未开始。

### 任务5：worker/episode/segment GAE、跨段标签与塑形版本固定

**依赖：** 任务4通过。

**文件：**

- 修改：`models/work3/ppo_buffer.py`
- 修改：`models/work3/potential_shaping.py`或新增紧邻的episode预测器版本注册器（仅在现有塑形器无法持有版本引用时新增）
- 复核：`training/work3_vector_env.py`（无需改动：`DecisionSnapshot`已携带worker/episode标识；segment由采样协调器编号；未结清step仍返回缺失结果）
- 测试：`tests/work3/test_work3_training_runtime.py`

- [x] 写测试：两个worker、两个episode及两个采样段交错加入奖励；GAE不跨任一复合键串接。非终止段使用对应bootstrap value；成功/失败真实终止均不接到reset状态。
- [x] 写测试：待补标签缓存与PPO rollout buffer独立，跨rollout清理仍保留，真实cycle transfer后才产生训练样本；未揭示周期可按worker/episode丢弃且不生成伪标签。
- [x] 写测试：一个episode结束并刷新latest预测器时，另一个运行中episode继续使用原版本；新episode使用新版本；旧版本最后一个引用释放后回收。
- [x] 写测试：PPO transition的图、状态/时间张量、掩码与记录内标签在加入时复制到CPU，不受后续源对象修改影响。
- [x] TDD红灯：`-k "gae or label_cache or predictor_version"`首轮`2 failed`，复现PPO转移缺worker/segment键和标签缓存不接受worker标识；episode版本/样本副本组`2 failed`，复现缺少episode版本绑定且buffer持有可变输入；补充worker限定discard后`1 failed`，复现标签清理接口未隔离worker。
- [x] 最小实现：按`(worker_id, episode_id, segment_id)`分别计算GAE并要求非终止段提供匹配bootstrap；PPO缓冲区与待补标签缓存各自持有递归CPU副本；缓存按worker/episode/cycle隔离；塑形器按episode引用冻结的Actor/时间头版本，最后引用释放后回收旧版本。现有快照已含worker/episode标识，segment由采样协调器编号；故核查后未改向量运行时文件。
- [x] 定向验证：新增4项`4 passed, 25 deselected in 8.39s`；任务5主回归`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py tests/work3/test_ppo_trainer.py tests/work3/test_work3_f07_time_integration.py tests/work3/test_work3_f11_termination_reporting.py tests/work3/test_potential_shaping.py -q`为`54 passed in 171.34s`；R08复现`10 passed in 38.23s`；F06签名残差回归`5 passed in 10.40s`。
- [x] 代码/测试提交`056e314`；任务表随文档提交回填。未运行多环境训练或完整批次实验；当前训练入口尚未消费多worker复合键，向量化PPO整合留待任务6 Lightning生命周期。

### 任务6：工作三Lightning生命周期与PPO优化器所有权

**依赖：** 任务5通过；本任务先用FP32，不启用AMP。

**文件：**

- 新增：`training/work3_lightning.py`
- 修改：`models/work3/ppo_trainer.py`
- 修改：`scripts/work3/train_ppo_work3.py`
- 测试：`tests/work3/test_work3_training_runtime.py`
- 不修改：`training/lightning_module.py`

- [x] 写测试：工作三Lightning模块由`Trainer.fit()`驱动至少一个PPO优化步，报告的Lightning optimization step与environment step分别计数。
- [x] 写测试：`configure_optimizers()`是Actor、Critic、共享图编码器和时间头参数的唯一优化器所有者；参数去重，不允许共享图参数进入两个优化器。
- [x] 写测试：PPO损失纯计算路径不创建optimizer、不调用`backward()`/`step()`；Lightning手动优化路径用Lightning管理的优化器/`manual_backward`完成更新，梯度裁剪和有限值检查有效。
- [x] 写测试：DataModule的DataLoader `num_workers=0`；训练入口通过独立spawn CPU环境worker取快照、执行动作（本任务实际入口为`num_envs=1`）。
- [x] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k lightning -q`记录红灯；完整runtime文件的最终回归见任务记录。
- [x] 从`PPOTrainerWork3`提取PPO/时间监督损失与重放计算，保留原损失公式/条件熵语义；Lightning模块拥有唯一优化器和更新步骤，训练入口改为`Trainer.fit()`编排。
- [x] 用CPU FP32跑至少一次Lightning更新，重放概率与采样概率一致；运行原`test_ppo_trainer.py`和训练入口回归。
- [x] 任务表记录旧Lightning文件边界、优化器所有权检查、CPU更新结果和提交号。

**任务6验收边界：** 完成的是Lightning FP32生命周期及单环境训练入口。双worker协调器、预算、GAE和episode版本此前已分别在任务4/5验收；但`train_ppo_work3.run_training()`尚未消费`runtime.num_envs`，不能据此声称训练入口已完成多环境采样。AMP前增加任务6B，将多worker FP32采样接入同一个Lightning更新入口。

### 任务6B：多环境FP32训练入口接线（AMP前置）

**依赖：** 任务6；在AMP和完整批次试点前完成。

- [ ] 先写`num_envs=2`训练入口测试：单一主进程Actor/Lightning优化器、两个spawn CPU环境worker；相同固定worker/episode场景计划可追踪。
- [ ] 聚合预算精确计算所有worker实际`step()`，剩余预算按稳定顺序派发；worker/episode/segment独立缓存并正确bootstrap后合并PPO更新。
- [ ] 对齐`runtime.num_envs`配置，不改变单环境FP32语义；C/D在相同worker/episode事件映射下执行。
- [ ] 检查跨worker同一决策编号的时间标签键、episode势函数版本和worker异常清理；不把Lightning/DataLoader worker用于环境并行。
- [ ] 使用两个环境跑FP32短段，验证实际environment step数、Lightning优化步数、快照掩码重放、GAE和进程CUDA状态；通过前不启用AMP。
- [ ] 先观察新缺陷测试红灯，再修复并运行任务6回归；记录结果并独立提交代码/测试。

### 任务7：AMP一致精度路径、配置/profile检查点

**依赖：** 任务6与任务6B FP32通过；在双环境训练入口验收前不得启用AMP。

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
- 第10节阶段顺序与试点门槛：任务3→4→5→6→6B→7→8严格覆盖。
- 非目标/用户工作区边界和逐任务提交：全局约束与提交规则覆盖。
