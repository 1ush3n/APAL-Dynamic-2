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

- [x] 写测试：默认配置含profile、`num_envs=1`、FP32、事件清单/散列字段、总步数/墙钟边界、线程限制、PPO参数和`pathlib`输出路径。
- [x] 写测试：CLI式OmegaConf覆盖进入resolved YAML并改变配置哈希；缺字段、非法精度、非正预算、越界环境数均明确失败。
- [x] 写测试：相同主种子和`(worker_id, episode_index)`派生相同环境种子，不同worker或episode不会复用同一派生种子；Python/NumPy/PyTorch种子初始化可重复。
- [x] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -q`执行新测试，记录预期失败原因；若测试通过，检查是否已有配置实现或测试前提已变化，不人为制造失败。
- [x] 使用OmegaConf现有依赖实现配置加载、值域校验、resolved配置/散列和统一种子函数；不安装第二套配置框架。
- [x] 重跑该测试文件，并运行`tests/work3/test_work3_r08_training_reproducibility.py`与`tests/work3/test_train_ppo.py`。
- [x] 在任务表记载配置字段、测试首轮结果/最终结果、实际修改文件和提交号；仅提交本任务文件。

**任务1记录核对：** 主任务表已记录首次新增配置反例`8 failed`及后续gamma/pilot目标与空白路径反例；修复后合并定向回归`26 passed in 45.33s`，配置与种子实现提交`35eb658`。本计划勾选状态据此补齐，不重复修改运行代码。

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

- [x] 先写`num_envs=2`训练入口测试：单一主进程Actor/Lightning优化器、两个spawn CPU环境worker；相同固定worker/episode场景计划可追踪。
- [x] 聚合预算精确计算所有worker实际`step()`，剩余预算按稳定顺序派发；worker/episode/segment独立缓存并正确bootstrap后合并PPO更新。
- [x] 训练入口接收并记录`num_envs`，增加CLI参数；C/D按同一worker/episode事件映射可复现。YAML配置加载仍留在任务7，不在本任务声称已接通。
- [x] 检查跨worker时间标签复合键、episode势函数版本和worker异常清理；环境并行仍由spawn worker承担，不使用Lightning/DataLoader worker。
- [x] 使用两个环境运行FP32短段，核验聚合步数、Lightning更新、采样/重放样本、episode版本、GAE分段与worker CUDA状态；未启用AMP。
- [x] 先观察新缺陷测试红灯，再修复并运行任务6回归；代码/测试已独立提交。

**任务6B执行记录（2026-09-25）：** 首个双环境训练入口用例在旧入口上按预期以`TypeError: unexpected keyword argument 'num_envs'`失败。复核期间另发现入口收到worker错误后未关闭同批其他worker；新增异常回收反例先红（异常后`workers_alive=(True, True)`），入口关闭向量环境后通过。实现将采样按固定worker顺序批量派发，`steps_per_iter`及全局决策预算按所有worker实际step合计；每worker维护独立episode/标签/塑形版本，GAE按`(worker, episode, segment)`结算；报告记录固定事件计划指纹、worker实际步数和势函数版本。同步episode wave是有意简化：有worker先完成时会空闲至同wave其他worker结束，以换取事件映射和塑形版本边界简单可审计；若实测轨迹长度差导致吞吐明显损失，再单独改为异步补位。`run_training(num_envs=...)`和`--num-envs`现已接通，YAML加载仍属于任务7。验证：`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_train_ppo.py -q`为`4 passed in 64.49s`；`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -q`为`34 passed in 172.55s`；相关`py_compile`及`git diff --check`通过。双worker短测聚合3步为`[2, 1]`、完成1次Lightning优化更新；未运行完整批次、未启用AMP、未声称正式训练效果。实现/测试提交`baa4336`；用户既有`training/lightning_module.py`未修改、未暂存。

### 任务7：AMP一致精度路径、配置/profile检查点

**依赖：** 任务6与任务6B FP32通过；在双环境训练入口验收前不得启用AMP。

**文件：**

- 修改：`training/work3_lightning.py`
- 修改：`training/work3_runtime_config.py`
- 修改：`scripts/work3/train_ppo_work3.py`
- 修改/新增：工作三检查点加载器（优先复用`PPOTrainerWork3`现有检查点契约）
- 测试：`tests/work3/test_work3_training_runtime.py`

- [x] 训练CLI默认载入工作三OmegaConf YAML；`--set`与显式CLI选项作为覆盖值，并在覆盖后重新校验。
- [x] 将配置中的预算、C/D profile、路径、PPO参数、随机确定性、主/环境线程数和worker结算时限接入训练入口；smoke的实际决策数受单段长度约束。
- [x] 报告和检查点保存解析后YAML及SHA256；训练入口校验两者配对且哈希一致。

**子阶段7A执行记录（2026-09-25）：** 初始CLI反例因`--config/--set`未注册而失败；配置留存反例修正smoke夹具后，旧报告在`resolved_runtime_config_yaml`处以`KeyError`失败；worker线程配置反例因向量环境不接受`worker_torch_num_threads`而失败。修复后训练入口`5 passed in 81.34s`、训练运行时`35 passed in 173.18s`、R08复现性`10 passed in 113.27s`；关键CLI/检查点复测`2 passed in 23.83s`，相关编译与`git diff --check`通过。代码提交`8c4864c`。本子阶段只接通FP32配置入口和元数据，不宣称AMP已启用；选择fp16/bf16会明确拒绝，下一子阶段实现一致精度路径。

- [x] 写测试：CUDA AMP可用时，采样与PPO重放用相同dtype/设备/autocast上下文并记录log-prob最大误差；CPU不允许伪报fp16/bf16训练，bf16按硬件能力检查。
- [x] 写测试：PPO统计/比率/时间辅助损失保持FP32，AMP前向与梯度有限；既有运行时测试覆盖FP32奖励、费用、GAE路径。
- [x] 写测试：Actor/Critic/时间头共享单一去重Lightning optimizer；FP16 GradScaler仅由Lightning精度插件持有，BF16无scaler。
- [x] 写测试：C profile缺时间头仍可严格加载Actor/Critic/图版本/配置元数据；D profile未训练时间头、跨profile、错误配置哈希及缺失/格式错误指纹均明确失败；D加载有符号时间头并校验冻结塑形预测器快照。
- [x] 写测试：检查点保存resolved YAML/哈希、代码提交、场景散列、图特征版本、C/D profile、Actor初始指纹、模型权重、Lightning优化器/精度状态和随机状态；不包含环境进度时标`non_exact`续训。
- [x] 使用`D:\Conda\envs\rag_env\python.exe -m pytest tests/work3/test_work3_training_runtime.py -k 'amp or checkpoint or profile' -q`及新增指纹反例运行红灯用例；红灯实际分别记录在子阶段7B/7C执行记录。
- [x] 用Lightning精度插件启用AMP；rollout前向显式使用`torch.amp.autocast`同一精度路径；统计/GAE等保持FP32。当前只记录本机RTX 4060 Laptop GPU上的通路测试，不对任意配置外推资源结论。
- [x] 实现profile严格载入并重跑新测试、`tests/work3/test_work3_f09_experiment_groups.py`、`tests/work3/test_work3_r08_training_reproducibility.py`和Lightning回归。
- [x] 更新任务表并提交；此时仍不启动完整批次试点。

**子阶段7B执行记录（2026-09-25）：** 先新增FP16/BF16采样重放、PPO FP32统计、Lightning scaler所有权、梯度更新计数和训练入口烟测。RED复现旧实现FP16在手动裁剪/有限值检查处中断，Lightning scaler尚未获得执行跳步和降scale的机会。修复后精度解析拒绝CPU AMP并核验BF16硬件支持；Actor图聚合显式匹配累加器dtype；rollout与Lightning按同一配置精度前向，PPO统计及时间监督损失维持FP32。梯度有限性检查和裁剪移入Lightning `on_before_optimizer_step`，由插件先反缩放；GradScaler仍仅由插件持有，日志区分step尝试、成功更新和AMP跳过。FP16/BF16插件测试`2 passed`，AMP采样/重放及D入口烟测`4 passed`，完整运行时`43 passed in 199.80s`，相关Actor/PPO/时间头/训练入口/势函数组合`24 passed, 1 skipped in 87.74s`；skip是缺正式检查点的既有条件项，未计为通过。`py_compile`和`git diff --check`通过。代码/测试提交`c315188`；未改用户的`training/lightning_module.py`，未启动完整批次试点。下一项为Task7C profile严格加载、检查点精度/续训状态与对应测试。

**子阶段7C执行记录（2026-09-25）：** 新增正式检查点契约测试后，旧评测器先被复现为接受未训练D时间头、跨C/D profile和错误resolved-config哈希；复核加载边界时又补出三种非法运行指纹反例，旧实现均未拒绝（`3 failed`）：Actor初始指纹含非十六进制字符、事件计划哈希缺失、worker事件哈希格式错误。实现后，训练检查点保存profile、Actor初始指纹、来源提交、完整场景/事件计划指纹、成功批次与评测资格；同时保存Lightning优化器状态、精度和GradScaler状态、随机数状态、更新计数、总环境步数及worker事件计划位置。继续明确标记`resume_capability=non_exact`，不声称可恢复环境进度。D的在线时间头状态仅在真实转站标签存在且辅助优化器成功更新后标为`trained_online`；AMP跳过不计入成功更新。正式加载校验C/D profile与配置哈希、源/初始/数据/事件指纹、训练状态及AMP精度一致性；C允许无时间头，D必须有兼容的有符号时间头和冻结塑形预测器快照，错误检查点只有显式debug模式可退化为随机策略。绿灯：`test_work3_f09_experiment_groups.py`为`13 passed in 35.38s`；完整训练运行时`44 passed in 201.21s`；相关训练/检查点回归组合`42 passed in 214.47s`；`py_compile`与`git diff --check`通过。代码/测试提交`901796f`，任务表与本计划随后单独提交。未修改用户的`training/lightning_module.py`，未启动完整批次训练或C/D性能实验；下一项为Task8小预算端到端运行门槛。

### 任务8：端到端小预算运行门槛与试点协议冻结

**依赖：** 任务1—7全部通过；不作为正式性能实验。

**文件：**

- 修改：`conf/work3/train_pilot.yaml`
- 修改：`scripts/work3/train_ppo_work3.py`及必要的工作三runtime组件
- 测试：`tests/work3/test_work3_training_runtime.py`
- 记录：`docs/工作三_代码一致性问题与修复任务表.md`

- [x] 写集成测试：同一固定tau=0训练事件、种子和有限交互预算下，C/D运行入口保留相同Actor初始指纹，并分别报告事件触发/命中、转站标签可用性、优化更新、step预算、截断/退出原因、独立可行性状态及原始费用分项。
- [x] 写C/D门禁测试：C/D Actor初始指纹相同；D端到端用例在真实转站后观察到正、负残差并成功更新时间辅助头，记录episode内冻结的塑形版本；截断周期未产生伪标签；Task7C测试拒绝缺少已训练兼容时间头的正式D检查点。
- [x] 写资源记录测试：CPU线程数、环境数、rollout/batch、AMP dtype、主机峰值内存、GPU峰值显存、墙钟和step结算请求数有字段；CPU仅报主机值、CUDA分别报主机和显存值，未用设备字段为`null`；smoke/pilot研究结果资格为false。
- [x] 以`rag_env`执行完整工作三测试集（包含全部runtime测试）；结果`304 passed, 3 skipped in 1576.95s`。3个检查点/轨迹条件skip不计通过；未运行全仓`pytest -q`，不声称全仓通过。
- [x] RTX 4060上执行明确受限的双worker BF16 smoke：2个聚合step均实际命中固定目标，完成1次Lightning优化更新，worker未初始化CUDA；严格检查点/方法profile规则由Task7C测试覆盖。此项只证明链路，不代表正式训练。
- [x] 运行`num_envs=2` FP32与RTX 4060 BF16双worker AMP集成验收；均保留截断/未完成状态，不据此作性能比较。FP16探索曾发生非有限梯度并被GradScaler跳过（成功更新0、跳过1、scale降至32768）；当前硬件通路测试用BF16，不据此冻结完整试点精度。
- [x] 所有门槛通过后，只冻结下一步完整批次试点的配置、训练集事件清单及哈希、种子、总聚合step/墙钟限制、`num_envs`、精度、batch/rollout和C/D初始权重指纹；提交冻结记录。此任务不自动授权正式测试集评测或方法优劣结论。

**Task8执行记录（2026-09-25）：** 资源遥测反例先证明CUDA和CPU训练报告都缺少分设备字段；新增字段保留旧`memory_peak_bytes`兼容行为，并将`host_memory_peak_bytes`明确限定为训练主进程峰值常驻内存。定向CUDA BF16及CPU C/D固定事件用例`2 passed`。新增D端到端有限预算测试由固定命中事件触发真实转站，检查缓存中的残差同时包含正负值、时间监督有成功优化更新、episode内势函数版本固定；不完整周期的真实转站标签继续缺失。该测试首轮因错误地断言所有周期都已完成而失败，修正为核验未完成周期不伪造标签后通过。代码提交`cd90e23`，D有符号残差集成测试提交`98b031e`。定向测试`1 passed, 47 deselected in 54.50s`；整套`tests/work3`（含该测试）`304 passed, 3 skipped in 1576.95s`；`py_compile`和`git diff --check`通过。试点预算/设备/场景清单及C/D真实初始权重指纹尚未冻结，未启动完整批次或正式比较；未推送。

**成功目标退出边界复核（2026-09-25）：** 对照设计稿“达到成功批次目标后停止分配新episode；已启动episode在预算允许时继续到终态”，发现旧入口在第一个成功episode后立即结束当前rollout wave，并将仍活动worker截断。新增测试先因缺少“活动episode数参与停止判定”的逻辑失败（`1 failed`）；修复后`tests/work3/test_work3_training_runtime.py`、`tests/work3/test_work3_r08_training_reproducibility.py`及`tests/work3/test_train_ppo.py`合计`64 passed in 642.15s`，相关`py_compile`与`git diff --check`通过。实现/测试提交`13ad3c9`。现在只有成功目标已达且活动episode数为0才正常停止；若步数或墙钟预算先耗尽，活动episode仍按预算截断语义记录。完整批次试点协议随后完成冻结，见下文。

**完整批次试点协议冻结（2026-09-25）：** 新增`conf/work3/pilot_trial_20260925.yaml`，仅冻结训练集可行性试点，不构成正式性能实验：C先于D；seed 42；2个spawn环境；RTX 4060 Laptop GPU、BF16；总交互上限12000个所有环境实际step、墙钟上限5400秒、成功批次目标1；每段64步、PPO epoch 1、batch 64；线程均为1，worker结算时限2秒。达到目标后不派发新wave，但同一轮已启动的轨迹会在预算允许时继续到终态。

训练输入仅用train拆分的25个场景；完整事件ID顺序和manifest位于冻结YAML，`episode_plan_sha256=b5f331765ecc7f9db8e706ed26c338ccd7998ecf7008521402527d153d27f724`，双worker固定映射`worker_event_plan_sha256=1d70bbe08cd7f12000b55edfb364e89c5cd433c14726b9556c0fe7d8816bbfc6`。场景池、拆分及基线SHA分别为`ecd2ff2fbca9f535195c1b2257896efc1571a9e9e39f9e946c04c856861ab10b`、`f43fd5145eb5cc97e04540e9145d7add6fdacdcb0ea0e3057e1777b0e97ec98f`、`d38f827460ec7142493da940bb48648d694234f414193e8d536ecf8e8d175a56`。C/D唯一配置覆盖差异为`runtime.method_profile`；resolved YAML SHA分别为`15b492dd265fef6ce5836503330fccf1bedc0164217fda6f103715ca1934878d`和`4fc7eff7b14523d0b7e970c469938c81f4e0e58b988ffc7ddf0682d935917418`。共同初始Actor指纹为`1b3525dbbf0d67ff9dce95f18f9812e64bd11bc4b715bb24d23da170f4877bb2`。D不加载来源不明的旧时间头，随机初始化后仅使用训练期间获得的真实转站标签在线监督；配置测试验证该预训练路径不存在。

配置文件SHA256=`8b7aff26f63d99b0e4d2b4fa0b7f3db28cf5ee09726a0a05c96612e476a77191`；首轮协议测试先以`3 failed`确认冻结配置缺失，配置固化后`tests/work3/test_work3_pilot_protocol.py`为`3 passed in 20.65s`，覆盖预算/硬件、D无预训练、原始文件/事件计划哈希、actor指纹及C/D除profile外配置一致。另用真实C训练入口执行1步CPU FP32 smoke，报告初始Actor指纹与冻结值一致、单次更新完成；该截断烟测不计为完整生产试点。实机预核验Python 3.11.15、PyTorch 2.12.0+cu132、Lightning 2.6.5、NumPy 2.4.6，RTX 4060硬件BF16可用。配置/测试提交`6b84d0f`；冻结记录另行提交。完整C/D试点尚未运行。

**C完整批次试点执行与诊断（2026-09-25）：** 按上述冻结YAML运行C，实际耗时4094.6秒、聚合交互12000步（两个worker各6000）、完成188次Lightning优化更新；因达到`decision_limit`，两个episode均截断，成功批次数`0/2`。完成工序分别为`2627/2830`和`2569/2830`，各有13次实际转站；独立可行性均为`incomplete_not_assessed`，因此本轮不能判为完整或可行。两条事件均触发但实际命中数为0；目标`3_27`实际完工时刻为745.3793和844.8804，固定扰动时刻为1463.7837，说明该固定晚时事件在C轨迹中到达过晚、目标已完成。账本独立核对12000步原始奖励和累计费用，绝对差`1.1e-14`；逐条费用分项与累计值一致。没有OOM或训练异常，采样—重放最大概率误差为0。由此只能确认运行链路稳定至冻结步数，不能确认C能完整完成批次或扰动命中；12000聚合步不足以让两条并行episode均完成。C报告与仅用于追溯、不可精确续训/评测的权重提交`b504d1c`，报告SHA256=`2d09612c6693fe5c13c7029bd3aa9109c50e2fffdb78c399bf4cd1d2de1a810a`。D及任何性能比较尚未开始；本次不事后扩大预算或替换事件，后续若修改协议须先另行冻结并记录。

**D完整批次试点执行与OOM定位（2026-09-25）：** D按相同冻结YAML和事件计划启动，在聚合step `6848`、迭代107处的PyG `GATv2Conv`前向触发CUDA OOM；错误信息为单次申请2 MiB失败，但当时仍报告约5.78 GiB device free、约1.08 GiB PyTorch allocated及38.44 MiB reserved-unallocated，故不能简单归因为GPU总容量用尽。主进程和两个spawn worker在未捕获异常后仍存活，人工中断后退出；D运行报告和权重检查点均未生成。本轮不推断未观测的D最终命中、完成数、耗时或总标签数。

源码追踪发现时间辅助标签批次持有整批CPU `HeteroData`快照，而共享图编码器调用`graph.to(device)`会经PyG对象`apply`原地替换其图内张量；每个已处理快照因此仍由批次列表引用并留在CUDA。真实实例图快照为14个张量、633196字节；在约1100个已标注快照时，图本体可累计约0.7 GB，尚未计入图网络中间激活。新增CUDA快照不可变反例首次`1 failed`（旧实现确实将输入图移至CUDA）；将特征和边索引分别映射到设备后，专项`3 passed in 15.78s`，图策略/PPO/训练入口组合`17 passed, 1 skipped in 141.20s`。另对12张完整快照做4轮重复编码，确认输入留在CPU且预热后的CUDA分配不随轮数继续增长。代码和测试提交`576f0da`。PyG设备迁移语义可见[官方HeteroData文档源码](https://pytorch-geometric.readthedocs.io/en/stable/_modules/torch_geometric/data/hetero_data.html)及当前环境PyG 2.7.0安装实现。

**当前阶段门槛及下一项：** C未完整完成，D因OOM中断；二者均未通过完整批次门槛，也不得进入配对费用比较。精确OOM上层调用帧未保留，快照累计迁移是源代码和小案例确认的缺陷，但须由后续固定协议D全程复跑检验。异常路径未统一调用`env.close()`是已确认的worker泄漏原因；下一项先写异常清理测试并实现统一回收，更新任务表、提交后再按原YAML重跑D，不改冻结预算或场景，也不把中断模型称作完整D检查点。

## 任务提交和记录规则

1. 开始每项前检查`git status --short --branch`；不覆盖或暂存与当前任务无关的工作区变更。
2. 新缺陷测试首次执行结果原样写入任务表；已有回归测试允许首次即通过。若新测试首次通过，先核实既有修复和前提，不人为制造红灯。
3. 只有当前任务的反例、定向回归和必要兼容测试通过，才提交该任务；可把测试与修复同一提交。任务表同步写入修改方式、命令、结果、提交号和未通过/未运行范围。
4. 一项未通过时留在该项定位，不开始下一项；不因局部通过声称完整方法或全仓通过。
5. 全部runtime门槛通过后，才可按训练前冻结协议开展受限完整批次可行性试点；预算或事件计划若需调整，必须先记录并冻结新版本。正式C/D效果比较与GitHub推送仍须满足总体测试和评测门槛。

## 计划自审

- 设计稿第4节快照、预算、事件暴露：任务2、4、8覆盖。
- 第5节GAE、标签跨段、塑形副本episode版本：任务5覆盖。
- 第6节Lightning唯一优化器和工作三模块：任务6覆盖。
- 第7节AMP精度边界、线程/资源：任务7、8覆盖。
- 第8节配置、种子、profile检查点：任务1、7覆盖。
- 第10节阶段顺序与试点门槛：任务3→4→5→6→6B→7→8严格覆盖。
- 非目标/用户工作区边界和逐任务提交：全局约束与提交规则覆盖。
