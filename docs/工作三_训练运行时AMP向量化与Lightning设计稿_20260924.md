# 工作三训练运行时：AMP、向量化环境与Lightning设计稿

**日期：** 2026-09-24
**状态：** 用户已完成设计审阅并批准按下述边界进入实施；本文档为当前设计基线。尚未修改训练代码，也未启动完整批次试点。
**适用范围：** 工作三训练/试点运行时，不改变工作三已确认的方法定义、奖励、动作分支或实验分组。

## 1. 目标与边界

用户确认完整批次训练试点前必须落实：

1. AMP；
2. 向量化环境并行采样；
3. YAML 配置；
4. PyTorch Lightning 训练生命周期。

保留一个事件驱动生产环境、现有条件分支自回归PPO及其事件/奖励语义。试点决策预算按**所有环境实际执行的动作步数之和**计数。训练参数、事件清单和保护上限仍须在开跑前另行冻结；本文中的 `num_envs=2` 仅是首轮运行通路测试建议，不是正式实验超参数。梯度累积只在显存实测证明必要时启用。

不修改用户工作区中未提交的 `training/lightning_module.py`，也不把它作为工作三新运行时的必须依赖。该文件当前接口服务于另一套 Agent 生命周期，工作三需有独立、可测试的 Lightning 集成边界。

## 2. 仓库现状与根因

- `scripts/work3/train_ppo_work3.py::run_training` 当前在单个进程、单个 `AirLineEnvWork3` 上承担场景调度、采样、时间标签缓存、塑形、PPO更新和报告，无法直接通过给旧循环套一层 `Trainer.fit()` 得到正确的向量化/Lightning生命周期。
- `ActorCriticWork3.select_action()` 直接读取环境对象、候选工序、分支掩码及工人信息。子进程环境不能把活的环境对象交给主进程策略；必须改为传递采样时刻的不可变CPU决策快照，并让环境仍在子进程内校验和执行动作。
- `PPOTrainerWork3` 当前同时计算损失、持有优化器并直接调用 `backward/step`。这会绕开Lightning的手动优化和混合精度管理；需将PPO损失/重放计算与Lightning优化器生命周期分开，禁止同一Actor参数被两个优化器重复管理。
- 仓库已有通用Lightning模块，但其 `training_step` 面向另一套 `agent.update()` 契约；本方案不修改该文件，也不复用其 Agent 训练模块。
- `requirements.txt` 已包含Lightning和Hydra依赖，运行环境已有OmegaConf；优先使用现有OmegaConf能力，不增加Ray、Stable-Baselines3或第二套配置框架。
- 新增Python代码沿用仓库强制约定：完整类型标注；核心张量/数据边界作形状与完整性校验；张量维度变换注释写清输入/输出形状；路径只用`pathlib`；训练入口不恢复成单体原生训练循环。

## 3. 方案比较与选择

| 方案 | 取舍 |
|---|---|
| 保留原训练大循环，只在外层调用Lightning | 不选。优化器、AMP和训练生命周期仍由脚本私自管理，Lightning会成为装饰层；向量环境也没有明确所有权。 |
| 每个环境子进程复制Actor/训练器 | 不选。复制模型和GPU上下文会增加显存、同步及策略版本不一致风险，也使PPO重放更难核验。 |
| **环境子进程，主进程唯一Actor/优化器，工作三专用Lightning数据与训练模块** | **采用。** CPU环境并行推进；主进程统一采样决策、构造PPO批次并更新模型，确保一个rollout段内策略版本一致。 |

## 4. 目标架构与数据流

```text
OmegaConf YAML ──> scripts/work3/train_ppo_work3.py（入口、种子、构造Trainer）
                              │
                 PyTorch Lightning Trainer（单训练进程/单Actor）
                    ├─ Work3DataModule：输出固定的rollout请求
                    ├─ Work3LightningModule：Actor/Critic/时间头、AMP、PPO更新、检查点
                    └─ Work3RolloutCoordinator：主进程汇总观测、调用唯一Actor、整理批次
                              │ spawn进程间命令/快照
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       环境工作进程0      环境工作进程1       环境工作进程N-1
       AirLineEnvWork3    AirLineEnvWork3     AirLineEnvWork3
       无Actor/无CUDA     无Actor/无CUDA      无Actor/无CUDA
```

每个环境工作进程常驻一个独立 `AirLineEnvWork3`，只负责基线实例、预先分配的场景、环境事件、`step()`、物理约束与原始奖励。Windows和Linux均显式采用 `spawn`；工作函数放在可导入模块顶层，入口受 `if __name__ == "__main__"` 保护。worker失败、超时或非法消息必须向主进程显式报错并清理进程，不能静默重启后继续同一轨迹。

环境每次准备好策略决策时，返回不可变 `DecisionSnapshot`：只含CPU可序列化数据，足以复原该时刻图特征、候选任务、留站/后移掩码、工人候选与团队补全约束、任务/工人节点索引映射、时间与周期标识。逐人选队必须基于快照动态生成每一步掩码：快照至少携带本站工人及稳定ID、技能/效率、日历可用性、任务技能需求、人数和团队合法性条件，使主进程能判断“加入当前工人后，剩余候选是否仍能补全合法团队”。每一步生成的掩码都存入PPO重放记录。快照不能包含活环境引用、CUDA张量或会在环境变化后被改写的对象。动作回传对应环境进程，由 `env.step()` 对完整团队再作最终权威校验。

快照验收必须对同一物理状态逐项比较单环境环境端规则与快照端规则：留站/后移分支掩码、每一工人指针位置的团队补全掩码及候选顺序均一致；以这些快照采样后，PPO重放概率与采样记录一致。

采样采用同步向量轮次：每轮收集所有仍活动环境的决策快照，在主进程由同一Actor产生动作，再并行发送给各环境并等待本轮结果。rollout段内不更新策略参数，避免异步worker使用不同策略版本造成PPO样本陈旧。聚合交互预算精确定义为所有worker实际完成的 `env.step()` 调用总数，包含 `ADVANCE_TO_NEXT_EVENT` 强制推进；Lightning优化步数和向量轮次数单独计数，不能代替交互步数。每轮派发前计算剩余额度，末轮只向确定顺序中额度允许的活动环境发送动作，绝不超发。

墙钟上限到达时停止发出新动作，并进入有界的在途请求结算期。收到完整响应的动作正常记入转移；若响应在配置的结算期限内仍未返回，丢弃该未完整动作转移并将worker/episode标为基础设施中断。只允许用最后一个完整收到、与最后一条已记录转移匹配的观测做bootstrap；没有该观测时不得合成后继状态或价值。报告预算触发时刻、结算耗时、已结算步数、丢弃请求数和中断原因。

训练场景按冻结清单预先映射到 `(worker_id, episode_index)`，不按哪个worker先完成来临时抽场景。C/D使用同一预生成映射、训练种子集合和聚合交互预算；由于策略完成速度可能不同，不假定两组在预算终止时实际经历了相同数量/序号的episode。每个方法分别报告计划映射、已启动/已执行事件、预算时在途事件、预设目标数、实际命中数和未命中原因。`successful_batch_target`仅用于完整批次试点的保护性停机，不用于正式C/D配对性能比较。正式配对评测必须对冻结测试清单中的每个事件分别运行C、D两组，保留双方失败/未命中结果；费用比较不得基于不同事件暴露或仅筛选成功轨迹。

达到 `successful_batch_target` 后停止分配新episode；其他已经开始的episode在剩余交互/墙钟预算允许时继续至终态，否则按上述在途结算规则明确记为截断或基础设施中断并保留记录。不得为了凑成功数而丢弃在途episode结果。

## 5. PPO、标签、塑形和终止语义

- 每个worker/episode独立保存转移和 `PendingTimeLabelCache`；标签键至少区分环境、episode、cycle。GAE段键为 `(worker_id, episode_id, segment_id)`；先各自结束/截断并计算bootstrap，再合并已完成样本用于一次策略更新，绝不跨worker、episode或段边界串接优势值。真实成功终止与不可恢复失败均不接到下一条reset轨迹；预算截断不伪造时间标签。
- 标签缓存跨多个rollout段保留到真实转站或轨迹失败/结束，保存CPU图快照、启发式值及采样时显式时间输入，不持有可变环境对象。对墙钟时限内未结算的动作，不生成时间/奖励标签，也不伪造后继价值。
- 环境的原始奖励及分项账本随对应 `step()` 返回。塑形只使用该次转移的原始奖励和同一冻结预测器版本的前后势函数，不在rollout协调器中重新估算或重复记费。
- 预测器副本按现有确认规则在完整生产轨迹边界刷新，但版本以episode为单位引用：episode启动时绑定不可变预测器版本，记录其版本号；episode完成/失败后释放引用。环境A完成触发新预测器版本时，仅改变随后启动的episode版本，不得替换环境B仍在使用的副本。旧副本保留至最后一个引用它的episode结束。C不启用学习时间预测时，记录启发式版本并不创建虚假时间头。
- PPO采样时的图快照、候选集合、动作分支掩码、工人掩码和已修正时间输入随样本保存；重放只使用该快照，不从当前环境重建旧状态。

## 6. Lightning与优化器所有权

新增工作三专用 `Work3DataModule`、`Work3LightningModule` 和rollout协调组件，不触碰旧 `training/lightning_module.py`。DataModule提供按冻结事件计划构造的rollout请求，使用 `num_workers=0`；CPU并行由独立环境worker池完成，避免DataLoader再复制环境池。LightningModule持有注册后的Actor/Critic、时间头及所需归一化状态，训练入口使用 `Trainer.fit(model, datamodule=...)`。

工作三PPO具有多轮mini-batch重放和辅助时间监督，采用Lightning手动优化。`configure_optimizers()`是Actor、Critic与时间头共享参数的唯一优化器所有者；共享图编码器参数去重后只注册一次。损失模块只计算策略、价值、熵与时间监督损失；由Lightning训练步骤取得其优化器，调用 `manual_backward`、Lightning优化器步进和梯度裁剪。不得在 `PPOTrainerWork3` 内另行创建优化器或直接 `.backward()`。

## 7. AMP、精度边界与资源限制

- CUDA训练的策略/价值/时间头前向和PPO重放使用混合精度；rollout策略推理路径显式使用 `torch.amp.autocast`，训练更新使用Lightning精度插件与同一配置精度，不创建第二个GradScaler。支持的dtype在启动时检查；设备不支持所选精度时明确失败，不静默退化成“AMP训练”。首轮本机通路测试优先用RTX 4060可用的BF16；若另一硬件需FP16，梯度缩放由Lightning精度插件统一管理。
- 奖励账本、时间/资源比较、GAE、优势、目标价值、重要性采样比率及报告统计保持FP32；环境模拟保持Python数值语义，不受autocast影响。
- worker不创建CUDA上下文；CPU快照及回放图数据留在主机内存，按mini-batch搬运到GPU。默认DataLoader worker数为0；`num_envs`、worker线程数、rollout步数和PPO batch size均由YAML配置并记录，报告主机峰值内存和GPU峰值显存。
- 梯度累积不是默认启用项。只有实际峰值显存或有效batch需求证明必要时，才添加独立任务、配置与测试；不得将它作为未经验证的性能优化一并引入。
- DataModule、worker和环境数由配置显式限额；主训练进程与环境worker的PyTorch CPU线程数也需显式记录，避免每个子进程各自展开默认线程池。

## 8. 配置、随机性与检查点

新增工作三专用YAML，例如 `conf/work3/train_pilot.yaml`，用OmegaConf读取并校验必填项。配置至少覆盖运行模式、C/D方法开关、种子、事件清单与散列、环境数、每段聚合步数、总交互/墙钟上限、设备、AMP dtype、PPO参数、路径及日志/检查点目录。路径统一通过 `pathlib.Path`解析。配置展开后保存resolved YAML和SHA256，不把命令行覆盖隐藏在日志之外。

入口在模型初始化和创建worker前统一锁定Python、NumPy、PyTorch/CUDA及环境随机源，记录确定性算法与worker派生种子。场景清单按worker/episode固定映射。C/D的 `initial_actor_fingerprint` 必须继续相同；Actor与时间头完整初始化指纹分别报告。

评测检查点采用方法profile条件校验：C严格加载Actor/Critic/图编码器、归一化状态、图特征版本、C profile、resolved config及散列、源码提交和数据版本；C关闭学习时间预测时，不得因缺少时间头被拒绝。D除上述共享组件外，还必须严格加载已训练且模型/图特征版本匹配的在线时间头和所需预测器快照；缺失、随机初始化或版本不匹配均须拒绝，不能静默替代。C、D均记录完整配置、图版本和`initial_actor_fingerprint`。Lightning训练检查点另保存优化器、学习率状态（若配置）、精度插件状态/GradScaler（FP16时）、Trainer步数、随机状态及事件计划位置。worker当前环境状态若未实现并验证可恢复，检查点必须标为非精确续训，不能宣称位级/轨迹级断点续训；这不影响完整模型评测加载。

## 9. 文件边界

预计修改或新增的最小范围：

- 新增 `conf/work3/train_pilot.yaml`；
- 新增工作三向量环境worker/协调模块及不可变决策快照类型；
- 修改 `models/work3/actor_critic.py`，让采样与重放可以直接消费快照；
- 修改 `models/work3/ppo_trainer.py`，拆出无优化器所有权的PPO损失/重放计算；
- 新增工作三LightningModule/DataModule运行时模块；
- 精简 `scripts/work3/train_ppo_work3.py` 为配置加载、种子设置、组件构造与Trainer启动/报告入口；
- 新增聚焦配置、AMP、spawn向量环境、PPO重放、Lightning检查点及集成smoke的 `tests/work3/test_work3_training_runtime.py`（如测试范围增长，再按职责拆分文件）。

文件归属可在实现计划中按现有目录结构细化；不得扩大到其他工作、旧训练入口迁移或用户未提交文件。

## 10. 验收门槛与顺序

严格按以下顺序实现和验收，不得跳步：

1. **单环境FP32语义对照：** 先以`num_envs=1`、FP32和固定动作序列运行旧入口与新快照/协调路径，逐步比较事件序列、原始奖励及分项、转站记录、终止/截断状态和最终可行性；差异未解释前不进入多环境。
2. **双环境FP32：** 再用`spawn`启动两个worker，核验跨平台进程关闭、固定`(worker_id, episode_index)`场景分配、聚合步数和最后一轮不超发、墙钟在途请求结算；同一状态逐分支比较环境端与快照端掩码，特别覆盖每一步逐人团队补全掩码；比较PPO采样/重放概率，并验证GAE严格按`(worker, episode, segment)`隔离。另验证A完成不改变B绑定的塑形版本。
3. **AMP与Lightning：** 前两阶段通过后，接入Lightning优化生命周期与AMP。采样与重放使用完全相同的配置精度路径核对动作对数概率容差；检查FP32奖励/GAE/重要性比率、AMP前向/辅助损失、梯度和关键日志均有限；验证唯一优化器确实更新共享图参数、FP16时只有Lightning一个GradScaler所有者。
4. **配置与检查点：** 对配置解析/覆盖/散列、worker种子、版本信息做自动测试；按C/D profile分别验证严格加载规则（特别是C缺少时间头可加载、D缺少训练时间头必须失败）；缺环境进度的恢复明确标非精确。
5. **小预算闭环与资源记录：** 固定明确的`num_envs`、batch size、rollout和AMP dtype，在RTX 4060 8GB记录主机/GPU峰值；小预算smoke须完成至少一次Lightning优化更新，逐worker记录成功/失败/截断、实际命中、事件暴露、奖励分项和独立可行性。之后才冻结完整批次试点协议。Lightning optimizer steps另行报告，不替代聚合环境交互数；小预算smoke不替代完整轨迹或性能结论。

实施严格按“先写反例/测试、运行记录红灯、再改实现、定向验证、单任务提交并回填任务表”执行。每次只完成一个有可运行结果的任务，先不开始完整批次或正式C/D性能比较。

## 11. 非目标与未决参数

- 不改变F01–F12及W3-14既有方法/环境修复，不重定义奖励或C/D算法。
- 不做Ray、DDP、多Actor副本、Lightning迁移到其他工作、自动worker重启、AMP性能基准比赛或任意硬件显存承诺。
- 不默认梯度累积；不改变前述已记录的事件清单、种子和试点预算。最终训练设备、AMP dtype、`num_envs`、有效batch与完整批次协议须在通路测试后、首次完整运行前冻结并提交。

## 12. 技术依据

- PyTorch AMP：<https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- PyTorch multiprocessing与CUDA进程启动：<https://docs.pytorch.org/docs/stable/multiprocessing.html>
- Lightning手动优化：<https://lightning.ai/docs/pytorch/stable/common/optimization.html>
- OmegaConf配置解析与展开：<https://omegaconf.readthedocs.io/en/latest/usage.html>
