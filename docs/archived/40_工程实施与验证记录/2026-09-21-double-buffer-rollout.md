# APAL 环境双缓冲流水线实施计划

> 执行说明：按任务顺序实施，每项先建立失败用例，再修改生产代码并验证；不得跳过正确性门槛直接开展长训练。本次仅交付计划，未实施训练代码修改。

**目标：** 在现有 Lightning、HB-GAT-PN、PPO 和多进程 APAL 环境基础上，使一组环境的 CPU 步进与另一组的 GPU 推断重叠，保持单环境时序及当前 PPO 更新边界。

**架构：** 复用一个 VectorEnv 环境池、一个 PPOAgent 和一个 GPU 推断执行线程，将环境索引固定分为 A/B 两组。拆分现有融合步进接口为提交与接收，采用固定 A→B 调度；只在本轮采样完全收尾后交给 Lightning 更新。

**技术栈：** Python multiprocessing/Pipe、PyTorch AMP、PyG、既有 LightningModule/DataModule、YAML 配置。无需新增 Ray、线程池、模型副本或通用任务调度框架。

**需求依据：** 用户本次提出的双缓冲、严格单环境顺序、无陈旧策略要求；本文包含对应设计约束和实施步骤。

## 约束与承诺边界

- 不能在未实现、未运行验证前承诺“零 bug”。将风险落实为不变量、故障注入、回归测试与默认关闭的开关。
- CPU 耗时完全被掩盖、GPU 利用率达到 70% 都是待测目标，不是架构自然保证的结果。
- 语义目标是同一行为策略版本下的 On-Policy 采样；不承诺与原来整批推断逐位相同、同一种子逐动作相同。
- 严格保持 APAL 的工位、工人、资源等待、死锁处罚、脉动与重调度语义，不替换环境动力学。
- 所有 Python 命令在 rag_env 内执行。已读取本机包元数据：Python 3.11.15、torch 2.12.0+cu132、PyG 2.7.0、Lightning 2.6.5、pytest 9.0.3；这不构成 GPU 运行验证。不能由用户所报 CUDA Toolkit 13.1 推断实际 PyTorch 运行时。
- Windows 使用现有 spawn 方案；Linux 复用已配置且验证过的启动方式，至少补测 spawn。路径使用 pathlib，注释使用中文，张量变形写明形状。
- 保留现有 torch.amp.autocast 封装、梯度累积、全局种子入口、Lightning 手动优化接口和 DataLoader(num_workers=0)。
- 8GB 显存、16GB 内存是上线约束；先验收 4 环境的 2+2，再测 8 环境的 4+4。不能只因 CPU 核数多就默认增加环境数。

## 已核对的代码与设计依据

| 位置 | 已有行为 | 对改造的影响 |
|---|---|---|
| `utils/vector_env.py:981` | `step_rollout_indices()` 先发送，再阻塞收齐融合结果 | 拆分此接口，复用 worker 的 `step_rollout` 指令 |
| `utils/vector_env.py:821` | `_recv_workers_unordered()` 使用 connection.wait 收齐指定环境 | 保留按环境索引归位，不按到达顺序写轨迹 |
| `utils/vector_env.py:400` | EnvProxy 自身也直接收发 Pipe | 仅在 VectorEnv 的异步接口加锁不够，必须覆盖代理入口 |
| `training/rollout_service.py:213` | `_collect_episode()` 整批推断后等待所有活跃环境 | 将调度拆成固定两组，保留状态处理逻辑 |
| `training/rollout_service.py:126` | `_append_action()` 在步进前写动作等字段 | 新流水线应先暂存动作记录，收到有效反馈再完整写入 |
| `training/rollout_service.py:633` | `collect()` 按更新选数据集，收集若干回合后合并 | 两组共享原更新边界与数据集，禁止跨更新预取 |
| `training/lightning_module.py:108,210` | 主进程采样，随后 `agent.update()` | 复用现有生命周期；增加边界检查，不引入后台采样 |
| `training/worker_pointer_v2_behavior.py` | 按原行为组同形重放，避免 bf16 批形状漂移 | 每次 A/B 推断产生独立行为组，不能重新合成原 8 环境组 |
| `ppo_agent.py:2621` | `select_actions_batch()` 覆写多个 `last_*` 字段 | 下一组前向前必须捕获本组所有行为元数据 |
| `ppo_agent.py:1866` | GAE 遇 terminal 或 truncated 清零递推，当前截断也使用零末值 | 本次保持该规则；改为非终止截断 bootstrap 属于独立算法变更 |
| `runtime/seed.py` | 确定性算法采用 warn_only=True | 严格验收时改为报错模式运行测试；不能把警告模式称为严格保证 |

共享调用者还包括 literature PPO/DQN、基准脚本及安全测试。同步接口必须保留原签名与返回结构，不能只覆盖 Lightning 主入口。

## 调度设计

固定划分：N≥2 时 A 为前 ceil(N/2) 个全局环境索引，B 为剩余索引。支持奇数环境；组内结束后不补入新回合、不动态迁移成员。N=1 且显式启用双缓冲时在初始化前报错。

```text
启动：  推断 A₀ → 提交 A₀ 的 CPU 步进
重叠：  推断 B₀ → 提交 B₀ 的 CPU 步进       此时 A₀ 可在 CPU 运行
稳态：  收齐 A₀ → 写入 A₀ → 推断 A₁ → 提交 A₁   与 B₀ 重叠
        收齐 B₀ → 写入 B₀ → 推断 B₁ → 提交 B₁   与 A₁ 重叠
收尾：  停止提交 → 收齐两组已提交结果 → 标记轨迹边界 → 合并 → PPO 更新
```

这里只需要进程间异步提交。默认 CUDA stream 已足以让 GPU 推断与另一组 CPU worker 重叠；首版不引入双 CUDA stream 或异步 D2H。Python 动作发往 worker 前必须已可安全读取。CUDA 异步执行与同步规则依据 [PyTorch 2.12 CUDA 文档](https://docs.pytorch.org/docs/2.12/notes/cuda.html)。

每组每轮最多推进一次原有决策轮。组内保留原顺序：读取当前 masks → 对全掩码环境调用资源等待 → 必要时刷新 snapshot/图 → 推断 → 提交真实动作。每组独立计算轮数上限，不能把 A/B 两次调度计为原来的两轮，也不能按最快组的样本数提前终止另一组。

一组全部完成后，另一组正常运行至自己的边界，尾部可能退化为串行。空组不调用模型；不得因为一个组没有合法动作就结束整个 rollout。

## 正确性不变量

1. 同一环境最多一个未接收的命令；上一动作反馈未收齐前不能读取新状态、再次 step、reset 或切数据集。
2. 每条转换由 `(update_index, episode, env_index, local_step)` 唯一识别；同一环境 local_step 单调递增，提交一次、接收一次、写入一次。
3. 状态、mask、动作、旧 logprob、旧 value、行为 trace 同属于决策前状态；reward、done 属于该动作的返回。严禁保存动作后的 snapshot 作为动作前状态。
4. `collect()` 整段执行期间不能优化模型、切换 EMA 或 ScheduleFree 的评估权重。应在第一组推断前进入既有训练权重模式，再建立审计基准，避免把初次 optimizer.train() 的切换误判为违规。
5. 两组推断在主进程顺序执行；同一 GPU builder 可以复用，但持久记录不得别名引用其可变工作区。
6. 每组推断返回后立即捕获 `last_gated_team_traces`、`last_anchor_proposal_traces`、`last_v2_behavior_logprobs`、`last_v2_behavior_values` 和诊断结果；随后才能调用另一组推断。
7. 仅按真实 `select_actions_batch()` 调用建立行为组。继续使用 `(episode, rollout_call_index)`，A/B 共享递增计数；保留全局 env_index 和有效成员的 group_position。沿用既有无效动作过滤规则并补测。
8. 写入成功后 Memory 的所有逐动作字段等长；合并继续按回合→环境→本地步顺序，禁止按 Pipe 返回顺序交错合并 GAE 序列。
9. 所有在途命令清空、所有转换写入、末端标记完成后才能返回 RolloutUpdate、评估、保存更新边界检查点或开始下一版本采样。
10. worker 异常、超时、非法动作或部分提交失败时使整个环境池失效，丢弃本次未提交给训练器的 rollout；禁止将部分轨迹当作成功样本交给 PPO，禁止盲目重试非幂等 step。

在固定策略参数 θₖ、环境转移相互独立、推断不使用会跨组改变的状态统计量的条件下，每个环境仍按 `aₜ ~ πθₖ(·|sₜ)` 再执行 `sₜ₊₁ ~ P(·|sₜ,aₜ)`。跨环境改变执行顺序不需要陈旧策略修正；行为概率必须仍是实际采样概率。PPO 的基本更新边界参考 [OpenAI PPO 文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

实现前测试完整模型中是否存在跨样本统计、训练态随机层或可变缓冲；若存在导致同一行为概率无法重算的情况，阻止该模式启用并先修复，不擅自切 eval() 改变现有策略。拆批数值差异与跨设备复现限制分别见 [数值精度](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html) 和 [可复现性](https://docs.pytorch.org/docs/2.12/notes/randomness.html)。

## 任务一：建立回归基线和配置门禁

**文件：** 修改 `configs.py`、`conf/rollout/fastpath.yaml`、`runtime/configuration.py`；扩展 `tests/test_runtime_config_guards.py`、`tests/test_config_loader.py`。

- [ ] 在 rag_env 记录当前目标测试结果、实际 CUDA 可用性、设备型号及可用内存。已有失败单独归档，不将其当成新功能通过。
- [ ] 新增一个配置 `rollout_double_buffer: bool = False`，YAML 放在 rollout 节点；两个固定组由 num_envs 推导，不另加可矛盾的组大小参数。
- [ ] 先写配置测试：旧配置缺失字段时为 False；开启且 num_envs<2 报错；开启且 enable_rollout_ipc_fusion=False 报错；4/5/8 环境可配置；Hydra 加载与保存清单均包含最终值。
- [ ] 在 `validate_runtime_config()` 增加门禁，同时在服务构造处调用同一个检查，覆盖直接构造服务的脚本。Fast-Exact 维持现有硬件要求；不添加 CPU 生产 fallback。
- [ ] 验证关闭开关时仍走原整批同步路径；不默认把现有 4 环境改成 8 环境。

计划配置示例：

```yaml
rollout:
  rollout_double_buffer: false
  enable_rollout_ipc_fusion: true
```

**交付门槛：** 配置边界测试通过，旧配置可加载；本任务不宣称产生吞吐收益。

## 任务二：将融合步进拆成安全的提交和接收

**文件：** 修改 `utils/vector_env.py`；扩展 `tests/test_vector_env_safety.py`。

**新增接口契约：**

```python
# 以下为计划中的类型与接口契约，并非本次已实现代码。
from dataclasses import dataclass
from typing import Any
import torch

RolloutMasks = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
RolloutStepResult = tuple[RolloutMasks, dict[str, Any], float, bool, dict[str, Any]]

@dataclass(frozen=True)
class RolloutStepTicket:
    request_id: int
    indices: tuple[int, ...]
    deadline: float
```

在 VectorEnv 上新增 `step_rollout_async(actions: dict[int, Any]) -> RolloutStepTicket` 和 `step_rollout_wait(ticket: RolloutStepTicket) -> dict[int, RolloutStepResult]`。ticket 必须由创建它的池登记并按对象身份确认；不能只凭编号接收别的池的请求。同步 `step_rollout_indices(actions)` 组合调用两者。

- [ ] 先测试：两组互不重叠的请求能同时在途；重复环境、越界索引、重复 wait、伪造/跨池 ticket 均在发送前或接收前明确报错；空请求返回空结果且不得阻塞。
- [ ] 验证整批 actions 后再发送第一条，避免参数错误时已推进部分环境。一次提交中途 Pipe 失败则关闭池并抛错，不允许继续复用。
- [ ] 每个 EnvProxy 持有本环境在途状态，VectorEnv 与该代理使用同一状态来源。审计所有 `.send/.recv` 入口，保证代理的 reset/get_masks/get_snapshot/wait/switch 等也不能越过在途保护；本地只读属性不能被误当新状态使用。
- [ ] wait 复用既有融合结果解析及动态属性刷新。完成后释放对应环境占用；不同组结果始终通过全局索引归位。
- [ ] deadline 使用 monotonic，在提交时确定，传到接收层；不得每次 wait 重置超时。到达 wait 时先检查已就绪结果，避免仅因主进程在推断、回包尚未读取就误报超时；仍未就绪且期限到达才失败。
- [ ] worker 死亡、EOF、错误状态、超时均关闭池。关闭流程不能向已有在途数据且可能堵塞的 Pipe 无界发送 close；对在途 worker 走有界回收/终止，释放连接，重复 close 无副作用。
- [ ] 同步旧接口的返回结构、异常类型和融合 snapshot 更新行为保持兼容。

**可运行测试设计：** 用 multiprocessing.Event 控制测试 worker：A 收到动作后发出“已开始”信号并等待释放；主进程在未释放 A 时提交 B，确认 B 已开始，再依次释放并收齐。用事件和测试总超时验证并行性，不用“sleep 后希望已经完成”的测试。大响应包另测 close，覆盖 worker 阻塞于 send 的情况。

**依据：** Pipe 的并发读写和 connection.wait 约束见 [Python 3.11 multiprocessing](https://docs.python.org/3.11/library/multiprocessing.html)。单一主线程管理每条连接，不引入多接收者。

## 任务三：接入 A/B 调度与完整转换写入

**文件：** 修改 `training/rollout_service.py`；新增 `tests/test_double_buffer_rollout.py`。

- [ ] 先用假环境和假 Agent 测试调度：每个环境状态带 `(env_index, local_step)`，动作必须引用相同标识；任意错配立即失败。
- [ ] 从现有 `_collect_episode()` 提取本地职责：准备指定索引的决策状态、推断并暂存动作记录、接收并写入反馈。仅提取同步与双缓冲共同使用的代码，不复制一整套等待/奖励/死锁逻辑。
- [ ] 暂存记录包含全局 env_index、本地步号、动作前 snapshot/masks、动作、logprob/value、全部 trace 和分量值。变动数组须有稳定所有权；静态只读模板可共享，不深拷贝整个环境。
- [ ] 每组推断结束立即捕获元数据、构造行为组，然后提交 step；收到有效返回后调用现有 `_append_action()` 并连续追加 reward/terminal/truncated，避免一组 Memory 只有动作而没有反馈。
- [ ] 原资源等待成功时使用刷新后的 snapshot/masks；失败时保留原死锁处罚且仅处罚一次。模型无效动作不向 worker 发送 None 伪装真实 transition；环境拒绝已允许动作时沿用硬失败。
- [ ] 保留每组相同的 max_steps 轮数预算，终止环境不再发送命令。一组全终止时只停止该组；所有停止路径均收齐另一组已经提交的动作。
- [ ] 保留重调度奖励各分量、完成率、makespan、诊断等输出。SPS 的分母是完整 rollout 墙钟，分子为真实成功执行的环境动作数量。
- [ ] 对外继续返回现有 `list[Memory], RolloutMetrics`，collect 与更新批量大小、episodes_per_update、数据集选择保持原语义。

**必须覆盖：** 2/4/5/8 环境、乱序返回、A 先结束/B 先结束、空有效动作组、资源等待后可执行、持续全掩码、首步死锁、已有奖励后的死锁、步数截断、部分提交失败、异常收尾。

测试对每个环境验证字段数相同、步号连续、恰好一次写入、旧状态与新状态相邻；最终验证 `sum(len(m.actions) for m in memories) == environment_steps`。

## 任务四：验证 On-Policy、行为组与更新屏障

**文件：** 修改 `training/lightning_module.py`、`training/rollout_service.py`；扩展 `tests/test_lightning_architecture.py`、`tests/test_worker_pointer_v2_behavior_group.py`、`tests/test_worker_pointer_v2_fast_exact_rollout.py` 和新流水线测试。

- [ ] 在返回采样结果前提供 `assert_rollout_idle() -> None`，检查无在途请求、无未提交记录；按现有 RolloutService 协议接入更新入口检查，并更新协议测试替身。禁止对非空状态静默忽略。
- [ ] 构造更新 spy：当任一组仍 pending 时更新不可发生；两组收尾后更新恰好一次；update_every_episodes>1 时所有回合均先收齐。
- [ ] 审计测试在进入训练权重模式后保存参数及模型 buffer 副本，采样结束比较；生产热路径不逐步复制权重。优化器 OOM 跳过更新沿用既有事务语义，下一轮不残留旧 pending。
- [ ] A/B 分别产生不同的 group_id；有效成员数量、env_index 和顺序与原调用一致。按原组首次 PPO 重算 logprob，检查比率接近 1；沿用项目既有精度门槛，不因双缓冲放宽。
- [ ] 测试不同组及组内缩小批次不覆盖先前 trace，尤其是 fast-exact 模板重用、bf16、条件 value 和 anchor/gated trace。
- [ ] 测试完整 Memory 合并后的 GAE 不跨环境/回合边界；本次保持现有 truncated 零末值行为，单列其算法限制，不混入 bootstrap 修正。
- [ ] 完成一次真实小规模 PPO 更新，验证 loss/梯度有限、有效动作合法、所有已启用 head 的梯度合同未退化。

**两级等价性判据：**

1. 同样的 A/B 分组、A→B 前向顺序、相同参数和随机状态下，对比“提交后立即等待”的测试参考调度与流水线调度，逐转换比较；改变 worker 完成顺序不能改变采样决策顺序。测试参考模式只在测试内，不新增生产配置。
2. 与原 N 环境整批路径比较 APAL 转移、边界、行为概率重算、完整更新合同及多种子质量，不要求逐位一致。用固定动作回放隔离环境语义，避免把采样分歧误判为 Pipe 错配。

严格确定性检查使用 `torch.use_deterministic_algorithms(True)`；若实际 PyG 算子无确定性实现，明确报告阻塞算子，不静默降级。跨 Windows/Linux 分别验证功能与容差，不承诺跨平台逐位一致。

## 任务五：内存、时间线与吞吐验收

**文件：** 扩展 `scripts/benchmark_rollout_fastpath.py`、`training/rollout_service.py`；仅在验证证实需要时修改 `training/v2_fast_exact_batch.py` 的缓存生命周期。

- [ ] 基准增加与 YAML 同语义的双缓冲开关，记录最终配置、软件/设备、种子、环境数、图规模和策略模式。保留既有参数；分别支持计时与诊断运行。
- [ ] 每种模式在独立进程中加载相同初始权重，预热 2 个 rollout、测量 5 个完整 rollout，种子 42/43/44；对比顺序交替，避免温度和缓存偏差。
- [ ] 比较 4 同步/2+2 双缓冲、8 同步/4+4 双缓冲；单次性能结果按实际完成 transitions/s 和每次更新总耗时报告。8 环境不满足内存门槛时停止扩容，保留 4 环境结果。
- [ ] 用完整窗口的 CPU 墙钟统计总耗时，GPU 事件用于诊断设备执行时间；计时边界确保设备工作完成。关闭现有 detailed_profile 的逐阶段 synchronize 后再测性能；诊断和吞吐结果分开报告。
- [ ] CPU worker 记录本地 step 起止区间，主进程记录推断、发包和收包区间，结合 GPU 时间线证明重叠。IPC 等待耗时、CPU step 实际耗时、主进程前向墙钟和 GPU kernel 忙碌时间分别命名，不能将重叠分项直接相加当总时长。
- [ ] 记录 GPU allocated/reserved 峰值、系统可用 RAM、所有相关进程内存及 pending 峰值。禁止 GPU 保存整段计算图；复用 AMP/no_grad 和 CPU snapshot 存储，持久标量在 CPU 保存。每环境最多一份在途转换，禁止无界队列与每步 empty_cache。
- [ ] worker_threads 初始保持 1，DataLoader worker 保持 0；不同时增加环境数和进程内线程数。异步评估若与训练争用 GPU，基准固定关闭或隔离并明确记录，随后另测实际生产配置。
- [ ] 预设资源门槛：GPU 总占用保留至少约 1 GiB 余量、系统可用 RAM 保留至少约 2 GiB；发生换页增长、OOM 或占用持续上升时停止扩容。连续至少 30 个小规模更新检查资源趋势和子进程回收。

**性能模型，仅用于判断何时可能收益：**

设 F 是一组前向、必要的数据传输及动作准备耗时，E 是一组提交后到结果就绪的 CPU/通信耗时，H 是接收、记录及调度开销；均用相同硬件实测，F 不是单纯 GPU kernel 时间。

对于等长、稳定的两组，忽略 H 和启动/收尾时，两组前向至少需要 2F，同一组下一次前向又必须等待自身的 F+E，因此一个 A/B 循环的理想下界为 `max(2F, F+E)`。固定交替且无额外竞争时可接近这个下界。

`T_流水线 ≈ max(2F, F+E) + H`，而相同分组的串行参考为 `2(F+E)`。若 E≤F，环境步进可能在另一组前向期间完成；若 E>F，仍有等待。实际原基线是整批 N 环境，耗时约 `F_N+E_N`，必须与它直接测量比较；不能拿拆组串行参考替代原基线宣称加速。拆批前向效率下降、大结果包、图重建与短回合均可能使新方案变慢。

**上线门槛：** 所有正确性测试通过后，以目标工作负载的中位 SPS 至少提高 10%、完整更新耗时无明显退化、资源门槛通过作为建议启用标准。10% 是预设工程门槛而非预测；不达标就保持关闭并报告瓶颈。GPU 忙碌率 70% 只作观察目标，不能牺牲正确性或吞吐去追数字。

## 任务六：跨平台回归、启用与回退

**文件：** 复用上述测试和基准；启用时仅修改目标实验 YAML，不全局切换默认值。

- [ ] Windows rag_env 跑下面目标测试；Linux 在其 rag_env 内跑同一集合，记录系统、驱动、包版本及启动方式。Linux 环境未经检查前不假定虚拟环境绝对路径。
- [ ] 回归共享接口的 literature PPO/DQN、worker-pointer v2、fast-exact、重调度和 OOM 事务测试；默认关闭时不得改变原行为。
- [ ] 先运行 2+2 的短训练，再运行目标图规模的完整回合和小规模真实更新，最后在内存允许时验收 4+4。
- [ ] 保存运行清单中的开关、分组规则、实际分组和版本；检查点仍只在既有更新边界保存，不序列化在途 Pipe。恢复时重建环境池，从既有支持的更新边界继续。
- [ ] 回退方式为关闭 rollout_double_buffer 并从最后有效更新边界重启。发生在途异常时不得在同一已损坏环境池内“自动切回同步”。

Windows 执行示例（仓库根目录；其中新测试需在实施后才存在）：

```powershell
& 'D:/Conda/envs/rag_env/python.exe' -m pytest tests/test_config_loader.py tests/test_runtime_config_guards.py tests/test_vector_env_safety.py tests/test_double_buffer_rollout.py tests/test_lightning_architecture.py tests/test_literature_vector_semantics.py tests/test_worker_pointer_v2_behavior_group.py tests/test_worker_pointer_v2_fast_exact_rollout.py tests/test_ppo_loop_fixes.py tests/test_ppo_oom_recovery.py -q
```

Linux 在 rag_env 激活并核验 sys.executable 后，使用同一 `python -m pytest` 测试列表。服务器连接和远端执行属于实施阶段，本次计划没有连接服务器。

## 最终验收清单

- [ ] 请求不串读、不重复、无单环境并发 step；所有代理入口也受约束。
- [ ] 超时、断管、部分发送、在途 close 和大响应包均有有界退出测试。
- [ ] A/B 的状态、mask、动作、reward、value、logprob、trace 一一对应。
- [ ] 资源等待、死锁、组提前结束、截断、奇数环境均通过。
- [ ] 所有组无在途任务后才更新；同一 collect 行为权重及相关 buffer 没有非预期改变。
- [ ] 同形重放、首次概率比率、GAE 边界和真实 PPO 更新通过。
- [ ] 默认关闭路径、共享调用者、Windows/Linux 回归通过。
- [ ] 实测重叠、SPS、更新耗时与内存趋势达标后才启用目标实验配置。

本次状态：已完成代码路径检查、包元数据读取和官方文档核验；未实施功能、未运行训练回归或性能测试，不将上述待执行验收项标为通过。
