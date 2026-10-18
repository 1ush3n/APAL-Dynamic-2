# Work-3：RTX 4060 8GB 本地试训与问题排查方案

适用基准：APAL-Dynamic-2，分支 work3-next-round-p2-1-p1-3-p2-2，提交 2d9cda24013b159ddc99fb4a7ef2afd659320309。日期：2026-10-18。

## 1. 推荐直接采用的方案

**先跑 70 工序烟测，再以 210 工序作为主要调试实例，最后选做 566 工序的完整工艺桥接。默认使用 FP32、单环境、短 rollout、小批量。**

这次目标是暴露真实 CUDA、PPO、时间监督、跨段缓存和物理执行问题，不要求训练收敛，也不据此比较 C/D 的研究效果。无需先跑 2,830 工序综合训练，或完整 Section 9.3——现有综合脚本仍绑定大实例。

| 数据 | 构成 | 保留的诊断能力 | 用途 |
|---|---|---|---|
| mini70 | 五站、七架、每架 10 工序，共 70 | 填线/满线/排空、同步转站、站内前序链、双人团队 | 快速检查 GPU 和训练入口 |
| mini210 | 五站、七架、每架 30 工序，共 210 | 五工种、1/2 人需求、完整前序闭包、多架次与当前/下一周期学习 | **主要试训实例** |
| real566 | 五站、两架、每架完整 283 工序，共 566 | 保留完整工艺 DAG、五工种及 1/2/3/5 人需求 | 可选的真实工艺桥接 |

mini70 只覆盖工种 0，不足以代替五工种验证；mini210 是主要验收实例。real566 只有两架，不能代替七架实例的满线验证。

210 工序约为原 2,830 工序的 7.4%，这是任务节点规模比例，不能直接当成显存比例。**尚未在你的 RTX 4060 上实测峰值，不能保证所有配置绝不 OOM。**工程目标是先将 PyTorch 分配峰值控制在约 6 GiB 内，并用 nvidia-smi 同时观察设备总占用，再决定是否放大。

## 2. 已准备好的文件

将压缩包里的 work3_4060 文件夹解压到仓库根目录，与 scripts、models、conf 同级。保持文件夹名称不变。

- data：三个实例、各自的扰动事件和 split、指纹清单。
- configs：8 份完整配置，可直接被当前训练入口读取。
- preflight.py：检查实际 Python、Torch/PyG/Lightning、CUDA 小型反传和全部配置；记录依赖与 Git 状态。
- verify_local_data.py：在你的真实依赖环境中检查七条小实例物理轨迹。
- run_local.py：通过现有训练 CLI 启动各阶段，保存控制台、命令、运行 JSON 和权重。
- inspect_report.py：检查实际报告的基本训练证据，不为缺失记录伪造 PASS。
- prepare_local_data.py：可选的可复现数据重建脚本，默认不需要运行。
- verification：本次隔离物理验证记录与验证边界说明。

此包不修改训练算法、网络宽度、工序时长、技能人数、工人池或五站规则。三个实例均从当前基准 JSON 抽取，并检查所选任务的完整前序闭包；real566 只减少飞机数。没有随机截取前 N 行，也没有重编号或删除仍然需要的前序边。

mini210 每架工序为 2–30 和 125。选取 125 是为了覆盖工种 3，并同时保留它的全部前序；五站每架分别含 8、5、5、4、8 道任务。

各阶段独立初始化模型。它们不是接续训练；现有检查点仍为 model_weights/non_exact，不能把分阶段运行说成精确断点恢复。

## 3. 小实例专用扰动

默认 pilot 使用各实例自己的 LOCAL_*_INITIAL_HIT：tau=0、恢复时刻 20.0、目标 0_16。该任务有前序 0_15，在首次决策前仍未开工；本次三个实例均确认实际命中。它便于先稳定检查事件入图、时间预测、奖励与缓存，不依赖策略偶然走到某个时刻。

另外提供 middle-d：仍用 mini210，在固定 tau=130.20533674370247 对中间站目标 2_13 注入扰动，恢复时刻 226.7626745432108。参数根据固定启发式无扰动轨迹离线标定：该轨迹第五周期从 129.7053 开始，目标原定 131.3066 开工；受扰参照轨迹确实命中并延至恢复时刻，且全线合法完工。

**middle-d 的学习策略可能与参照轨迹不同。必须检查实际命中数和发生时的周期/占位，不能仅凭场景名称宣称“满线中间站命中”。**未命中先查实际原因，按覆盖不足记录；不要在训练中根据策略状态偷偷改 tau 或换目标。

这些事件是诊断夹具，不是论文正式的训练/测试场景划分。不要沿用大实例的 scenarios_9class.json 与原 split，否则可能有目标不存在、扰动时刻过晚、始终未命中等情况。

## 4. 先做环境检查

建议沿用你已经运行通过综合验证的依赖组合，先记录版本，不为本轮试训盲目升级 Torch、CUDA 或 PyG。以下命令均在仓库根目录运行，Windows PowerShell 和常规终端都可使用：

~~~text
git branch --show-current
git rev-parse HEAD
python work3_4060/preflight.py
python work3_4060/verify_local_data.py
~~~

preflight 应显示 CUDA 可用、实际设备是目标显卡、小型 FP32 反传 PASS、全部配置通过。若这里失败，先处理环境或依赖问题，再进入策略训练。数据验证应显示 mini70/mini210 的 11 次转站和 real566 的 6 次转站，全部独立可行，受扰参照轨迹实际命中。

包内新增文件和生成结果会让 git status 出现未跟踪内容，这是正常记录。保留真实 Git 状态与配置指纹；不要把它伪写成完全干净的源码工作区。

## 5. 分阶段执行

所有命令顺序运行，**一次只开一个训练进程**。run_local.py 会自动创建带阶段、种子和时间戳的独立输出目录，不覆盖上次结果。每个目录含 console.log、launch.json、run.json、weights.pt。

| 阶段 | 数据 | rollout 步数 | PPO 批量 / 辅助批量 | epoch | 决策预算 / 完成批次目标 | 墙钟预算 |
|---|---|---:|---|---:|---|---|
| smoke-c | 70 | 8 | 8 / 2 | 1 | 8 / 不设 | 300 秒 |
| smoke-d | 70 | 8 | 8 / 2 | 1 | 8 / 不设 | 300 秒 |
| pilot-c | 210 | 8 | 2 / 2 | 1 | 1024 / 1 | 900 秒 |
| pilot-d | 210 | 8 | 2 / 2 | 1 | 1024 / 1 | 900 秒 |
| middle-d | 210 | 8 | 2 / 2 | 1 | 1024 / 1 | 900 秒 |
| stability-d | 210 | 8 | 2 / 2 | 1 | 4096 / 5 | 1800 秒 |
| bridge-c / bridge-d | 566 | 8 | 1 / 1 | 1 | 2048 / 1 | 1800 秒 |

所有配置：单环境、spawn、CPU 环境 worker、主策略 cuda:0、FP32、模型 hidden_dim=64（沿用当前入口）、DataLoader worker=0、无 warmup。两个学习批量分别控制 PPO 和时间监督，降低 PPO 批量不会自动降低辅助批量。

表中秒数是有界运行预算，不是预计耗时；正在执行的环境步或优化步可能稍超预算。达到完成目标会提前结束。预算先到而目标未达，不等于已经证明算法错误，但不能记为完整验收通过。

### 第一步：只验证真实 CUDA 训练能否启动

~~~text
python work3_4060/run_local.py smoke-c
python work3_4060/inspect_report.py --latest smoke-c
python work3_4060/run_local.py smoke-d
python work3_4060/inspect_report.py --latest smoke-d
~~~

smoke 的 PPO 批量为 8，是为了满足当前入口“覆盖本轮所有采样”的合同。不要直接把 smoke 批量改为 2。8 步内 D 可能尚无可用真实标签，这一阶段允许跳过辅助更新，但不能据此宣称 D 的时间学习已验收。

若只需隔离 CPU/GPU 差异，可另跑 smoke-c --device cpu；CPU 成功不能替代 CUDA 验证。

### 第二步：主要训练检查

~~~text
python work3_4060/run_local.py pilot-c
python work3_4060/inspect_report.py --latest pilot-c
python work3_4060/run_local.py pilot-d
python work3_4060/inspect_report.py --latest pilot-d
python work3_4060/run_local.py middle-d
python work3_4060/inspect_report.py --latest middle-d
~~~

先 C 后 D 能帮助区分基础 PPO/图策略问题与时间预测、辅助更新、冻结预测器相关问题。middle-d 再检查真正位于 episode 中途的扰动处理。两组有相同种子和外生事件，但策略轨迹可能不同，本轮不将费用差解释为方法优劣。

### 第三步：短稳定性检查

前面通过后运行：

~~~text
python work3_4060/run_local.py stability-d
python work3_4060/inspect_report.py --latest stability-d
~~~

目标是看多个 episode/rollout 后是否出现缓存累计、显存持续增长、NaN、重放偏差或 worker 中断。若发现疑点，再单独用 --seed 43、--seed 44 复现；不必先安排长时间多种子实验。

### 第四步：可选的完整工艺桥接

~~~text
python work3_4060/run_local.py bridge-c
python work3_4060/inspect_report.py --latest bridge-c
python work3_4060/run_local.py bridge-d
python work3_4060/inspect_report.py --latest bridge-d
~~~

这一级保留完整 283 工序模板，只缩减为两架，并使用批量 1。通过前不要扩大到 10 架；这一级本身仍不是正式研究实验。

## 6. 怎么判断“真实训练出了问题”

inspect_report 的退出码：0 为已记录的基本检查通过，2 为失败，3 为覆盖不足。0 对应 BASIC_RECORDED_CHECKS_PASS，不是全部设计合同已验收。

| 检查 | 预期 | 异常时优先定位 |
|---|---|---|
| GPU 使用 | report.device 为 cuda:0，gpu_memory_peak_bytes 有效 | 当前 Python 是否为 CUDA 版 Torch；是否启动了错误环境 |
| PPO 实际更新 | optimizer steps >0，history 中 ppo_updates >0 | 是否预算太短、buffer 没有结算、数据进入损失前失败 |
| 数值 | 损失和诊断值有限，无 NaN/Inf | 第一处出现非有限值的状态、奖励、时间目标、log_prob |
| 概率重放 | 每个有 PPO 更新的 rollout：误差存在、样本数 >0；FP32 暂用 <1e-4 | 图快照、动作掩码、团队指针、时间输入或参数更新顺序 |
| D 辅助更新 | pilot 中 time_supervision_optimizer_updates >0，状态 trained_online | 是否实际转站、是否产生有效 mask、标签是否跨段保留 |
| 跨段 | 至少两个 fit；同一 episode 有至少两次真实转站 | rollout 是否过长、事件推进是否停住、预算是否过早截断 |
| 完整生产 | successful_batch_count >=1，导出完整批次的 independent_feasibility 为 feasible/true | 若仅 incomplete_not_assessed，是未完成，不是可行性 PASS |
| 扰动 | actual_disturbance_hit_count >0 | 目标是否仍未开工、事件是否到达、是否用错大实例清单 |
| worker 隔离 | environment_worker_cuda_initialized 全为 false | 子进程是否错误初始化 CUDA、是否复制模型进 worker |
| 基础设施 | 无 worker_infrastructure_interrupted | spawn、文件路径、CPU/RAM、线程、实际异常与超时日志 |

PPO loss、value loss、entropy、KL 或训练奖励不要求逐轮单调下降。小实例上的单轮起伏、C/D 费用接近，不能单独证明有 bug；也不能用“损失下降”替代完工、可行性和重放验证。

当前报告的 sampling_replay_scope 是 first_pre_update_minibatch_per_rollout，不覆盖更新后所有 mini-batch。现有聚合报告也没有充分记录 next-only 的 decision 身份、其实际更新次数，以及实际绑定冻结 Actor/head 在跨更新前后的完整校验和。**本包会明确保留这些验收缺口，不用总标签数或两个 fit 自动推断 E04 全闭环。**若需关闭该项，只补只读审计记录，沿用现有训练算法。

## 7. 显存观察与 OOM 处理

另开一个终端持续观察：

~~~text
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1
~~~

主要对照训练报告的 gpu_memory_peak_bytes 和设备总占用。前者是本次主训练进程 PyTorch 的 max_memory_allocated；后者还包含缓存分配、CUDA 上下文及其他程序，二者不应机械相等。PyTorch 官方分别提供 allocated/reserved 指标；empty_cache 只释放未使用的缓存，不会释放仍被张量引用的显存。[1]

优先按下列顺序处理，保留失败日志后用新进程重试：

1. 确认没有另一份训练进程或其他大型 GPU 程序同时占用。
2. **PPO 和辅助批量同时降为 1**：

~~~text
python work3_4060/run_local.py pilot-d --ppo-batch 1 --aux-batch 1
~~~

3. 若仍在采样/缓存阶段增长，将 rollout 从 8 降为 4；保留跨多个 rollout 的实际事件覆盖：

~~~text
python work3_4060/run_local.py pilot-d --ppo-batch 1 --aux-batch 1 --steps 4
~~~

4. 若连 70 工序 smoke 都 OOM，将 smoke 同时降至 4 步和批量 4：

~~~text
python work3_4060/run_local.py smoke-d --steps 4 --ppo-batch 4
~~~

5. 桥接阶段 OOM 时回到 210 工序查明原因；不要同时改图结构、工人数量、工序时长和精度。小实例也反复 OOM，优先检查是否意外加载了 2,830 工序基准、是否持有计算图、GPU 快照未释放，或辅助批量没有实际改小。

总环境步数主要控制运行长度；单次反向峰值更直接受图大小和 mini-batch 影响。rollout 长度和待结算标签还会影响 CPU 快照/缓存占用。**短一点的数据集文件或较少训练文件，不一定降低一次前向的图大小**，本包实际缩小了单实例任务图。

不要每步调用 empty_cache 当作解决方案。nvidia-smi 上 reserved 占用趋于稳定不等于泄漏；应在同等图规模/相同采样阶段比较。若要判断持续增长，可在调试日志中只读记录每个 rollout 更新前后的 memory_allocated、memory_reserved、缓存待结算数，同时观察主机 RAM。当前报告只含整体峰值，不能单靠它判断增长趋势。

FP32 先通过后，才可在 preflight 与当前仓库都确认 BF16 支持时单独测试 --amp bf16。AMP 的算子会使用混合精度，不等价于把整个模型和输入强制改为半精度。[2] 本轮不建议先用 FP16，因为之前的跨 rollout scaler 连续性事项尚未完成。

## 8. 发回哪些结果最有用

第一轮先完成 preflight、数据验证、两个 smoke，再跑 pilot-c、pilot-d、middle-d。暂时不需要一口气跑完所有阶段。

发回每个实际运行目录的：

- console.log、launch.json、run.json；
- preflight.json 与 pip-freeze.txt；
- fixture_validation.json；
- OOM/异常的完整 traceback，以及该时刻 nvidia-smi 记录。

正常情况下无需先上传权重。没有生成 run.json 时，优先发 console.log 和 launch.json；异常退出不应被误写成成功报告。运行目录自动分开，保留失败现场便于复现。

## 9. 本次准备和验证边界

已完成：三个数据实例的构建、原始物理 DAG 前序闭包核对、七条固定启发式无扰动/受扰轨迹、独立可行性、费用账本和命中检查；8 份配置的 YAML/原校验逻辑检查、实际 CLI 参数解析，以及报告检查器的正常/缺失重放/NaN/预算不足负例。

物理结果：70/210 工序实例各 11 次转站，566 工序实例 6 次转站；全部完工并通过独立可行性。初始事件将目标 0_16 的参照开工时刻从约 3.2169 推迟至 20.0；中间站参照事件使 2_13 从约 131.3066 延至 226.7627。

这些是隔离环境与控制流验证：本地缺少 Torch/PyG/Lightning，数据加载末尾的张量包装由 NumPy 替代；配置检查使用原验证函数的控制流，不替代你本机实际 OmegaConf/Torch 运行。**尚未运行真实 GPU 神经训练，也未测得你的 4060 显存或耗时。**请以本地 preflight、smoke 和 pilot 的实际结果为准。

参考：

[1] [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)：allocated/reserved、峰值与缓存区别。

[2] [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)：autocast 与混合精度原则。仅作原理参考，不要求升级为网页所示版本。

源码依据：[固定审查提交](https://github.com/1ush3n/APAL-Dynamic-2/commit/2d9cda24013b159ddc99fb4a7ef2afd659320309)。

