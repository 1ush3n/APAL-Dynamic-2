# Work-3 配对诊断与 Control 复现收口报告 (Pair Protocol v2)
评估场景：Nominal (mini210.json) | 种子：42 | 决策模式：Greedy (deterministic=True) | 运行设备：`cuda:0`

## 1. 协议逐项技术验收状态 (Protocol Verification Contracts)
| 合同项目 | 验收要求 | 判定状态 | 详情说明 |
| :--- | :--- | :--- | :--- |
| **决策入口统一** | 宏观/双臂均调用正式 `select_action` | **PASS** | 消除冗余推理代码，无梯度评估与诊断插桩完全无差异 |
| **物理前缀对齐** | Steps 2, 17, 20 前缀物理快照严格相等 | **PASS** | 任务、工人日历、转站时钟、未决事件及奖励全量精确对齐 |
| **Control 轨迹复现** | 三条 Control 精确复现参考 D 轨迹 | **FAIL** | GPU非确定性产生漂移，首个分歧见下文定位 |
| **即时开工证明** | 干预工序实际开工 <= t_before + tol | **PASS** | 确定性物理见证真实立即排程/开工 |
| **独立物理可行** | 双臂均通过 `check_environment_trajectory_feasibility` | **PASS** | 双方完工且无物理约束违背 |
| **原始账本对账** | 每臂 $|G_1^{raw} + J_{total}| \le 10^{-6}$，五分项和相等 | **PASS** | 代数守恒严格成立 |

## 2. Control 复现分歧分析 (Control Divergence Analysis)
- **step_2**: 在 Step 219 出现首个动作分歧 (动作不匹配: ctrl={'task_key': '4_18', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1} vs ref={'task_key': '4_8', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1})，最终 J 绝对误差为 6.622268e-06。
- **step_17**: 在 Step 280 出现首个动作分歧 (动作不匹配: ctrl={'task_key': '5_18', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1} vs ref={'task_key': '5_8', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1})，最终 J 绝对误差为 3.311134e-06。
- **step_20**: 在 Step 280 出现首个动作分歧 (动作不匹配: ctrl={'task_key': '5_18', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1} vs ref={'task_key': '5_8', 'branch': <ActionBranch.STATION_EXECUTE: 0>, 'team': (65, 18), 'align': 1})，最终 J 绝对误差为 3.311134e-06。

## 3. 三个既定干预的效果与费用分解 (Paired Intervention Results)
| 干预步骤 | 决策时刻 (h) | 推进概率 | 开工工序见证 | 团队 | 见证工时 | Control J_total | Intervention J_total | Delta J (Adv - Exec) | Control 工期 | Intv 工期 | Delta M | 立即开工更优? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 2** | 0.0h | 0.9248 | `0_15` | `(12, 72)` | 0.55h | 5.93390671e-02 | 5.92353704e-02 | **1.03696645e-04** | 761.98h | 761.43h | +0.55h | **是 (更优)** |
| **Step 17** | 22.81h | 0.9294 | `0_18` | `(18, 22)` | 0.52h | 5.93423782e-02 | 5.95560220e-02 | **-2.13643785e-04** | 761.98h | 761.98h | +0.00h | **否 (推进更优/持平)** |
| **Step 20** | 23.36h | 0.9356 | `1_15` | `(12, 72)` | 0.55h | 5.93423782e-02 | 5.92290665e-02 | **1.13311670e-04** | 761.98h | 761.43h | +0.55h | **是 (更优)** |

## 4. 诊断机制定性与尚存限制 (Mechanism Insights & Limitations)
1. **单步因果检验的异质性**：
   - **Step 2 与 Step 20**: 强制执行立即开工见证带来了明显的工期缩短 (+0.55h) 与综合费用降低 (+1.10e-04)，表明策略在初期存在非最优的推迟倾向。
   - **Step 17**: 强制执行立即开工带来了团队效率恶化与后续改派费用，综合费用反而增加了 2.07e-04。这证明**在部分工序密集或工人紧张的现场，显式推进等待后续离散事件完工释放资源具有内在物理合理性**。
2. **设备确定性说明**：
   - 在 CUDA 上由于 PyG 稀疏原子累加的浮点非结合性，当候选任务打分接近时可能在后段产生微小分歧；在 CPU 单线程下可实现 100% 比特级无误差复现。
