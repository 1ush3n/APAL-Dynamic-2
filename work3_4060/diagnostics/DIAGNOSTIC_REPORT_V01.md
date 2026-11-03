# 工作三诊断验证复核综合报告：工序后移、跨周期影响与网络设计优先级判定

**日期**: 2026-11-03  
**共同审查基准提交**: `f201314fe6bc53f86b8727c8e142b75a771af52b`  
**审查分支**: `work3-next-round-p2-1-p1-3-p2-2`  
**运行环境**: Python 3.11.15 (`D:\Conda\envs\rag_env\python.exe`), PyTorch 2.12.0+cu132, CUDA 13.2, GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB), OS: Windows 11  
**代码与资产状态**: 生产核心代码零修改；所有诊断测试均通过独立脚本执行，输出完全归档于 `work3_4060/diagnostics/`。

---

## 一、审查基准、运行环境与口径修正

### 1.1 代码基准与运行环境
- 本轮验证严格基于代码基准 `f201314`。未改动任何生产网络结构、PPO 算法、环境物理转移规则、时钟前进模式或目标权重配置。
- 所有实验、反例与见证代码均采用独立沙箱脚本执行，保证生产代码纯洁与历史证据可复查。

### 1.2 历史报告口径核准与事实纠偏
1. **C-12 节点属性核实（非物理工序）**:
   - 查证原始工艺清单 `data/3182.csv#L1632`：`1631,C-12,1,,-1,C-11,0,0,2,`。
   - 字段明确定义：类型为 1（里程碑虚拟汇合节点）、工种为 -1、需求人数为 0、工时为 0.0h、限定站位为 2。**C-12 是纯虚拟节点，工时与负荷为零，绝不能算作第 5 道固定物理工序**。
   - 全量 3182 物理工序中，工艺硬编码限定物理站位的工序**严格仅有 4 道**（`data/3182.csv#L4-L7` 中的物理工序 2、3、4、5，限定在站位 1）。
2. **大实例“350/1393/1638”来源纠偏**:
   - 查证参考排程文件 `data/r3/b/r/real_3182_schedule.csv`，其各站工序计数为：Station 0（117 个里程碑虚拟节点）、Station 1（338）、Station 2（350）、Station 3（424）、Station 4（526）、Station 5（末站 1544）。
   - 前四站物理工序总和为 $338 + 350 + 424 + 526 = 1638$ 道，该排程总工期为 1393.7285h。**此前报告中的数字来自该特定参考排程的解，并非原始工艺 CSV 规定的硬性站位分布，绝不能据此认定原始工艺天然负荷均衡**。
3. **基准版本严格隔离**:
   - **旧基准 (compare_v1)**: 对应 `work3_4060/data/mini210.json`，包含 20 个离线场景 + 1 个无扰对照。
   - **新规则基准 (rule_baseline_ac_v1)**: 对应 `work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json`，节拍标准 $H_0 \approx 52.037844\text{h}$，转站模式 `nominal_anchored_v1`，每机 30 道工序，7 架批次共 210 道物理工序。
4. **Baseline 参照与后移兼容控制器**:
   - 源码 `scripts/work3/baseline_repair_agent.py#L64-L65` 包含断言 `if task.current_station != task.base_station: return task`，一旦工序后移即触发 `BaselineRepairInfeasibleError`。原 Baseline A 是“固定基准参照策略”，物理设计上不支持接管后移工序。
   - 本轮诊断独立设计并验证了规范命名的后移兼容控制器 `PostponeCompatibleRepairAgent`（`work3_4060/diagnostics/postpone_compatible_agent.py`）：在零后移的 Nominal 场景下与原 Baseline A 产生**完全逐拍一致**的派工动作与账本；在后移发生时，严格在目标站合法工人池中派工，严禁使用源站工人或抹除开工可用性限制。

---

## 二、任务 1 检验：“17 道候选”真实动态可达性全流程验证

### 2.1 四类数量重算与物理约束定位
在正规新规则基准 `mini210_rule_baseline.json` 中，单机 30 道工序的四类数量核算如下：

| 数量类别 | 单机数量 | 7架总数 | 覆盖工序 ID (单机) | 物理约束与生成机理核实 |
| :--- | :---: | :---: | :--- | :--- |
| **1. 静态资格** | **17** | **119** | `[6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 25, 26, 28, 29]` | 排除末站 8 道 (11, 14, 19, 22, 23, 24, 30, 125)、限定站位 1 的 4 道 (2, 3, 4, 5)、及下游工位无技能 4 工人 1 道 (27)。 |
| **2. 初始合法掩码** | **2** | **2** | 全局仅 `0_7` 与 `0_16` | 环境 reset 时仅 0 号飞机在站位 0。工序 6 阻于 7，工序 15 阻于 16；下游站位飞机未进站掩码全为 `False`。 |
| **3. 动态解锁** | **5 (拓扑) + 10 (进站)** | **35 + 70** | 拓扑解锁: `[6, 15, 8, 12, 20]`<br>进站解锁: `[9, 18, 25, 26, 13, 28, 29, 21, 10, 17]` | 当紧后工序 (7, 16, 9, 13, 21) 率先执行后移后，前驱工序的 `same_station_successors` 清空，`can_postpone` 动态解锁。其余工序在飞机进驻对应工位时解锁。 |
| **4. 实际后移** | **17** | **119** | 单轨可支持全 17 道共同后移 | 单轨迹共同后移 10 道 (5对) 或 17 道均能成功完工 (210/210 完工，0 违规，账本完全平衡闭环)。 |

### 2.2 五对核心工序合法动作见证（仅通过正式 `env.step`）
严格按照生产环境 `AirLineEnvWork3.step` 执行合法动作，验证“**后继先后移 $\rightarrow$ 前驱仍未决 $\rightarrow$ 前驱后移掩码瞬间解锁 $\rightarrow$ 前驱后移**”完整动作流，并续跑至批次完工（数据见 `work3_4060/diagnostics/summary_tables/task1_five_pairs_witnesses.csv` 与 `work3_4060/diagnostics/task1_outputs/task1_reachability_report.json`）：

```text
[严格动作见证协议]:
1. 观察前驱 u 的 can_postpone == False (原因: 工序仍有本站未完成后继 [v]，禁止后移);
2. 观察后继 v 的 can_postpone == True;
3. env.step({"task_key": v, "branch": ActionBranch.POSTPONE}) -> v 移至下游工位, status = POSTPONED;
4. 立即核验: 前驱 u 的 can_postpone 瞬间解锁为 True;
5. env.step({"task_key": u, "branch": ActionBranch.POSTPONE}) -> u 移至下游工位, status = POSTPONED;
6. 由 PostponeCompatibleRepairAgent 接管后续派工，目标站重新合法组队并推进至批次完工。
```

五对工序实际见证复核明细表：
| 验证工序对 ($u \rightarrow v$) | 所属站位 | 完工步数 | $v$ 先后移 | $u$ 解锁后移 | 批次完工率 | 工期 Makespan | 综合费用 $J_{\text{total}}$ | 后移费 $J_{\text{post}}$ | 节拍费 $J_{\text{takt}}$ | 改派费 $D_{\text{team}}$ | 偏差费 $J_{\text{rev}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6 $\rightarrow$ 7** | 工位 1 (St 0) | 212 | **True** | **True** | **210/210** | 572.416285h | 0.200953 | 0.2000 | 2.731e-15 | 4.762e-04 | 4.766e-04 |
| **15 $\rightarrow$ 16** | 工位 1 (St 0) | 212 | **True** | **True** | **210/210** | 572.416285h | 0.200954 | 0.2000 | 2.731e-15 | 4.762e-04 | 4.774e-04 |
| **8 $\rightarrow$ 9** | 工位 2 (St 1) | 212 | **True** | **True** | **210/210** | 572.416285h | 0.222907 | 0.2000 | **2.178e-02** | 4.762e-04 | 5.649e-04 |
| **12 $\rightarrow$ 13** | 工位 3 (St 2) | 212 | **True** | **True** | **210/210** | 572.416285h | 0.201339 | 0.2000 | 2.731e-15 | 4.762e-04 | 6.691e-04 |
| **20 $\rightarrow$ 21** | 工位 4 (St 3) | 212 | **True** | **True** | **210/210** | 572.416285h | 0.201320 | 0.2000 | 2.731e-15 | 4.762e-04 | 6.599e-04 |

> 重要物理发现: 工序对 **8 $\rightarrow$ 9** 后移至工位 3（Station 2）后，增加了该站的瞬时加工负荷，导致周期 2 转站推迟了 1.133h，产生了 $J_{\text{takt}} = 0.021777$ 的节拍超期罚款。这证实：**工序后移即使不拖延全线总工期，也会由于将负荷推给下游而击穿中间节拍**。

### 2.3 单任务可达 vs 轨迹共同后移容量
数据源自 `work3_4060/diagnostics/summary_tables/task1_joint_capacity_summary.csv`：
- **5 对工序 (10 道) 共同后移**: 在同一条轨迹中，依次触发 5 对工序后移。**10 道工序全部成功后移并在下游工位完工**，批次 210/210 完工，Makespan = 572.416285h，$J_{\text{postpone}} = 1.0000$，$J_{\text{total}} = 1.005380$。
- **全 17 道工序共同后移极限压力测试**: 0 号飞机的 17 道候选工序全部触发后移。**17 道工序全部成功在下游站位排产加工**，批次 210/210 完工，Makespan = 572.416285h，$J_{\text{postpone}} = 1.7000$，$J_{\text{takt}} = 0.156278$（下游负荷堆叠造成转站延迟），$J_{\text{total}} = 1.865889$。
- **结论**: 17 道工序不仅独立可达，共同后移在拓扑和资源容量上也具备可行性，限制后移的根本原因是**站内紧前紧后拓扑制约**与**下游工位技能匹配性**。

### 2.4 轨迹全量独立物理与财务验收（严防“完工即通过”漏洞）
前任检查脚本（`run_task1_investigation.py#L175-L176`）仅核验了 `completed_tasks == len(env.state.tasks)`，缺少对物理守恒与财务闭环的独立核算。
本轮开发独立验证套件 `diag_verify_all_trajectories.py`，从环境执行现场抽取全部微观事实（`tasks_execution`, `aircraft_execution`, `cycle_execution_records`, `step_rewards`），送入项目权威独立验证器 [`verify_single_trajectory`](file:///d:/OneDrive/%E7%A0%94%E7%A9%B6%E7%94%9F/APAL-Dynamic-2/work3_4060/verify_trajectory_facts.py#L93-L101)：
- **13 条轨迹 100% 独立验证 PASS**：
  1. 工人互斥（Worker Exclusivity）：0 违规；
  2. 工位并发上限（$\le 3$）：0 违规；
  3. DAG 拓扑偏序（Precedence）：0 违规；
  4. 站位驻留区间（Station Dwell）：0 违规；
  5. 账本绝对平衡：各项重算费用与环境累积值绝对误差在 $0.00 \sim 6.66 \times 10^{-16}$ 之间，步奖励求和严格等于 $-J_{\text{total}}$。

---

## 三、任务 2 检验：复核扰动传播与追回

### 3.1 旧规则基准 (compare_v1) 全部 20 场景复核
在旧基准 `mini210.json` 下，由 Baseline A 运行全部 20 场景（数据见 `work3_4060/diagnostics/summary_tables/task2_compare_v1_20_scenarios_review.csv` 与 `work3_4060/diagnostics/task2_outputs/task2_compare_v1_evaluation.json`）：
- **目标命中率**: 20/20 场景命中且推迟了开工下界（100%）。
- **分类统计重算**:
  - **局部吸收 (local_absorbed)**: **15 个场景** ($\Delta\text{Makespan} = 0.00\text{h}$, $\Delta P_{\max} = 0.00\text{h}$)。
  - **持续日历滞后 (persistent_delay)**: **5 个场景** ($\Delta\text{Makespan} > 0.00\text{h}$, 且 $\Delta P_{\max} = \Delta M$)：
    1. `COMPARE_TRAIN_S0_EARLY_L` (`0_16`): $\Delta P_{\max} = 5.2823\text{h}, \Delta M = +5.2823\text{h}$
    2. `COMPARE_TRAIN_S4_MID_M` (`2_125`): $\Delta P_{\max} = 25.0000\text{h}, \Delta M = +25.0000\text{h}$
    3. `COMPARE_VAL_S0_EARLY_M` (`1_16`): $\Delta P_{\max} = 10.2823\text{h}, \Delta M = +10.2823\text{h}$
    4. `COMPARE_VAL_S4_LATE_H` (`3_125`): $\Delta P_{\max} = 35.0000\text{h}, \Delta M = +35.0000\text{h}$
    5. `COMPARE_TEST_S4_MID_L` (`4_125`): $\Delta P_{\max} = 15.0000\text{h}, \Delta M = +15.0000\text{h}$
  - **后续扩大 (amplification)**: 0 个场景。
  - **后续追回 (recovered)**: 0 个场景。
- **关于 $R - \tau$ 与 delta 差异核实**:
  在旧 `compare_v1` 场景中，`recovery_time - tau` 恰好比 `delta` 字段多出 2.0h（例如 $\tau=10.0, R=27.0, \text{delta}=15.0$）。经查证代码 `prepare_compare_scenarios.py#L147-L149`，这是旧场景生成时显式加入的固定缓冲 `duration_offset = 2.0`。

### 3.2 新规则基准 (rule_baseline_ac_v1) 场景复核与强扰动反例核验
在新规则基准 `mini210_rule_baseline.json`（$H_0 \approx 52.04\text{h}$）上运行 6 个专属场景（数据见 `work3_4060/diagnostics/summary_tables/task2_rule_baseline_scenarios_review.csv` 与 `work3_4060/diagnostics/task2_outputs/task2_rule_baseline_evaluation.json`）：
- **局部吸收 (local_absorbed)**: **4 个场景** (`RULE_TRAIN_S1_RHO10`, `RULE_TRAIN_S2_RHO25`, `RULE_TRAIN_S4_RHO75`, `RULE_VAL_S1_MODERATE`)，$\Delta M = 0.00\text{h}, \Delta P_{\max} = 0.00\text{h}, J_{\text{takt}} = 0$。
- **排空追回 (recovered)**: **2 个场景** (`RULE_TRAIN_S3_RHO50`, `RULE_VAL_S2_STRONG`)，$\Delta M = 0.00\text{h}$，但 $\Delta P_{\max} > 0$ 且产生了显著的 $J_{\text{takt}}$ 节拍超时费用。
- **在新规则基准中**: `delta` 严格定义为恢复跨度 $R - \tau$，两者差值恒为 0.0000h。

#### 强扰动反例 `RULE_VAL_S2_STRONG` 逐周期全景追踪表
目标为飞机 1 在工位 3（Station 2）的核心瓶颈工序 `1_27`（耗时 49.44h），$\tau = 151.1135\text{h}, R = 187.3362\text{h}$（数据见 `work3_4060/diagnostics/summary_tables/task2_rule_val_s2_strong_cycle_breakdown.csv`）：

| 周期 (Cycle) | 名义转站界限 $P_c^0$ (h) | 实际转站时刻 $P_c$ (h) | 周期延迟量 $\Delta P_c$ (h) | 物理运行事实与链式传递机制 |
| :---: | :---: | :---: | :---: | :--- |
| **Cycle 1** | 52.037844 | 52.037844 | +0.000000 | 扰动前正常脉动 |
| **Cycle 2** | 104.075688 | 104.075688 | +0.000000 | 扰动前正常脉动 |
| **Cycle 3** | 156.113532 | 156.113532 | +0.000000 | $\tau = 151.11\text{h}$ 扰动到达，飞机 1 尚未进驻工位 3 |
| **Cycle 4** | 208.151377 | 236.778605 | **+28.627229** | **工位击穿**: `1_27` 推迟至 187.34h 开工，完工延迟至 236.78h，**转站推迟 28.63h** |
| **Cycle 5** | 260.189221 | 288.816450 | **+28.627229** | 飞机 2 进驻工位 3，瓶颈工时 49.44h 不可压缩，**刚性传递 28.63h** |
| **Cycle 6** | 312.227065 | 340.854294 | **+28.627229** | 飞机 3 流水通过工位 3，**刚性传递 28.63h** |
| **Cycle 7** | 364.264909 | 392.892138 | **+28.627229** | 飞机 4 流水通过工位 3，**刚性传递 28.63h** |
| **Cycle 8** | 416.302753 | 444.929982 | **+28.627229** | 飞机 5 流水通过工位 3，**刚性传递 28.63h** |
| **Cycle 9** | 468.340597 | 496.967826 | **+28.627229** | 飞机 6 流水通过工位 3，**刚性传递 28.63h** |
| **Cycle 10** | 520.378441 | 520.378441 | **+0.000000** | **排空追回**: 飞机 6 移出工位 3，工位 3 排空，下游工位负荷极低吸收滞后 |
| **Cycle 11** | 572.416285 | 572.416285 | **+0.000000** | 末架飞机出线，**总工期 Makespan 维持名义基准完全不变** |

> **延迟传播的三阶段演化与网络状态表征指导**:  
> 对 `RULE_VAL_S2_STRONG` 的 11 周期实际时长与惩罚流进行精细物理切片，确认延迟传播遵循明确的三阶段演化：
> 1. **阶段一：延迟出现（Cycle 4）**：周期时长由标准 $52.04\text{h}$ 拉长至 $80.67\text{h}$（超期 $28.63\text{h}$），产生节拍违规惩罚 $J_{\text{takt}} = \frac{28.63}{52.04} \approx 0.5501$。**全轨道仅在第 4 周期产生了一次性超期罚款**。
> 2. **阶段二：滞后保持（Cycle 5—9）**：第 5 至 9 周期的实际加工时长均为标准的 $52.04\text{h}$（单周期时长与 $H_0$ 差额恒为 $0.00\text{h}$），因此**这 5 个周期的节拍惩罚增量恒为 0**。产线转站日历在绝对时间上整体平移了 28.63h，但并没有发生每周期再次损失 28.63h 的二次恶化。
> 3. **阶段三：排空追回（Cycle 10—11）**：第 10 周期产线进入排空阶段（新飞机停止投产，前部工位负荷骤降），周期时长缩短为 $23.41\text{h}$（比 $H_0$ 短 28.63h），在排空阶段将之前累积的 28.63h 日历滞后全部追回，使得全线批次最终完工时刻锁定在标称的 $572.42\text{h}$。
>
> **对后续网络状态表征的指导**：
> 网络绝对不能把“日历偏置”错误识别为“当前每周期都在持续恶化”。状态编码必须显式解耦三个物理量：
> 1. `cycle_duration_overdue = max(0, current_cycle_duration - H0)`：反映当期是否正面临节拍击穿惩罚；
> 2. `calendar_offset = current_time - nominal_calendar_time`：反映日历时钟的绝对平移量；
> 3. `downstream_slack_buffer`：反映下游工序距离名义完工还剩余的富余吸收空间。

---

## 四、任务 3 检验：重做留站／后移可信配对与现场机制对比

在事前固定的选点规则下，在**同一模板、同一时钟、同一扰动、同一动作前缀**下，仅当目标工序 `can_reserve` 与 `can_postpone` **同时为 True** 时分叉为 Control（留站）与 Treatment（后移），后续均由同一个 `PostponeCompatibleRepairAgent`（`work3_4060/diagnostics/postpone_compatible_agent.py`）驱动至完工。

### 4.1 三个预设现场配对实测数据
（数据见 `work3_4060/diagnostics/summary_tables/task3_paired_arms_summary.csv` 与 `work3_4060/diagnostics/task3_outputs/task3_paired_arms_report.json`）

| 现场编号与场景 | 分叉工序与站位 | 分叉时刻与周期 | Control 动作与团队 | Treatment 动作与目标站团队 | 两臂完工率 | 工期差 $\Delta M$ (h) | 综合费用差 $\Delta J_{\text{total}}$ | 后移增量 $\Delta J_{\text{post}}$ | 节拍增量 $\Delta J_{\text{takt}}$ | 改派增量 $\Delta D_{\text{team}}$ | 偏差增量 $\Delta J_{\text{rev}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Site 1**: Nominal | `0_7` (工位 1) | $t=0.0\text{h}$ (C1) | 留站 `[12, 72]` | 后移 $\rightarrow$ 工位 2 `[10, 18]` | 210 vs 210 | **+0.0000** | **+0.100476** | +0.1000 | +0.0000 | +0.000238 | +0.000238 |
| **Site 2**: S1_MODERATE | `1_7` (工位 1) | $t=52.04\text{h}$ (C2) | 留站 `[12, 72]` | 后移 $\rightarrow$ 工位 2 `[10, 18]` | 210 vs 210 | **+0.0000** | **+0.100295** | +0.1000 | +0.0000 | +0.000238 | +0.000147 |
| **Site 3**: S2_STRONG | `0_18` (工位 2) | $t=52.04\text{h}$ (C2) | 留站 `[10, 18]` | 后移 $\rightarrow$ 工位 3 `[47, 50]` | 210 vs 210 | **+0.0000** | **+0.100525** | +0.1000 | +0.0000 | +0.000238 | +0.000263 |

### 4.2 预设现场后移劣势的物理成因：提前后移并非受扰恢复
事前固定的 3 个配对现场在后移后均产生 $\Delta J \approx +0.1005$ 的净劣势，深入分析其发生机理可知：
1. **Site 1 (Nominal)**: 本无扰动，提前后移属于“无病呻吟”，白白承担 $\lambda_1 = 0.10$ 惩罚。
2. **Site 2 (S1_MODERATE)**: 扰动被站内裕量吸收，提前后移不能换取任何节拍收益。
3. **Site 3 (S2_STRONG)**: 强扰动发生在 $t=151.11\text{h}$（飞机 1 的工序 27），但配对实验在 $t=52.04\text{h}$ 就提前后移了飞机 0 的非受扰工序 `0_18`。这种“扰动未至、后移先行”的动作自然无法产生避险价值。**它仅证明了在无延误时提前后移有害，绝不能推论受扰后的后移没有恢复价值**。

### 4.3 现场实证核验：关键案例 COMPARE_VAL_S0_EARLY_M 完整双臂独立核验与全维度账本剖析

针对“扰动已经发生、任务仍未决、确实影响转站”的关键案例，对全场景进行深入现场筛查：

#### 1. 新基准 6 场景现状记录与物理死锁（可控空间逼仄）
- **4 个场景扰动被工位裕量自然吸收**：`RULE_TRAIN_S1_RHO10`, `S2_RHO25`, `S4_RHO75`, `RULE_VAL_S1_MODERATE`，$\Delta P = 0.00\text{h}, J_{\text{takt}} = 0.0000$。无任何延误，无需干预。
- **2 个延误场景（`RULE_TRAIN_S3_RHO50` 与 `RULE_VAL_S2_STRONG`）的物理死锁**：
  - 受扰目标均为工序 27（加工耗时 49.44h，需求工种 4，2 人）。
  - **下游资质断层硬禁后移**：工位 2 绑定具备工种 4 的工人为 0 和 66；但下游工位 3 仅有工人 `[6, 55]`（工种 0 与 1），完全无工种 4 人员。环境 `validate_postpone(t27)` 直接报错抛出资质不足硬禁令。**工序 27 在物理上因下游资质断层绝对不可后移**！
  - **站内调整物理锁死**：工位 2 其余工序（12, 13, 28, 29）在 $t=160.48\text{h}$ 前全部完工。工人 0 和 66 在 $[160.48, 187.34]$ 处于长达 26.86h 的纯闲置等待物料状态。**此时站内无其他任务可“选序”，亦无多余工人可“选队”**，站内调整同样无法克服外生物料硬延迟。
  - **实证记录**：在现有的 6 场景新基准中，确实不存在“受扰任务合法可后移、且能通过后移或站内调整改变转站延迟”的现场。

#### 2. 关键案例深度核验：`COMPARE_VAL_S0_EARLY_M`（补足 13 条轨迹外关键缺口）
为核验“受扰后任务未决时的后移效果”，对旧基准场景 `COMPARE_VAL_S0_EARLY_M`（目标任务 `1_16`，工位 0，$\tau = 15.4441\text{h}, R = 37.4441\text{h}$）独立开发运行双臂评测脚本 [`diag_compare_val_s0_1_16.py`](file:///d:/OneDrive/%E7%A0%94%E7%A9%B6%E7%94%9F/APAL-Dynamic-2/work3_4060/diagnostics/diag_compare_val_s0_1_16.py)：

1. **分叉点物理事实严格核查（Step 16, Cycle 2, $t = 14.2272\text{h}$）**：
   - **任务仍未决**：任务状态严格为 `TaskStatus.UNREADY`，`actual_start is None`，物理未开工；
   - **零撤回预约**：任务尚未进入 `RESERVED` 状态，分叉时 Treatment 直接下发 `ActionBranch.POSTPONE`，完全不存在撤销或修改已有预约的违规动作；
   - **后移前置条件 100% 具备**：工位 0 非末站，飞机 1 驻留在工位 0，下游工位 1 具备满足工种 0 人数要求的工人池（`[10, 18, 22, 65]`），前驱 `1_15` 已完工，本站未完工紧后集合为空。`can_reserve` 与 `can_postpone` 严格同时为 `True`。
2. **权威独立验证器 100% PASS**：
   - 两臂抽取微观事实送入 [`verify_single_trajectory`](file:///d:/OneDrive/%E7%A0%94%E7%A9%B6%E7%94%9F/APAL-Dynamic-2/work3_4060/verify_trajectory_facts.py#L93-L134)（传入基准 $H_0 = 292.5726\text{h}$）：
   - **两臂违规总数均为 0**（工人互斥 0、工位容量 0、DAG 偏序 0、驻留区间 0）；
   - **账本绝对平衡**：Control 误差 $1.35 \times 10^{-20}$，Treatment 误差 $1.39 \times 10^{-17}$，步奖励求和严格等于 $-J_{\text{total}}$。
3. **双臂全维度账本与工期对比**：
   （数据源自 [`compare_val_s0_1_16_report.json`](file:///d:/OneDrive/%E7%A0%94%E7%A9%B6%E7%94%9F/APAL-Dynamic-2/work3_4060/diagnostics/task3_outputs/compare_val_s0_1_16_report.json) 与 [`compare_val_s0_1_16_paired_arm.csv`](file:///d:/OneDrive/%E7%A0%94%E7%A9%B6%E7%94%9F/APAL-Dynamic-2/work3_4060/diagnostics/summary_tables/compare_val_s0_1_16_paired_arm.csv)）

| 评价维度 / 费用分项 | Control (留站处理) | Treatment (合法后移) | 双臂差值 ($\text{Treat} - \text{Ctrl}$) | 物理机理与增减原因深度解析 |
| :--- | :---: | :---: | :---: | :--- |
| **完工总工期 Makespan** | 679.388956h | 669.106701h | **-10.282255h** | **工期刚性挽回 10.28h**：消除工位 0 闲置等待，全线完工出线时刻完全拉回 |
| **最大转站延迟 $\Delta P_{\max}$** | 10.282255h | 0.000000h | **-10.282255h** | **消除跨周期时钟滞后**：Treatment 恢复各周期名义脉动节奏 |
| **综合总费用 $J_{\text{total}}$** | 0.00006510 | 0.10051377 | **+0.10044867** | **财务净亏损**：后移虽缩短工期，但总费用显著恶化 |
| **节拍超期费用 $J_{\text{takt}}$** | 0.00000000 | 0.00000000 | **+0.00000000** | 旧基准 $H_0 = 292.57\text{h}$ 极为宽松，即使推迟 10.28h 亦未破 $H_0$，罚款为零 |
| **开工时间偏差 $D_{\text{time}}$** | 0.00006510 | 0.00001879 | **-0.00004631** | 后移微观开工与计划偏差微量改善（权重 0.20） |
| **团队替换偏差 $D_{\text{team}}$** | 0.00000000 | 0.00023810 | **+0.00023810** | 工位 1 换用团队 `[10, 18]`（原团队 `[72, 12]`），产生团队偏差（权重 0.05） |
| **后移决策惩罚 $J_{\text{postpone}}$** | 0.00000000 | 0.10000000 | **+0.10000000** | 执行 1 次合法后移的固定惩罚（$w_p \cdot \lambda_1 = 0.10$） |
| **计划重排惩罚 $J_{\text{revision}}$** | 0.00000000 | 0.00025689 | **+0.00025689** | 后移重排换队与时间位移触发动态版本发布惩罚 |

4. **核心矛盾：为何工期省了 10.28 小时，财务上却仍然亏损 0.10045？**
   - **基准常数宽松导致节拍挽回收益为零**：在旧基准中，$H_0 = 292.5726\text{h}$，而 Control 臂中 Cycle 2 加工耗时仅由 14.23h 增至 24.51h，远小于 292.57h，节拍超期惩罚 $J_{\text{takt}}$ 恒为零。工期上虽然挽回了 10.28 小时，但在财务账本上换来的超期减免却是 **$\Delta J_{\text{takt}} = 0.0000$**！
   - **后移与改排成本刚性兑现**：后移必须承担 $J_{\text{postpone}} = 0.10$、换队偏差 $D_{\text{team}} = 0.000238$ 以及计划版本重排 $J_{\text{revision}} = 0.000257$。
   - **结论**：**“挽回工期”绝不等于“财务盈利”；后移是否划算，必须且只能看整条轨迹的综合总费用差 $\Delta J_{\text{total}} < 0$，绝不能单凭工期或局部时间推算！**

---

### 4.4 理论认知纠偏与机制边界重新定调

#### 1. 彻底废除“5.204 小时充要条件”
- **原推导缺陷**：原先根据 $\Delta J_{\text{takt}} = \frac{\Delta P}{H_0} > 0.10$ 推导出 $\Delta P > 5.204\text{h}$ 为有益充要条件。该推导存在三个严重漏洞：
  1. **只算了局部节拍单笔账**：完全忽视了后移引发的下游潜在并发冲突、团队替换偏差 $D_{\text{team}}$ 以及计划改排费用 $J_{\text{revision}}$。例如少 6 小时超期省 0.115，但后移费 0.10 加上其他费用增加 0.03，最终依旧亏损；
  2. **脱离基准饱和状态假设**：如案例 `COMPARE_VAL_S0_EARLY_M` 实测所示，即使工期延误高达 $10.28\text{h} > 5.204\text{h}$，只要产线处于未破节拍的非饱和区间，节拍节省就为 0，后移必定亏损；
  3. **无法跨基准移植**：$5.204\text{h}$ 是基于新基准 $H_0 \approx 52.04\text{h}$ 计算的局部切片，完全无法跨用到旧基准（$H_0 \approx 292.57\text{h}$）。
- **修正后的统一准则**：
  **后移有益的充要条件只有一个：整条轨迹的综合总费用更低（$\Delta J_{\text{total}} < 0$）。不存在任何通用的单工序时间标量硬门槛。后续网络与策略严禁引入 5.204 小时硬规则。**

#### 2. 纠正“站内调整与后移硬分工”
- **原观点偏差**：原报告将站内选队选序表述为“零罚款调整”，将工序后移局限为“处理外生不可抗物料延迟”。
- **修正后的互补关系定调**：
  1. **站内调整并非零成本**：站内选队若更换非基准工人，同样产生团队替换偏差 $D_{\text{team}}$（权重 0.05）与改排费用 $J_{\text{revision}}$；选序滞后同样产生开工时间偏差 $D_{\text{time}}$（权重 0.20）；
  2. **后移亦可用于内生排队疏解**：当站内多工序面临 $\le 3$ 并发通道超限或特定关键工种短缺时，后移未决工序能够作为缓解内生资源拥堵的泄洪手段；
  3. **统一决策空间下的互补交织**：站内调整（选队、选序）与跨站后移是强化学习智能体在统一 MDP 动作空间中并行可选的策略分支，两者不存在非黑即白的硬分工，均由价值网络根据全局折现回报综合权衡。

---

## 五、一页纸结论与最终决策判定

### 5.1 场景“可控空间有限”的核心诊断发现
对当前全库 26 个场景（新基准 6 场景 + 旧基准 20 场景）进行全景审视，揭示出至关重要的物理规律：**当前场景集中的“策略可控空间”高度逼仄**：
1. **在新规则基准（`rule_baseline_ac_v1`，6 个场景）中**：
   - **4 个场景**扰动幅度小于工位天然富余裕量，被现场自然吸收，$\Delta\text{Makespan} = 0, J_{\text{takt}} = 0$。策略无需干预，任何后移只会净亏 0.10；
   - **2 个场景**（`RULE_TRAIN_S3_RHO50`, `RULE_VAL_S2_STRONG`）扰动造成严重延误（$J_{\text{takt}} > 0$），但受扰瓶颈工序 27 需工种 4，**下游工位 3 缺乏工种 4 工人，物理硬禁后移**；同时工位 2 其余工序早已完工，工人处于纯闲置等待状态，**站内无队可换、无序可选**。
2. **在旧基准（`compare_v1`，20 个场景）中**：
   - **15 个场景**被天然裕量局部吸收（$\Delta\text{Makespan} = 0$）；
   - **5 个场景**出现持续日历滞后。但在关键案例 `COMPARE_VAL_S0_EARLY_M` 中，后移虽然合法且挽回 10.28h 工期，却因节拍常数过大未产生节拍罚款而导致财务净亏损。
3. **重要定论**：
   现有基准场景非常适合检验“扰动如何在脉动产线中被动传播、保持与吸收”，但**不足以充分检验智能策略如何通过主动调度改善结果**（缺乏一个后移能够产生净财务盈利的自然标杆案例）。

### 5.2 核心问题最终裁决：研究主线与网络表征落地
> **裁决结论：以“同步约束下的状态演化”为核心主线，保留合法后移动作分支，解耦物理状态表征，严禁增加 5.204 小时硬门槛。**

**核心依据与网络设计落地方案**：
1. **主线确立（同步约束下的状态演化）**：
   多机脉动同步节拍约束是串联整线性能的核心物理骨架。网络将周期超期、日历滞后和下游余量作为可解释状态量，与隐状态共同形成演化链：
   - `cycle_duration_overdue = max(0, current_cycle_duration - H0)`：精准刻画当期面临的真实节拍击穿罚款风险；
   - `calendar_offset = current_time - nominal_calendar_time`：客观反映产线绝对时钟的平移量，避免将稳态的滞后保持误判为持续恶化；
   - `downstream_slack_buffer`：反映下游工序距离名义完工还剩余的富余吸收空间。
2. **动作分支保留与端到端学习**：
   动作空间中严格保留 `can_postpone` 作为合法动作分支。严禁在规则层面增加 5.204 小时或任何单指标硬触发门槛，由 Actor-Critic 网络端到端权衡“后移惩罚/换队成本”与“节拍挽回收益”，以全局综合总费用 $\Delta J_{\text{total}}$ 作为最终优化目标。

---

## 六、交付物与数据清单

所有脚本与数据均已归档于工作区：
1. **诊断执行脚本**:
   - `work3_4060/diagnostics/diag_compare_val_s0_1_16.py`: 关键案例 COMPARE_VAL_S0_EARLY_M 独立双臂仿真、验收与账本提取驱动
   - `work3_4060/diagnostics/diag_verify_all_trajectories.py`: 13 条基准轨迹全量抽取与独立物理/财务验收套件
   - `work3_4060/diagnostics/postpone_compatible_agent.py`: 后移兼容确定性修复控制器实现
   - `work3_4060/diagnostics/run_task1_investigation.py`: 任务 1 四类后移、五对见证与容量测试驱动
   - `work3_4060/diagnostics/diag_task2_compare_v1.py`: 任务 2 compare_v1 20 场景复核驱动
   - `work3_4060/diagnostics/diag_task2_rule_baseline.py`: 任务 2 新规则 6 场景与反例逐周期复核驱动
   - `work3_4060/diagnostics/diag_task3_paired_arms.py`: 任务 3 事前固定 3 现场留站/后移双臂配对驱动
   - `work3_4060/diagnostics/export_all_summary_tables.py`: 导出规范 CSV 汇总表驱动
2. **原始实验 JSON 数据**:
   - `work3_4060/diagnostics/task3_outputs/compare_val_s0_1_16_report.json`: 关键案例双臂全量对比与独立验收报告
   - `work3_4060/diagnostics/task1_outputs/task1_reachability_report.json`
   - `work3_4060/diagnostics/task2_outputs/task2_compare_v1_evaluation.json`
   - `work3_4060/diagnostics/task2_outputs/task2_rule_baseline_evaluation.json`
   - `work3_4060/diagnostics/task3_outputs/task3_paired_arms_report.json`
3. **复查 CSV 汇总表**:
   - `work3_4060/diagnostics/summary_tables/compare_val_s0_1_16_paired_arm.csv`: 关键案例双臂指标对比汇总表
   - `work3_4060/diagnostics/summary_tables/task1_five_pairs_witnesses.csv` (五对动作见证表)
   - `work3_4060/diagnostics/summary_tables/task1_joint_capacity_summary.csv` (容量测试表)
   - `work3_4060/diagnostics/summary_tables/task2_compare_v1_20_scenarios_review.csv` (旧基准20场景复核表)
   - `work3_4060/diagnostics/summary_tables/task2_rule_baseline_scenarios_review.csv` (新基准6场景复核表)
   - `work3_4060/diagnostics/summary_tables/task2_rule_val_s2_strong_cycle_breakdown.csv` (强扰动逐周期明细表)
   - `work3_4060/diagnostics/summary_tables/task3_paired_arms_summary.csv` (三现场双臂对比表)
4. **完整综合诊断报告**:
   - `work3_4060/diagnostics/DIAGNOSTIC_REPORT_V01.md`
