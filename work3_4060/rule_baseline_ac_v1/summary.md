# Work-3 规则基准重建与最小 A/C 对比实验报告

## 1. 实验概况与环境配置
- **基准版本**: `work3_mini210_rule_baseline_v1` (SHA-256: `f461b887335b76bb...`)
- **转站机制**: `nominal_anchored_v1` ($P_q = \max\{P_q^0, P_{q-1}, F_q\}$)
- **规则排程节拍 $H_0$**: `52.037844h`（相比旧随机基准 292.57h 压缩 82.21%）
- **训练方法**: Method C（标准 PPO + 启发式势函数奖励塑形 $\Phi_{\text{heur}}$，$\beta = 1.0$）
- **训练预算**: 4096 决策步，单进程 spawn 向量环境，设备 `cuda:0` (RTX 4060 Laptop 8GB)
- **评估协议**: Nominal + 2 个重新标定的扰动场景（6 条完整正式轨迹，CPU 单线程贪心）
- **八项塑形审计**: `PASS`

## 2. 最小对比指标结果表

| 场景 ID | 场景类型 | A 总费用 $J_{\text{total}}$ | C 总费用 $J_{\text{total}}$ | 费用胜者 | A 节拍惩罚 $J_{\text{takt}}$ | C 节拍惩罚 $J_{\text{takt}}$ | A 完工工期 (h) | C 完工工期 (h) | A/C 可行性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NOMINAL` | 名义场景 | 2.751822e-15 | 3.822568 | **A** | 0.000000 | 3.554090 | 572.42 | 677.82 | PASS / PASS |
| `RULE_VAL_S1_MODERATE` | 适度扰动 | 0.000566 | 3.822568 | **A** | 0.000000 | 3.554090 | 572.42 | 677.82 | PASS / PASS |
| `RULE_VAL_S2_STRONG` | 强扰动 | 0.551171 | 3.822568 | **A** | 0.550123 | 3.554090 | 572.42 | 677.82 | PASS / PASS |

## 3. 详细分项费用诊断

| 场景 ID | 算法 | $J_{\text{total}}$ | $J_{\text{takt}}$ | $D_{\text{time}}$ | $D_{\text{team}}$ | $J_{\text{postpone}}$ | $J_{\text{revision}}$ | 决策数 | 命中数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NOMINAL` | **A** | 2.751822e-15 | 2.730869e-15 | 2.095292e-17 | 0.000000 | 0.000000 | 0.000000 | 210 | 0 |
| `NOMINAL` | **C** | 3.822568e+00 | 3.554090e+00 | 1.200722e-01 | 0.014167 | 0.000000 | 0.134239 | 395 | 0 |
| `RULE_VAL_S1_MODERATE` | **A** | 5.660589e-04 | 2.730869e-15 | 2.830295e-04 | 0.000000 | 0.000000 | 0.000283 | 210 | 1 |
| `RULE_VAL_S1_MODERATE` | **C** | 3.822568e+00 | 3.554090e+00 | 1.200722e-01 | 0.014167 | 0.000000 | 0.134239 | 396 | 1 |
| `RULE_VAL_S2_STRONG` | **A** | 5.511711e-01 | 5.501233e-01 | 5.239269e-04 | 0.000000 | 0.000000 | 0.000524 | 210 | 1 |
| `RULE_VAL_S2_STRONG` | **C** | 3.822568e+00 | 3.554090e+00 | 1.200722e-01 | 0.014167 | 0.000000 | 0.134239 | 397 | 1 |

## 4. 八项塑形审计详情

| 检查编号 | 审计项名称 | 状态 | 详细诊断说明 |
| :--- | :--- | :--- | :--- |
| `single_step_formula` | 单步塑形公式 | **PASS** | shaped_reward - raw_reward == beta * (gamma * phi_next - phi_current) 逐步闭合 |
| `step_to_step_continuity` | 跨步势连续性 | **PASS** | 同一 episode 内相邻决策步 phi_next 与下一行 phi_current 完全相等 |
| `predictor_version_invariance` | 预测器版本不变 | **PASS** | 同一 episode 内预测器版本严格固定，无跨版本漂移 |
| `terminal_zero_potential` | 终局零势合同 | **PASS** | 自然完工 (terminated=True) 时 phi_terminal == 0.0 |
| `completed_episode_telescope` | 望远镜求和恒等式 | **PASS** | sum gamma^t (r_shaped - r_raw) == -beta * phi_start 严格对账 |
| `config_consistency` | 参数配置一致性 | **PASS** | run.json 参数与事件流中 gamma/beta 严格一致，中途无篡改 |
| `boundary_contract` | 边界与截断合同 | **PASS** | 段尾结算记录存在且有限，预算截断与完工终局正确分离 |
| `ledger_reconciliation` | 原始账本对账 | **PASS** | sum(raw_reward) + J_total == 0 且 五分项之和 - J_total == 0 严格成立 |

## 5. 核心结论与机制剖析
1. **新基准物理真实性成立**：节拍 $H_0 = 52.04\text{h}$ 紧密贴合 Station 3 物理瓶颈，A 在 Nominal 下 0 违规、0 偏离、0 延迟，$J_{\text{total}} \le 10^{-9}$，名义排程物理闭环完全通过。
2. **扰动敏感性验证有效**：适度扰动落在非瓶颈站位时被工序缓冲吸收 ($J_{\text{takt}} = 0$)，而强扰动落在瓶颈站位 (Station 3) 立即产生严格正的节拍惩罚 ($J_{\text{takt}} > 0$)，虚假吸收问题彻底解决。
3. **A 与 C 表现对比**：在紧凑基准与严密时钟下，A 策略通过静态基准保持展现出强基准对齐能力；C 在 4096 决策试训后，物理可行率保持 100%，对比两者的绝对费用与工期分项，如实呈现学习策略与规则策略在抗扰动与基准保持上的差异。
