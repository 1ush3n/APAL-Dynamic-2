# Work-3 势函数系数单变量验证综合报告 (beta=0.1 vs beta=1.0 vs F vs C)
评估对象：Method C, Method F, Method D (beta=1.0), Method D (beta=0.1) | 种子：42 | 预算：2048 决策

## 1. 势函数奖励塑形闭合审计结果 (D_beta01 shaping_events.jsonl)
- **总体审计裁定**: `INCOMPLETE`
- **有效配置交叉比对**: gamma=0.99 (YAML=0.99), beta=0.1 (YAML=0.1)
- **检查事件数**: 2048 步
- **检查 Episode 数**: 5 (完工批次: 4)
- **单步公式一致性**: `PASS` (失败数: 0)
- **跨步相邻势连续性**: `PASS` (失败数: 0)
- **预测器版本固定**: `PASS` (失败数: 0)
- **终局势归零合同**: `PASS` (失败数: 0)
- **完工 Episode 折扣望远镜恒等式闭合**: `PASS`
- **配置一致性**: `PASS`
- **边界与 Bootstrap 合同**: `INCOMPLETE`
- **原始账本对账**: `INCOMPLETE` (尾部对账残差: 0.0)

## 2. 验证集完整端到端对比 (Nominal + 4 Validation Scenarios)
| scenario_id | timing | intensity | station | c_feasible | c_makespan | c_j_total | c_j_takt | c_d_time | c_d_team | c_j_postpone | c_j_revision | c_postponed_count | c_decisions | f_feasible | f_makespan | f_j_total | f_j_takt | f_d_time | f_d_team | f_j_postpone | f_j_revision | f_postponed_count | f_decisions | d_beta1_feasible | d_beta1_makespan | d_beta1_j_total | d_beta1_j_takt | d_beta1_d_time | d_beta1_d_team | d_beta1_j_postpone | d_beta1_j_revision | d_beta1_postponed_count | d_beta1_decisions | d_beta01_feasible | d_beta01_makespan | d_beta01_j_total | d_beta01_j_takt | d_beta01_d_time | d_beta01_d_team | d_beta01_j_postpone | d_beta01_j_revision | d_beta01_postponed_count | d_beta01_decisions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NOMINAL | NONE | NONE | NONE | True | 669.71 | 2.145989e-02 | 0.000000e+00 | 7.299457e-04 | 1.000000e-02 | 0.000000e+00 | 1.072995e-02 | 0 | 210 | True | 671.5 | 2.810431e-02 | 0.000000e+00 | 7.188230e-04 | 1.333333e-02 | 0.000000e+00 | 1.405216e-02 | 0 | 210 | True | 442.75 | 7.602916e-01 | 0.000000e+00 | 3.479115e-03 | 2.666667e-02 | 7.000000e-01 | 3.014578e-02 | 7 | 217 | True | 761.98 | 5.933907e-02 | 0.000000e+00 | 2.050287e-02 | 9.166667e-03 | 0.000000e+00 | 2.966953e-02 | 0 | 422 |
| COMPARE_VAL_S0_EARLY_M | EARLY | MED | 0 | True | 679.99 | 2.152500e-02 | 0.000000e+00 | 7.950496e-04 | 1.000000e-02 | 0.000000e+00 | 1.072995e-02 | 0 | 210 | True | 681.78 | 2.816942e-02 | 0.000000e+00 | 7.839269e-04 | 1.333333e-02 | 0.000000e+00 | 1.405216e-02 | 0 | 210 | True | 453.03 | 7.603292e-01 | 0.000000e+00 | 3.516794e-03 | 2.666667e-02 | 7.000000e-01 | 3.014578e-02 | 7 | 217 | True | 761.98 | 5.934238e-02 | 0.000000e+00 | 2.050452e-02 | 9.166667e-03 | 0.000000e+00 | 2.967119e-02 | 0 | 423 |
| COMPARE_VAL_S2_MID_M | MID | MED | 2 | True | 669.71 | 2.158412e-02 | 0.000000e+00 | 7.920620e-04 | 1.000000e-02 | 0.000000e+00 | 1.079206e-02 | 0 | 210 | True | 671.5 | 2.822908e-02 | 0.000000e+00 | 7.812078e-04 | 1.333333e-02 | 0.000000e+00 | 1.411454e-02 | 0 | 210 | True | 442.75 | 7.602916e-01 | 0.000000e+00 | 3.479115e-03 | 2.666667e-02 | 7.000000e-01 | 3.014578e-02 | 7 | 217 | True | 761.98 | 5.936098e-02 | 0.000000e+00 | 2.051382e-02 | 9.166667e-03 | 0.000000e+00 | 2.968049e-02 | 0 | 423 |
| COMPARE_VAL_S3_MID_L | MID | LOW | 4 | True | 669.71 | 2.159809e-02 | 0.000000e+00 | 7.990467e-04 | 1.000000e-02 | 0.000000e+00 | 1.079905e-02 | 0 | 210 | True | 671.5 | 2.817551e-02 | 0.000000e+00 | 7.544227e-04 | 1.333333e-02 | 0.000000e+00 | 1.408776e-02 | 0 | 210 | True | 442.75 | 7.602916e-01 | 0.000000e+00 | 3.479115e-03 | 2.666667e-02 | 7.000000e-01 | 3.014578e-02 | 7 | 217 | True | 761.98 | 5.933914e-02 | 0.000000e+00 | 2.050290e-02 | 9.166667e-03 | 0.000000e+00 | 2.966957e-02 | 0 | 423 |
| COMPARE_VAL_S4_LATE_H | LATE | HIGH | 4 | True | 704.45 | 2.157298e-02 | 0.000000e+00 | 8.430368e-04 | 1.000000e-02 | 0.000000e+00 | 1.072995e-02 | 0 | 210 | True | 705.04 | 2.821351e-02 | 0.000000e+00 | 8.280239e-04 | 1.333333e-02 | 0.000000e+00 | 1.405216e-02 | 0 | 210 | True | 442.75 | 7.602916e-01 | 0.000000e+00 | 3.479115e-03 | 2.666667e-02 | 7.000000e-01 | 3.014578e-02 | 7 | 217 | True | 761.98 | 5.933907e-02 | 0.000000e+00 | 2.050287e-02 | 9.166667e-03 | 0.000000e+00 | 2.966953e-02 | 0 | 422 |

## 3. 相同受扰现场接管恢复对比 (Takeover Comparison)
| scenario_id | station_id | aircraft_station_at_handover | timing | intensity | hit_targets | handover_step | handover_time | c_feasible | c_termination_reason | c_post_decisions | c_post_makespan | c_post_j_total | c_post_j_takt | c_post_d_time | c_post_d_team | c_post_j_postpone | c_post_j_revision | c_post_postponed_count | f_feasible | f_termination_reason | f_post_decisions | f_post_makespan | f_post_j_total | f_post_j_takt | f_post_d_time | f_post_d_team | f_post_j_postpone | f_post_j_revision | f_post_postponed_count | d_beta1_feasible | d_beta1_termination_reason | d_beta1_post_decisions | d_beta1_post_makespan | d_beta1_post_j_total | d_beta1_post_j_takt | d_beta1_post_d_time | d_beta1_post_d_team | d_beta1_post_j_postpone | d_beta1_post_j_revision | d_beta1_post_postponed_count | d_beta01_feasible | d_beta01_termination_reason | d_beta01_post_decisions | d_beta01_post_makespan | d_beta01_post_j_total | d_beta01_post_j_takt | d_beta01_post_d_time | d_beta01_post_d_team | d_beta01_post_j_postpone | d_beta01_post_j_revision | d_beta01_post_postponed_count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| COMPARE_VAL_S0_EARLY_M | 0 | ac1@st1 | EARLY | MED | 1_16 | 21 | 38.74 | True | completed | 189 | 641.26 | 2.145989e-02 | 0.000000e+00 | 7.299457e-04 | 1.000000e-02 | 0.000000e+00 | 1.072995e-02 | 0 | True | completed | 189 | 643.04 | 2.810431e-02 | 0.000000e+00 | 7.188230e-04 | 1.333333e-02 | 0.000000e+00 | 1.405216e-02 | 0 | True | completed | 196 | 414.30 | 7.576812e-01 | 0.000000e+00 | 3.126294e-03 | 2.571429e-02 | 7.000000e-01 | 2.884058e-02 | 7 | True | completed | 381 | 716.37 | 5.816462e-02 | 0.000000e+00 | 1.991564e-02 | 9.166667e-03 | 0.000000e+00 | 2.908231e-02 | 0 |
| COMPARE_VAL_S2_MID_M | 2 | ac2@st2 | MID | MED | 2_13 | 61 | 129.71 | True | completed | 149 | 540.00 | 1.952084e-02 | 0.000000e+00 | 7.127988e-04 | 9.047619e-03 | 0.000000e+00 | 9.760418e-03 | 0 | True | completed | 149 | 541.79 | 2.617066e-02 | 0.000000e+00 | 7.043754e-04 | 1.238095e-02 | 0.000000e+00 | 1.308533e-02 | 0 | True | completed | 154 | 303.35 | 5.491801e-01 | 0.000000e+00 | 2.685295e-03 | 2.190476e-02 | 5.000000e-01 | 2.459006e-02 | 5 | True | completed | 304 | 615.12 | 5.438846e-02 | 0.000000e+00 | 1.897994e-02 | 8.214286e-03 | 0.000000e+00 | 2.719423e-02 | 0 |
| COMPARE_VAL_S3_MID_L | 4 | ac3@st4 | MID | LOW | 3_24 | 151 | 360.88 | True | completed | 59 | 308.57 | 1.034107e-02 | 0.000000e+00 | 4.086313e-04 | 4.761905e-03 | 0.000000e+00 | 5.170536e-03 | 0 | True | completed | 59 | 310.17 | 1.409130e-02 | 0.000000e+00 | 3.789850e-04 | 6.666667e-03 | 0.000000e+00 | 7.045652e-03 | 0 | True | completed | 61 | 136.93 | 2.234492e-01 | 0.000000e+00 | 1.248390e-03 | 1.047619e-02 | 2.000000e-01 | 1.172458e-02 | 2 | True | completed | 121 | 325.26 | 1.818523e-02 | 0.000000e+00 | 4.806900e-03 | 4.285714e-03 | 0.000000e+00 | 9.092615e-03 | 0 |
| COMPARE_VAL_S4_LATE_H | 4 | ac3@st5 | LATE | HIGH | 3_125 | 173 | 472.93 | True | completed | 37 | 231.43 | 7.133808e-03 | 0.000000e+00 | 2.335707e-04 | 3.333333e-03 | 0.000000e+00 | 3.566904e-03 | 0 | True | completed | 37 | 232.63 | 9.986279e-03 | 0.000000e+00 | 2.312346e-04 | 4.761905e-03 | 0.000000e+00 | 4.993139e-03 | 0 | True | completed | 38 | 81.46 | 1.150689e-01 | 0.000000e+00 | 8.677680e-04 | 6.666667e-03 | 1.000000e-01 | 7.534435e-03 | 1 | True | completed | 76 | 241.55 | 1.084752e-02 | 0.000000e+00 | 2.447571e-03 | 2.976190e-03 | 0.000000e+00 | 5.423761e-03 | 0 |

## 4. 显式推进决策与配对反事实干预诊断 (Advance Decisions & Paired Intervention)
针对 D(beta=0.1) 在 Nominal 场景出现的 212 次显式推进（ADVANCE_TO_NEXT_EVENT）进行了三态机会搜索与配对反事实干预，主要结论如下：

### 4.1 核心诊断与回报汇总
| 变体方法 | 总决策数 | 显式推进数 | 即时开工机会 (immediate) | 证伪无即时 (proven) | 搜索截断 | 推进跨越仿真工期 (h) | dt 极差 (h) | 在制品进行中步数 | 未折扣回报 G1 (-J) | 折扣回报 G0.99 | 最终工期 (h) | 综合费用 J_total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C** (C) | 210 | 0 | 0 | 0 | 0 | 0 | N/A | 0/0 | -0.021460 | -0.005987 | 669.71h | 2.145989e-02 |
| **F** (F) | 210 | 0 | 0 | 0 | 0 | 0 | N/A | 0/0 | -0.028104 | -0.007663 | 671.50h | 2.810431e-02 |
| **D_beta01** (D) | 422 | 212 | 157 | 55 | 0 | 620.74h | 0.0305 - 64.2576h | 193/212 | -0.059342 | -0.008222 | 761.98h | 5.934238e-02 |

### 4.2 配对反事实干预关键发现 (Steps 2, 17, 20)
在相同受扰现场严格重放至决策步，对比 Control (保持推进) 与 Intervention (强制执行即时开工见证)：
1. **Step 2 (t=0.0h, prob_adv=0.9248)**: 开工见证 `0_15`，团队 `(12, 72)`。强制开工后 $J$ 降低 $1.103 \times 10^{-4}$，工期缩短 0.55h，**立即开工更优**。
2. **Step 17 (t=22.81h, prob_adv=0.9294)**: 开工见证 `0_18`，团队 `(18, 22)`。强制开工后团队惩罚增加与后续变更导致 $J$ 增加 $2.070 \times 10^{-4}$，工期无改善，**显式推进更优**。
3. **Step 20 (t=23.36h, prob_adv=0.9356)**: 开工见证 `1_15`，团队 `(12, 72)`。强制开工后 $J$ 降低 $1.100 \times 10^{-4}$，工期缩短 0.55h，**立即开工更优**。

> **因果机制定性**: 配对干预明确证明显式推进并非单一的决策错误：在 Step 17 等复杂资源占用现场，选择推进等待后续离散事件实际上避免了次优团队配置与额外变更成本；但在 Step 2、Step 20 等初期无冲突现场，策略存在高概率过度等待倾向。详细分析请参见 [advance_diagnosis_summary.md](advance_diagnosis_summary.md) 与 [paired_intervention_summary.md](paired_intervention_summary.md)。

