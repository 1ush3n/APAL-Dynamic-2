# Work-3 小规模 C/D 效果对比结果简报 (compare_v1)
日期: 2026-10-18 | 数据集: mini210 (210 工序, 7 架, 5 站) | 设备: cuda:0 | 精度: FP32

## 1. 核心结论摘要
- **测试场景总对数**: 24 (3 seeds × 8 测试场景)
- **双向独立可行对数**: 24 / 24 (100% 可行)
- **绝对费用差均值 (J_C - J_D)**: `-2.585690e-01` (正值表示 D 优于 C)
- **工期差均值 (Makespan_C - Makespan_D)**: `+48.57h` (正值表示 D 工期更短)

## 2. 逐 Seed 统计表
| Seed | 测试总数 | 可行对数 | 双方命中数 | 单方命中数 | 费用差均值 (J_C - J_D) | 工期差均值 (h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 8 | 8 | 5 | 3 | -7.3878e-01 | +228.79 |
| 43 | 8 | 8 | 4 | 4 | +9.6813e-04 | -261.49 |
| 44 | 8 | 8 | 6 | 2 | -3.7897e-02 | +178.42 |

## 3. 详细逐场景对比
| Seed | 场景 ID | 时机 | 强度 | 命中状态 | C Makespan | D Makespan | C J_total | D J_total | J_C - J_D |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | NOMINAL_NO_DISTURBANCE | NONE | NONE | NONE | 669.7 | 442.7 | 2.1460e-02 | 7.6029e-01 | -7.3883e-01 |
| 42 | COMPARE_TEST_S0_EARLY_M | EARLY | MED | BOTH | 669.7 | 442.7 | 2.1541e-02 | 7.6035e-01 | -7.3880e-01 |
| 42 | COMPARE_TEST_S1_EARLY_L | EARLY | LOW | ONE_SIDED | 669.7 | 442.7 | 2.1508e-02 | 7.6029e-01 | -7.3878e-01 |
| 42 | COMPARE_TEST_S2_MID_H | MID | HIGH | BOTH | 669.7 | 442.7 | 2.1574e-02 | 7.6048e-01 | -7.3891e-01 |
| 42 | COMPARE_TEST_S3_MID_M | MID | MED | ONE_SIDED | 669.7 | 442.7 | 2.1529e-02 | 7.6029e-01 | -7.3876e-01 |
| 42 | COMPARE_TEST_S4_MID_L | MID | LOW | ONE_SIDED | 684.4 | 442.7 | 2.1508e-02 | 7.6029e-01 | -7.3878e-01 |
| 42 | COMPARE_TEST_S1_MID_H | MID | HIGH | BOTH | 669.7 | 442.7 | 2.1908e-02 | 7.6060e-01 | -7.3870e-01 |
| 42 | COMPARE_TEST_S0_LATE_L | LATE | LOW | BOTH | 669.7 | 442.7 | 2.1509e-02 | 7.6029e-01 | -7.3878e-01 |
| 42 | COMPARE_TEST_S2_LATE_M | LATE | MED | BOTH | 669.7 | 442.7 | 2.1583e-02 | 7.6029e-01 | -7.3871e-01 |
| 43 | NOMINAL_NO_DISTURBANCE | NONE | NONE | NONE | 421.7 | 683.1 | 4.3572e-02 | 4.2560e-02 | +1.0123e-03 |
| 43 | COMPARE_TEST_S0_EARLY_M | EARLY | MED | BOTH | 421.7 | 683.1 | 4.3626e-02 | 4.2641e-02 | +9.8484e-04 |
| 43 | COMPARE_TEST_S1_EARLY_L | EARLY | LOW | ONE_SIDED | 421.7 | 683.1 | 4.3544e-02 | 4.2567e-02 | +9.7651e-04 |
| 43 | COMPARE_TEST_S2_MID_H | MID | HIGH | BOTH | 421.7 | 683.1 | 4.3719e-02 | 4.2744e-02 | +9.7519e-04 |
| 43 | COMPARE_TEST_S3_MID_M | MID | MED | ONE_SIDED | 421.7 | 683.1 | 4.3551e-02 | 4.2633e-02 | +9.1825e-04 |
| 43 | COMPARE_TEST_S4_MID_L | MID | LOW | ONE_SIDED | 421.7 | 684.3 | 4.3572e-02 | 4.2564e-02 | +1.0083e-03 |
| 43 | COMPARE_TEST_S1_MID_H | MID | HIGH | BOTH | 421.7 | 683.1 | 4.3812e-02 | 4.2829e-02 | +9.8287e-04 |
| 43 | COMPARE_TEST_S0_LATE_L | LATE | LOW | BOTH | 421.7 | 683.1 | 4.3558e-02 | 4.2631e-02 | +9.2678e-04 |
| 43 | COMPARE_TEST_S2_LATE_M | LATE | MED | ONE_SIDED | 421.7 | 683.1 | 4.3565e-02 | 4.2593e-02 | +9.7232e-04 |
| 44 | NOMINAL_NO_DISTURBANCE | NONE | NONE | NONE | 672.5 | 493.7 | 2.3752e-02 | 6.1730e-02 | -3.7977e-02 |
| 44 | COMPARE_TEST_S0_EARLY_M | EARLY | MED | BOTH | 672.5 | 493.7 | 2.3798e-02 | 6.1784e-02 | -3.7986e-02 |
| 44 | COMPARE_TEST_S1_EARLY_L | EARLY | LOW | BOTH | 672.5 | 492.9 | 2.3781e-02 | 6.1731e-02 | -3.7950e-02 |
| 44 | COMPARE_TEST_S2_MID_H | MID | HIGH | BOTH | 672.5 | 491.6 | 2.3970e-02 | 6.1642e-02 | -3.7672e-02 |
| 44 | COMPARE_TEST_S3_MID_M | MID | MED | ONE_SIDED | 672.5 | 493.7 | 2.3834e-02 | 6.1730e-02 | -3.7895e-02 |
| 44 | COMPARE_TEST_S4_MID_L | MID | LOW | ONE_SIDED | 685.6 | 493.7 | 2.3795e-02 | 6.1730e-02 | -3.7935e-02 |
| 44 | COMPARE_TEST_S1_MID_H | MID | HIGH | BOTH | 672.5 | 493.9 | 2.4076e-02 | 6.1854e-02 | -3.7778e-02 |
| 44 | COMPARE_TEST_S0_LATE_L | LATE | LOW | BOTH | 672.5 | 493.7 | 2.3684e-02 | 6.1730e-02 | -3.8045e-02 |
| 44 | COMPARE_TEST_S2_LATE_M | LATE | MED | BOTH | 672.5 | 512.5 | 2.4045e-02 | 6.1957e-02 | -3.7912e-02 |
