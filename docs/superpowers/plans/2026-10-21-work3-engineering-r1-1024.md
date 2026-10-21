# Work-3 工程收口与 R1 1024 步复核实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复三个不符合当前物理合同的测试夹具，固定可复现源码并完成本地依赖归档、唯一 R1 1024 步复核和下一阶段恢复学习准备。

**Architecture:** 复用现有 Work-3 环境、Actor-Critic、Lightning PPO、轨迹与塑形核验器；只修测试入口和本轮运行/报告编排，不改物理环境、费用口径或 PPO 算法。工程修复先独立提交并冻结，训练只写入新输出根目录。

**Tech Stack:** Python、PyTorch、PyG、Lightning、pytest、Git、ZIP64、现有 Work-3 验证脚本。

**Spec:** 用户在 2026-10-21 提供的 Work-3 执行提示词（当前会话）。

## 全局约束

- 仅使用 `D:\Conda\envs\rag_env\python.exe` 及其现有依赖，不升级或重建环境。
- 不 reset/clean、不覆盖历史实验、不删除本地文件、不上传文件、不提交 `.pt` 或凭据。
- PPO seed=43，R1，FP32，Nominal，无扰动，warmup=none，AdamW，实际学习率 3e-5。
- 总训练交互上限 1024；固定评估点 0、128、256、512、768、1024；费用与物理验收阈值保持既有定义。
- 先完成工程测试与来源核验，提交并推送训练源码冻结版本，再运行训练。
- 074fe07 冻结版的首次启动在校准初始化前因直接脚本入口未加入仓库根路径而失败；训练子进程未启动，环境交互/优化/评估均为 0。预检清单原样保留于仓库外备份目录，修复后仅允许这一次同来源、同 seed、同预算重启。

---

### 任务 1：基线归因与测试夹具修复

**文件：**
- 修改：`tests/work3/test_work3_training_runtime.py`
- 新增：`docs/superpowers/plans/2026-10-21-work3-engineering-r1-1024.md`

**步骤：**
- [x] 用同一 rag_env 在 `0e2e0bb` 与 `2c434ff` 隔离工作树上运行原三项测试并记录真实模块路径。
- [x] 用正式合法的未来预约构造显式推进前缀，并在单 worker、双 worker测试中回放。
- [x] 在未预约现场采样留站/后移动作；执行后断言 RESERVED 不在候选集且两个分支都禁用。
- [x] 运行三项修复测试并记录结果；基线与修复前各 3 项失败，修复后目标覆盖 16 项通过。

### 任务 2：AMP/FP32 离线等价与依赖归档

**文件：**
- 新增：`work3_4060/engineering_closeout_r1_1024_v1/ppo_seed43/engineering/fp32_equivalence.json`
- 新增：`work3_4060/engineering_closeout_r1_1024_v1/ppo_seed43/backup/*`

**步骤：**
- [x] 核对历史 autocast 源码能否精确恢复；用保存的真实首段和 mini-batch 身份做独立诊断更新。历史临时源码无法精确恢复，重建等价检查已通过，精确复现为 INCOMPLETE。
- [x] 清点 34 个历史 `.pt` 及本轮必需依赖，当前逐项存在且哈希匹配。
- [x] 建立 pretraining/postrun ZIP64 本地归档并逐成员回读 SHA256；各包含并核验34/34个历史登记 `.pt`。

### 任务 3：训练入口、契约测试与冻结提交

**文件：**
- 修改/新增：最小化的 Work-3 复核驱动、相关测试和工程清单。
- 结果目录：`work3_4060/engineering_closeout_r1_1024_v1/ppo_seed43/`

**步骤：**
- [x] 复用 seed43 首段、配置和轨迹核验函数，新增显式 1024 预算与六点覆盖。
- [x] 验证来源哈希、R1 实际 optimizer 参数组、无 R4 hook、评估隔离及路径边界的代码合同；训练实测仍待执行。
- [x] 运行受影响回归，提交并推送训练相关源码冻结版本（`074fe07`、入口修复 `792ad60`）。训练完成后另以报告专用提交修复异构 CSV 列并集及其源绑定（`0da4da6`、`1797556`），未更动模型或物理轨迹。

### 任务 4：唯一 R1 运行与独立核验

**步骤：**
- [x] 从校准权重重新初始化，seed43 串行运行至1024步预算。
- [x] 在六个固定点执行独立 CPU greedy 评估，运行现有轨迹事实与塑形审计。
- [x] 复核真实交互1024、optimizer.step 512、独立评估1260决策、训练耗时247.29秒、PyTorch峰值allocated/reserved 76.36/78MiB、来源权重和旧输入未变。

### 任务 5：离线目标核对、恢复学习草案与交付

**文件：**
- 新增：`objective_alignment_note.md`、`recovery_learning_protocol_draft.md`、`summary.md` 及机器可读验证记录。

**步骤：**
- [x] 独立核对 raw 折扣回报、未折扣费用、塑形边界和实际奖励发生位置。
- [x] 草拟下一阶段固定恢复协议；未生成扰动数据、未执行扰动训练。
- [x] 归档本轮新增产物并生成动态状态报告；结果文档与计划状态一并提交并推送。
