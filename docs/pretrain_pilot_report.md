# 工作三：真实扰动、示例预训练与PPO有限对照报告

日期：2026年11月17日。结论分级：**工程接入通过；本轮未观察到预训练的总体优势，并出现闭环退化。PPO改善了部分BC结果，但两种子均未超过同在线预算的从头PPO。** 不据此断言该方法普遍无效，也不把损失下降当作排程收益。

## 版本与环境

- 起点：`68434ebf9e2ff74110ac04f32e113ccc6b0bca6f`，独立分支：`work3-pretrain-pilot-20261117`。
- 扰动回归：`863216b`；示例生成与独立核证：`c699944`；完整学习接入及实验源码：`7ec0a38a01bcd41b19c37fd14422672221bffe0f`。全部正式训练在该源码、干净工作区运行。
- 收尾修订：`1a1d824`，仅固定导出CLI的随机来源、兼容Windows相对清单路径；不改冻结数据、网络、损失或已训权重。原52例解析出的绝对输入路径全部保持，另复验10项相关检查与3项双worker检查。
- Windows；`D:\Conda\envs\rag_env\python.exe`；Python3.11.15、PyTorch2.12.0+cu132、PyG2.7.0、Lightning2.6.5、NumPy2.4.6；RTX4060 Laptop 8GB。未安装或升级依赖。
- 原QUERY结构：hidden_dim64、每站指针宽度16、32维事实、14维查询；GPU FP32，主进程与两个CPU spawn环境各1线程。AMP上下文保留，FP32不启用低精度。无LLM服务调用。

本轮全部本机工件位于 `runs/work3_pretrain/pretrain_pilot_20261117_01/`，下文称本轮目录。权重和大批轨迹不进入Git；仓库保留代码、本报告与[逐实例结果表](pretrain_pilot_results.csv)。GitHub默认main未合并，本分支才包含本轮实现。

本轮目录15:50:39创建，证据记录于17:08:15，期间77.61分钟包含实现、核对、实验、分析和等待，不等于GPU训练时间，也不含此前等待用户下载资料的时间；交付整理与推送另计。经外层控制器启动的训练、评估、重建和条件概率进程，其精确argv/预算/退出码/墙钟保存在`*_execution.json`。

## 实际修改

| 原行为 | 新行为与关键位置 | 原因 |
|---|---|---|
| 准备包只在旧a710e34核验 | `apal/data/work3/certified_examples.py`生成、原生见证回放；`certificates.py`从公开输入独立重推下界 | 不能仅改import沿用旧最优性结论 |
| 没有正式教师条件记录入口 | `demonstrations.py::export_demonstrations`；`ActorCriticWork3.select_snapshot(teacher_action=...)`在各层校验并直接选教师动作 | 候选、掩码、查询和前缀对应真实教师选择，未先随机采样再换标签 |
| 可微重放只返回PPO总概率 | `evaluate_action_log_probs(return_components=True)`返回实际发生层；共享`_worker_logits`，集合监督另给`team_set` | 只监督证书允许的层，后移不伪造团队/对齐损失 |
| 人员顺序容易被误当团队质量 | CPU冻结全部团队真子集条件，当前参数递推求已证团队的全部排列概率 | 双人/五人分别核对2/120排列的概率和全部参数梯度；分母包含团队外合法工人，PPO仍用原序列概率 |
| 没有同结构BC入口 | `apal/training/work3/behavior_cloning.py`复用原Lightning优化器、有限梯度守卫；保存W0、验证最佳及末尾权重 | 不增加网络、critic目标、时间头或预测塑形 |
| 固定单实例、pilot绑定扰动 | `runtime_config.load_work3_episode_manifest`；`vector_env`显式换实例重建环境/图/事件追踪；允许`scenario=None` | 清除旧人员、日历、任务索引、完工集合和查询事实；无计划时保留旧行为 |
| PPO总是随机建Actor | `train.load_query_initial_weights`和`paths.initial_checkpoint`，在Lightning优化器创建前严格加载 | 完整继承图、查询与动作层，拒绝错误语义，不调用会跳过投影的旧warm start；新建PPO动量 |
| 评估只接一个工艺实例 | `evaluation.query.evaluate_query_instances`及`--instance-manifest/--split` | 使用清单的工艺、人员和场景，记录C_task、输入身份及峰值分配显存；BC/W0使用真实阶段标记 |

`conf/work3/pretrain_pilot.yaml`是独立实验配置。原`train_query.yaml`、三项研究的生产规则、图结构、state_dict键、后移/对齐、忙碌人员资格和资源窗口均未修改。

导出边界另修正一项实测问题：若清单改了转站模式等条件，旧导出器仍会读取case默认现场。现在核对三个公开文件、场景及环境条件，超出证书现场的覆盖直接拒绝；该反例修复前失败、修复后通过。实际52例原本一致，未受该问题影响。

## 准备包与独立证明范围

实际从Windows Downloads取得：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| APAL_certified_examples_preparation_20261116.zip | 312675 | `1e3e76cd5560881aa8485340200f03ca3e1f1e082965a56cbf603a3dcef33642` |
| APAL_optimal_diversity_preparation_20261117.zip | 4552524 | `41d8e083337511f6505cbf173adfc16b9331d43935f6cde15651fd27feeecfcd` |

两包CRC通过，包内175项与2158项的大小/SHA逐项一致；六个成熟生成器来源SHA另行核对，保存在`GENERATOR_SOURCE_SHA256`。没有把整套研究脚本或旧hidden_dim32输出当作生产训练输入。

主实验固定五站、两架次、`actual_cleared_v0`，共52个受限构造母参数实例，32/8/12分为训练/验证/测试，来源组不交叉；编号、K或教师变体不算新母例。训练为15个结构/团队族、8个顺序族、8个后移族及1个五人族；验证4/2/2、测试6/3/3，没有五人留出样本。它们不是52个工业实例，也不是未见拓扑泛化。

每例先从公开资格、工时、固定前序链及同步规则重推有效下界，再核私有声明，最后在当前原生环境完整执行见证、独立检查物理和费用。52例均达到相应下界。改变公开工时并更新哈希仍会被语义核证拒绝；非法教师动作也被拒绝。

public只含工艺、人员、独立公开参考baseline和运行条件；private保存证书与教师，模型不读取最优值或隐藏映射。baseline分别由公开拓扑/编号、资格和效率规则生成，没有填入私有最优排程。主目标仅纯C_exit：`w_makespan=1`，其他实际费用及两项修订权重全部0，gamma1。lambda_1/2保留原值，但w_p=0使该惩罚不参与费用。原始C_exit、C_task和每例固定H0分开；不能把该证书推广至默认混合费用目标。

另有9个开发实例，覆盖留站/后移、对齐0/1、运行中揭示等条件。名义时钟的揭示族单列，未混入actual训练；两个不同未来R的揭示前图、状态、候选、掩码、查询完全相同，揭示后才改变。条件证书按各自前提核对，没有统一套用静态下界。

## 真正生效的扰动补验

使用真实283工艺的五站、两机小夹具，严格加载上轮兼容QUERY诊断权重；共用合法前缀正常完成`0_15`，之后`reset=False`自由执行QUERY后缀。此次受控验收不是工业2830任务重复长评估。

| 目标0_18在事件前的状态 | tau与恢复R | 正变化与承诺 | 名义/受扰C_exit |
|---|---|---|---:|
| READY、尚未指派 | tau=0.5457256030603916，R=25 | material_ready_time 0→25；实际开工25，不早于新条件 | 6.72316483417 / 31.17477107596 |
| RESERVED、已指派未开工 | 同上 | 原开工10.54572560306→25；原站位1（0-based）、团队[18,65]、align1保持；generation0→1，旧start事件失效 | 16.73493460497 / 31.18920900191 |

两例actual_hit均含目标，且释放确有正增量，不只检查“命中”字段。四条对照后缀均10/10完成、全部飞机离线，日历无重叠、目标只开工/完工一次，可行性与全程/增量账本均通过。RESERVED受扰支路C0=0.001865265586087081、C1=0.11593474035253723，后续奖励和=-0.11406947476645014。这里沿用生产验收的混合账本，不冒称为主学习实验纯C_exit费用。

原始四臂报告、事件、动作、前缀与登记时刻位于本轮目录`artifacts/{reports,traces}/disturbance/`。没有改变同刻先开工规则、撤回团队、强制后移或替换策略。

## 数据覆盖与训练规则

训练1578条原生记录，1352条有监督；验证398条、342条有监督。没有把测试教师记录送入优化。训练中有多个合法选择的层计数：工序1183、团队1046、分支16、对齐1336、派工/推进798；这些层计数重叠，不能相加当独立样本。对齐有两种合法动作不代表每次后果都不同，主训练教师均align0；有意义的align1只在独立开发条件例核验。

原训练任务需求1/2/3/5人分别1196/232/132/10，明显偏单人。按当前283物理需求127/147/5/4校准BC采样权重为0.62/3.0/0.18/1.89；只对团队监督记录生效，其余权重1，保留实例长度控制。不能称原始数据已经“双人为主”。实际BC团队样本：seed42单/双/三/五=199/279/9/10，seed43=362/412/10/8；双人分别56.1%、52.0%。

损失只读真实label_scope；完整团队汇总所有合法人员排列，但只监督这条已证物理团队，不宣称穷尽所有最优团队。强制推进没有学习标签。BC每决定平均有效层并控制实例长度；需求校准后不再称严格全实例均匀目标。无辅助critic目标，图/查询/动作参数有梯度并实际更新，critic专属参数保持。

12条双人多选择开发记录经25次真实GPU更新，验证团队集合NLL约0.082655→0.001028，证明通路可拟合，不替代闭环收益。双人/五人集合概率及全参数梯度对照容差为FP32 atol1e-6、rtol1e-5，原PPO总概率/价值/熵保持。

正式参数在看测试结果前冻结于`configs/pre_registration.json`：seed42/43、BC最多200步、每25步验证、4次不改善停止，以最小验证NLL选模型；两个PPO均2048环境转移、rollout64、epoch1、batch8。每seed共用同一W0和一次BC。PPO新采样，不混入示范，不继承BC优化器；成功目标100000避免提前停止，但实际有完整episode。所有模型选完后才评估测试。

## 真正更新与费用

| 阶段 | 实际/选中BC更新或PPO更新 | 在线转移 | 完整episode | 内部选人次数 | 报告内/外层秒 | 峰值分配显存MiB |
|---|---:|---:|---:|---:|---:|---:|
| BC42 | 125 / 25 | 0 | — | — | 76.88 / 85.84 | 74.26 |
| 从头PPO42 | 256 | 2048 | 37 | 2511 | 195.28 / 203.59 | 79.87 |
| BC-PPO42 | 256 | 2048 | 40 | 2602 | 191.42 / 200.22 | 79.77 |
| BC43 | 200 / 150 | 0 | — | — | 118.63 / 127.22 | 74.38 |
| 从头PPO43 | 256 | 2048 | 26 | 1792 | 181.42 / 189.78 | 79.73 |
| BC-PPO43 | 256 | 2048 | 40 | 2724 | 197.34 / 205.72 | 79.72 |

四PPO的成功/尝试更新均256，AMP跳步0、fit32次；每组原守卫核256条记录（32轮各首批8条），最大重放误差0，不宣称全部2048条均重放抽查。loss、梯度、参数有限，保存Lightning optimizer/RNG/计数。两worker均CPU spawn、CUDA未初始化。BC初始全部张量等于同seed W0；严格加载来源SHA/指纹相符。

两PPO共同episode序列前缀和公开哈希一致；策略不同导致进入后续episode的时刻/数量不同，不能把episode数当独立母例数。PPO检查点评估资格自然为true、research_result_eligible仍false；BC及W0保留各自false身份。BC检查点未独立保存Git字段，该来源由启动记录/预登记追溯，不补造原元数据。

一次新目录冷生成与核证52例外层9.67秒，211份公开输入与清单SHA均与原数据相同；这不是追补最初未单独计时的开发耗时。教师数值导出内部训练17.84、验证3.41秒，不含解释器启动。两seed BC外层合计213.06秒；两seed从头PPO393.37秒，BC后PPO405.94秒，后者还需加BC和教师处理成本。16次模型评估外层共312.67秒，额外条件概率诊断外层101.55秒。

计时包含各自边界的启动/验证/保存成本，不同边界不混加成精确总吞吐；宿主不宣称独占，也不将这些单次成本称为稳定加速。如此有限成本下已没有总体收益，当前不支持仅以“省在线探索”作为扩大理由。

## 留出闭环结果

gap=(C_exit−证书最优值)/证书最优值。表中为12个测试母实例的平均百分比，越低越好；最优绝对容差1e-9。160条模型评估（20个留出母例×4状态×2seed）及20条固定公开规则评估全部自然完成、独立可行、无约束违规、账本一致，没有截断或负gap。它们不是180个独立工业样本。合法性受环境硬规则保护，不能归因于BC学会约束。

| seed | 未训练W0 | BC-only | 从头PPO | BC-PPO |
|---|---:|---:|---:|---:|
| 42 平均gap | 126.106% | 129.860% | 41.174% | 64.812% |
| 43 平均gap | 40.547% | 182.880% | 87.116% | 139.988% |
| 42 最优例数/12 | 0 | 1 | 1 | 1 |
| 43 最优例数/12 | 1 | 1 | 0 | 1 |

原有`HeuristicAgentWork3`固定公开规则为40.056%、1/12最优；没有因神经结果不好而换规则。两seed平均gap：W0 83.326%、BC156.370%、从头PPO64.145%、BC-PPO102.400%。只有两个训练种子，不给统计稳定优势结论。

各测试族平均gap（结构/团队6母例、顺序3、后移3）：

| seed/状态 | 结构与团队 | 工序顺序 | 留站/后移 |
|---|---:|---:|---:|
| 42 W0 | 198.739% | 68.056% | 38.889% |
| 42 BC | 250.553% | 13.889% | 4.444% |
| 42 从头PPO | 66.584% | 27.083% | 4.444% |
| 42 BC-PPO | 122.194% | 10.417% | 4.444% |
| 43 W0 | 62.552% | 32.639% | 4.444% |
| 43 BC | 347.913% | 31.250% | 4.444% |
| 43 从头PPO | 132.218% | 39.583% | 44.444% |
| 43 BC-PPO | 263.519% | 28.472% | 4.444% |
| 公开规则 | 62.264% | 31.250% | 4.444% |

BC在顺序/后移族有局部收益，主要退化集中于资源竞争和必经链联合选择。BC-PPO对BC的改善/持平/退化为seed42的8/4/0、seed43的7/4/1；对从头PPO为3/3/6、6/0/6。BC对W0为8/0/4、1/5/6，从头PPO对W0为12/0/0、0/0/12。没有隐藏无收益或初始模型更好的结果。

条件于教师工序的验证已证团队贪心命中：seed42 W0/BC/PPO/BC-PPO为104/97/104/84（分母264），seed43为104/121/104/106。这只衡量已证团队子集，其他最优团队可能未列入，不是自由闭环准确率。分层概率、人数和覆盖详见`auxiliary_labels.json`；验证NLL改善不能替代上述闭环质量。

## 代表机制

- 受益：seed43/test_007，BC-PPO留站达到30，从头PPO为48。公开条件中F源站工时3，目标站F→G同工人需7，而必经anchor为5；两次错误后移至少增加4，另有迟派等待。原周期序列[5,12,12,5,9,5]解释实际损失，不能将18差值全归因于单个动作。
- 退化：seed43/test_000，从头PPO237.24275724275728，BC-PPO477.85064935064935，证书108。首个可证明损失为必经0_25选择align1，将可行起点0推至2.136；之后0_8选择效率1的工人15而非效率6的10，且与已有预约串行，排定[20.136,38.136)。必经链实际工时6/18/18/36，对应最短6/3/3/6，首周期98.136。14维条件已显示更早更短的合法候选及worker/alignment等待，未发现该例查询漏算或非法动作。

该退化轨迹仅74次决策，从头PPO为136次：更少动作不等于更短生产时间。首个与教师不同的旁支选择本可最优，未据此判错；完整差距是多次联合选择结果，未做单动作因果消融。两个机制例是冻结评估结束后事后选取，不用于调参或总体胜率。四条GPU代表重放的19个科学字段与正式评估精确一致，保留原始动作/掩码/条件/事件/日历轨迹。

下一轮最值得做两件事：先补必经链、团队与对齐在保留承诺的偏离现场中的有证最优后缀覆盖，人数校准不能替代有效选择覆盖；再预登记用验证闭环完成率/C_exit辅助选BC模型的小对照，核对价值误差和更新幅度后决定PPO预算。当前不直接扩大工业长训，也不叠加新API或辅助损失来追求正结果。

## 实际检查与复现

在rag_env从项目根执行的集中检查：

```powershell
python -m pytest tests/work3/test_certified_examples.py tests/work3/test_behavior_cloning.py tests/work3/test_team_set_probability.py tests/work3/test_bc_training.py tests/work3/test_instance_training.py tests/work3/test_scheduling_queries.py tests/work3/test_worker_mask_replay.py tests/work3/test_same_timestamp_events.py -q -p no:cacheprovider --tb=short
```

实验前76 passed、125.90秒；收尾仅受影响导出/评估10 passed、9.41秒，实例/严格加载/真实CPU更新3 passed、43.20秒。后两集与前集有重叠，不累加。另20 passed/54 deselected为实例与既有配置组合，也有重叠。GPU四次正式PPO及两次BC是真更新，不以forward或有限backward代替。

额外旧势函数波次用例`tests/work3/test_work3_training_runtime.py::test_training_preserves_rollout_boundary_potential_and_refreshes_episodes`的夹具`snapshots()`不接生产原有`allow_postpone`参数，在固定68434eb完整模块也复现；原错误只留在工具输出，未单独落盘，因此不将其作为本轮可复核通过项。未改生产迎合、删断言或加skip，未跑全库或重跑既有11项失败。

代表重放的临时脚本首次因漏传episode_manifest而停止，补齐配置后完成4条重放；没有改变模型或生产规则。其首次配置错误未作为一次成功执行记账，原始控制台错误未另行封存。

当前可直接复核本机既有工件，需使用新报告名避免覆盖：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$run = 'runs/work3_pretrain/pretrain_pilot_20261117_01'
python -m apal.evaluation.query --config conf/work3/pretrain_pilot.yaml --checkpoint "$run/checkpoints/bc_seed42.pt" --instance-manifest "$run/data/pilot/test_instances.json" --split test --report "$run/eval/recheck_bc42_test.json" --max-decisions 1000 --set runtime.device=cuda --set runtime.seed=42 --set runtime.max_wall_seconds=120 --set "paths.episode_manifest=$run/data/pilot/train_instances.json"
```

完整复现须选一个新目录，按顺序生成并核证52例，导出CPU事实，生成同seed W0，再分别运行BC和两组PPO。固定参数及所有实际argv在`configs/pre_registration.json`、`*_execution.json`；`artifacts/traces/experiment_driver.py`包含本轮训练/评估/规则/汇总的完整调用，输入权重由下列标准入口建立：

```python
from pathlib import Path
import torch
from apal.runtime.artifacts import create_run_context
from apal.data.work3.certified_examples import generate_dataset
from apal.data.work3.demonstrations import export_demonstrations
from apal.models.work3.actor_critic import ActorCriticWork3
from apal.models.work3.ppo_trainer import PPOTrainerWork3
from apal.training.work3.runtime_config import seed_work3_runtime

# 使用现有run_id函数创建新运行，保留既有结果。
torch.set_num_threads(1)
c = create_run_context(Path.cwd(), experiment_name="work3_pretrain", create_dirs=True)
manifests = generate_dataset(c.run_dir / "data" / "pilot", train_count=32,
                            validation_count=8, test_count=12, seed=20261117)
seed_work3_runtime(20261117, deterministic=True)
export_demonstrations(manifests["train"], c.run_dir / "data" / "train_set_records.pt", split="train")
export_demonstrations(manifests["val"], c.run_dir / "data" / "validation_set_records.pt", split="val")
for seed in (42, 43):
    seed_work3_runtime(seed, deterministic=True)
    actor = ActorCriticWork3(condition_queries=True)
    PPOTrainerWork3(actor, create_optimizer=False).save_checkpoint(
        str(c.checkpoint_dir / f"initial_seed{seed}.pt"),
        metadata={"stage": "initial", "method_profile": "QUERY", "seed": seed,
                  "checkpoint_evaluation_eligible": False, "research_result_eligible": False})
print(c.run_dir)
```

将打印目录作为`$run`，以下为一个种子的完整学习命令，另一种子将42统一换成43；源文件已由上段建立，没有隐藏检查点依赖。PowerShell启动前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`、OMP/MKL线程1；从项目根在rag_env执行。

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m apal.training.work3.behavior_cloning --config conf/work3/pretrain_pilot.yaml --output "$run/checkpoints/bc_seed42.pt" --report "$run/artifacts/reports/bc_seed42.json" --set runtime.device=cuda --set runtime.seed=42 --set "paths.initial_checkpoint=$run/checkpoints/initial_seed42.pt" --set "paths.train_records=$run/data/train_set_records.pt" --set "paths.validation_records=$run/data/validation_set_records.pt"
python -m apal.training.work3.train --config conf/work3/pretrain_pilot.yaml --device cuda --seed 42 --output "$run/checkpoints/ppo_scratch_seed42.pt" --report "$run/artifacts/reports/ppo_scratch_seed42.json" --set "paths.initial_checkpoint=$run/checkpoints/initial_seed42.pt" --set "paths.episode_manifest=$run/data/pilot/train_instances.json" --set "artifacts.runs_root=$run/artifacts/children" --set artifacts.run_id=ppo_scratch_seed42
python -m apal.training.work3.train --config conf/work3/pretrain_pilot.yaml --device cuda --seed 42 --output "$run/checkpoints/bc_ppo_seed42.pt" --report "$run/artifacts/reports/bc_ppo_seed42.json" --set "paths.initial_checkpoint=$run/checkpoints/bc_seed42.pt" --set "paths.episode_manifest=$run/data/pilot/train_instances.json" --set "artifacts.runs_root=$run/artifacts/children" --set artifacts.run_id=bc_ppo_seed42
```

所有权重都保留真实阶段与资格，读取旧输入不覆盖旧模型。检查点保存了权重与各自训练状态，不承诺恢复生产现场的精确断点续训。

## 工件与边界

本轮目录保留公开/私有划分及生成参数、数值事实记录、W0/BC最佳/BC末尾/PPO末尾、解析配置、原始报告、代表轨迹和必要日志。17个小检查点包括开发诊断与严格继承核验所需快照，并非17组独立实验。关键证据：`training_integrity.json`、`paired_analysis.json`、`family_results.csv`、`representative_analysis.json`、`data_reproduction.json`。

已登记并清理6个本轮具名目标：两包解压副本、旧序列训练/验证记录及其JSON、冷重建副本，共2849文件、171512854字节；另移除本轮新模块139319字节可重建pyc。原ZIP已再次核字节，仍在Downloads；旧归档、既有模型及用户文件不动。默认系统pytest临时目录未记录具名归属，没有批量清理。`artifacts/reproduction.zip`为精简复核包，不带模型/大型Tensor记录；源数据、原始JSON/CSV、脚本、SHA和必要复核轨迹在包内，精确限制见包内说明。

未验证：真实LLM、Linux运行与GPU跨平台一致性、完整工业规模/长训、默认混合目标学习收益、五人留出泛化、更多种子及继续BC匹配额外计算的对照。没有宣称无bug、收敛、总体加速或方法优越。
