# APAL：飞机脉动装配线调度

研究一采用同构图进行初始排程；研究二采用异构图进行延迟开工后的计划修复；研究三采用异构图、分层融合、准确调度条件查询和自回归联合决策。三项研究保留各自物理环境和资源配置。

## 安装与目录

在 `rag_env` 中，从仓库根运行：

```powershell
conda activate rag_env
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

已有可用依赖时只执行 `pip install --no-deps --no-build-isolation -e .`，不升级原环境。PyTorch/CUDA 版本由实际驱动与官方安装矩阵决定。`apal` 是唯一生产代码包；安装时不复制数据、配置或模型。当前支持从仓库根运行命令，显式相对路径以项目原约定解释；不承诺安装 wheel 后脱离仓库运行。

| 目录 | 用途 |
|---|---|
| `apal/` | 环境、模型、训练、运行支持、基线和工具；内部保留三项工作的职责边界 |
| `conf/` | Hydra配置组、工作三配置及已核验查询计划 |
| `data/` | 工艺输入、数据协议、清单和固定场景；生成训练池不进Git |
| `scripts/common、work1、work2、work3/` | 按工作归类的工具及少量兼容命令 |
| `tests/` | 分组回归与必要固定fixtures |
| `docs/` | 当前工程导航、研究成果索引、迁移记录及历史恢复说明 |
| `runs/` | 本机新运行输出，默认不进Git |

## 工作三：可直接运行的CPU通路检查

以下命令使用已跟踪的283基准、工艺数据、场景划分和默认离线计划。先产生诊断权重，再加载评估；每次使用新运行ID，重复ID会拒绝覆盖。

```powershell
$runId = 'query_smoke_' + [guid]::NewGuid().ToString('N')
$run = Join-Path (Join-Path 'runs' 'work3_query') $runId
$checkpoint = Join-Path (Join-Path $run 'checkpoints') 'smoke_method_query_weights.pt'
python -m apal.training.work3.train --config conf/work3/train_query.yaml --set "artifacts.run_id=$runId"
python -m apal.evaluation.query --checkpoint $checkpoint --report (Join-Path (Join-Path $run 'eval') 'nominal.json') --max-decisions 8
python -m apal.evaluation.query --checkpoint $checkpoint --report (Join-Path (Join-Path $run 'eval') 'disturbed.json') --scenario-id EARLY_LOW_S0 --max-decisions 8
python -m pytest tests/work3/test_scheduling_queries.py tests/common/test_namespace.py -q -p no:cacheprovider
```

默认CPU FP32、两个spawn worker、8步、一个Lightning主优化器更新。训练与评估遇到步数或时间上限会报告 `truncated`，终局指标留空。短评估可能尚未触发登记的未来扰动；诊断权重没有策略收益结论。

受扰试点仍使用同一网络与入口，将预算和运行模式显式改为：

```powershell
python -m apal.training.work3.train --config conf/work3/train_query.yaml --set runtime.run_mode=pilot --set runtime.successful_batch_target=1 --set runtime.total_env_steps=10000 --set runtime.max_wall_seconds=3600
```

这是一小时上限的试点命令，本轮目录整理不执行长训。工作三只在模板站位和已有合法后移范围内生成安排，不能替代工作一的自由初始分站。运行中普通事件由原事件引擎结算；已提交站位、团队和对齐方式保持原规则。

QUERY不创建时间预测头、时间标签缓存或预测势函数。查询读取已揭示的前序、物料、日历和容量，忙碌但合资格的工人仍能选择；PPO只用采样记录的原始条件与图重放。完整团队枚举的大规模吞吐仍需独立测量。

LLM暂不配置API。默认程序为 `conf/work3/condition_program.json`；可用 `--set paths.query_program=conf/work3/program_material_focus.json` 选择经核验的有限计划。新生成程序仍须语义核验，真实服务未验证。编译入口：`python -m apal.envs.work3.query_program --help`。

继续调用方持有的中段现场使用 `apal.evaluation.query.evaluate_query_trajectory(env, actor, reset=False, max_decisions=100)`；默认 `reset=True` 从零开始。中段评估保留原承诺、事件与费用，并核对后续增量账本。

## 工作一、二

代表训练入口如下；它们会真正启动训练，使用前须确认下述数据与模型存在：

```powershell
python train.py experiment=initial_homogeneous_graphsage_strict
python train.py experiment=reschedule_task_delay_r5_full_x
```

工作一该配置使用 `data/scale_400_800_datasets` 的680训练池；本机另有 `data/generated/initial_283`，不能替代不同配置的池。工作二R5使用 `data/r5_task_delay_v1/manifest.json` 及其绑定的初始模型；通用重调度配置还引用 `results/final_schedule.csv`、`results/reschedule_eval_scenarios.csv` 和初始模型，这些通用输入本机缺失。缺失输入须恢复或显式配置，不能把工作三基准充作替代。

三项工作入口帮助：

```powershell
python train.py --help
python evaluate_model.py --help
python evaluate_reschedule_model.py --help
python -m apal.training.work3.train --help
```

根级三个命令仅委托 `apal` 中的唯一实现。旧 `scripts/work3/train_ppo_work3.py`、`scripts/work3/evaluate_query.py` 作为已公开命令保留薄入口；内部调用统一使用包模块，不依赖同名文件的搜索顺序。

## 运行与恢复

所有受维护默认新输出统一为 `runs/<experiment>/<run_id>/`，内部沿用 `checkpoints/、logs/tensorboard/、eval/、configs/、artifacts/reports/、artifacts/traces/`。显式输出参数继续有效；读取旧checkpoint不会自动把新输出写回旧结果根。

当前Hydra入口先建立本次runs目录；需要恢复时显式提供有效检查点路径，并为新输出选择运行ID。旧解析器在无run_id恢复时先固定旧检查点输入，再建立新的runs输出；显式恢复源优先。显式选择legacy布局仍保留其历史语义。工作三目前保存诊断权重与Lightning优化器/RNG状态，不承诺完整生产现场的精确断点续训。

历史B/C/D/E/F/G仍通过 `conf/work3/train_pilot.yaml` 和 `scripts/work3/evaluate_c_vs_d.py` 访问，原方法字母含义及时间预测实现不变。APCF等扩展保留其数据和权重依赖；没有这些输入时不能宣称可直接运行。

已知旧类工件通过 `apal.runtime.legacy_pickle` 在真实读取点显式迁移，范围仅为已登记的DDQN经验及工作三状态/快照/查询结果类；不创建旧顶级包别名。未登记的整对象pickle须在原固定版本导出标准state_dict，来源不明的pickle不要加载。

[文档导航](docs/README.md) · [数据入口](docs/data.md) · [研究成果及验证边界](docs/research.md) · [实施与验收](docs/maintenance.md) · [唯一路径映射](docs/directory_migration.csv) · [历史材料恢复](docs/history.md)

旧README的完整实验说明按 [dbc33c9固定版本](https://github.com/1ush3n/APAL-Dynamic-2/blob/dbc33c90728249f590a5b4293ec6f237a63160b1/README.md) 查阅，不作为当前命令导航。

2026-11-17已补真实扰动、带证书示例、集合行为克隆和多实例PPO接入，专用配置为`conf/work3/pretrain_pilot.yaml`。两种子等在线预算的有限对照未观察到预训练的总体收益；完整结果、数据范围及复现命令见[预训练试点报告](docs/pretrain_pilot_report.md)。这是独立纯C_exit实验，原QUERY默认混合目标和网络保持。
