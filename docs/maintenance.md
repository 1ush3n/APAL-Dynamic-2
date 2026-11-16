# 目录与命名维护记录

本轮从`a710e3427ab87f01d27fca13238172894de6fa09`开始，独立分支`project-layout-20261116`；原工作三容错/性能审阅分支保留。这里只处理归属、路径和输出接口，不改变网络、物理规则、奖励、精度、随机种子和数据划分。

## 本轮顺序

- [x] 冻结源码起点，检查当前文件与可见任务；仅发现本工具自身Python进程，未停止用户任务。CIM进程查询受宿主限制，进程结果不代表宿主独占。
- [x] 分别盘点Git跟踪、未跟踪、被忽略文件；本机清单保存于本轮外部核验目录。
- [x] 用户选择项目外普通文件夹；1259个资料和历史运行副本逐文件SHA核对，保留原文件名、字节与恢复身份。
- [x] 根据唯一迁移表完成归属和引用更新；原始清单及封存材料不改写。
- [x] 复用原RunContext，将受维护入口的新运行指向`runs/<experiment>/<run_id>`。
- [x] 完成代表入口、权重、确定性轨迹、原生spawn/IPC及一次真实PPO更新核验。
- [x] 审阅实际差异，清理本轮具名临时产物，记录提交和剩余边界。

## 维护规则

维护源码目录保留`envs/core/models/training/runtime/baselines/utils`；配置、命令、输入、回归及工程文档分别归`conf/scripts/data/tests/docs`。不增加`src/apal`包装层，不把不同研究的环境或训练器合并。文件位置与科学接口身份分开：state_dict键、模型/数据协议版本、场景ID和查询API不因目录整理改名。

脚本按真实用途逐步归`common/work1/work2/work3`，不为分类创建空目录。同组目录已经说明工作内容时，文件名不再重复`work3_`前缀。原根级正式薄入口和仍广泛导入的模块保留。只对确有外部调用/序列化需要的入口考虑兼容壳，不保留两份实现。

受维护新运行复用`runtime/artifacts.py`的RunContext和既有ID/时间戳函数；内部保留`checkpoints`、`logs/tensorboard`、`eval`、`configs`、`artifacts/reports`、`artifacts/traces`。显式输出参数继续有效，旧运行路径与原始报告不批量改写。既有本地`outputs/results/checkpoints`不是本轮临时目录，不做全库清理。

材料按[历史索引](history.md)查找。逐文件复制和校验完成后才处理项目内对应副本；复制、停止跟踪、删除本地副本分别记录。Git历史不重写，不能把当前树减少的字节说成克隆历史立即变小。

临时核验脚本、清单、对照权重和日志统一置于仓库外；最后只留下可复核的必要证据。本机原有`research_diagnostics/`和其它尚未确认用途的未跟踪/忽略文件保留，不用清理命令掩盖来源不明的内容。

## 2026-11-16实施结果

整理代码提交：`712dc62958d4681acb8bccd8d1a734874601e4e5`；本节与迁移表随后以独立文档提交保存。没有合并或改写原`work3-query-migration-20261115`分支，原分支仍为`a710e34`。没有将目录整理计作前一轮性能收益。

### 改动与科学边界

| 原位置或行为 | 当前归属/行为 | 原因与验收 |
|---|---|---|
| docs混放完整开题资料、文献、旧稿；旧根审计与benchmark结果散落 | 普通资料目录保存原文件；docs只留当前导航、数据入口、维护/历史索引、已有迁移报告及唯一CSV表 | 1259份普通副本、1,049,718,664字节，逐文件SHA核验；不重打包论文/PPT，不改旧报告 |
| `work3_4060`混合工具/配置/数据/运行 | 19工具→`scripts/work3`；8配置→`conf/work3/local_*.yaml`；12输入→`data/work3/local_diagnostics`；必要固定参照→`tests/fixtures/work3/local_history` | 删除旧位置前910物理文件、367,544,765字节全部再次与普通副本逐SHA一致 |
| `diagnostics/next_suite.py`、`experiments/main_screen/screen_models.py` | `scripts/common/next_suite.py`、`models/main_screen.py` | 更新真实导入/测试，不留第二份实现；筛查模型state_dict键与计算不改 |
| `benchmarks/baseline_283_v0.json` | `tests/fixtures/baseline_283_v0.json` | 固定回归标尺保留，更新工具默认save/check路径 |
| `training/work3_*.py`四模块 | `training/work3/{lightning,runtime_config,vector_env,r3_e04_audit}.py` | 更新脚本、模型、测试与spawn模块路径；包初始化不引入循环依赖 |
| 根scripts中的四个明确入口 | `scripts/work1/{build_training_pool,train_main_screen,evaluate_main_screen}.py`和`scripts/work2/build_r5_data.py` | 引用及直接执行时的根路径同步更新，4个真实CLI帮助均正常 |
| 三个重复工作前缀的测试名 | `test_transfer_semantics.py`、`test_same_timestamp_events.py`、`test_worker_mask_replay.py` | 纯重命名，内容及断言保留；旧报告通过CSV查当前节点 |
| 维护配置/工具分散向results、outputs、docs或models写新产物 | 共享RunContext：`runs/<experiment>/<run_id>`及既有checkpoints/logs/eval/configs/reports/traces角色 | QUERY CLI和直接API、历史本地启动器、综合验证及三处报告生成器接通；显式输出参数保留 |

必要历史参照123文件、24,279,055字节，包括7个二进制共约13.34MB。它们用于固定契约回归，不表示旧检查点都符合当前正式方法profile。`.gitignore`仅为这7个具名fixture设例外，新训练权重仍默认忽略。原始JSON/CSV、数据字段、数组顺序、数值类型、float位模式及版本身份未为整理而改写。

新增两条窄范围`.gitattributes`规则，仅冻结新固定输入和fixtures的原字节；保留原CRLF并告知Git其行终止含义，不统一全库行尾。重新读取暂存索引后，136个固定输入（123参照、12输入及1个基准）与本机实际字节逐SHA相同；其中135份亦与普通副本匹配，另1份基准与原Git blob相同。原模板及旧清单没有为通过守卫而转换换行。

QUERY默认`paths.checkpoint_dir/report_dir=null`由RunContext解析；`--output/--report`及显式目录继续有效。解析真实ID和位置后才计算resolved指纹；清单包含Git、设备、精度、seed、程序指纹、图/数值输入契约、数据指纹和实际更新/终止状态。首次运行目录独占创建，重复ID拒绝覆盖。审计用真实run_id，Lightning输出归logs。共享旧Config接口保留，未把工作一二的ModelSpec强塞给QUERY。

当前Git维护顶层由16个目录减为12个，根文件24→17（增加标准`.gitattributes`，迁出8个旧报告/结果）；跟踪文件2015→1187。当前树展开Git blob总量由477,676,990字节（约455.55MiB）降至约59.8MiB；文档提交会小幅增加文本量。**这不是Git历史或克隆下载量减少**，没有重写历史。用户原有ignored输出、PPT交付目录、`research_diagnostics`等仍在本机原位，不能声称本机所有文件夹都只剩12个。

[唯一迁移表](directory_migration.csv)记录176个当前路径映射，并列原Git blob、普通副本SHA及本机当前SHA；本机未确认文件只列分组，完整私有清单留在外部。Git自动相似度重命名可能把相近的两份YAML显示成交叉匹配，应以CSV的原名映射及8配置逐字段核验为准，不用Git的猜测改变科学身份。

### 实际验证

环境：Windows、`D:\Conda\envs\rag_env\python.exe`，Python3.11.15、PyTorch2.12.0+cu132、PyG2.7.0、Lightning2.6.5、OmegaConf2.3.0；CPU FP32，主/环境线程1。未改依赖，未用GPU或连接服务器，未停止用户任务。下面各集合独立记录，重叠检查不相加成更大通过数。

- 当前入口/事件/重放/筛查模型/R5组合：**71 passed，177.76秒**，无skip。
- 历史工具11文件：**113 passed、11 failed、0 skipped，67.39秒**。11节点均在隔离的原`a710e34`源码、原测试/配置、相同物理数据和历史文件上实际复现：**11 failed，21.73秒**。10项是原C检查点profile不匹配，一项是原工具对外部tmp路径执行`relative_to(REPO_ROOT)`；节点及异常原文逐项对应，仅该位置异常中的必然变化路径按声明归一。没有松断言、加skip或改守卫。
- 两项代表Hydra配置实际组合：工作一`homogeneous_graphsage_strict/operation_station_worker/reschedule=False`，工作二`hetero_gat/operation_station_worker/reschedule=True`；新runs根正确，数据/原manifest/基准模型输入引用保留。未创建运行或启动工作一二训练。
- 四个迁移后CLI及根`train.py --help`：全部退出码0。综合验证CLI帮助、路径回归及受控C/D参数probe通过；没有执行其完整历史长流程。
- 路径/API/清单的集中回归分阶段运行，包含重复ID防覆盖、显式覆盖、最终truncated清单及真实ID。最后新增输出字节断言先红后绿：新resolved YAML显式写LF，3种输出方式**3 passed、67未选入**，原始字节SHA均等于声明指纹。此前受控probe明确标记没有训练，不当作真实更新。

主组合命令如下；实际使用rag_env，设置`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONDONTWRITEBYTECODE=1`，并用仓库外本轮目录作为`--basetemp`：

```powershell
python -m pytest tests/work3/test_transfer_semantics.py tests/work3/test_same_timestamp_events.py tests/work3/test_worker_mask_replay.py tests/work3/test_scheduling_queries.py tests/test_full_joint_next_diagnostics.py tests/test_dual_attention_context_screen.py tests/test_reschedule_r5_protocol.py tests/test_worker_pointer_v2.py::test_v2_experiment_config_is_isolated_and_explicit -q -p no:cacheprovider --tb=short
```

历史集合使用如下文件，不修改原11项失败以制造全绿；单节点基线命令及原始日志见证据包：

```powershell
python -m pytest tests/test_alignment_bc_robustness.py tests/test_bc_ppo_retention.py tests/test_critic_calibration_retention.py tests/test_c_reproduction_diagnostic.py tests/test_ppo_retention_repeat.py tests/work3/test_baseline_a_evaluation_contracts.py tests/work3/test_budget_evaluation_contracts.py tests/work3/test_nominal_anchored_clock_contracts.py tests/work3/test_rule_baseline_contracts.py tests/work3/test_shaping_boundary_contracts.py tests/work3/test_trajectory_verifier_loopholes.py -q -p no:cacheprovider --tb=short
```

### 真实更新及确定性对照

使用正式`train_query.yaml`，固定seed42、CPU FP32、2个spawn worker、8步、一个完整Lightning/PPO更新。整理前`aa_283_a1`与整理后`after_final`：fit=1、成功优化=1、尝试=1、AMP跳过=0，replay样本8/误差0；138个参数张量确实更新。初始/最终Actor、完整Lightning训练状态、optimizer及Python/NumPy/Torch RNG精确相同；59条有序边界摘要全部一致，比较浮点/张量字节，不用allclose。没有比较整份.pt文件SHA，也没有缓存旧embedding绕过重放梯度。

用整理前真实诊断权重通过新模块加载，实际调用`evaluate_query_trajectory`完成受扰两机/五站10工序：整理前后20条动作/环境边界及完整结果精确相同，`success=true`、全部离线、独立可行性和费用账本通过。边界包含图/原始条件和原因、前缀/掩码、动作、日历、完整状态事实、事件堆/代数及reward/费用。所属状态的反向指针`_state_ref`不展开；Tensor别名、store父引用及未捕获内部对象不在本次摘要证明范围。

启动器迁移曾真实触发父/子重复占run目录的`FileExistsError`，已经修复：子进程独占目录，父进程通过manifest中的PID/运行ID/命令核对后归入console/launch记录，不取消防覆盖。真实CPU `smoke-c`两步、batch2亦完成一次更新、样本2/replay误差0、AMP跳过0，并保存Actor及Lightning优化器状态。两类烟测均诚实标记`truncated/decision_limit`、成功生产批次0；不据此评价策略收益。

训练对照完成后另修一行新配置文件的LF输出；原`after_final`证据保持CRLF，没有手工回填。该次报告/manifest的canonical YAML指纹一致，新文件原始字节一致性由上述3项回归单独核验，不把旧文件原始SHA说成已匹配。

### 清理、资料查找与未验范围

旧资料现位于普通目录`D:\OneDrive\研究生\归档\APAL-Dynamic-2`，位置与单文件恢复方法见[历史索引](history.md)。现有开题报告/PDF/PPT可直接打开；原先已有ZIP保留原格式。完整逐文件清单SHA为`db049c1c21935a819fcd46a45a3370a8bcdae2d278e49a79d60909ed00e12813`。

本轮核验脚本、命令、输入身份、原始日志、输出字节样本及对照摘要保留在普通资料根的`verification/APAL_directory_20261116_evidence.zip`；不包含论文材料、大权重、基线源码副本或完整探索trace。普通资料本身没有打成新压缩包。隔离原版、诊断模型、pytest树及一次性展开目录在证据复读后按具名路径清理，原用户输出和研究目录未动。

未验证GPU、本轮Linux原生IPC、收敛/收益及全部历史长实验；没有做全库全绿声明。11项既有问题、原工作区固定源SHA/CRLF守卫边界如实保留。根级可导入模块、数据r3/r4/r5路径及其原始manifest按本方案较后阶段规则原位保留；旧公开工作三入口名也保留，避免只为缩短名字增加兼容壳。目录整理不等于再次设计APAL方法或完成全部科研实验。
