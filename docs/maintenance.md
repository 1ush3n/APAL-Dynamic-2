# 当前目录与实施记录

## 2026-11-17：单一apal包、本机资产及入口整理

起点为 `dbc33c90728249f590a5b4293ec6f237a63160b1`，独立分支 `apal-namespace-20261117`。资产提交 `482d8ba68f59ac9f686546c7f3adcf1f0e28b4a9`；代码提交 `215caace0db0c6ce8a8ea6994babfeeef051fd81`；本报告、导航与唯一迁移表以随后文档提交保存。没有合并main、改写历史、删除其它分支/worktree/stash或浅克隆替换工作目录。GitHub默认main没有自动变更，审阅应选择本分支。

本轮不实施最优示例预训练，不改图结构、网络层、参数键、动作顺序、约束、人员忙碌资格、事件优先级、奖励、精度、超参数或数据划分；没有把目录改动记作性能收益。下文旧日期章节保留原阶段身份，现行规则以本节和根README为准。

### 目录与实际改动

| 原行为 | 新行为 | 文件与原因 |
|---|---|---|
| 7个实现目录及根实现混合导入 | 唯一生产包apal，内部原职责保留；不加src层，不合并三套环境/训练逻辑 | `apal/{core,envs,models,training,runtime,baselines,utils}`；根实现迁入apal，训练编排位于`apal/training/entry.py`及`work3/train.py`，评估位于`apal/evaluation` |
| 多处按文件层级猜项目根 | 轻量`apal.project.PROJECT_ROOT/resolve_project_root`，默认验证conf/data，允许显式项目根 | configs、环境、Hydra、运行路径、基线及工具调用同步更新；相对路径语义不因新包层级漂移 |
| 同名评估脚本可能导入自身 | 显式导入`apal.evaluation.initial/reschedule` | 根与分组薄入口统一实现；筛查评估修改真实实现模块的模型类，不修改包装器副本 |
| old模块名写入少数可信pickle | 仅真实旧类在读取点显式映射到apal类 | `apal/runtime/legacy_pickle.py`：DDQN经验1类、工作三状态8类、快照3类、查询结果3类；没有旧顶级包别名或torch全局补丁，不是供不可信pickle执行的安全沙箱 |
| scripts根106文件、tests根112文件 | scripts根0文件，tests根只留conftest；scripts按common/work1/work2/work3，性能工具与APCF在所属组内 | 64个危险/一次性历史脚本原字节外部保存；所有科学回归及fixtures保留，helper及借用构造器只迁一份 |
| 多个维护工具默认写results或源码目录 | 复用同一RunContext，默认进入runs，显式输出仍尊重 | 12工具集中调整；公共logger及旧GA导出也接同一运行。12工具包括sensitivity、三机会审计、DDQN replay/GPU汇总、四manifest评估、joint筛查和消融编排 |
| 旧解析器resume无run_id时连新输出也沿用旧根 | 先固定旧last.ckpt输入，再创建新runs输出；显式恢复路径优先 | `apal/runtime/configuration.py::initialize_training_config`；仅输入/输出职责调整，存在性与严格模型恢复校验继续由原路径执行 |
| 时间头API默认覆盖源码目录中的best文件 | helper默认只返回模型；CLI默认独立runs保存，显式路径/显式None含义清楚 | `train_time_head(checkpoint_path=None)`不保存；`scripts/work3/train_time_predictor.py`负责默认保存位置。APCF预训练CLI也从根checkpoints改为独立run；未训练APCF或恢复QUERY时间头 |
| pytest固定根临时目录并无条件创建空目录 | 默认使用pytest系统临时目录；核验均显式仓库外basetemp | 删除pytest.ini旧basetemp及conftest无用mkdir；没有修改断言或跳过规则 |

打包只增加`pyproject.toml`，在原rag_env中用 `pip install --no-deps --no-build-isolation -e .` 登记；不升级依赖。发现范围包含apal和scripts命令工具命名空间，后者让直接文件执行时的工具互导不依赖PYTHONPATH或新增sys.path补丁；生产实现仍只有apal。采用[Setuptools可编辑安装](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)，当前支持从仓库根运行，不承诺脱离conf/data的wheel部署。

保留的有限薄入口：根`train.py/evaluate_model.py/evaluate_reschedule_model.py`；工作三原训练/QUERY评估/轨迹收集命令；分组后的两评估包装及四个已有数据/排程命令。没有为其余每个旧源码路径创建壳。旧`train_lightning.py`实现变为`apal.training.entry`，工具命令同步指向真实位置。旧checkpoint参数键不随文件路径改名。

Git维护树根目录12→6（apal/conf/data/docs/scripts/tests），根文件17→9；scripts158→94，根平铺106→0；tests301→302，根平铺112→1。测试数量变化来自新增集中回归及preflight工具归位，不是删除科学断言。源码提交当前树1141文件，本文档提交增加研究索引后为1142文件；自定义目录角色只有6个，runs是额外的本机输出根，.git是标准历史库。起点Git blob展开62,741,668字节，源码提交62,182,563字节；这些不是磁盘占用或历史克隆大小，文档提交会增加少量文本。r3/r4/r5等真实数据协议保持原位。

### 本机资产、研究材料与清理

本轮普通目录仍为 `D:\OneDrive\研究生\归档\APAL-Dynamic-2`。先完整复制、逐文件SHA复核，再确认源路径/mtime/大小/SHA未变，最后逐文件移除登记副本及已空目录；没有全库clean或递归删除未登记目录。

- 839文件、624,047,083字节：旧PPT、outputs/results、saved_local_run_283、两组研究诊断；另361文件、378,999,016字节：旧性能JSON、历史探针、混合测试及手工烟测产物。共1200文件保留普通副本，其中研究诊断46模型和补充产物140模型均完整保留，不把它们当垃圾或全部声称可重建。
- 64个历史脚本保留原文件名及字节；旧cleanup硬编码其它项目路径且默认执行、旧批次自动关机，均未运行。compute_statistical_metrics无维护消费者，原44,446字节/SHA也与固定起点物理记录一致。
- 两份旧参考权重移至 `runs/work3_legacy/imported_dbc33c9/checkpoints/`，附configs/source_manifest；不是QUERY权重，未来新时间头不会覆盖它们。原有效历史运行仍在 `runs/initial_schedule_283/`。
- `data/generated/initial_283`仍17文件、154,322字节；16CSV与manifest逐项SHA一致。工作二通用final_schedule/reschedule_eval_scenarios及APCF资产本机原本缺失，未伪造替代；R5仍按原manifest与初始模型依赖读取。
- 当前确认Word/PPT/图源未在本机找到，用户明确同意本轮跳过；旧PPT不能提升为当前开题稿，不建立空proposal目录。研究准备成果仅作[带来源的索引](research.md)，未下载或复验其资料库包。
- 51个具名旧pyc及已确认空目录已清；本轮产生的缓存、诊断模型、外部pytest目录和唯一固定原树在保留关键证据后具名清理。其它worktree、Git历史和旧本地分支未动。

完整资产校验身份与按文件恢复方法见[历史索引](history.md)；唯一工程映射见[directory_migration.csv](directory_migration.csv)。没有把本机私有逐文件清单提交GitHub。原先1259份普通资料副本及前两轮证据包没有重压或删除。

### 实际核验

Windows，严格使用 `D:\Conda\envs\rag_env\python.exe`；Python3.11.15、Torch2.12.0+cu132、PyG2.7.0、Lightning2.6.5、OmegaConf2.3.0、NumPy2.4.6。CPU FP32，主/环境线程1，实际两个spawn worker；无GPU、服务器或真实LLM调用。进程可见性受CIM权限限制，不宣称宿主独占。以下重叠集合分别记账，不相加成独立总数。

| 实际检查 | 结果与范围 |
|---|---|
| 核心迁移组合 | 81 passed，178.07秒；转站、同刻事件、查询/重放、工作一模型与工作二R5、根定位/有限旧类/旧契约及路径 |
| 配置与resume集中回归 | 整文件46 passed，12.83秒；新2参数resume反例在修复前均失败，修复后旧输入保留、新输出进入runs |
| 维护工具路径及旧回归 | 44 passed，20.62秒，其中13新路径分派、31关联旧回归；旧默认路径先红，显式覆盖保留 |
| 最后路径/旧头兼容组合 | 62 passed，16.69秒；包含namespace、配置、logger、现有时间头显式保存训练检查及G3初始化约束，不是QUERY恢复时间预测 |
| 直接脚本R1与路径回归 | 6 passed，37.87秒；修复安装范围只含apal而导致工具互导失败的问题 |
| 最终全体收集 | 1618 tests collected，27.96秒，无收集错误；不等于执行1618项通过 |
| 历史11文件组合 | 113 passed、11 failed，73.30秒；仍为10个C检查点profile不匹配、1个外部路径relative_to旧边界，未删断言或加skip |
| 七次直接CLI帮助/两项Hydra组合 | 全部成功；两评估包装各验证root在后位和清空PYTHONPATH。工作一同构严格联合动作/reschedule false；工作二异构联合动作/reschedule true。未启动工作一二训练 |
| 固定数据与源检查 | 634份data/fixtures共55,418,359字节的原SHA保持；原科学断言/跳过规则保持；403活跃Python实际compile及git diff --check通过 |

核心命令（仓库根，rag_env，仓库外basetemp）为：

```powershell
python -m pytest tests/work3/test_transfer_semantics.py tests/work3/test_same_timestamp_events.py tests/work3/test_worker_mask_replay.py tests/work3/test_scheduling_queries.py tests/work1/test_full_joint_next_diagnostics.py tests/work1/test_dual_attention_context_screen.py tests/work2/test_reschedule_r5_protocol.py tests/common/policy/test_worker_pointer_v2.py::test_v2_experiment_config_is_isolated_and_explicit tests/common/test_namespace.py tests/common/runtime/test_legacy_check_contract.py tests/common/runtime/test_pathlib_compat.py -q -p no:cacheprovider --tb=short
```

实际命令额外传本轮外部`--basetemp`；13新工具分派与6关联文件、历史11文件及其全部参数保存在外部证据JSON和日志。README给出当前三项工作入口、新QUERY无扰动/受扰评估及最短检查命令；缺失资产的位置与身份在[data.md](data.md)，不把需模型的命令称为无条件可复制运行。

### 真实更新、完整调度与模型边界

整理前、整理后API、整理后实际 `python -m apal.training.work3.train` 三次各从相同seed42/数据/参数开始：2worker、8步、一次真实rollout→GAE→PPO损失→backward→Lightning主优化器step→checkpoint。每次fit/成功/尝试均1，AMP跳过0，重放样本8/绝对误差0；无时间头/标签监督更新。loss=-0.019621893763542175，grad norm=11.951152801513672，均有限；138参数张量实际更新。三次不是顺序累积训练。

API前后59条有序边界全部精确一致；初始/最终Actor、完整Lightning训练状态、optimizer及RNG逐字节摘要一致，未冻结旧embedding或使用allclose。实际CLI保存状态也与整理前精确一致。最终Actor摘要 `d7f384aeea6e749216644105fa4c041347c2584f79b4f41cb071cac22c6f9b76`。两步历史C启动器亦成功完成一次完整更新，重放2/误差0，父子运行目录归属核验通过。所有烟测诚实标记decision_limit/truncated、成功生产批次0，不伪造工期终值。

同一真实诊断权重完成两机/五站/10工序受扰调度：20条有序动作/环境边界与完整结果精确一致，全部离线、费用及可行性核验通过。查询数值/原因、候选/前缀/掩码、事件、承诺、日历与reward/费用包含在边界摘要内；不展开_state_ref反向指针，未单独证明全部Tensor别名与PyG store父指针的合同。比较器的一ULP、候选顺序和原因负控均能识别差异。已有reset=False回归通过实际模块调用保留中段现场，没有AST摘函数代替。

实际CLI用新权重评估283的名义与预登记扰动8步前缀，均正确截断、空终局工期、完整/增量费用账本一致；不据此声称扰动已触发或批次已完工。旧 `data/work3/val_trajectories.pt` 原字节工件通过有限类迁移实际读取16条；其SHA为 `ff3e30ca54c0cf15ff5e7380fc2f4f60f56badae4e326d1ad8f02aa570abaad2`，没有回填或改写原件。

两份本机旧Lightning best（194011与194402）可读format v2，策略568键、优化器568项/2组、agent/scaler有保存，scheduler为空，未见独立RNG字段；当前严格Agent加载均缺相同60个baseline_team参数。用206个原文件重建唯一固定dbc33c9临时原树，实际导入原完整helper，也复现同一严格错误。没有补随机权重/strict=False，不把可读取误写成精确恢复成功。该边界与11个历史测试失败分别登记。

开发期曾发现并修复配置键误当模块字符串、复合代码路径叶名误改、动态__import__返回层级、future导入顺序及直接工具包导入问题；修正后完成上述复验。安装首次还遇到宿主临时目录字符/权限问题，改本轮临时目录后仅登记editable包，依赖未变；原始失败日志保留，不计成科学测试通过。

### 复现证据与未验范围

本轮最小证据包保存于普通资料根 `verification/APAL_namespace_20261117_evidence.zip`；含前后有序摘要、真实CLI/测试原文、输入身份/配置、可复用核验脚本及SHA清单，不含大权重、全部探索trace或原树副本。包内README说明从新空临时目录复核，避免覆盖原证据。证据包62项、369,953字节，SHA256为 `5468dc844b71df532922047a778e1b2e2ed27a7989f6f823a3c277fb691330f3`；相邻.sha256文件可复核，这不是论文资料压缩包。

未验证GPU、Linux原生IPC、长训/收敛/策略收益、全部1618项执行、真实LLM服务、用户其它完整对象pickle及旧Lightning架构迁移。未实施预训练或新的性能候选；原数据协议、旧时间方法及三项研究有效实现仍保留。只有本机可访问目录实际完成整理，其它clone/worktree及服务器未处理。

---

## 2026-11-16已封存阶段记录

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
