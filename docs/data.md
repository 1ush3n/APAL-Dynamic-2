# 当前数据入口

本页只说明受维护入口使用的数据，不改变任务编号、字段、划分或文件内容。路径均相对仓库根目录。

| 研究内容 | 当前入口配置 | 输入与边界 |
|---|---|---|
| 工作一：无扰动初始排程 | `conf/experiment/initial_homogeneous_graphsage_strict.yaml`，继承`initial_strict_common` | 原始模板`data/680.csv`，训练池及清单`data/scale_400_800_datasets/`；以实际Hydra组合配置为准 |
| 工作二：延迟开工修复 | `conf/experiment/reschedule_task_delay_r5_full_x.yaml` | `data/r5_task_delay_v1/manifest.json`及其train/validation/test实例、固定场景和基准；配置还引用已有工作一检查点，启动前须确认文件存在 |
| 工作三：多架次QUERY | `conf/work3/train_query.yaml` | `data/work3/real_283_k10_baseline.json`、`data/283.csv`、`data/work3/scenarios_9class.json`及`experiment_splits/train.json` |
| 工作三：680规模检查 | 同一个QUERY配置同时覆盖两条输入 | `paths.baseline=data/work3/real_680_k10_baseline.json`和`paths.raw_data=data/680.csv`；只换其中一条会混用模板和工艺元数据 |

原始283/680/2338/3182工艺CSV暂保留原名和原路径。人员资格/效率及不同研究的资源配置仍由各自正式加载器解释，不能因为目录归并而混用。QUERY的14维条件契约、图特征和查询程序版本不随文件位置变化。

工作一训练池生成入口为`scripts/work1/build_training_pool.py`；工作二R5数据构建入口为`scripts/work2/build_r5_data.py`。这次仅搬入口，不重新生成训练池，也不改变生成种子。构建器的覆盖选项仍会操作生成目录，整理任务没有执行这些覆盖操作。

`r3`、`r4`和`r5_task_delay_v1`有不同的实验协议，保持各自身份。原manifest中的SHA、相对路径、划分、旧`results/.../best.ckpt`引用不批量重写。新运行统一写`runs`，并不意味着历史模型输入也自动改到`runs`；使用新权重时应显式指定其真实路径。

回归使用的固定轨迹、报告和必要小权重是测试数据；已封存的完整旧实验在[历史资料索引](history.md)中查询。文件名相近、内容重复或当前未被QUERY调用，都不是删除数据和有效检查点的依据。
