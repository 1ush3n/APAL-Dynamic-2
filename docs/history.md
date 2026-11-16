# 历史资料、运行证据与恢复

本轮整理从 `a710e3427ab87f01d27fca13238172894de6fa09` 开始。当前代码与运行命令见[仓库说明](../README.md)，工作三方法及实际验证见[迁移记录](work3_query_migration.md)。旧审计、设计稿和测量结果保留其原始日期、来源提交及适用范围，不能作为当前代码的缺陷清单或性能结论直接重用。

[逐项路径映射](directory_migration.csv)是本轮源码、配置、夹具与历史材料的唯一迁移表；普通副本的逐文件校验身份由下述 `archive_manifest.json` 保留。

## 项目外普通目录

本机资料目录为 `D:\OneDrive\研究生\归档\APAL-Dynamic-2`，保存普通文件，不依赖压缩包才能读取。目录中的 `archive_manifest.json` 是逐文件清单，包含原相对路径、来源提交、原文件类别、字节数、SHA256、普通副本绝对路径及核验结果。清单登记1259份文件；对应副本在移除原路径前逐项核验，当前实现所需源码、数据与小型回归夹具另留项目内。

本轮没有把这些材料重新压成一个大包。原先就存在的原始ZIP仍以原格式保留；Word、PDF、PPT及展开资料沿用各自文件名和层级，可以直接查阅。

| 原位置 | 普通副本位置（相对上述资料目录） | 用途 |
|---|---|---|
| `docs/开题报告/`、`docs/archived/`、`docs/superpowers/`及docs根原始资料包 | `materials/a710e34/`下的原相对路径 | 正文历史版本、批注、文献、PPT、图稿、制作源与旧实施计划 |
| 根 `CODEBASE_AUDIT.md`、`PERFORMANCE_AUDIT.md`、`REFACTOR_PLAN.md` | `materials/a710e34/`下的原相对路径 | 特定旧提交的审计与计划 |
| 根 `benchmark_*.json`、`gpu_utilization_283.json`及 `docs/section9_comprehensive_validation_report.json` | `runs/a710e34/`下的原相对路径 | 历史测量与旧方法验收证据 |
| `experiments/work3/`及清单登记的旧本地运行材料 | `runs/a710e34/`下的原相对路径 | 原始报告、模型、轨迹及复现材料 |

清单本身的SHA256为 `db049c1c21935a819fcd46a45a3370a8bcdae2d278e49a79d60909ed00e12813`。原始报告内的旧路径和解析配置保持原样，实际保存位置由清单映射；这些字段不表示当前运行仍向旧目录写入。未登记的用户文件与 `research_diagnostics/` 不在本轮移除范围。

GitHub审阅者可从[整理前提交](https://github.com/1ush3n/APAL-Dynamic-2/tree/a710e3427ab87f01d27fca13238172894de6fa09)读取当时已跟踪的文件；本地ignored文件仅在普通资料目录中保存。外部副本不随新代码重复提交。

## 留在代码中的内容

- `scripts/common/next_suite.py`：诊断任务编排，被命令入口和回归测试实际复用。
- `models/main_screen.py`：筛查模型定义，被训练、评估和测试使用；模型参数名与state_dict不因文件搬移改变。
- `tests/fixtures/baseline_283_v0.json`：固定小型回归标尺，继续供仿真基准的 `--save`、`--check` 使用。
- `docs/README.md`和 `docs/work3_query_migration.md`：当前导航与迁移记录。

筛查模型原模块名为 `experiments.main_screen.screen_models`，当前为 `models.main_screen`。仓库受维护保存路径使用state_dict及模型元数据；清单中126份 `.pt/.pth/.ckpt` 的pickle载荷经静态扫描，未发现旧筛查模块引用，检查没有反序列化模型。没有为未登记的完整Python对象pickle保留旧模块壳。若另有外部完整对象pickle，应先在整理前版本转换成state_dict，再用当前模型显式加载，不能将模块搬移视为完整对象pickle的自动兼容保证。

## 按文件核验和恢复

先从清单按原相对路径选择一份文件，核对普通副本SHA256，再复制到原位置。不要把整个历史资料树重新覆盖当前源码。下面在仓库根目录运行；输入需要恢复的真实相对路径，例如 `CODEBASE_AUDIT.md`。目标已存在时立即停止。

```powershell
$repo = (Get-Location).Path
$archive = Join-Path (Join-Path (Split-Path -Parent $repo) '归档') 'APAL-Dynamic-2'
$manifestPath = Join-Path $archive 'archive_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$relativePath = Read-Host '输入清单中的原相对路径'
$entries = @($manifest.files | Where-Object { $_.path -eq $relativePath })
if ($entries.Count -ne 1) { throw '清单中未找到唯一文件' }
$entry = $entries[0]
if ((Get-FileHash -LiteralPath $entry.archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.source_sha256) { throw '副本SHA256不一致' }
$target = [IO.Path]::GetFullPath((Join-Path $repo $entry.path))
if (-not $target.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw '恢复目标超出当前仓库' }
if (Test-Path -LiteralPath $target) { throw '目标已存在，停止覆盖' }
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
Copy-Item -LiteralPath $entry.archive -Destination $target
```

该恢复命令只复制用户明确选中的文件，不改写当前运行配置。新输出统一进入 `runs/`，历史报告的原路径可用于定位来源，不能作为新的输出目录约定。

## 历史工具的完整输入

`scripts/work3`中保留19个基础或回归需要的旧工具；必要固定参照在`tests/fixtures/work3/local_history/`，实际运行输入在`data/work3/local_diagnostics/`。这些目录不是新的训练输出位置。完整历史实验的其它中间件、教师/MC快照和源码版本仍保存在普通资料目录，不能将现有小范围回归通过解释成全部历史长实验已经重跑。

BC工具的既有教师读取链可显式配置只读文件；在PowerShell中设置如下，然后再调用相应工具：

```powershell
$env:APAL_WORK3_TEACHER_SNAPSHOTS = 'D:\OneDrive\研究生\归档\APAL-Dynamic-2\runs\a710e34\work3_4060\c_reproduction_diagnostic_v1\seed42\teacher_snapshots.pt'
```

该文件已通过公开`fixed_snapshot_sample`真实读取两条，读取前后SHA一致；它是旧C契约的32维状态/12维时间输入，不是QUERY的14维查询条件。Linux使用复制到本机后的真实路径配置同一变量。critic的MC快照是该流程生成的输出，仍写当前run的traces，没有新增平行读取接口或向外部原件写入。

`prepare_local_data`只为新生成内容写`generated_fixture_manifest.json`，保留原`fixture_manifest.json`。原固定源SHA守卫仍要求当时的Git原始字节；本机CRLF版本可能因SHA不同被拒绝。本轮用固定a710e34原Git blob在隔离目录核验新清单路径，没有转换工作区输入或放宽守卫。完整复现旧实验应在对应固定提交和输入身份下进行。
