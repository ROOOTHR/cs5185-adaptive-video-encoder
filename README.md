# CS5185 内容自适应视频编码（v2）

本仓库以 FFmpeg/libx264 完成固定体积预算的视频编码，使用逐帧特征与开发集扫描验证内容→参数规则。规则冻结后才运行独立 TEST。所有正式运行保存在各自的 results/runs/<run>/ 目录，含原始尝试、逐帧指标、视频产物和 index.html。

## 当前状态

| 项目 | v2 状态 |
|---|---|
| 数据源 | 外部只读数据集；CS5185_DATASET 可覆写路径 |
| 数据划分 | DEV 01/02/03/04/07/08；VAL 11/12；TEST 05/06/09/10 |
| 体积验收 | -3% ≤ size_error ≤ 0；不超过目标 |
| 完整性 | 解码成功、帧数一致、时长误差不超过一帧、分辨率/帧率不变 |
| MS-SSIM | FFmpeg 原始 Y 平面输入；接入评测并做 PyTorch 参考对齐 |
| 规则/最终实验 | 以 docs/experiment_v2.md 和 frozen-v2 tag 为准；不得用 TEST 调参 |

旧版汇总结果已从活动结果目录移除；不得把旧的“24/24 在 ±3% 内”表述当成 v2 上限验收。旧日志在 archive/v1_logs/ 保留，唯一修改过的日志行也原样保留。归档目录 legacy/ 与数据集不由本实验改动。

## 环境

主程序已在 Python 3.14.7 验证；依赖见 requirements.txt。FFmpeg 由 imageio-ffmpeg 提供，必须包含 libx264。E1 对齐校验使用独立的 Python 3.12 环境：D:/gcg/Multimedia/project/_venv_msssim，其中安装 CPU PyTorch 与 pytorch-msssim；不加入主程序依赖。

PowerShell 示例：

```powershell
$env:CS5185_DATASET = 'D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset'
python run.py check-retry
python run.py sweep --run sweep_v2
python run.py align-check --run sweep_v2
python run.py derive --run sweep_v2
python run.py validate --run val_v2
# 代码/规则/文档提交后，确保 HEAD == frozen-v2 且 git status 干净
# git tag -a frozen-v2 -m 'Freeze v2 rules after DEV and VAL'
python run.py final --run final_v2
python run.py report --run final_v2
```

完成后的实验命令、参数、阈值、扫描结果、留一法、VAL 和 TEST 指标详见 docs/experiment_v2.md；HTML 索引在 results/runs/final_v2/index.html。

## 分阶段命令

- `features --run <name>`：提取全数据集特征（不把 TEST 特征用于拟合任何阈值）。
- `sweep --run <name>`：只在 DEV 上对锚点逐个改变一个参数维度并跑反馈闭环。
- `derive --run <sweep-name>`：只读 DEV 扫描表，生成至多 3 条规则及 LOOCV。
- `validate --run <name>`：只跑 VAL，对比锚点，不再改规则。
- `final --run <name>`：要求 frozen-v2 tag 指向当前 HEAD、工作区干净；串行跑 12×2×{B0,B1,A} 一次。
- `report --run <name>`：生成表格、图、TEST 三联帧、并排短片、主观观察表和 HTML。
- `align-check --run <sweep-name>` / `check-retry`：独立 E1/E3 小检查，保存在 results/checks/。

E1 对齐校验、扫描和最终实验产物均不写回原始数据集。results/runs/ 与 results/checks/ 是本地交付产物，不纳入 Git，以避免提交大量视频；报告和摘要保存在工作区供查看。
