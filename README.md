# CS5185 内容自适应视频编码（v2）

以 FFmpeg/libx264 执行两档固定文件大小预算的视频编码。规则由 DEV 扫描推导，VAL 仅检查，规则冻结后 TEST 只运行一次。v2 当前只支持一条 bpp→AQ-mode 规则；bpp 是预算率代理，不等价于证明运动或纹理内容规则。每个 run 保存在独立 `results/runs/<run>/`，保留尝试记录、逐帧指标、编码视频、图和 HTML。

## 当前状态

| 项目 | v2 结果 |
|---|---|
| 分支 / 冻结 | `v2`（远程 `origin/v2`）；冻结 tag `frozen-v2` → `55516859a134dfbdf64a5811a33ed9a344fb9820`（tag 仅在本机，远程没有，复现时按下文用该 commit 重建） |
| 划分 | DEV 01/02/03/04/07/08；VAL 11/12；TEST 05/06/09/10 |
| 体积验收 | 72 组无超目标；B0 24/24、B1 24/24、A 23/24 完全通过 `-3%..0`，A 有 1 组低于下限 |
| MS-SSIM | FFmpeg 灰度/Y 平面；30 帧 PyTorch 对齐最大差 `1.8034227472485398e-6` |
| 规则 | 1 条：`bpp > 0.26137514467592593 → aq-mode=2`，否则 1；非纯运动/纹理规则 |
| TEST A−B1 | 7 个有效配对；平均 ΔPSNR-Y `-0.07447467630169465 dB`，ΔMS-SSIM `+0.06614078252185861 dB` |
| TEST A−B0 | 7 个有效配对；平均 ΔPSNR-Y `+0.08884749671659366 dB`，ΔMS-SSIM `+0.2526249232827765 dB` |

正式实验在 `frozen-v2` 后只执行一次。`10_Stockholm` 5% 的 A 在 4 次尝试后 size_error 为 `-0.032729066206098764`，未达到最小目标体积，判失败并排除出配对均值；没有 TEST 超目标输出。详细逐视频值、偏离、限制和相对路径图片见 [`docs/experiment_v2.md`](docs/experiment_v2.md)。

旧版“24/24 在 ±3% 内”不等于不超目标，不作为 v2 验收。旧版（v1）的设计与结果归档在 [`results/legacy_v1/`](results/legacy_v1/)，旧日志在 `archive/v1_logs/`；它们受测试集调参影响，只作探索阶段记录。

## 环境和复现

主环境 Python 3.14.7、FFmpeg 7.1/libx264；实际依赖版本以 `results/runs/final_v2/run_config.json` 为准（旧 `requirements.txt` 与实测版本不同）。独立 E1 环境含 CPU PyTorch 与 pytorch-msssim，不属于主程序依赖。

```powershell
git clone https://github.com/ROOOTHR/cs5185-adaptive-video-encoder
cd cs5185-adaptive-video-encoder
git checkout v2
$env:CS5185_DATASET = 'D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset'
python run.py check-retry
python run.py features --run features_v2
python run.py sweep --run sweep_v2
python run.py align-check --run sweep_v2
python run.py derive --run sweep_v2
python -m pytest -q tests/test_decide.py
python run.py validate --run val_v2
# final 必须在冻结提交上执行（要求 frozen-v2 tag 指向 HEAD 且工作区干净）。
# 远程没有该 tag，先在 v2 的历史里按 commit 重建，不要指向别处：
git tag frozen-v2 55516859a134dfbdf64a5811a33ed9a344fb9820
git checkout frozen-v2
python run.py final --run final_v2
# 展示材料用 v2 分支的报告器生成
git checkout v2
python run.py report --run final_v2
```

> 上面的流程没有在全新克隆上完整重放验证过。切换到冻结提交前请先确认工作区干净（`final` 会拒绝有未提交或未跟踪文件的工作区）；重现时请为 run 换新名字，程序不会覆盖已有目录。

## 实验结果与交付

- DEV 扫描：168 候选，164 通过；唯一规则是预算相关 bpp 规则，纯运动/纹理规则没有数据支持。
- LOOCV：40/96 参数预测与 oracle 一致；平均 MS-SSIM(dB) 损失 `0.04362710941110234`。
- VAL：8/8 通过；均在规则阈值低侧，未验证高侧泛化。
- TEST：A 组 7/8 通过完整体积窗口，B0/B1 均 8/8；全部 24 个 TEST 输出均未超过目标。完整逐行表见 `results/runs/final_v2/comparisons.csv`。
- 主观材料和 HTML：`results/runs/final_v2/index.html`（16 张三联帧、8 个并排短片、观察表）。已在 Edge 核验首页渲染，138 个本地引用均存在。运动流畅度未确认，未整段播放短片。

- 入库内容与目录约定（哪些结果入库、哪些由程序重新生成而被忽略）：见 [`results/README.md`](results/README.md)。

## 命令

- `features --run <name>`：提取 12 视频的报告用特征，不参与规则拟合。
- `sweep --run <name>`：DEV-only 扫描。
- `derive --run <sweep-name>`：DEV-only 规则、oracle 与 LOOCV。
- `validate --run <name>`：VAL-only 检查，不调参。
- `final --run <name>`：需 frozen tag 指向 HEAD 且工作区干净；串行运行 12×2×{B0,B1,A}。
- `report --run <name>`：图、TEST 主观材料、CSV 和 HTML。
- `align-check --run <sweep-name>` / `check-retry`：独立 E1/E3 检查。

所有编码与展示片均由 FFmpeg 执行；输出不写回数据集。`results/` 下入库的是结论与可追溯证据（汇总表、逐轮记录、逐帧指标、图、三联帧、观察表），程序可重新生成的体积项（`videos/`、x264 日志、`.npz`、并排短片）不入库，详见 [`results/README.md`](results/README.md)。
