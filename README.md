# CS5185 内容自适应视频编码（v2）

以 FFmpeg/libx264 执行两档固定文件大小预算的视频编码。规则由 DEV 扫描推导，VAL 仅检查，规则冻结后 TEST 只运行一次。v2 当前只支持一条 bpp→AQ-mode 规则；bpp 是预算率代理，不等价于证明运动或纹理内容规则。每个 run 保存在独立 `results/runs/<run>/`，保留尝试记录、逐帧指标、编码视频、图和 HTML。

## 当前状态

| 项目 | v2 结果 |
|---|---|
| 分支 / 冻结 | `v2`（远程 `origin/v2`）；入库结果的冻结点是 commit `55516859a134dfbdf64a5811a33ed9a344fb9820`（历史上由 git 标签把关，标签已移除；现由哈希检查把关，见复现一节） |
| 划分 | DEV 01/02/03/04/07/08；VAL 11/12；TEST 05/06/09/10 |
| 体积验收 | 72 组无超目标；B0 24/24、B1 24/24、A 23/24 完全通过 `-3%..0`，A 有 1 组低于下限 |
| MS-SSIM | FFmpeg 灰度/Y 平面；30 帧 PyTorch 对齐最大差 `1.8034227472485398e-6` |
| 规则 | 1 条：`bpp > 0.26137514467592593 → aq-mode=2`，否则 1；非纯运动/纹理规则 |
| TEST A−B1 | 7 个有效配对；平均 ΔPSNR-Y `-0.07447467630169465 dB`，ΔMS-SSIM `+0.06614078252185861 dB` |
| TEST A−B0 | 7 个有效配对；平均 ΔPSNR-Y `+0.08884749671659366 dB`，ΔMS-SSIM `+0.2526249232827765 dB` |

正式实验在规则冻结后只执行一次。`10_Stockholm` 5% 的 A 在 4 次尝试后 size_error 为 `-0.032729066206098764`，未达到最小目标体积，判失败并排除出配对均值；没有 TEST 超目标输出。详细逐视频值、偏离、限制和相对路径图片见 [`docs/experiment_v2.md`](docs/experiment_v2.md)。

旧版“24/24 在 ±3% 内”不等于不超目标，不作为 v2 验收。旧版（v1）的设计与结果归档在 [`results/legacy_v1/`](results/legacy_v1/)，旧日志在 `archive/v1_logs/`；它们受测试集调参影响，只作探索阶段记录。

## 环境和复现

主环境 Python 3.14.7、FFmpeg 7.1/libx264；实际依赖版本以 `results/runs/final_v2/run_config.json` 为准（旧 `requirements.txt` 与实测版本不同）。独立 E1 环境含 CPU PyTorch 与 pytorch-msssim，不属于主程序依赖。

```powershell
git clone https://github.com/ROOOTHR/cs5185-adaptive-video-encoder
cd cs5185-adaptive-video-encoder
git checkout v2
pip install -r requirements.txt
$env:CS5185_DATASET = 'D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset'   # 课程数据集目录
python run.py all --prefix r1      # 全流程，每个阶段写入 results/runs/r1_*/
```

`all` 按顺序执行 8 步：E3 超目标重试检查 → 特征 → DEV 扫描 → 由 DEV 推导规则 → MS-SSIM 对齐（可选）→ VAL 检查 → TEST（只此一次）→ 报告。全程约 3 小时（本机 32 线程的历史耗时：扫描约 1.7 小时、TEST 约 1.1 小时）。`--prefix` 必须是新名字，已有的目录不会被覆盖。

也可以分步执行：

```powershell
python run.py check-retry --run r1_retry
python run.py features --run r1_features
python run.py sweep --run r1_sweep          # 仅 DEV
python run.py derive --run r1_sweep         # 仅 DEV，写出 src/rules_v2.json
python run.py align-check --run r1_sweep    # 可选，需要 requirements-align.txt 的环境
python run.py validate --run r1_val         # 仅 VAL；记录规则与选参代码的哈希，等于“冻结”
python run.py final --run r1_final --val r1_val   # TEST，只允许一次
python run.py report --run r1_final
```

### 冻结与“只测一次”是怎么保证的（不依赖 git 标签）

`validate` 会把 `src/rules_v2.json` 和决定编码/评测的代码（`config/decide/encode/evaluate/features/metadata/ms_ssim/pipeline`）的哈希写进该 VAL run 的 `run_config.json`。`final` 开始前要求：① 指定的 VAL run 已完成且数据集校验通过；② 当前规则和代码的哈希与 VAL run 记录一致，改过就拒绝并要求重跑 `validate`；③ 同一组哈希下没有已经完成的 final run。`sweep/derive` 遇到非 DEV 视频直接报错。

> 重新跑出的数字会和入库的 `*_v2` 略有不同：多线程 x264 不是逐字节确定的，`derive` 也可能推出不同的规则。入库的 `*_v2` 结果是首次正式运行，历史上是在 git 标签 `frozen-v2`（commit `55516859a134dfbdf64a5811a33ed9a344fb9820`）上由旧版的标签检查把关；该标签现已移除，commit 仍在 `v2` 分支历史中，`results/runs/final_v2/run_config.json` 也记录了它。

### 可选：反馈策略 `center`

```powershell
python run.py --feedback-policy center all --prefix r1     # 或设置环境变量 CS5185_FEEDBACK_POLICY=center
```

默认策略是 `proportional`，不加选项时行为不变。启用 `center` 解决的问题：默认的比例修正把“目标体积”本身当作瞄准点，首轮偏低时下一轮容易冲过窗口上沿（超过目标体积），再回调，白白多用重试轮数。`center` 把瞄准点改为验收窗口 −3%..0 的中点（0.985 × 目标体积），其余逻辑不变。

在开发集和验证集 48 个任务 × 3 次重复的对比里（[`docs/feedback_experiment.md`](docs/feedback_experiment.md)），需要 3 轮以上的任务从 35 个降到 2 个，平均尝试轮数从 1.81 降到 1.57；通过率没有可区分的差别，也不是靠多花预算换来的（通过任务的平均体积误差 −1.66% 对 −1.70%）。它不能解决体积响应本身不稳定的格子（例如 08_Shields 2%），那类格子在各个策略下都会偶尔失败。`secant`（割线法）也可选，实验里没有比 `center` 更好。

每次运行的 `run_config.json` 记录所用策略，`final` 要求与对应 VAL 运行所用的策略一致。

## 实验结果与交付

- DEV 扫描：168 候选，164 通过；唯一规则是预算相关 bpp 规则，纯运动/纹理规则没有数据支持。
- LOOCV：40/96 参数预测与 oracle 一致；平均 MS-SSIM(dB) 损失 `0.04362710941110234`。
- VAL：8/8 通过；均在规则阈值低侧，未验证高侧泛化。
- TEST：A 组 7/8 通过完整体积窗口，B0/B1 均 8/8；全部 24 个 TEST 输出均未超过目标。完整逐行表见 `results/runs/final_v2/comparisons.csv`。
- 主观材料和 HTML：`results/runs/final_v2/index.html`（16 张三联帧、8 个并排短片、观察表）。已在 Edge 核验首页渲染，138 个本地引用均存在。运动流畅度未确认，未整段播放短片。

- 入库内容与目录约定（哪些结果入库、哪些由程序重新生成而被忽略）：见 [`results/README.md`](results/README.md)。
- 踩过的坑及 v2 的实际响应状态（预算公式、PDF 码率表错误、线程数导致的 +40% 过冲、规则表的取舍）：见 [`docs/pitfalls.md`](docs/pitfalls.md)。

## 命令

- `features --run <name>`：提取 12 视频的报告用特征，不参与规则拟合。
- `sweep --run <name>`：DEV-only 扫描。
- `derive --run <sweep-name>`：DEV-only 规则、oracle 与 LOOCV。
- `validate --run <name>`：VAL-only 检查，不调参。
- `final --run <name> --val <val-run>`：TEST 一次性正式实验，串行运行 12×2×{B0,B1,A}；规则或选参代码在 VAL 之后改动、或同一冻结状态已跑过 TEST，都会被拒绝。
- `all --prefix <name>`：按顺序执行上面所有步骤。
- `--feedback-policy {proportional,center,secant}`（写在子命令之前）：体积重试的修正规则，默认 `proportional`。
- `report --run <name>`：图、TEST 主观材料、CSV 和 HTML。
- `align-check --run <sweep-name>` / `check-retry --run <name>`：独立 E1/E3 检查；结果写在 `results/runs/<sweep-name>/checks/` 和 `results/checks/<name>/`，不覆盖已入库的检查结果。

所有编码与展示片均由 FFmpeg 执行；输出不写回数据集。`results/` 下入库的是结论与可追溯证据（汇总表、逐轮记录、逐帧指标、图、三联帧、观察表），程序可重新生成的体积项（`videos/`、x264 日志、`.npz`、并排短片）不入库，详见 [`results/README.md`](results/README.md)。
