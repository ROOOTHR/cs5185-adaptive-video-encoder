# CS5185 内容自适应视频编码 v2 实验报告

## 1. 改动摘要（Review1 问题 1–6）

| Review | 本次处理与证据 |
|---|---|
| 1 内容→参数 | 在扫描前预注册标准；sweep/derive 限定 DEV；仅以 DEV 扫描推导并冻结 1 条规则。它是 bpp（目标预算率）规则，不是已证明的运动/纹理内容规则。`src/experiment.py:24-29,259-348,371-456`、`src/rules_v2.json`、`src/decide.py:28-52`。 |
| 2 体积上限 | 验收为 `-3% ≤ size_error ≤ 0`，超过上限重试；4 次上限。`src/evaluate.py:69-94`、`src/pipeline.py:12-15,27-85`、`src/config.py:18-20`。 |
| 3 MS-SSIM | 以 FFmpeg 灰度/Y 平面逐帧比较，强制两路完整解码且帧数匹配；输出平均值、dB、最小值和逐帧值；对照 PyTorch 参考实现。`src/ms_ssim.py:12-61,64-144`、`src/evaluate.py:52-94`。 |
| 4 材料 | 生成 TEST 的 16 张三联帧、8 个原始/B0/A 三路并排短片、主观观察 CSV 和 HTML 索引；未通过验收的 Stockholm 5% 仅作为明确标记的诊断材料。`src/report.py:71-170,382-443`。 |
| 5 数据泄漏 | 划分常量位于 `src/config.py:22-26`；DEV-only guard 位于 `src/experiment.py:33-40`；final 检查 tag 与干净工作区位于 `src/experiment.py:522-567`。测试集只在冻结 tag 后运行一次。 |
| 6 完整性/隔离 | 每个 run 独立目录；候选必须通过解码、帧数、时长、分辨率、帧率和体积检查才算成功。`src/experiment.py:63-71`、`src/evaluate.py:69-94`、`src/pipeline.py:17-117`。 |

**归档偏离：**任务书要求把旧 `logs/*.jsonl` 移至外部只读 `legacy/logs/`；按调用者要求未改外部 `legacy/`，而将历史日志随 v2 框架提交移入仓库 `archive/v1_logs/`。日志未用于 v2 拟合。旧版 `docs/design.md` 和 `results/` 的有意删除随首个阶段提交。数据集目录保持只读。

## 2. 环境与复现

最终运行记录环境：Windows 11、Python 3.14.7、AMD64 32 逻辑 CPU、FFmpeg 7.1 essentials/libx264；编码任务 4 threads，final 串行；NumPy 2.5.3、OpenCV headless 5.0.0.93、Matplotlib 3.11.2、imageio-ffmpeg 0.6.0。E1 独立环境为 Python 3.12、CPU PyTorch 2.15.0、pytorch-msssim 1.0.0。未安装 ffprobe。扫描 run_config 的 OpenCV 版本在采集修正之前记录为 null；VAL/final run_config 已记录 headless 5.0.0.93。

```powershell
$env:CS5185_DATASET = 'D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset'
python run.py check-retry
python run.py features --run features_v2
python run.py sweep --run sweep_v2
python run.py align-check --run sweep_v2
python run.py derive --run sweep_v2
python -m pytest -q tests/test_decide.py
python run.py validate --run val_v2
# 历史正式运行在干净的 frozen-v2 提交；独立复现应在该 tag 执行 final，不要移动 tag
python run.py final --run final_v2
# 生成展示材料时使用 exp-v2 的报告器
python run.py report --run final_v2
```

TEST 只在冻结 tag 后运行一次。重现时请为已存在的 run 目录选择新名称，程序不会覆盖已有目录。数据集哈希验证在 final run_config 中为通过；不使用仓库旧 `data/raw/`。v1 日志只作归档，不进入 v2 规则。

## 3. MS-SSIM 与对齐校验

视频输入由 FFmpeg 解码为灰度/Y 平面帧，按 metadata 分辨率逐帧对齐；预期帧数为 06 的 75 帧、其他视频各 150 帧，任一路少帧、多帧或解码失败都会使测量失败。单帧参数：data range 255、5 尺度、11×11 Gaussian 窗口、sigma 1.5、权重 `(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)`，逐级下采样。视频 ms_ssim 是逐帧算术平均；`ms_ssim_db=-10 log10(1-ms_ssim)`；同时记录最小值和逐帧序列。实现：`src/ms_ssim.py`。

`results/checks/msssim_alignment.csv` 有 3 个 DEV 视频各 10 帧，共 30 对；相对 pytorch-msssim 的平均绝对差 `6.4532750951862285e-7`、最大绝对差 `1.8034227472485398e-6`（阈值 `1e-4`）。自比较为 1.0；加噪标准差 1、2、5、10 后分数依次下降，见 `results/checks/msssim_alignment_basic.json`。

无 ffprobe，所以 `file_kbps` 用完整文件大小/时长计算；`actual_video_kbps` 是 FFmpeg stream-copy 汇总估算，不是 ffprobe packet-size 精确值，两列不应混为一谈。

## 4. 划分、防泄漏与冻结

- DEV：01、02、03、04、07、08，仅用于扫描和规则推导。
- VAL：11、12，仅用于冻结前检查，不选参。
- TEST：05、06、09、10，v2 final 前没有用于规则拟合；冻结后执行唯一一次 `final_v2`。
- `sweep/derive` 只允许 DEV IDs；`final` 要求 `frozen-v2` 指向当前 HEAD 且开始时工作区干净。
- 冻结 tag `frozen-v2` 指向 commit `55516859a134dfbdf64a5811a33ed9a344fb9820`（短 hash `5551685`）。Final 的 `run_config.json` 记录相同 commit、`git_dirty=false`、数据集哈希通过。分支为 `exp-v2`，未 push。

冻结后仅对 `src/report.py` 作报告修正：让未通过验收的 A 候选也生成标有诊断用途的三联帧/短片，并在 HTML 标红失败行、标注诊断短片。同时保留已有人工观察表，避免重建报告时覆盖。没有改编码器决策、参数、输入或重跑 final；因此 tag 保持在实际编码使用的冻结提交，分支最新报告提交在 tag 之后。

## 5. DEV 扫描结果

扫描前固定锚点 medium / g=150 / bf=3 / refs=3 / AQ-mode=1 / strength=1.0 / deblock=0:0 / psy-rd=1.0:0；每次只改变一维。6 个 DEV × 2 档，锚点加 13 变体，合计 168 候选。预注册“显著改善”为 ΔMS-SSIM(dB) ≥ +0.10 且 ΔPSNR-Y ≥ −0.30 dB。候选和图：[`sweep_summary.csv`](../results/runs/sweep_v2/sweep_summary.csv)、[`sweep_heatmap_msssim.png`](../results/runs/sweep_v2/figs/sweep_heatmap_msssim.png)、[`sweep_heatmap_psnr.png`](../results/runs/sweep_v2/figs/sweep_heatmap_psnr.png)；每维最优值见 [`best_values.csv`](../results/runs/sweep_v2/best_values.csv)。

![DEV 扫描 ΔMS-SSIM 热力图](../results/runs/sweep_v2/figs/sweep_heatmap_msssim.png)

![DEV 扫描 ΔPSNR-Y 热力图](../results/runs/sweep_v2/figs/sweep_heatmap_psnr.png)

通过 164/168，失败 4 个：均为 08_Shields 5% 的 bf=8、deblock=1:1、psy-rd=0.5、preset=slow；经过 4 次仍无法进入 `-3%..0` 区间。成功候选的 size_error 最小/最大为 `-0.029251073237191845` / `-0.00005993498087936329`。

下表为每变体相对锚点的平均差，只对通过硬性验收的配对计数；显著数按预注册的双指标门槛：

| 维度 | 变体 | 成功配对 | 平均 ΔMS-SSIM(dB) | 平均 ΔPSNR-Y(dB) | 显著数 |
|---|---:|---:|---:|---:|---:|
| bf | 5 | 12 | -0.012162353 | -0.020888058 | 1 |
| bf | 8 | 11 | -0.048203815 | -0.063518430 | 1 |
| refs | 6 | 12 | +0.045830630 | +0.045739629 | 1 |
| AQ-mode | 2 | 12 | +0.028581306 | -0.049896265 | 2 |
| AQ-mode | 3 | 12 | +0.055479503 | -0.183082617 | 1 |
| AQ-strength | 0.6 | 12 | -0.126266418 | +0.168997983 | 0 |
| AQ-strength | 1.4 | 12 | +0.039249585 | -0.286892838 | 2 |
| deblock | -1:-1 | 12 | -0.031660688 | -0.051961193 | 0 |
| deblock | 1:1 | 11 | -0.007761941 | +0.022633085 | 0 |
| psy-rd | 0.5 | 11 | +0.152632981 | +0.159533027 | 9 |
| psy-rd | 1.5 | 12 | -0.109730864 | -0.112181883 | 0 |
| preset | slow | 11 | -0.077144605 | -0.058574499 | 0 |
| g | 60 | 12 | -0.137493206 | -0.145068827 | 0 |

psy-rd=0.5 的 11 个有效配对中平均改善突出，但 08_Shields 5% 未通过大小验收；不能声称该变体在全部 DEV 样本上一致更好，也未找到通过单一预注册特征中点阈值分隔改善与非改善组的规则。`mafd`、`edge_ratio`、`luma_std` 都未得到满足判据的决策规则。没有编造内容映射；未支持的维度退回锚点。

## 6. 冻结规则、LOOCV、VAL

唯一规则：`bpp > 0.26137514467592593 → AQ-mode=2；否则 AQ-mode=1`，见 `src/rules_v2.json`。阈值来自 DEV 样本中点。AQ-mode=2 两个显著样本为 07_Mobcal 5%（ΔMS-SSIM `+0.12048569826709254 dB`、ΔPSNR-Y `-0.18495767514929184 dB`）和 08_Shields 5%（`+0.11830530203227951 dB`、`-0.18967054982965692 dB`）。其他预注册特征没有得到此规则。**因此 bpp 是将视频空间预算率和分辨率合成后的预算相关代理；本结果不证明纯运动或纹理特征改变了参数。**

留一法共 96 行（每留出视频、预算和参数维度），参数与 oracle 一致 40/96；MS-SSIM(dB) 损失平均 `0.04362710941110234`，最大 `0.2931913676143374`。AQ-mode 子集一致 2/12，平均损失 `0.03826607410551238`。明细：[`loocv.csv`](../results/runs/sweep_v2/loocv.csv)。该总一致率包含没有规则的各参数维度，不能解释为只测 AQ 规则。

VAL 8/8 组 A/B1 编码通过硬性验收。4 个 A−B1 配对的平均 ΔPSNR-Y `-0.09698377371118916 dB`、平均 ΔMS-SSIM(dB) `-0.0896528456999155 dB`；两个指标的胜/负/平计数均为 1/3/0。全部 VAL bpp 低于阈值，规则高侧没有触发；因此 VAL 仅检查了锚点回退分支，没有提供阈值另一侧的独立验证。明细：[`val_check.csv`](../results/runs/val_v2/val_check.csv)。

## 7. Final 实验：TEST 主结果、DEV/VAL 附表与大小验收

final 串行执行 12 视频 × 2 档 × B0/B1/A，共 72 组；正式总耗时 `3949.417858300003 秒`。摘要和逐行结果：[`summary.csv`](../results/runs/final_v2/summary.csv)、[`comparisons.csv`](../results/runs/final_v2/comparisons.csv)、[`phase_timing.json`](../results/runs/final_v2/phase_timing.json)。比较只使用两组均 `status=ok` 的配对；TEST 为 8 个预算组合中 7 个有效配对，因为 10_Stockholm 5% 的 A 失败。

| TEST 比较（有效配对 n=7） | 平均 ΔPSNR-Y | PSNR 胜/负/平 | 平均 ΔMS-SSIM(dB) | MS-SSIM 胜/负/平 |
|---|---:|---:|---:|---:|
| A−B1 | -0.07447467630169465 dB | 2/4/1 | +0.06614078252185861 dB | 3/3/1 |
| A−B0 | +0.08884749671659366 dB | 6/1/0 | +0.2526249232827765 dB | 7/0/0 |

### TEST 编码明细

每组均为 medium / bf=3 / refs=3 / AQ-strength=1.0 / deblock=0:0 / psy-rd=1.0:0；B0 使用 g=60、AQ=1，B1 使用 g=150、AQ=1，A 使用 g=150 且按冻结规则设置 AQ。`file kbps` 为整文件码率；`video kbps` 为 FFmpeg stream-copy 估算。每行数值直接由 final `summary.csv` 格式化。

| 视频 | 档位 | 组 | 状态 | 目标/实际字节 | 目标/file/video kbps | size_error | AQ / g | PSNR-Y dB | MS-SSIM dB | total_time s |
|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|
| 05_Jockey | 5% | B0 | ok | 3555036/3515418 | 5574/5624.669/5621.350 | -1.1144% | 1 / 60 | 41.6844 | 21.1979 | 34.95 |
| 05_Jockey | 5% | B1 | ok | 3555036/3512796 | 5574/5620.474/5618.074 | -1.1882% | 1 / 150 | 41.7168 | 21.2435 | 27.90 |
| 05_Jockey | 5% | A | ok | 3555036/3514432 | 5574/5623.091/5619.712 | -1.1422% | 1 / 150 | 41.7256 | 21.2525 | 27.21 |
| 05_Jockey | 2% | B0 | ok | 1422014/1392139 | 2230/2227.422/2224.947 | -2.1009% | 1 / 60 | 38.3797 | 17.5383 | 61.81 |
| 05_Jockey | 2% | B1 | ok | 1422014/1395757 | 2230/2233.211/2229.862 | -1.8465% | 1 / 150 | 38.4558 | 17.6251 | 58.74 |
| 05_Jockey | 2% | A | ok | 1422014/1398706 | 2230/2237.930/2234.778 | -1.6391% | 1 / 150 | 38.4422 | 17.6165 | 56.72 |
| 06_ShakeNDry | 5% | B0 | ok | 2059164/2018435 | 6458/6458.992/6452.019 | -1.9779% | 1 / 60 | 38.6448 | 20.8067 | 48.10 |
| 06_ShakeNDry | 5% | B1 | ok | 2059164/2049963 | 6458/6559.882/6553.600 | -0.4468% | 1 / 150 | 38.9219 | 21.1820 | 31.38 |
| 06_ShakeNDry | 5% | A | ok | 2059164/2054869 | 6458/6575.581/6569.984 | -0.2086% | 1 / 150 | 38.9240 | 21.1807 | 30.69 |
| 06_ShakeNDry | 2% | B0 | ok | 823665/808944 | 2583/2588.621/2582.118 | -1.7873% | 1 / 60 | 35.6604 | 17.0145 | 15.32 |
| 06_ShakeNDry | 2% | B1 | ok | 823665/823569 | 2583/2635.421/2631.270 | -0.0117% | 1 / 150 | 35.9687 | 17.3958 | 28.17 |
| 06_ShakeNDry | 2% | A | ok | 823665/804780 | 2583/2575.296/2569.011 | -2.2928% | 1 / 150 | 35.9040 | 17.3016 | 41.31 |
| 09_Parkrun | 5% | B0 | ok | 5798513/5780267 | 9092/9248.427/9243.853 | -0.3147% | 1 / 60 | 28.5218 | 17.0832 | 64.27 |
| 09_Parkrun | 5% | B1 | ok | 5798513/5679423 | 9092/9087.077/9083.290 | -2.0538% | 1 / 150 | 28.6246 | 17.2119 | 95.57 |
| 09_Parkrun | 5% | A | ok | 5798513/5654451 | 9092/9047.122/9042.330 | -2.4845% | 2 / 150 | 28.1719 | 17.7691 | 91.22 |
| 09_Parkrun | 2% | B0 | ok | 2319405/2314823 | 3637/3703.717/3699.507 | -0.1976% | 1 / 60 | 25.0952 | 13.7460 | 57.83 |
| 09_Parkrun | 2% | B1 | ok | 2319405/2265902 | 3637/3625.443/3620.864 | -2.3068% | 1 / 150 | 25.3117 | 13.9033 | 58.78 |
| 09_Parkrun | 2% | A | ok | 2319405/2265902 | 3637/3625.443/3620.864 | -2.3068% | 1 / 150 | 25.3117 | 13.9033 | 58.00 |
| 10_Stockholm | 5% | B0 | ok | 4965464/4831823 | 7786/7730.917/7726.694 | -2.6914% | 1 / 60 | 34.4349 | 18.1978 | 83.45 |
| 10_Stockholm | 5% | B1 | ok | 4965464/4952056 | 7786/7923.290/7918.387 | -0.2700% | 1 / 150 | 34.5564 | 18.3448 | 62.32 |
| 10_Stockholm | 5% | A | failed | 4965464/4802949 | 7786/7684.718/7680.819 | -3.2729% | 2 / 150 | 34.4415 | 18.3551 | 121.41 |
| 10_Stockholm | 2% | B0 | ok | 1986185/1951565 | 3114/3122.504/3117.875 | -1.7430% | 1 / 60 | 33.4705 | 17.0350 | 28.83 |
| 10_Stockholm | 2% | B1 | ok | 1986185/1938262 | 3114/3101.219/3096.576 | -2.4128% | 1 / 150 | 33.6006 | 17.1655 | 85.18 |
| 10_Stockholm | 2% | A | ok | 1986185/1940220 | 3114/3104.352/3099.853 | -2.3142% | 1 / 150 | 33.5994 | 17.1663 | 86.49 |

以下为 TEST 逐预算配对差，括号依次为 ΔPSNR-Y / ΔMS-SSIM(dB)。10_Stockholm 5% 不纳入正式差值。

| 视频 | 预算 | A size_error / 状态 | A−B1 | A−B0 |
|---|---:|---|---|---|
| 05_Jockey | 5% | -0.01142154397311307 / ok | +0.008833197037958485 / +0.009038390334804802 | +0.041201310370624356 / +0.05461570172557728 |
| 05_Jockey | 2% | -0.016390837221011888 / ok | -0.013593329550424471 / -0.008512088126153117 | +0.06245485198996903 / +0.07825786651221733 |
| 06_ShakeNDry | 5% | -0.002085797925760163 / ok | +0.002047076412594606 / -0.0013162513660738284 | +0.27921237968098467 / +0.3739684392930194 |
| 06_ShakeNDry | 2% | -0.022928010781082114 / ok | -0.0647487713838828 / -0.09421798185752195 | +0.24356918593407073 / +0.28708277950557104 |
| 09_Parkrun | 5% | -0.024844645515151902 / ok | -0.4526374526135726 / +0.5572125465099269 | -0.3498666558856911 / +0.6858727953409876 |
| 09_Parkrun | 2% | -0.023067553963193148 / ok | 0 / 0 | +0.21650742006633195 / +0.15723332539447377 |
| 10_Stockholm | 5% | -0.032729066206098764 / **failed** | —（不配对） | —（不配对） |
| 10_Stockholm | 2% | -0.023142355822846312 / ok | -0.0012234540145357187 / +0.0007808621580274178 | +0.128853984859866 / +0.13134355520758945 |

硬性体积验收：B0 为 24/24 通过，size_error 范围 `-0.02948160920349638` 至 `-0.001526415642361433`；B1 为 24/24 通过，范围 `-0.026419410556939894` 至 `-0.0001165522390777804`；A 为 23/24 通过，范围（包含失败候选）`-0.032729066206098764` 至 `-0.0008864089689799849`。所有 72 个结果均未超目标；A 的 Stockholm 5% 经过 4 次尝试仍低于目标超过 3%，故作为失败而非合格候选。TEST 子集 B0、B1 都是 8/8 通过，A 为 7/8，且无超目标结果。

最终 TEST A 参数分支：05/06 两预算均回退 AQ=1；09_Parkrun 5% 和 10_Stockholm 5% 因 bpp 高于阈值选 AQ=2。10_Stockholm 5% 的 A 失败不参加指标均值。DEV 与 VAL final 分别列于 `comparisons.csv`，不与 TEST 主结论混算。

附表：以下为 final 中的 DEV/VAL 配对均值，来自 `comparisons.csv`，不计入 TEST 结论。这里的 VAL 与冻结前单独运行的 `val_check.csv` 不同。

| 划分 | 配对数 | 比较 | ΔPSNR-Y (dB) | ΔMS-SSIM (dB) |
|---|---:|---|---:|---:|
| DEV | 12 | A−B1 | -0.030603302900779877 | +0.023269583769153595 |
| DEV | 12 | A−B0 | +0.11900868927703101 | +0.16469027383360135 |
| VAL | 4 | A−B1 | -0.08198814288738454 | -0.07226860775328703 |
| VAL | 4 | A−B0 | +0.010002939264179389 | +0.057927385437706036 |

独立的 [E3 重试检查](../results/checks/over_target_retry_v3/over_target_retry.csv)：01_FourPeople 5% 首轮按 1.15 倍码率编码，`size_error=+0.10531368094753112`，触发重试，第 2 次通过；未计入正式实验。

## 8. 主观材料与观察

已逐张检查 16 张 TEST 三联帧；8 个 30 fps 并排短片均通过 FFmpeg `-xerror` 解码，并抽样查看多个时点。报告页收录帧、特征图、对比图及短片：[`index.html`](../results/runs/final_v2/index.html)。已在 Edge 无头浏览器中打开本地 HTML，核验首页链接和汇总表实际渲染；另静态检查 138 个本地引用，缺失引用为 0。短片采样只用于检查内容与布局，**未对完整短片连续播放做流畅度判定**。

代表帧（另一个代表帧以及所有剩余图可由 HTML/frames 目录打开）：

| 视频与预算 | 代表帧 |
|---|---|
| 05_Jockey 5% | ![05 Jockey 5%](../results/runs/final_v2/frames/05_Jockey_05pct_lowest_msssim.png) |
| 05_Jockey 2% | ![05 Jockey 2%](../results/runs/final_v2/frames/05_Jockey_02pct_lowest_msssim.png) |
| 06_ShakeNDry 5% | ![06 ShakeNDry 5%](../results/runs/final_v2/frames/06_ShakeNDry_05pct_lowest_msssim.png) |
| 06_ShakeNDry 2% | ![06 ShakeNDry 2%](../results/runs/final_v2/frames/06_ShakeNDry_02pct_lowest_msssim.png) |
| 09_Parkrun 5% | ![09 Parkrun 5%](../results/runs/final_v2/frames/09_Parkrun_05pct_lowest_msssim.png) |
| 09_Parkrun 2% | ![09 Parkrun 2%](../results/runs/final_v2/frames/09_Parkrun_02pct_lowest_msssim.png) |
| 10_Stockholm 5% | ![10 Stockholm 5% — 诊断材料，A 未通过体积验收](../results/runs/final_v2/frames/10_Stockholm_05pct_lowest_msssim.png) |
| 10_Stockholm 2% | ![10 Stockholm 2%](../results/runs/final_v2/frames/10_Stockholm_02pct_lowest_msssim.png) |

人工观察表：[`subjective_notes.csv`](../results/runs/final_v2/subjective_notes.csv)，共 80 行（每个视频预算两帧 × 五个观察项目）。所选局部中：05_Jockey 的栏杆/衣纹略软；06_ShakeNDry 的毛发和林地纹理在 2% 更软；09_Parkrun 的细枝在 2% 出现更明显融合；10_Stockholm 的窗框与街面细节软化。A 与 B0 的多数静帧差异不明显；标注区域未见明确块边界或双边重影，但这些单局部观察不代表全帧。运动流畅度标为“未确认”，因为没有连续播放判定。Stockholm 5% 所有观察均注明仅诊断、不计正式比较。

## 9. 案例：检测到什么 → 参数 → 理由

以 TEST 09_Parkrun 5% 为例：该视频的元数据预算结合 1280×720/30 fps 得到 `bpp=0.32884837962962965`，高于只由 DEV 决定的阈值 `0.26137514467592593`，决策选择 AQ-mode=2，其余参数保持锚点。`summary.csv` 中理由记录了检测值、`>` 比较、命中分支、`aq=2` 和 `sweep_summary.csv` 来源。作为测试结果，A 相对 B1 的 PSNR-Y 变化为 `-0.4526374526135726 dB`，MS-SSIM(dB) 变化为 `+0.5572125465099269 dB`；相对 B0 分别为 `-0.3498666558856911 dB` 和 `+0.6858727953409876 dB`。这些是报告性观测，规则没有根据 TEST 结果修改。

案例图实际展示另一侧的 05_Jockey 5%：`bpp=0.2016059027777778` 未超过阈值，A 保持 AQ=1。图中运动、纹理和亮度随时间变化，但它们不是冻结规则的触发条件。

![05_Jockey 5%：特征时间线、参数与逐帧质量](../results/runs/final_v2/figs/case_feature_to_parameters.png)

## 10. 局限与未完成项

- 本次没有从 DEV 证明纯运动/纹理特征驱动参数；唯一规则依赖 bpp 预算代理，故“内容自适应”核心证据有限。不得把结果写成已满足强内容分类要求。
- 高 bpp 规则侧只有两个 DEV 显著样本，VAL 没有触发，泛化证据不足。
- Final A 有 1 个失败（10_Stockholm 5%）；虽然没有超目标，但未满足 `-3%` 下限，所有正式比较已排除该配对。
- ffprobe 不可用，`actual_video_kbps` 只能使用 FFmpeg 汇总估算；`file_kbps` 则按文件长度计算。
- 扫描 run_config 的 OpenCV 包版本字段为 null，是发生在环境版本收集修正之前的记录；原始 run_config 保留，VAL 和 final 记录正确版本。
- 全 12 视频的纯特征提取是在 DEV 扫描启动后补跑；规则拟合仍只读取 DEV，VAL 独立检查，TEST 在冻结前未用于调参。
- 报告器的失败候选展示修正在冻结后完成；没有触碰冻结规则或 final 编码数据。HTML 已核验首页渲染，未逐一交互播放所有嵌入短片。
- 本次未 push。最终结果文件大（视频/图片），按 `.gitignore` 保持本地；报告和代码在 Git 中提交。
