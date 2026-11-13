# CS5185 内容自适应视频编码系统

给定文件体积上限，先判断画面是「静」还是「动」、细节是「多」还是「少」，
再决定把有限码率花在哪里；压完量一下，不达标就调，每轮都留记录。

---

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 数据准备 | ✅ 已跑通 | 12 个 mp4 + metadata 已解压到 `data/raw/` |
| 内容特征提取 | ✅ 已跑通 | 12 个视频共 1800 帧，30 秒跑完 |
| 参数决策 | ✅ 已跑通 | 24 组（12 视频 × 2 档）已导出 CSV |
| 报告图 | ✅ 已生成 | 特征时间线、分类散点 |
| FFmpeg 编码 | ✅ 已跑通 | 经 `imageio-ffmpeg` 提供，48 次两遍编码全部成功 |
| 结果评测 | ✅ 已跑通 | 体积误差 24/24 在 ±3% 内，解码校验全通过 |

> `python run.py check-env` 会列出 ffmpeg 的实际路径、来源、libx264 是否可用。

---

## 实验结果

自适应（`--mode adaptive`）与统一基线（`--mode fixed`）在同一 12 个视频、同一 5% / 2% 预算下各跑 24 组。

| 指标 | 自适应 | 统一基线 |
|---|---|---|
| 平均体积误差 | −1.75% | −1.68% |
| 误差范围 | −2.98% ~ +0.96% | −3.78% ~ +0.83% |
| 落在自设 ±3% 带内 | **24 / 24** | 23 / 24 |
| 平均 PSNR-Y | **38.26 dB** | 38.09 dB |
| 平均单任务耗时 | 10.9 s | 11.5 s |
| 解码校验 | 24 / 24 通过 | 24 / 24 通过 |

**逐组胜负：胜 20 / 负 1 / 平 3，平均 +0.17 dB。** 详见 `results/compare.csv` 与
`docs/design.md` 第 6 节。增益不大，但方向稳定，且自适应是在体积误差更偏负
（花更少字节）的前提下取得的。

产物：

```
results/compare.csv                  逐视频逐档对比
results/figs/compare_psnr_0.02.png   绝对画质 + 增益双面板
results/figs/compare_psnr_0.05.png
results/figs/compare_gain_vs_detail.png   增益 vs 细节密度（r = +0.60）
```

---

## 目录结构

```
mediaproject/
├── run.py                 命令行入口
├── requirements.txt
├── data/
│   ├── raw/               原始数据集（12 mp4 + metadata + licenses）
│   └── out/               编码输出，按 video_id/ 分层，每轮候选都保留
├── logs/                  每次尝试一行 JSONL（附录 E2 的字段）
├── results/
│   ├── features/          12 个视频的逐帧特征
│   ├── figs/              报告用图
│   ├── decision_table.csv 24 组参数决策
│   └── summary.csv        编码汇总（跑完 encode 后生成）
├── src/
│   ├── config.py          路径与常量
│   ├── metadata.py        元数据 + 预算公式（附录 B2）
│   ├── features.py        内容特征（第 1 步）
│   ├── decide.py          参数决策（第 2 步）
│   ├── encode.py          FFmpeg 两遍编码（第 3 步）
│   ├── evaluate.py        结果检查（第 4 步）
│   ├── pipeline.py        五步闭环编排
│   └── report.py          画图
└── docs/design.md         设计决策与依据（写报告时直接用）
```

---

## 环境

| 组件 | 是否必需 | 本机状态 |
|---|---|---|
| Python 3.9+ | 必需 | ✅ `.venv`（3.11.3，位于项目内，不占 C 盘） |
| numpy / opencv-python / matplotlib | 必需 | ✅ 已装 |
| **FFmpeg + libx264** | **必需** | ✅ 经 `imageio-ffmpeg` 提供（gyan.dev essentials 9.0.2） |
| ffprobe | 可选 | ➖ 没有；`evaluate.py` 已改为正则解析 `ffmpeg -i` 输出 |
| torch / pytorch-msssim | 可选 | ➖ 没有；`src/ms_ssim.py` 为纯 numpy 实现 |

**为什么不用系统 FFmpeg？** C 盘只剩 6GB，装便携包要往 C 盘塞。`imageio-ffmpeg`
通过 pip 装进项目内的 venv，几十 MB，自带 libx264，且能把路径写死、不依赖 PATH。
它就一条 `pip install imageio-ffmpeg`，换机器复现也简单。

装依赖（已做过，重装才需要）：

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

想换成系统 FFmpeg 也可以：把 `ffmpeg.exe` 放进 PATH 即可，`src/encode.py` 的
`_exe()` 会优先用 PATH 里的版本。换完务必验证 libx264 存在：

```bash
ffmpeg -hide_banner -h encoder=libx264
```

看到参数列表 = 可用；报 `Unknown encoder` = 下错了构建，换 gyan.dev 的 essentials/full
或 BtbN 的 win64-gpl。

---

## 使用

```bash
# 0. 检查环境
.venv/Scripts/python.exe run.py check-env

# 1. 提取内容特征（约 30 秒）
.venv/Scripts/python.exe run.py features

# 2. 生成参数决策表 -> results/decision_table.csv
.venv/Scripts/python.exe run.py plan

# 3. 画报告图 -> results/figs/
.venv/Scripts/python.exe run.py report

# 4. 跑完整闭环
.venv/Scripts/python.exe run.py encode --ids 01_FourPeople --ratios 0.05
.venv/Scripts/python.exe run.py encode            # 全部 12 个 × 2 档（自适应）
.venv/Scripts/python.exe run.py encode --mode fixed   # 基线对照组

# 5. 生成自适应 vs 基线的对比表与图 -> results/compare.csv、figs/compare_*.png
.venv/Scripts/python.exe -c "from src import compare; compare.report(compare.compare())"
```

> `encode` 的汇总表是**按 (视频, 档位) 合并更新**的，用 `--ids` 只重跑一个视频
> 不会抹掉其余行。

---

## 三条不能违反的约束

1. **只能用 FFmpeg 的 libx264**，不许改 x264 源码，不许换成别的编码器。
2. **输出必须完整时长、可解码**，不许靠删帧、剪短、改分辨率凑体积。
3. **单位是十进制**：1 MB = 1,000,000 字节。项目 PDF 的 "63.90 MB" 就是 63,899,977 字节。

---

## 踩过的坑（写报告时可以直接用）

### 1. PDF 表格里的 bitrate 列不能直接用

它是按「整数 MB」倒算的，且没扣容器开销。以 09_Parkrun 为例：表里写 8.0 Mbps，
按精确公式应是 9.09 Mbps —— 相差 13%，照表里的数编码会白白浪费八分之一预算。

### 2. 12_ControlledBurn 的 2% 目标「18 Mbps」是笔误

按公式 1.1 MB × 8 ÷ 5s ≈ 1.76 Mbps，应为 1.8 Mbps。本项目一律按 `metadata.json`
里的准确字节数重算，不受影响。

### 3. 纯 ABR 两遍编码在 5 秒短片上会过冲 40%

02_Beauty 的 5% 任务编出来比目标大 40%。逐项排查 preset / GOP / bf / refs / AQ，
甚至只留 `-b:v` 也照样超 —— 说明不是参数问题，是两遍 ABR 本身没有码率上限约束，
短片段里 x264 会先花后补，尾巴补不回来。

解法是加 VBV：`-maxrate <kbps> -bufsize <kbps>`（1 秒缓冲）。加完立刻降到 −6.5%，
反馈循环也从「+40% → −10.6% → +8.0%」的震荡变成两轮收敛。

### 4. 「按内容分类调参」的 3×3 规则表输给了固定参数

第一版按运动量（mafd）和细节量（edge_ratio）分成九宫格，每格给不同 preset / GOP / bf。
结果平均 PSNR **−0.23 dB**（胜 9 / 负 13 / 平 2，共 24 组）。

做了控制变量扫描才明白：**12 个视频里只有 GOP 显著**，g=30→150 能带来
+0.51 dB（高运动）、+1.33 dB（12_ControlledBurn）、+0.1~0.36 dB（低运动）；
preset / bf / refs / AQ 的差异全在 ±0.2 dB 以内，属于噪声。

所以最终版本干脆不用九宫格，全部走同一套校准配置。改回 `g=150` 之后，
同一批 24 组的平均增益从 **−0.23 dB 翻到 +0.17 dB**（胜 20 / 负 1 / 平 3）。
这个过程本身比「调出一个好参数」更值得写进报告 —— 见 `docs/design.md` 第 4 节和第 6 节。

