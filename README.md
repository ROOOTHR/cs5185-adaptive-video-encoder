# CS5185 内容自适应视频编码

给定文件体积上限（原文件的 **5%** 和 **2%**），用 **FFmpeg + libx264** 压缩 12 个 5 秒 720p 视频：
先分析画面内容，再决定编码参数，压完检查体积，超标或偏离就按反馈重编码，每一轮都留记录。

> 重点不是“压得够多够小”，而是**是否严谨、稳定**：数据集划分在代码里强制执行，规则冻结后测试集只能跑一次，
> 同参数重复运行的噪声被量化，所有结论都附带可追溯的证据文件。

---

## 一、流程

```
原视频 ─► 内容特征 ─► 参数决策 ─► 两遍编码 ─► 检查 ─┬─ 体积落在 −3%..0 且完整可解码 ─► 通过
          mafd        规则 +      libx264    体积      │
          edge_ratio  目标码率    (VBV)      解码      └─ 否 ─► 按反馈修正码率 ─► 重编码（最多 4 轮）
          luma_std
          bpp
```

**数据划分**（项目说明规定）：

| 集合 | 视频 | 用途 |
|---|---|---|
| 开发集 DEV | 01、02、03、04、07、08 | 扫描参数、推导规则；唯一允许调参的集合 |
| 验证集 VAL | 11、12 | 冻结前检查规则能否推广，不选参 |
| 测试集 TEST | 05、06、09、10 | 规则冻结后**只跑一次**，主结论只引用它 |

**对照组：** B0 = 统一基线（g=60）；B1 = 全局校准参数（g=150，所有视频相同）；A = 内容自适应（B1 加上 DEV 推导出的规则）。

---

## 二、结果摘要

数字来自 `results/runs/final_v2/`，完整逐行表见 [`comparisons.csv`](results/runs/final_v2/comparisons.csv)。

| 项目 | 结果 |
|---|---|
| 体积验收（−3%..0，不超目标） | 72 组中 B0 24/24、B1 24/24、A 23/24 通过；全部无超目标。唯一的失败是 10_Stockholm 5% 的 A（−3.27%，略低于自设下限，项目的硬性要求仍满足），已排除出配对均值 |
| 唯一规则 | `bpp > 0.2614` 时 `aq-mode=2`，否则 1。bpp 是预算与分辨率的组合，**不是纯运动/纹理规则** |
| TEST：A 相对 B0（7 个有效配对） | ΔPSNR-Y **+0.089 dB**，ΔMS-SSIM **+0.253 dB**，MS-SSIM 7/7 全胜 |
| TEST：A 相对 B1 | ΔPSNR-Y −0.074 dB，ΔMS-SSIM +0.066 dB；规则只在 1 对上有效触发（09_Parkrun 5%），其余参数相同，差异只是噪声 |
| 噪声底（20 对同参数配对） | ΔMS-SSIM 标准差 0.046 dB、最大 0.144 dB；预注册的显著门槛 +0.10 dB 只有约 2.2 倍标准差 |
| DEV 扫描 | 168 个候选，164 通过；`psy-rd=0.5` 平均 +0.153 dB（9/11 达标）是最强的信号，但它对所有内容都有效，不是内容规则 |
| 留一法 / VAL | 参数预测与 oracle 一致 40/96（平均 MS-SSIM 损失 0.044 dB）；VAL 8/8 通过，但全在规则阈值低侧，没有验证高侧 |
| MS-SSIM 实现 | 与 pytorch-msssim 逐帧对齐，最大差 1.8e-6 |
| 主观材料 | 16 张三联帧、8 个并排短片、观察表，汇总在 `results/runs/final_v2/index.html`；观察表由人工看短片填写 |

**怎么读这些结果：**

1. **系统层可靠**：闭环完整，体积验收严格，全部输出可完整播放。相对统一基线 MS-SSIM 全胜，其中大部分收益来自 DEV 上校准出的全局参数（B1 相对 B0 约 +0.18 dB）。
2. **内容规则层证据弱**：只推出 1 条规则，依据的样本落在噪声范围内；TEST 上有效触发的 1 对 PSNR 跌幅超出预注册下限，不算显著更好。
3. **正式实验的跑法是亮点**：划分强制、预注册、冻结后只测一次、噪声量化。

详细分析、案例、主观观察和局限见 [`docs/experiment_v2.md`](docs/experiment_v2.md)。

---

## 三、快速开始

**环境：** 在 Windows 上验证过（Python 3.14.7）；FFmpeg 7.1 + libx264 由 `imageio-ffmpeg` 提供，也可使用 PATH 中的 ffmpeg；不需要 ffprobe。代码对 Linux/macOS 的空设备做了处理，但没有测试过。实际依赖版本以 `results/runs/final_v2/run_config.json` 为准。

```powershell
git clone https://github.com/ROOOTHR/cs5185-adaptive-video-encoder
cd cs5185-adaptive-video-encoder
git checkout main
pip install -r requirements.txt
$env:CS5185_DATASET = '...\CS5185_Project_Video_Dataset'   # 课程数据集目录，只读，需要另外下载
python run.py all --prefix r1
```

- `all` 约 3 小时（实机 32 线程的历史耗时：扫描约 1.7 小时、TEST 约 1.1 小时），每个阶段写入 `results/runs/r1_*/`。
- `--prefix` 必须是**新名字**，已存在的目录不会被覆盖。
- 跑完后用浏览器打开 `results/runs/r1_final/index.html` 查看汇总表、图、三联帧和并排短片。
- 重新跑出的数字会和入库的 `*_v2` 略有不同：多线程 x264 不是逐字节确定的，`derive` 也可能推出不同的规则。

**`all` 的 8 个步骤：** E3 超目标重试检查 → 特征 → DEV 扫描 → 由 DEV 推导规则 → MS-SSIM 对齐（可选）→ VAL 检查 → TEST（只此一次）→ 报告。

### 分步执行

```powershell
python run.py check-retry --run r1_retry            # E3：故意触发一次超目标重试，独立于正式结果
python run.py features   --run r1_features          # 12 个视频的内容特征
python run.py sweep      --run r1_sweep             # 参数扫描，仅 DEV
python run.py derive     --run r1_sweep             # 推导规则与留一法，仅 DEV，写出 src/rules_v2.json
python run.py align-check --run r1_sweep            # 可选：与 pytorch-msssim 对齐，需要 requirements-align.txt 的环境
python run.py validate   --run r1_val               # 仅 VAL；同时记录规则与选参代码的哈希，等于“冻结”
python run.py final      --run r1_final --val r1_val  # TEST，只允许一次
python run.py report     --run r1_final             # 图、三联帧、并排短片、观察表、index.html
```

### 命令速查

| 命令 | 作用 |
|---|---|
| `features --run <名>` | 提取特征；报告用，不参与规则拟合 |
| `sweep --run <名>` | DEV 参数扫描；遇到非 DEV 视频直接报错 |
| `derive --run <扫描名>` | DEV 规则、oracle、留一法 |
| `validate --run <名>` | VAL 检查，不调参，记录冻结哈希 |
| `final --run <名> --val <VAL 名>` | TEST 正式实验，串行跑 12 视频 × 2 档 × {B0, B1, A} |
| `report --run <final 名>` | 生成图、主观材料、CSV、HTML |
| `align-check --run <扫描名>` | MS-SSIM 对齐校验；写入 `results/runs/<扫描名>/checks/` |
| `check-retry --run <名>` | E3 重试检查；写入 `results/checks/<名>/`，不覆盖已入库的检查结果 |
| `all --prefix <名>` | 按顺序执行以上全部 |
| `--feedback-policy {proportional,center,secant}` | 写在子命令之前，选择体积重试规则，默认 `proportional` |

---

## 四、冻结与“只测一次”是怎么保证的

不依赖 git 标签，靠哈希检查：

- `validate` 把 `src/rules_v2.json` 和决定编码/评测的代码（`config / decide / encode / evaluate / features / metadata / ms_ssim / pipeline`）的哈希写进该 VAL 运行的 `run_config.json`。
- `final` 开始前要求：① 指定的 VAL 运行已完成且数据集校验通过；② 当前规则、代码和反馈策略与 VAL 运行记录一致，改过就拒绝，要求重跑 `validate`；③ 同一冻结状态下没有已经完成的 final 运行。
- `sweep` / `derive` 遇到非 DEV 视频直接报错。

入库的 `*_v2` 是首次正式运行，当时由 git 标签 `frozen-v2` 把关（对应 commit `55516859a134dfbdf64a5811a33ed9a344fb9820`；标签现已移除，该 commit 仍在 `v2` 历史里，`final_v2/run_config.json` 也记录了它）。

---

## 五、可选：反馈策略 `center`

```powershell
python run.py --feedback-policy center all --prefix r1     # 或设置环境变量 CS5185_FEEDBACK_POLICY=center
```

默认策略是 `proportional`，不加选项时行为不变。

**启用 `center` 解决的问题：**

默认的比例修正把“目标体积”本身当作瞄准点，首轮偏低时下一轮容易冲过窗口上沿（超过目标体积），再回调，白白多用重试轮数。`center` 把瞄准点改为验收窗口 −3%..0 的中点（0.985 × 目标体积），其余逻辑不变。

在开发集和验证集 48 个任务 × 3 次重复的对比里（[`docs/feedback_experiment.md`](docs/feedback_experiment.md)）：

| 策略 | 通过率 | 平均轮数 | 需要 3 轮以上 |
|---|---:|---:|---:|
| `proportional`（默认） | 98.6% | 1.81 | 35 |
| `center` | 99.3% | 1.57 | 2 |
| `secant` | 99.3% | 1.60 | 5 |

通过率没有可区分的差别，也不是靠多花预算换来的（通过任务的平均体积误差 −1.66% 对 −1.70%）。它不能解决体积响应本身不稳定的格子（例如 08_Shields 2%）。`secant`（割线法）也可选，实验里没有比 `center` 更好。每次运行的 `run_config.json` 记录所用策略，`final` 要求与对应 VAL 运行一致。

---

## 六、如何查看和分析结果

结果都在 `results/runs/<运行名>/` 下。入库的正式记录是 `final_v2`（冻结后的测试实验）、`sweep_v2`（开发集扫描）、`val_v2`（验证集检查）；自己跑出来的是 `r1_*`，结构相同。**先从 `index.html` 开始，再按下表查找所需要的结果。**

| 我想知道… | 看这里 | 怎么看 |
|---|---|---|
| 整体情况：体积、画质、哪些没通过 | `final_v2/index.html` | 浏览器打开。汇总表里超目标或失败的行标红，下面是全部图和三联帧。并排短片不入库，克隆后短片链接是空的，要自己 `report` 重新生成 |
| 某个视频某一档的具体数字 | `final_v2/summary.csv` | 每行 = 视频 × 档位 × 对照组（B0 / B1 / A），含体积、`size_error`、`psnr_y_db`、`ms_ssim_db`、轮数、所用参数。Excel 可直接打开（UTF-8），也可以用下面的命令过滤 |
| A 比 B0 / B1 好多少 | `final_v2/comparisons.csv` | `comparison` 列区分 A−B0 与 A−B1，`delta_psnr_y_db` / `delta_ms_ssim_db` 是 A 减对照。**主结论只看 `split=TEST`**；没通过验收的配对没有 delta |
| 某次编码为什么重试了几轮 | `final_v2/attempts.jsonl` | 每行一轮：请求码率 → `size_error` → `next_action` → 下一轮请求码率。示例见下 |
| 画质随时间怎么变 | `final_v2/perframe/<视频>_<组>_<档>pct.json` | 逐帧 `psnr_y_per_frame`、`ms_ssim_per_frame`。`figs/case_feature_to_parameters.png` 把特征曲线、所用参数和逐帧 MS-SSIM 画在同一张图上 |
| 肉眼看画面差异 | `final_v2/frames/*.png`、`clips/*.mp4` | 三联帧是 原始 \| B0 \| A，下面是同一区域的局部放大；每个测试视频每档 2 张，`lowest_msssim` 是 A 最差的帧，`edge_peak` 是纹理最多的帧。短片用能逐帧步进的播放器看运动和重影 |
| 规则是怎么推出来的 | `sweep_v2/figs/sweep_heatmap_*.png`、`sweep_summary.csv`、`rules.csv`、`loocv.csv` | 热力图每格是某个变体相对锚点的 Δ；`rules.csv` 是最终规则；`loocv.csv` 是留一法预测。`figs/oracle_*.png` 是各参数在每个视频上的最优值 |
| 这个结果是怎么跑出来的、能不能复现 | `run_config.json`、`attempts.jsonl` | commit、FFmpeg 与依赖版本、数据集哈希；每个候选的完整 ffmpeg 参数和输出 sha256 |
| 结论和解读 | [`docs/experiment_v2.md`](docs/experiment_v2.md) | 实验报告：改动、划分与冻结、扫描、规则与留一法、最终结果、主观材料、案例、局限 |
| 实现中踩过的坑 | [`docs/pitfalls.md`](docs/pitfalls.md) | 预算公式、PDF 码率表的多处错误、默认线程数导致的 +40% 过冲、规则表的取舍，以及它们在 v2 的状态 |
| 反馈策略 `center` 的依据 | [`docs/feedback_experiment.md`](docs/feedback_experiment.md) | 预先登记的方案、判定标准、结果 |
| 每个文件夹放什么、哪些入库、整个仓库的目录结构 | [`results/README.md`](results/README.md) | 目录约定与追溯方式；文末是完整的仓库目录结构 |

**可以用命令来过滤一些结果，下面是两个ai用例**（PowerShell）：

例 1：查某个视频的所有结果：
```powershell
Import-Csv results/runs/final_v2/summary.csv | Where-Object { $_.video_id -eq '09_Parkrun' } |
  Format-Table group,target_ratio,status,actual_bytes,size_error,psnr_y_db,ms_ssim_db -AutoSize
```

例 2：查某一次编码为什么重试了几轮
```powershell
Get-Content results/runs/final_v2/attempts.jsonl -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.video_id -eq '10_Stockholm' -and $_.group -eq 'A' -and $_.target_ratio -eq 0.05 } |
  Select-Object attempt_id, planned_kbps, @{n='size_error';e={$_.measurements.size_error}}, next_action, next_kbps |
  Format-Table -AutoSize
```

第二条输出的就是 10_Stockholm 5% A 组的四轮：`-6.0% → -4.2% → +5.1% → -3.3%`，对应报告里对这一格失败的分析。

**指标怎么读：** `psnr_y_db` 是亮度 PSNR，`ms_ssim_db = -10·log10(1 − MS-SSIM)`，两者都是越大越好；`size_error` 是 (实际 − 目标) ÷ 目标，验收窗口是 −3%..0。

**分析时要注意：**

- **主结论只引用测试集。** `comparisons.csv` 里的开发集和验证集行是附表，不要和测试集混着算均值。
- **0.1 dB 以内的差异要谨慎。** 同参数重复运行的 ΔMS-SSIM 标准差约 0.046 dB，这个量级的差别和噪声分不开。
- **A 相对 B1 的差异几乎全是噪声。** 规则只在 1 个测试配对上有效触发，其余 A 与 B1 参数相同；10_Stockholm 5% 的 A 没通过验收，已排除出配对。
- **比较两个结果时同时看 `size_error`。** 体积相差 1% 以上的两次编码，差异不能全算成参数的效果。
- **自己重跑的数值会和已入库的结果略有不同。** 多线程 x264 不是逐字节确定的。

---

## 七、本实验的已知限制

- **规则证据弱**：只有 1 条 bpp 规则，不是纯内容规则；样本只有 12 个 5 秒片段。
- **编码不是逐字节确定的**：`-threads 4` 下同一条命令的输出会不同，同参数对比有约 ±0.05 dB 的噪声；`-threads 1` 才确定，但更慢。
- **验收窗口是自设的**：硬性要求只有“不超目标”和“完整播放”，−3% 下限是实验自己加的接近度要求。
- **PDF 里的码率表不可直接使用**：多处取整或折算错误，程序只用元数据里的精确字节数（见 [`docs/pitfalls.md`](docs/pitfalls.md)）。
- **约束**：只用 FFmpeg 的 libx264，不改分辨率、帧率、时长；输出不写回数据集；1 MB = 1,000,000 字节。
