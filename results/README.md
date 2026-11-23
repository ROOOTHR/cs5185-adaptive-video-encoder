# results/ 目录约定

入库的是**结论与可追溯证据**；程序能重新生成的大文件不入库（规则见仓库根目录 `.gitignore`）。

```
results/
├── runs/<run>/                 每次运行一个独立目录，之后不再覆盖
│   ├── summary.csv             每行 = 视频 × 档位 × 对照组：体积、码率、PSNR、MS-SSIM、耗时、所用参数    入库
│   ├── comparisons.csv         A−B1、A−B0 的逐行配对差                                                入库
│   ├── attempts.jsonl          每一轮的参数、ffmpeg 参数、测量值、调整原因、输出 sha256                  入库
│   ├── run_config.json         git commit、FFmpeg/依赖版本、硬件、规则文件与数据集校验                    入库
│   ├── features/  perframe/    逐帧内容特征、逐帧 PSNR / MS-SSIM（体积小，图可据此重绘）                 入库
│   ├── figs/  frames/          统计图、原始/B0/A 三联帧                                                入库
│   ├── clips/                  原始/B0/A 并排短片（主观对比材料）                                       入库
│   ├── index.html  subjective_notes.csv                                                                入库
│   └── videos/                 全部候选输出 mp4 + x264 统计文件（final_v2 约 430 MB，sweep_v2 约 1.1 GB）  不入库
└── checks/                     附录 E3 小检查与 MS-SSIM 对齐校验（与正式结果分开）
    ├── msssim_alignment.csv, *_torch.json, msssim_alignment_basic.json                                  入库
    ├── over_target_retry*/     超目标重试演示的 csv、attempts.jsonl                                         入库（其中 videos/ 不入库）
    └── msssim_alignment/*.npz  对齐校验用的逐帧数组（每个约 18 MB）                                       不入库
```

## 关于"不入库的视频还能不能追溯"

- 本机 libx264 在 `-threads 4` 下**不是逐字节确定的**：同一条命令跑两遍，输出的 sha256 通常不同。
  所以 `videos/` 不能靠重跑原样恢复，追溯靠的是：
  - `attempts.jsonl` 里每个候选的 `output_sha256`、完整 ffmpeg 参数和测量值；
  - 入库的 `perframe/`、`frames/`、`clips/` ——它们由当时的那批输出生成。
- 要完整保留视频，请单独归档（例如压缩后放网盘或 GitHub Release），不要放进 git 历史。

## 各 run 的用途

| run | 内容 | 阶段 |
|---|---|---|
| `features_v2` | 12 个视频的内容特征（报告用，不参与规则拟合） | 准备 |
| `sweep_v2` | 开发集（01/02/03/04/07/08）参数扫描、规则推导、留一法 | 冻结前，仅 DEV |
| `val_v2` | 验证集（11/12）上检查规则回退分支 | 冻结前 |
| `final_v2` | 规则冻结之后一次性正式实验，12 视频 × 2 档 × {B0, B1, A} | 正式结果；主结论只引用其中的 TEST |

旧版（v1）结果在仓库外的 `legacy/`，旧日志在 `archive/v1_logs/`；它们受测试集调参影响，不作为 v2 结论依据。

## 复现

命令序列见 `docs/experiment_v2.md` 第 2 节。重新运行会在 `results/runs/<新名字>/` 下生成同样结构的目录，
其中 `videos/` 自动被 `.gitignore` 排除，不需要再改忽略规则。

## 附：仓库整体目录结构

```
cs5185-adaptive-video-encoder/
├── run.py                    命令行入口
├── requirements.txt          主流程依赖（实际使用的版本）
├── requirements-align.txt    可选：MS-SSIM 对齐校验用的独立环境
├── src/
│   ├── config.py             路径、常量、数据划分、反馈策略
│   ├── metadata.py           元数据读取与预算公式
│   ├── features.py           内容特征
│   ├── decide.py             规则驱动的参数决策与 FFmpeg 命令
│   ├── encode.py             两遍编码、解码检查
│   ├── evaluate.py           体积、完整性、PSNR、MS-SSIM 验收
│   ├── ms_ssim.py            MS-SSIM 实现（Y 通道，5 尺度）
│   ├── pipeline.py           单个候选的反馈环（含三种修正规则）
│   ├── experiment.py         各阶段编排、划分与冻结检查、all
│   ├── report.py             图、三联帧、并排短片、HTML
│   └── rules_v2.json         DEV 推导出的规则
├── tests/                    19 个测试（决策规则、冻结检查、反馈策略）
├── tools/                    反馈策略实验脚本、pytorch-msssim 对齐助手
├── docs/
│   ├── experiment_v2.md      v2 实验报告
│   ├── pitfalls.md           踩过的坑及 v2 的状态
│   └── feedback_experiment.md  反馈策略实验
├── results/                  实验结果与证据（本文上半部分）
├── archive/v1_logs/          v1 的逐轮日志
└── _diag/                    v1 阶段的诊断脚本（历史遗留）
```

怎么查看和分析这些结果，见根目录 `README.md` 的“如何查看和分析结果”。
