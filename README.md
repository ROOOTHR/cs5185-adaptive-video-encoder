# CS5185 内容自适应视频编码（v2）

以 FFmpeg/libx264 执行两档目标文件大小的视频编码，采用开发集扫描和预注册判据推导规则。v2 目前只从 DEV 支持一条 bpp→AQ-mode 规则；这代表预算率差异，不等于已经证明运动/纹理内容规则。每次实验保存在独立 `results/runs/<run>/`，含尝试记录、逐帧指标、视频产物和 HTML 索引。

## 当前状态

| 项目 | v2 状态 |
|---|---|
| 数据源/划分 | 外部只读数据集；DEV 01/02/03/04/07/08；VAL 11/12；TEST 05/06/09/10 |
| 大小验收 | `-3% ≤ size_error ≤ 0`，任何超目标候选不接受 |
| 完整性 | 解码成功、帧数相等、时长误差≤1帧，分辨率/帧率不变 |
| MS-SSIM | FFmpeg Y 平面、接入评测、30 帧 PyTorch 参考对齐最大差 `1.8034227472485398e-6` |
| DEV/VAL | 168 个候选中 164 个通过；VAL 8/8 通过。结果详见实验报告 |
| TEST | 只能在规则冻结 tag `frozen-v2` 后运行一次；测试结果及索引将在最终实验后补录 |

旧版“24/24 在 ±3% 内”不等价于不超目标，不作为 v2 结果。旧日志保存在 `archive/v1_logs/`；外部 `legacy/` 和数据集均保持不变。

## 环境和复现

主环境：Python 3.14.7、FFmpeg 7.1/libx264；项目依赖见 `requirements.txt`。E1 对齐用独立 CPU PyTorch 环境，非主程序依赖。报告、逐阶段命令、限制及预注册标准见 [`docs/experiment_v2.md`](docs/experiment_v2.md)。

```powershell
$env:CS5185_DATASET = 'D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset'
python run.py check-retry
python run.py sweep --run sweep_v2
python run.py align-check --run sweep_v2
python run.py derive --run sweep_v2
python -m pytest -q tests/test_decide.py
python run.py validate --run val_v2
# 提交冻结代码/规则/预注册报告，确认工作区 clean 后创建 frozen-v2 tag
python run.py final --run final_v2
python run.py report --run final_v2
```

## 实验结果

DEV 规则为 `bpp > 0.26137514467592593 → aq-mode=2`，否则 `aq-mode=1`。规则来自 AQ-mode=2 的两个显著 DEV 样本；运动和边缘特征没有形成合格分割，因此不得把这条规则称为已验证的纯内容规则。留一法参数一致 40/96，平均 MS-SSIM(dB) 损失 `0.04362710941110234`。VAL 全部落在阈值低侧，未触发 AQ-mode=2 分支；A−B1 平均 ΔMS-SSIM(dB) 为 `-0.0896528456999155`。完整结果及扫描图请看实验报告。

冻结后的 TEST 对比表、主观图片与并排视频将保存在 `results/runs/final_v2/`，入口为 `index.html`；**最终实验尚未执行**。不要将 TEST 结果用于调整规则。

## 命令

- `features --run <name>`：提取 12 个视频特征，仅用于报告；TEST 特征不得用于拟合。
- `sweep --run <name>`：仅扫描 DEV。
- `derive --run <sweep-name>`：仅从 DEV 表生成规则和 LOOCV。
- `validate --run <name>`：冻结前只检查 VAL，不选参。
- `final --run <name>`：要求 `frozen-v2` 指向 HEAD 且工作区干净；按串行顺序运行 12×2×{B0,B1,A}。
- `report --run <name>`：生成对比 CSV、图、TEST 主观材料和 HTML。
- `align-check --run <sweep-name>` / `check-retry`：E1/E3 独立检查，写入 `results/checks/`。

所有编码与展示短片均由 FFmpeg 执行；结果不写回原始数据集。正式 run/check 产物为本地文件，按 `.gitignore` 不纳入 Git。
