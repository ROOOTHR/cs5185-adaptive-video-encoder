"""参数决策（项目第 2 步）。

## 这份表是扫出来的，不是想出来的

第一版按"高运动→短 GOP / 多 B 帧"的经验规则写了 3x3 表，跑完 24 个任务后
**平均输给统一基线 0.23 dB**（24 个里输 13 个）。于是做了控制变量扫描，
结论和直觉相反：

| 参数 | 实测影响 | 最优值 |
|---|---|---|
| `-g`   | **0.1 ~ 1.3 dB** | 150（= 整段只有一个 I 帧） |
| `-preset` | < 0.2 dB，slow 甚至略差 | medium |
| `-bf` / `-refs` | < 0.1 dB | 3 / 3 |
| `-aq-mode` / `-aq-strength` | < 0.2 dB（PSNR 口径） | 1 / 1.0 |

原因：这些片段只有 75–150 帧，场景切换由 x264 的 sc_threshold 自动处理，
手动缩短 GOP 只会往 5 秒里塞 5 个昂贵的 I 帧。12_ControlledBurn 上
g=30 比 g=150 低 1.3 dB，这一条就是第一版输掉的主要原因。

所以"自适应"在这里体现为：**先用特征把内容分类并解释难度差异，
再套用一组由数据校准的参数**。逐维度调参的收益已经被测死，
虚假地给每类内容塞不同参数只会更差。完整数据见 docs/design.md 4.3。
"""

from __future__ import annotations

# 实测校准配置。所有内容共用——这是扫描结果，不是偷懒。
CALIBRATED = dict(preset="medium", g=150, bf=3, refs=3, aq=1, aq_str=1.0)

# 统一基线：对照组，模拟"不做任何内容分析"的默认做法。
# 用 g=60 而不是 150，因为它才是常见教程里的默认值。
FIXED = dict(preset="medium", g=60, bf=3, refs=3, aq=1, aq_str=1.0)

# 分类阈值（来源：12 个样本实测分布的自然断点，见 docs/design.md 第 3 节）
MOTION_LOW, MOTION_HIGH = 8.0, 14.0       # mafd
DETAIL_LOW, DETAIL_HIGH = 0.09, 0.16      # edge_ratio


def classify(feat: dict) -> tuple[str, str]:
    m = feat["stats"]["mafd"]["mean"]
    d = feat["stats"]["edge_ratio"]["mean"]
    mo = "low" if m < MOTION_LOW else ("high" if m > MOTION_HIGH else "mid")
    de = "low" if d < DETAIL_LOW else ("high" if d > DETAIL_HIGH else "mid")
    return mo, de


def decide(feat: dict, bud: dict, mode: str = "adaptive") -> dict:
    """返回参数计划。

    mode="adaptive" 用实测校准配置并给出内容解释；
    mode="fixed"    是统一基线对照组（g=60）。
    """
    mo, de = classify(feat)
    mafd = feat["stats"]["mafd"]["mean"]
    edge = feat["stats"]["edge_ratio"]["mean"]
    cuts = len(feat.get("scene_cuts", []))
    kbps = bud["video_kbps"]

    if mode == "fixed":
        return {
            "motion": mo, "detail": de,
            "params": dict(FIXED),
            "reasons": ["统一基线：medium / g=60 / aq=1，只按预算改码率"],
            "mode": "fixed",
        }

    p = dict(CALIBRATED)
    reasons = [
        f"运动 mafd={mafd:.2f} → {mo}（仅描述：实测与增益相关 r=-0.04，无解释力）",
        f"细节 edge={edge:.4f} → {de}（实测与增益相关 r=+0.60，是主要影响因素）",
        "GOP 取 150（整段一个 I 帧）：扫描显示比 g=60 高 0.1–1.3 dB",
        "preset/bf/refs/AQ 已扫描，差异 < 0.2 dB，取统一值",
    ]
    if cuts:
        reasons.append(f"检出 {cuts} 次场景切换，交给 sc_threshold 自动插 I 帧"
                       f"（实测强插 GOP 会低 {1.3:.1f} dB）")

    note = None
    if kbps < 800:
        note = (f"目标仅 {kbps} kbps，720p 在此码率下画质受限；"
                f"下一步可考虑降分辨率，但需按附录 E1 对齐评测")
        reasons.append(note)

    return {
        "motion": mo, "detail": de,
        "params": p,
        "reasons": reasons,
        "note": note,
        "mode": "adaptive",
    }


def ffmpeg_args(d: dict, bud: dict, job: str, pass_no: int, out_path: str) -> list[str]:
    """拼一条两遍编码中的某一遍命令（不含 ffmpeg 本体）。"""
    p = d["params"]
    kbps = bud["video_kbps"]
    common = [
        "-c:v", "libx264",
        "-preset", p["preset"],
        "-pix_fmt", "yuv420p",
        "-b:v", f"{kbps}k",
        # VBV 不是可选项：实测 02_Beauty 在纯 ABR 两遍下超目标 40%，
        # 加 1 秒缓冲后落到 -6.5%，再靠反馈循环补回。见 docs/design.md 5.1。
        "-maxrate", f"{kbps}k",
        "-bufsize", f"{kbps}k",
        "-g", str(p["g"]),
        "-bf", str(p["bf"]),
        "-refs", str(p["refs"]),
        "-aq-mode", str(p["aq"]),
        "-aq-strength", str(p["aq_str"]),
    ]
    if pass_no == 1:
        return common + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"]
    return common + ["-pass", "2", "-passlogfile", job, "-an",
                     "-movflags", "+faststart", out_path]
