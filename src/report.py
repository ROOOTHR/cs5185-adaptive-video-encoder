"""报告用图：特征时间线 + 内容分类散点。

这两张图是"内容分析 → 参数决策"因果链的证据，报告第 2 节要用。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, decide, features
from .metadata import list_ids

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = config.RESULT_DIR / "figs"
SHORT = {
    "01_FourPeople": "01 FourPeople", "02_Beauty": "02 Beauty",
    "03_Bosphorus": "03 Bosphorus", "04_HoneyBee": "04 HoneyBee",
    "05_Jockey": "05 Jockey", "06_ShakeNDry": "06 ShakeNDry",
    "07_Mobcal": "07 Mobcal", "08_Shields": "08 Shields",
    "09_Parkrun": "09 Parkrun", "10_Stockholm": "10 Stockholm",
    "11_Sintel": "11 Sintel", "12_ControlledBurn": "12 ControlledBurn",
}


def feature_timelines() -> None:
    """12 个小图，每格是逐帧 mafd 曲线，标题标出分类结果。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_ids()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=False, sharey=False)

    for ax, vid in zip(axes.ravel(), ids):
        f = features.load(vid)
        y = np.asarray(f["per_frame"]["mafd"])
        x = np.arange(len(y)) / f["fps"]
        mo, de = decide.classify(f)

        ax.plot(x, y, color="#185FA5", lw=1.0)
        ax.axhline(y.mean(), color="#BA7517", lw=0.9, ls="--", alpha=0.9)

        for c in f.get("scene_cuts", []):
            ax.axvline(c / f["fps"], color="#A32D2D", lw=1.1, alpha=0.85)

        ax.set_title(f"{SHORT[vid]}  [{mo}/{de}]  μ={y.mean():.1f}", fontsize=9.2, pad=4)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, float(y.max()) * 1.18)

    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")

    for r in range(3):
        axes[r, 0].set_ylabel("mafd", fontsize=8)
    for c in range(4):
        axes[2, c].set_xlabel("时间 (s)", fontsize=8)

    fig.suptitle("逐帧内容变化强度（蓝线 = mafd，橙虚线 = 均值，红竖线 = 检出场景切换；各图纵轴独立）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "feature_timelines.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  已保存 {out}")


def decision_map() -> None:
    """运动 x 细节 二维散点，展示 3x3 分类与 12 个视频的落点。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_ids()
    pts = []
    for vid in ids:
        f = features.load(vid)
        pts.append((vid, f["stats"]["mafd"]["mean"], f["stats"]["edge_ratio"]["mean"]))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 8, 10, 15, 20, 30]))
    ax.xaxis.set_major_formatter(FixedFormatter(["2", "3", "4", "6", "8", "10", "15", "20", "30"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.015, 0.02, 0.03, 0.05, 0.08, 0.13, 0.2, 0.3, 0.45]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.015", "0.02", "0.03", "0.05", "0.08",
                                                 "0.13", "0.2", "0.3", "0.45"]))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.axvline(decide.MOTION_LOW, color="#B4B2A9", lw=1, ls="--")
    ax.axvline(decide.MOTION_HIGH, color="#B4B2A9", lw=1, ls="--")
    ax.axhline(decide.DETAIL_LOW, color="#B4B2A9", lw=1, ls="--")
    ax.axhline(decide.DETAIL_HIGH, color="#B4B2A9", lw=1, ls="--")

    colors = {"low": "#88B04B", "mid": "#E8A33D", "high": "#C0504D"}
    for vid, m, e in pts:
        mo, de = decide.classify({"stats": {"mafd": {"mean": m}, "edge_ratio": {"mean": e}}})
        ax.scatter(m, e, s=145, color=colors[mo], edgecolor="white",
                   linewidth=1.4, zorder=5)
        ax.annotate(SHORT[vid], (m, e), textcoords="offset points", xytext=(7, 6),
                    fontsize=8.5, color="#333333")

    ax.set_xlabel(f"运动强度  mafd（阈值 {decide.MOTION_LOW} / {decide.MOTION_HIGH}）", fontsize=10)
    ax.set_ylabel(f"细节密度  edge_ratio（阈值 {decide.DETAIL_LOW} / {decide.DETAIL_HIGH}）", fontsize=10)
    ax.set_title("12 个视频在「运动 × 细节」平面上的分布", fontsize=12.5)
    ax.grid(alpha=0.2, which="both", lw=0.5)
    ax.margins(0.16)

    for txt, xy in [("低运动·低细节", (1.6, 0.012)), ("高运动·高细节", (14, 0.5))]:
        ax.text(*xy, txt, fontsize=8.5, color="#777777", alpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "decision_map.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  已保存 {out}")
