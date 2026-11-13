"""自适应 vs 统一基线的对照分析。

输出：
    results/compare.csv        逐视频对比
    results/figs/compare_psnr.png   两档各一张柱状图
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .report import FIG_DIR, SHORT

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _load(name: str) -> dict:
    p = config.RESULT_DIR / name
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["video_id"], r["target_ratio"])
            out[key] = r
    return out


def _f(row, k, default=None):
    try:
        v = row.get(k)
        return default if v in (None, "", "None") else float(v)
    except (TypeError, ValueError):
        return default


def compare() -> list[dict]:
    ad = _load("summary.csv")
    bl = _load("summary_baseline.csv")
    if not ad or not bl:
        print("缺 summary.csv 或 summary_baseline.csv，先跑两种模式")
        return []

    rows = []
    for key in sorted(ad):
        a, b = ad[key], bl.get(key)
        if not b:
            continue
        pa, pb = _f(a, "psnr_y_db"), _f(b, "psnr_y_db")
        rows.append({
            "video_id": a["video_id"],
            "ratio": a["target_ratio"],
            "motion": a.get("motion"),
            "detail": a.get("detail"),
            "target_MB": _f(a, "target_MB"),
            "adaptive_MB": _f(a, "actual_MB"),
            "baseline_MB": _f(b, "actual_MB"),
            "adaptive_err": _f(a, "size_error"),
            "baseline_err": _f(b, "size_error"),
            "adaptive_psnr": pa,
            "baseline_psnr": pb,
            "psnr_gain": (pa - pb) if (pa and pb) else None,
            "adaptive_attempts": int(_f(a, "attempts", 0)),
            "baseline_attempts": int(_f(b, "attempts", 0)),
            "adaptive_time_s": _f(a, "total_time_s"),
            "baseline_time_s": _f(b, "total_time_s"),
        })
    return rows


def report(rows: list[dict]) -> None:
    if not rows:
        return
    out = config.RESULT_DIR / "compare.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  已写 {out}")

    print(f"\n{'video':<20}{'档':>5}{'自适应PSNR':>11}{'基线PSNR':>10}{'增益':>8}{'体积误差(自/基)':>20}")
    print("-" * 76)
    for r in rows:
        g = r["psnr_gain"]
        gt = f"{g:+.2f}" if g is not None else "-"
        ae = f"{r['adaptive_err']:+.1%} / {r['baseline_err']:+.1%}"
        print(f"{r['video_id']:<20}{r['ratio']:>5}"
              f"{r['adaptive_psnr'] or 0:>11.2f}{r['baseline_psnr'] or 0:>10.2f}"
              f"{gt:>8}{ae:>20}")

    gains = [r["psnr_gain"] for r in rows if r["psnr_gain"] is not None]
    if gains:
        print("-" * 76)
        win = sum(1 for g in gains if g > 0.05)
        lose = sum(1 for g in gains if g < -0.05)
        print(f"{'平均 PSNR 增益':<20}{np.mean(gains):+.2f} dB"
              f"   胜 {win} / 负 {lose} / 平 {len(gains) - win - lose}（共 {len(gains)}）")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # 两档共用同一纵轴范围，图之间可以直接对比
    gmax = max((r["psnr_gain"] or 0.0) for r in rows)
    gtop = max(0.35, gmax + 0.08)
    for ratio in sorted({r["ratio"] for r in rows}):
        sub = [r for r in rows if r["ratio"] == ratio]
        x = np.arange(len(sub))
        w = 0.38
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})

        # 左：绝对画质
        ax.bar(x - w / 2, [r["baseline_psnr"] for r in sub], w,
               label="统一基线", color="#B4B2A9")
        ax.bar(x + w / 2, [r["adaptive_psnr"] for r in sub], w,
               label="自适应", color="#185FA5")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(r["video_id"], r["video_id"]) for r in sub],
                           rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel("PSNR-Y (dB)")
        ax.set_title(f"绝对画质（{ratio} 档）", fontsize=11, pad=32)
        ax.legend(frameon=False, fontsize=9, ncol=2,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0))
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        hi = max(max(r["baseline_psnr"] or 0, r["adaptive_psnr"] or 0) for r in sub)
        ax.set_ylim(0, hi * 1.10)   # 从 0 起，避免柱状图夸大差异；差异交给右图

        # 右：增益（这才是被 0.1 dB 级差异藏起来的信号）
        gains = [r["psnr_gain"] or 0.0 for r in sub]
        cols = ["#C0392B" if g > 0.05 else "#1E8449" if g < -0.05 else "#B4B2A9"
                for g in gains]
        ax2.bar(x, gains, 0.62, color=cols)
        ax2.axhline(0, color="#444", lw=0.9)
        ax2.axhline(np.mean(gains), color="#185FA5", lw=1.1, ls="--",
                    label=f"平均 {np.mean(gains):+.2f} dB")
        ax2.set_xticks(x)
        ax2.set_xticklabels([SHORT.get(r["video_id"], r["video_id"]) for r in sub],
                            rotation=28, ha="right", fontsize=8.5)
        ax2.set_ylabel("PSNR 增益 (dB)")
        ax2.set_title("自适应相对基线的增益", fontsize=11)
        ax2.legend(frameon=False, fontsize=9, loc="upper left")
        ax2.grid(axis="y", alpha=0.25, lw=0.5)
        ax2.set_ylim(-0.12, gtop)

        fig.tight_layout()
        p = FIG_DIR / f"compare_psnr_{ratio.replace('%', 'pct')}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"  已保存 {p}")

    _gain_vs_detail(rows)


def _corr(rows: list[dict], key: str) -> float:
    """增益与某个内容特征（取 log）的皮尔逊相关系数。"""
    from . import features as feat_mod

    xs, ys = [], []
    for r in rows:
        try:
            v = feat_mod.load(r["video_id"])["stats"][key]["mean"]
        except Exception:
            continue
        if v and v > 0:
            xs.append(np.log10(v))
            ys.append(r["psnr_gain"] or 0.0)
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _gain_vs_detail(rows: list[dict]) -> None:
    """增益与画面细节量的关系。

    基线固定 g=60、自适应固定 g=150，两者唯一的差别就是 GOP 长度。
    长 GOP 省下的是 I 帧开销，而细节越密，帧内预测越差、I 帧越贵 ——
    所以「细节量 → 增益」应当正相关。这张图是对该机理的独立验证。
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator

    from . import features as feat_mod

    edge = {}
    for vid in {r["video_id"] for r in rows}:
        try:
            edge[vid] = feat_mod.load(vid)["stats"]["edge_ratio"]["mean"]
        except Exception:
            pass
    rows = [r for r in rows if r["video_id"] in edge]
    if not rows:
        return

    xs = np.array([edge[r["video_id"]] for r in rows], dtype=float)
    ys = np.array([r["psnr_gain"] or 0.0 for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for tag, color in (("0.02", "#C0392B"), ("0.05", "#185FA5")):
        m = np.array([r["ratio"] == tag for r in rows])
        if m.any():
            ax.scatter(xs[m], ys[m], s=54, color=color, label=f"{tag} 档",
                       edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(0, color="#444", lw=0.9)

    # 只标极值点，免得满屏标签；同一视频有两个点，只标一次
    per_vid = {}
    for i, r in enumerate(rows):
        xv = xs[i]
        if r["video_id"] not in per_vid or xv > per_vid[r["video_id"]][0]:
            per_vid[r["video_id"]] = (xv, ys[i], i)
    ranked = sorted(per_vid.items(), key=lambda kv: kv[1][0])
    for vid, (xv, _yv, i) in ranked[:1] + ranked[-1:]:
        ax.annotate(SHORT.get(vid, vid), (xv, _yv), textcoords="offset points",
                    xytext=(4, 4) if xv > xs.mean() else (4, 6),
                    ha="left", fontsize=7.5, color="#555")

    ax.set_xscale("log")
    ticks = [0.015, 0.03, 0.06, 0.12, 0.25, 0.5]
    ticks = [t for t in ticks if xs.min() * 0.8 <= t <= xs.max() * 1.2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("细节量 edge_ratio（Sobel 强边缘像素占比，对数轴）")
    ax.set_ylabel("PSNR 增益 (dB)")
    ax.set_title("细节越密，长 GOP 省下的 I 帧开销越值钱", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)

    lx = np.log10(xs)
    k = np.polyfit(lx, ys, 1)
    grid = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(10 ** grid, np.polyval(k, grid), color="#888", lw=1.2, ls="--")
    r = np.corrcoef(lx, ys)[0, 1]
    ax.text(0.98, 0.04, f"r = {r:+.2f}（n = {len(xs)}）",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F4F2", ec="#CCC", lw=0.7))

    fig.tight_layout()
    p = FIG_DIR / "compare_gain_vs_detail.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"  已保存 {p}")
