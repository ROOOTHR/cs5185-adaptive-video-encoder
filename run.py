"""命令行入口。

用法：
    python run.py features              提取全部视频的内容特征
    python run.py plan                  打印并导出 24 组参数决策
    python run.py encode [--ids ...]    跑完整闭环（需要 FFmpeg）
    python run.py report                画特征时间线
    python run.py check-env             检查 FFmpeg 是否可用
"""

from __future__ import annotations

import argparse
import csv
import sys

from src import config, decide, features
from src import metadata as meta_mod


def cmd_features(args):
    feat = features.run_all(ids=args.ids, stride=args.stride)
    print(f"\n{len(feat)} 个视频的特征已写入 {config.FEATURE_DIR}")


def cmd_plan(args):
    rows = []
    for vid in args.ids or meta_mod.list_ids():
        feat = features.load(vid)
        meta = meta_mod.load_meta(vid)
        for r in config.RATIOS:
            bud = meta_mod.budget(meta, r)
            plan = decide.decide(feat, bud)
            p = plan["params"]
            rows.append({
                "video_id": vid,
                "ratio": f"{r:.0%}",
                "motion": plan["motion"],
                "detail": plan["detail"],
                "target_kbps": bud["video_kbps"],
                "target_MB": round(bud["target_bytes"] / config.MB, 3),
                "preset": p["preset"], "g": p["g"], "bf": p["bf"],
                "refs": p["refs"], "aq_mode": p["aq"], "aq_strength": p["aq_str"],
                "reason": " | ".join(plan["reasons"]),
            })

    hdr = ["video_id", "ratio", "motion", "detail", "target_kbps", "target_MB",
           "preset", "g", "bf", "refs", "aq_mode", "aq_strength"]
    print(f"{'video':<20}{'比例':>4}{'运动':>6}{'细节':>6}{'kbps':>7}{'预设':>8}"
          f"{'g':>5}{'bf':>4}{'refs':>5}{'aq':>4}{'str':>5}")
    print("-" * 74)
    for r in rows:
        print(f"{r['video_id']:<20}{r['ratio']:>4}{r['motion']:>6}{r['detail']:>6}"
              f"{r['target_kbps']:>7}{r['preset']:>8}{r['g']:>5}{r['bf']:>4}"
              f"{r['refs']:>5}{r['aq_mode']:>4}{r['aq_strength']:>5}")

    out = config.RESULT_DIR / "decision_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已导出 {out}")


def cmd_encode(args):
    from src import pipeline
    ids = args.ids or meta_mod.list_ids()
    ratios = [float(x) for x in args.ratios.split(",")] if args.ratios else list(config.RATIOS)

    def _key(d):
        # target_ratio 在不同调用里可能是 float 也可能是 str，统一成定点字符串再比
        return (d["video_id"], f"{float(d['target_ratio']):.4f}")

    results = []
    for vid in ids:
        for r in ratios:
            results.append(pipeline.run_task(vid, r, mode=args.mode))

    name = "summary.csv" if args.mode == "adaptive" else "summary_baseline.csv"
    out = config.RESULT_DIR / name
    if results:
        # 与已有汇总合并，而不是整体覆盖：--ids 跑单个视频时不该抹掉其余行
        merged = {}
        if out.exists():
            with out.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    merged[_key(row)] = row
        for row in results:
            merged[_key(row)] = row
        rows_out = sorted(merged.values(),
                          key=lambda d: (d["video_id"], float(d["target_ratio"])))
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n汇总已写入 {out}（{len(rows_out)} 行，本次更新 {len(results)} 行）")


def cmd_report(args):
    from src import report
    report.feature_timelines()
    report.decision_map()


def cmd_check_env(args):
    import subprocess

    from src.encode import _exe

    ok = True
    try:
        ff = _exe("ffmpeg")
        src = "PATH" if ff == "ffmpeg" else "imageio-ffmpeg"
        print(f"{'ffmpeg':8s} OK   {ff}")
        print(f"{'':8s}      （来源：{src}）")
    except FileNotFoundError as e:
        print(f"{'ffmpeg':8s} 缺失 {e}")
        print("\n结论：还缺组件，见 README")
        return 1

    # ffprobe 不是必需：evaluate.py 用正则解析 ffmpeg stderr 代替
    import shutil
    fp = shutil.which("ffprobe")
    print(f"{'ffprobe':8s} {'OK   ' + fp if fp else '无（可选，本实现已不依赖）'}")

    r = subprocess.run([ff, "-hide_banner", "-h", "encoder=libx264"],
                       capture_output=True, text=True, errors="replace")
    has = "libx264" in (r.stdout + r.stderr)
    print(f"{'libx264':8s} {'可用' if has else '不可用（换一个带 GPL 的构建）'}")
    ok &= has

    # 版本号
    r2 = subprocess.run([ff, "-version"], capture_output=True, text=True, errors="replace")
    if r2.stdout:
        print(f"\n{r2.stdout.splitlines()[0]}")

    print("\n结论：" + ("环境就绪" if ok else "还缺组件，见 README"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="CS5185 内容自适应视频编码")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("features", help="提取内容特征")
    p1.add_argument("--ids", nargs="*", default=None)
    p1.add_argument("--stride", type=int, default=1)
    p1.set_defaults(func=cmd_features)

    p2 = sub.add_parser("plan", help="生成参数决策表")
    p2.add_argument("--ids", nargs="*", default=None)
    p2.set_defaults(func=cmd_plan)

    p3 = sub.add_parser("encode", help="跑完整闭环（需 FFmpeg）")
    p3.add_argument("--ids", nargs="*", default=None)
    p3.add_argument("--ratios", default=None, help="如 0.05,0.02")
    p3.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive",
                    help="fixed = 统一基线对照组")
    p3.set_defaults(func=cmd_encode)

    p4 = sub.add_parser("report", help="画特征时间线")
    p4.set_defaults(func=cmd_report)

    p5 = sub.add_parser("check-env", help="检查 FFmpeg")
    p5.set_defaults(func=cmd_check_env)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
