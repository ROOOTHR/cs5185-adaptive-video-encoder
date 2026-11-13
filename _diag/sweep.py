"""参数扫描：用控制变量法找出真正影响画质的参数。

背景：第一版规则表在 24 个任务上平均输给基线 0.23 dB，输的集中在 g=30 / bf=2 的组。
这里逐项验证，不猜。
"""

import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe
from src import metadata as M

ff = _exe("ffmpeg")


def encode_psnr(src, out, kbps, extra):
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps}k"] + extra
    job = out.replace(".mp4", "_log")
    r1 = subprocess.run(base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"],
                        capture_output=True, text=True, errors="replace")
    r2 = subprocess.run(base + ["-pass", "2", "-passlogfile", job, "-an", out],
                        capture_output=True, text=True, errors="replace")
    if r2.returncode != 0:
        return None, None

    r3 = subprocess.run([ff, "-hide_banner", "-i", out, "-i", src,
                         "-lavfi", "[0:v][1:v]psnr", "-f", "null", "NUL"],
                        capture_output=True, text=True, errors="replace")
    psnr = None
    for line in r3.stderr.splitlines():
        if "PSNR y:" in line:
            psnr = float(line.split("PSNR y:")[1].split()[0])
    size = os.path.getsize(out)
    return psnr, size


def scan(vid, ratio, variants, label):
    meta = M.load_meta(vid)
    bud = M.budget(meta, ratio)
    k = bud["video_kbps"]
    src = f"data/raw/CS5185_Project_Video_Dataset/{vid}.mp4"
    dur = meta["duration"]

    print(f"\n=== {vid} @ {ratio:.0%} (目标 {k} kbps, {dur}s) — 扫描 {label} ===")
    print(f"{'取值':<22}{'PSNR-Y':>9}{'体积MB':>9}{'实码率':>9}")
    best = None
    for tag, extra in variants:
        out = f"_diag/sw_{vid}_{ratio}_{tag}.mp4"
        p, sz = encode_psnr(src, out, k, extra)
        if p is None:
            print(f"{tag:<22}  失败")
            continue
        kb = 8 * sz / dur / 1000
        print(f"{tag:<22}{p:>9.3f}{sz/1e6:>9.2f}{kb:>9.0f}")
        if best is None or p > best[1]:
            best = (tag, p)
    if best:
        print(f"  → 最佳 {best[0]}  {best[1]:.3f} dB")
    return best
