import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe
from src import metadata as M

ff = _exe("ffmpeg")


def two_pass(src, kbps, tag, vbv=True):
    job = f"_diag/s_{tag}"
    out = f"_diag/s_{tag}.mp4"
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-b:v", f"{kbps}k"]
    if vbv:
        base += ["-maxrate", f"{kbps}k", "-bufsize", f"{kbps}k"]
    subprocess.run(base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"],
                   capture_output=True, text=True, errors="replace")
    subprocess.run(base + ["-pass", "2", "-passlogfile", job, "-an", out],
                   capture_output=True, text=True, errors="replace")
    return os.path.getsize(out) if os.path.exists(out) else 0


print(f"{'video':<20}{'目标k':>7}{'无VBV':>9}{'偏差':>8}{'有VBV':>9}{'偏差':>8}")
print("-" * 62)
tot_a = tot_b = 0
for vid in M.list_ids():
    meta = M.load_meta(vid)
    bud = M.budget(meta, 0.05)
    k = bud["video_kbps"]
    src = f"data/raw/CS5185_Project_Video_Dataset/{vid}.mp4"
    a = two_pass(src, k, f"{vid}_a", False)
    b = two_pass(src, k, f"{vid}_b", True)
    ka, kb = 8 * a / 5 / 1000, 8 * b / 5 / 1000
    tot_a += abs(ka - k) / k
    tot_b += abs(kb - k) / k
    print(f"{vid:<20}{k:>7}{ka:>9.0f}{(ka-k)/k:>+8.1%}{kb:>9.0f}{(kb-k)/k:>+8.1%}")
print("-" * 62)
print(f"{'平均绝对偏差':<20}{'':>7}{tot_a/12:>9.1%}{'':>8}{tot_b/12:>8.1%}")
