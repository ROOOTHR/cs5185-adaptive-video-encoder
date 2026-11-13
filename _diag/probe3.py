import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe

ff = _exe("ffmpeg")
SRC = "data/raw/CS5185_Project_Video_Dataset/02_Beauty.mp4"


def two_pass(src, target_k, tag, extra=()):
    job = f"_diag/r_{tag}"
    out = f"_diag/r_{tag}.mp4"
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", f"{target_k}k"] + list(extra)
    subprocess.run(base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"],
                   capture_output=True, text=True, errors="replace")
    subprocess.run(base + ["-pass", "2", "-passlogfile", job, "-an", out],
                   capture_output=True, text=True, errors="replace")
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return size


print("02_Beauty @ 6828k")
for extra, tag in [
    ((), "baseline"),
    (("-x264-params", "mbtree=0"), "no_mbtree"),
    (("-x264-params", "nal-hrd=cbr"), "cbr"),
    (("-x264-params", "mbtree=0:nal-hrd=cbr"), "both"),
    (("-maxrate", "6828k", "-bufsize", "6828k"), "vbv_1s"),
    (("-maxrate", "3414k", "-bufsize", "6828k"), "vbv_2s"),
]:
    sz = two_pass(SRC, 6828, tag, extra)
    print(f"  {tag:<12} {sz/1e6:6.2f}MB  {8*sz/5/1000:6.0f}k")

print("\n其他视频 @ 各自 5% 目标，mbtree=0 对比")
for vid, k in [("01_FourPeople", 5010), ("09_Parkrun", 9092), ("11_Sintel", 1269)]:
    src = f"data/raw/CS5185_Project_Video_Dataset/{vid}.mp4"
    a = two_pass(src, k, f"{vid}_on")
    b = two_pass(src, k, f"{vid}_off", ("-x264-params", "mbtree=0"))
    print(f"  {vid:<16} 目标{k:5d}k  mbtree开 {8*a/5/1000:6.0f}k   mbtree关 {8*b/5/1000:6.0f}k")
