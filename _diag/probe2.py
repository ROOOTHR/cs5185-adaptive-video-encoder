import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe

ff = _exe("ffmpeg")
src = "data/raw/CS5185_Project_Video_Dataset/02_Beauty.mp4"


def two_pass(target_k, tag, p1_tail=("NUL",)):
    job = f"_diag/q_{tag}"
    out = f"_diag/q_{tag}.mp4"
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", f"{target_k}k"]
    c1 = base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null"] + list(p1_tail)
    c2 = base + ["-pass", "2", "-passlogfile", job, "-an", out]
    subprocess.run(c1, capture_output=True, text=True, errors="replace")
    subprocess.run(c2, capture_output=True, text=True, errors="replace")
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{tag:<18} target={target_k:5d}k  ->  {size/1e6:6.2f}MB  {8*size/5/1000:6.0f}k")


def one_pass(target_k, tag):
    out = f"_diag/q_{tag}.mp4"
    subprocess.run([ff, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-b:v", f"{target_k}k", "-an", out],
                   capture_output=True, text=True, errors="replace")
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{tag:<18} target={target_k:5d}k  ->  {size/1e6:6.2f}MB  {8*size/5/1000:6.0f}k")


for k in (1000, 2000, 4000, 6828):
    one_pass(k, f"1pass_{k}")
for k in (1000, 2000, 4000, 6828):
    two_pass(k, f"2pass_{k}")
