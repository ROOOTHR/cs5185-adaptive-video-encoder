import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe

ff = _exe("ffmpeg")
CASES = [("01_FourPeople", 5010), ("02_Beauty", 6828), ("09_Parkrun", 9092),
         ("06_ShakeNDry", 6458)]


def two_pass(src, kbps, tag, maxr, bufs):
    job = f"_diag/t_{tag}"
    out = f"_diag/t_{tag}.mp4"
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-b:v", f"{kbps}k",
            "-maxrate", f"{int(kbps*maxr)}k", "-bufsize", f"{int(kbps*bufs)}k"]
    subprocess.run(base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"],
                   capture_output=True, text=True, errors="replace")
    subprocess.run(base + ["-pass", "2", "-passlogfile", job, "-an", out],
                   capture_output=True, text=True, errors="replace")
    return os.path.getsize(out) if os.path.exists(out) else 0


COMBOS = [(1.0, 1.0), (1.2, 1.0), (1.3, 1.2), (1.5, 1.0)]

print(f"{'video':<16}{'目标':>6}" + "".join(f"{f'{m}x/{b}x':>12}" for m, b in COMBOS))
print("-" * 66)
acc = {c: 0.0 for c in COMBOS}
for vid, k in CASES:
    src = f"data/raw/CS5185_Project_Video_Dataset/{vid}.mp4"
    line = f"{vid:<16}{k:>6}"
    for c in COMBOS:
        sz = two_pass(src, k, f"{vid}_{c[0]}_{c[1]}", *c)
        dev = (8 * sz / 5 / 1000 - k) / k
        acc[c] += abs(dev)
        line += f"{dev:>+12.1%}"
    print(line)
print("-" * 66)
print(f"{'平均绝对偏差':<16}{'':>6}" + "".join(f"{acc[c]/len(CASES):>12.1%}" for c in COMBOS))
