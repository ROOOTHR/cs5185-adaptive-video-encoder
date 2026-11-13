import os
import subprocess
import sys

sys.path.insert(0, ".")
from src.encode import _exe

ff = _exe("ffmpeg")
src = "data/raw/CS5185_Project_Video_Dataset/02_Beauty.mp4"
TARGET = 6828


def run(extra, tag):
    job = f"_diag/p_{tag}"
    out = f"_diag/o_{tag}.mp4"
    base = [ff, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", f"{TARGET}k"] + extra
    c1 = base + ["-pass", "1", "-passlogfile", job, "-an", "-f", "null", "NUL"]
    c2 = base + ["-pass", "2", "-passlogfile", job, "-an",
                 "-movflags", "+faststart", out]
    r1 = subprocess.run(c1, capture_output=True, text=True, errors="replace")
    r2 = subprocess.run(c2, capture_output=True, text=True, errors="replace")
    size = os.path.getsize(out) if os.path.exists(out) else 0
    kbl = [l for l in r2.stderr.splitlines() if "kb/s:" in l]
    rep = kbl[-1].split("kb/s:")[-1].strip() if kbl else "?"
    warn = [l for l in (r1.stderr + r2.stderr).splitlines()
            if "arning" in l or "not found" in l.lower()]
    print(f"{tag:<12} {size/1e6:6.2f}MB  {8*size/5/1000:6.0f}k  x264={rep}  warn={len(warn)}")
    for w in warn[:2]:
        print("    !", w.strip()[:130])


for extra, tag in [
    ([], "minimal"),
    (["-preset", "medium"], "preset"),
    (["-preset", "medium", "-g", "120"], "g120"),
    (["-preset", "medium", "-g", "120", "-bf", "3"], "bf3"),
    (["-preset", "medium", "-g", "120", "-bf", "3", "-refs", "5"], "refs5"),
    (["-preset", "medium", "-g", "120", "-bf", "3", "-refs", "5",
      "-aq-mode", "1", "-aq-strength", "0.8"], "full"),
]:
    run(extra, tag)
