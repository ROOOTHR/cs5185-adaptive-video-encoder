"""FFmpeg 调用封装（项目第 3 步）。

严格按附录 B3/B4 的命令形式，不做任何"聪明"的改写：
    两遍编码，第二遍带 +faststart
    解码校验用 -xerror
Windows 用 NUL，Linux/macOS 用 /dev/null。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import config
from .decide import ffmpeg_args

NULL = "NUL" if os.name == "nt" else "/dev/null"


def _exe(name: str) -> str:
    """找可执行文件：优先 PATH，其次 imageio-ffmpeg 自带的构建。

    本机走第二条路——PATH 里没有 FFmpeg，但 .venv 里的 imageio-ffmpeg
    携带的是 gyan.dev essentials 构建（ffmpeg 9.0.2，含 libx264）。
    """
    import shutil
    p = shutil.which(name)
    if p:
        return p
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    raise FileNotFoundError(
        f"找不到 {name}。装 FFmpeg 并加入 PATH，或在 .venv 里 pip install imageio-ffmpeg。"
    )


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout)


def two_pass(src: Path, out: Path, d: dict, bud: dict, job: str) -> dict:
    """执行两遍编码。返回每次调用的 returncode 与耗时。"""
    ff = _exe("ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    job_path = str(out.parent / job)

    results = []
    for pass_no in (1, 2):
        args = ffmpeg_args(d, bud, job_path, pass_no, str(out))
        cmd = [ff, "-y", "-i", str(src)] + args
        r = run(cmd)
        results.append({
            "pass": pass_no,
            "returncode": r.returncode,
            "cmd": cmd,
            "stderr_tail": (r.stderr or "")[-1200:],
        })
        if r.returncode != 0:
            break
    return {"passes": results, "output": str(out)}


def decode_check(path: Path) -> dict:
    """附录 B4：只校验可解码，不落盘重构帧。"""
    ff = _exe("ffmpeg")
    r = run([ff, "-v", "error", "-xerror", "-i", str(path),
             "-map", "0:v:0", "-f", "null", NULL])
    return {"ok": r.returncode == 0, "returncode": r.returncode,
            "stderr": (r.stderr or "")[-800:]}


def psnr(distorted: Path, reference: Path) -> dict:
    """亮度 PSNR。注意顺序：第一个输入是被测，第二个是参考。"""
    ff = _exe("ffmpeg")
    r = run([ff, "-hide_banner", "-i", str(distorted), "-i", str(reference),
             "-lavfi", "[0:v][1:v]psnr", "-f", "null", NULL])
    out = {"ok": False, "psnr_y_db": None, "psnr_avg_db": None}
    for line in (r.stderr or "").splitlines():
        if "PSNR y:" in line:
            try:
                seg = line.split("PSNR y:")[1].split()[0]
                out["psnr_y_db"] = float(seg)
                if "average:" in line:
                    out["psnr_avg_db"] = float(line.split("average:")[1].split()[0])
                out["ok"] = True
            except (IndexError, ValueError):
                pass
    if out["psnr_y_db"] == float("inf"):
        out["psnr_y_db"] = None
        out["note"] = "完全一致，PSNR 为无穷"
    return out


def vbv_retry_hint(bud: dict, actual_bytes: int) -> int:
    """附录 B2 的修正公式：next_bps = bps * target/actual * 0.98"""
    if actual_bytes <= 0:
        return bud["video_kbps"]
    nxt = bud["video_kbps"] * bud["target_bytes"] / actual_bytes * config.OVERHEAD
    return max(50, int(round(nxt)))
