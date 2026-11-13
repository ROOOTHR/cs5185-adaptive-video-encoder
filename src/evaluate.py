"""结果检查（项目第 4 步）。

指标口径按附录 E1：
    file_kbps  = 8 * 实际文件字节 / 时长        （含容器开销）
    video_kbps = 8 * 视频包字节 / 时长          （纯码流）
两者必须分开记录。

不依赖 ffprobe：本机 FFmpeg 来自 imageio-ffmpeg，只带 ffmpeg 不带 ffprobe。
所需信息从 `ffmpeg -i` 的 stderr、一次 null 解码计数、一次 -c copy 统计中提取。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import config
from .encode import _exe, decode_check, psnr, run

_RE_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RE_VIDEO = re.compile(r"Video:\s*(\w+)")
_RE_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")
_RE_FPS = re.compile(r"([\d.]+)\s*fps")
_RE_FRAME = re.compile(r"frame=\s*(\d+)")
_RE_VBYTES = re.compile(r"video:\s*(\d+)KiB")


def probe(path: Path) -> dict:
    """读时长、编码、分辨率、帧率、解码帧数。"""
    ff = _exe("ffmpeg")
    r = run([ff, "-hide_banner", "-i", str(path)])
    s = r.stderr or ""

    info = {"ok": False, "duration": 0.0, "frames": 0,
            "width": 0, "height": 0, "codec": None, "fps": None}

    m = _RE_DUR.search(s)
    if m:
        h, mi, sec = m.groups()
        info["duration"] = int(h) * 3600 + int(mi) * 60 + float(sec)
        info["ok"] = True

    m = _RE_VIDEO.search(s)
    if m:
        info["codec"] = m.group(1)

    # 只取 Stream 行里的尺寸，避免误匹配 SAR/DAR
    for line in s.splitlines():
        if "Video:" in line:
            mm = _RE_SIZE.search(line)
            if mm:
                info["width"], info["height"] = int(mm.group(1)), int(mm.group(2))
            mf = _RE_FPS.search(line)
            if mf:
                info["fps"] = float(mf.group(1))
            break

    # 帧数：解码一遍计数（150 帧约 0.16 秒）
    r2 = run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])
    frames = _RE_FRAME.findall(r2.stderr or "")
    if frames:
        info["frames"] = int(frames[-1])
    return info


def video_packet_bytes(path: Path) -> int:
    """视频流字节数（不含容器开销）。用 -c copy 到 null 的统计。"""
    ff = _exe("ffmpeg")
    r = run([ff, "-hide_banner", "-i", str(path), "-c", "copy", "-f", "null", "-"])
    m = _RE_VBYTES.search(r.stderr or "")
    return int(m.group(1)) * 1024 if m else 0


def evaluate(out: Path, src: Path, src_bytes: int, target_bytes: int,
             with_psnr: bool = True) -> dict:
    size = out.stat().st_size if out.exists() else 0
    pr = probe(out)
    dur = pr.get("duration") or 0
    vb = video_packet_bytes(out) if pr.get("ok") else 0

    res = {
        "output_path": str(out),
        "actual_bytes": size,
        "target_bytes": target_bytes,
        "actual_MB": round(size / config.MB, 3),
        "target_MB": round(target_bytes / config.MB, 3),
        "ratio_achieved": round(size / src_bytes, 5) if src_bytes else None,
        "size_error": round((size - target_bytes) / target_bytes, 4) if target_bytes else None,
        "duration_s": dur,
        "frame_count": pr.get("frames"),
        "width": pr.get("width"),
        "height": pr.get("height"),
        "codec": pr.get("codec"),
        "file_kbps": round(8 * size / dur / 1000, 1) if dur else None,
        "video_kbps": round(8 * vb / dur / 1000, 1) if dur and vb else None,
    }

    dc = decode_check(out)
    res["decode_ok"] = dc["ok"]
    if not dc["ok"]:
        res["decode_error"] = dc["stderr"]

    if dc["ok"] and with_psnr:
        res.update(psnr(out, src))
    return res
