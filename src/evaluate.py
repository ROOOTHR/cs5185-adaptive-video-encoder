"""Hard integrity checks and E1 quality measurements."""
from __future__ import annotations

import re
from pathlib import Path
from . import config
from .encode import _exe, run, decode_check
from .ms_ssim import compare_videos

_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO = re.compile(r"Video:\s*(\w+)")
_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"([\d.]+)\s*fps")
_FRAME = re.compile(r"frame=\s*(\d+)")
_VBYTES = re.compile(r"video:\s*(\d+)KiB")


def probe(path: Path) -> dict:
    ff = _exe("ffmpeg")
    head = run([ff, "-hide_banner", "-i", str(path)])
    text = head.stderr or ""
    result = {"duration": 0.0, "frames": 0, "width": 0, "height": 0,
              "fps": None, "codec": None, "decode_ok": False, "video_bytes": 0}
    m = _DUR.search(text)
    if m:
        h, minute, second = m.groups()
        result["duration"] = int(h) * 3600 + int(minute) * 60 + float(second)
    for line in text.splitlines():
        if "Video:" in line:
            m = _VIDEO.search(line)
            if m:
                result["codec"] = m.group(1)
            m = _SIZE.search(line)
            if m:
                result["width"], result["height"] = map(int, m.groups())
            m = _FPS.search(line)
            if m:
                result["fps"] = float(m.group(1))
            break
    decoded = run([ff, "-hide_banner", "-xerror", "-i", str(path),
                   "-map", "0:v:0", "-an", "-f", "null", "-"])
    frames = _FRAME.findall(decoded.stderr or "")
    result["frames"] = int(frames[-1]) if frames else 0
    result["decode_ok"] = decoded.returncode == 0 and result["frames"] > 0
    packet = run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0",
                  "-c", "copy", "-f", "null", "-"])
    m = _VBYTES.search(packet.stderr or "")
    result["video_bytes"] = int(m.group(1)) * 1024 if m else 0
    return result


def evaluate(out: Path, src: Path, meta: dict, target_bytes: int) -> dict:
    size = out.stat().st_size if out.is_file() else 0
    pr = probe(out) if size else {"duration": 0, "frames": 0, "width": 0,
                                  "height": 0, "fps": None, "codec": None,
                                  "decode_ok": False, "video_bytes": 0}
    duration = pr["duration"]
    result = {
        "output_path": str(out), "actual_bytes": size, "target_bytes": target_bytes,
        "actual_MB": size / config.MB, "target_MB": target_bytes / config.MB,
        "size_error": (size - target_bytes) / target_bytes if target_bytes else None,
        "duration_s": duration, "frame_count": pr["frames"],
        "expected_frames": meta["frames"], "width": pr["width"], "height": pr["height"],
        "fps": pr["fps"], "codec": pr["codec"],
        "file_kbps": 8 * size / duration / 1000 if duration else None,
        "actual_video_kbps": 8 * pr["video_bytes"] / duration / 1000 if duration and pr["video_bytes"] else None,
        "decode_ok": pr["decode_ok"],
    }
    errors = []
    if not pr["decode_ok"]:
        errors.append("decode failed")
    if pr["frames"] != meta["frames"]:
        errors.append(f"frame count {pr['frames']} != {meta['frames']}")
    if not duration or abs(duration - meta["duration"]) > 1 / meta["fps"] + 1e-6:
        errors.append(f"duration {duration} differs from {meta['duration']}s by more than one frame")
    if (pr["width"], pr["height"]) != (meta["width"], meta["height"]):
        errors.append("resolution changed")
    if pr["fps"] is None or abs(pr["fps"] - meta["fps"]) > 1e-6:
        errors.append("frame rate changed or could not be verified")
    if size > target_bytes:
        errors.append("actual_bytes exceeds target_bytes")
    if target_bytes and size < target_bytes * 0.97:
        errors.append("actual_bytes is more than 3% below target_bytes")
    result["integrity_errors"] = errors
    result["valid"] = not errors
    if pr["decode_ok"] and pr["frames"] == meta["frames"] and (pr["width"], pr["height"]) == (meta["width"], meta["height"]):
        metrics = compare_videos(src, out, meta["width"], meta["height"], meta["frames"])
        if metrics["ok"]:
            result.update(metrics)
        else:
            result["integrity_errors"].append("quality alignment failed: " + metrics["reason"])
            result["valid"] = False
    result["failure_reason"] = "; ".join(result["integrity_errors"])
    return result
