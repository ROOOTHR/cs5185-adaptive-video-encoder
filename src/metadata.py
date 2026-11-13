"""元数据读取与预算计算（附录 B2）。

不依赖 ffprobe：数据集已附带 metadata/*.json，内含完整 probe 结果。
若某文件缺 json，则回退到调用 ffprobe（此时才需要 ffmpeg 在 PATH 里）。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from . import config


def _probe_fallback(video_path: Path) -> dict:
    """没有 json 时用 ffprobe 现场读。需要 ffmpeg 已安装。"""
    exe = shutil.which("ffprobe")
    if not exe:
        raise FileNotFoundError(
            f"缺少 {video_path.name} 的 metadata json，且 ffprobe 不在 PATH 中"
        )
    out = subprocess.run(
        [exe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {"file": video_path.name, "probe": json.loads(out)}


def load_meta(video_id: str) -> dict:
    """读一个视频的元数据。video_id 形如 '01_FourPeople'。"""
    jf = config.META_DIR / f"{video_id}.json"
    if jf.exists():
        meta = json.loads(jf.read_text(encoding="utf-8"))
    else:
        meta = _probe_fallback(config.RAW_DIR / f"{video_id}.mp4")

    # 统一成后续代码依赖的字段
    probe = meta.get("probe", {})
    st = probe.get("streams", [{}])[0]
    fmt = probe.get("format", {})

    meta["file"] = meta.get("file") or f"{video_id}.mp4"
    meta["bytes"] = int(meta.get("bytes") or fmt.get("size") or 0)
    meta["duration"] = float(meta.get("duration") or fmt.get("duration") or 0)
    meta["width"] = int(meta.get("width") or st.get("width") or 0)
    meta["height"] = int(meta.get("height") or st.get("height") or 0)
    meta["fps"] = float(meta.get("fps") or 30)
    meta["frames"] = int(st.get("nb_frames") or round(meta["duration"] * meta["fps"]))
    meta["video_id"] = video_id
    return meta


def budget(meta: dict, ratio: float) -> dict:
    """附录 B2 的预算公式。

    target_bytes = floor(S * r)
    video_bps    = 0.98 * 8 * target_bytes / T
    """
    s = meta["bytes"]
    t = meta["duration"]
    target_bytes = math.floor(s * ratio)
    video_bps = config.OVERHEAD * 8 * target_bytes / t
    return {
        "ratio": ratio,
        "src_bytes": s,
        "target_bytes": target_bytes,
        "video_bps": video_bps,
        "video_kbps": round(video_bps / 1000),
        "duration": t,
    }


def list_ids() -> list[str]:
    """数据集里所有 video_id，按文件名排序。"""
    return sorted(p.stem for p in config.RAW_DIR.glob("*.mp4"))


def summary_table() -> list[dict]:
    rows = []
    for vid in list_ids():
        m = load_meta(vid)
        for r in config.RATIOS:
            b = budget(m, r)
            rows.append({
                "video_id": vid,
                "duration": m["duration"],
                "src_MB": round(m["bytes"] / config.MB, 2),
                "ratio": f"{r:.0%}",
                "target_MB": round(b["target_bytes"] / config.MB, 2),
                "target_kbps": b["video_kbps"],
            })
    return rows
