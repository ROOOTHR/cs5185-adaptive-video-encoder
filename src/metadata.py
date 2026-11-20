"""Dataset metadata and exact target-budget calculation."""
from __future__ import annotations

import json
import math
from . import config


def load_meta(video_id: str) -> dict:
    if video_id not in config.ALL_IDS:
        raise ValueError(f"unknown video id: {video_id}")
    path = config.META_DIR / f"{video_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    probe = meta.get("probe", {})
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = probe.get("format", {})
    meta.update({
        "video_id": video_id,
        "file": meta.get("file", f"{video_id}.mp4"),
        "bytes": int(meta.get("bytes") or fmt.get("size") or 0),
        "duration": float(meta.get("duration") or fmt.get("duration") or 0),
        "width": int(meta.get("width") or stream.get("width") or 0),
        "height": int(meta.get("height") or stream.get("height") or 0),
        "fps": float(meta.get("fps") or 30),
        "frames": int(stream.get("nb_frames") or round(float(meta.get("duration") or 0) * 30)),
    })
    if not all(meta[k] for k in ("bytes", "duration", "width", "height", "fps", "frames")):
        raise ValueError(f"incomplete metadata for {video_id}")
    return meta


def budget(meta: dict, ratio: float) -> dict:
    target_bytes = math.floor(meta["bytes"] * ratio)
    video_bps = config.OVERHEAD * 8 * target_bytes / meta["duration"]
    return {"ratio": ratio, "src_bytes": meta["bytes"], "target_bytes": target_bytes,
            "video_bps": video_bps, "video_kbps": round(video_bps / 1000),
            "duration": meta["duration"]}


def list_ids() -> list[str]:
    return list(config.ALL_IDS)
