"""OpenCV content features; feature extraction itself does not fit thresholds."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from . import config
from .metadata import list_ids

EDGE_THR = 100.0
GRID_ROWS, GRID_COLS = 9, 16


def _block_stats(diff: np.ndarray) -> tuple[float, float]:
    h, w = diff.shape
    bh, bw = h // GRID_ROWS, w // GRID_COLS
    blocks = diff[:bh * GRID_ROWS, :bw * GRID_COLS].reshape(
        GRID_ROWS, bh, GRID_COLS, bw).mean(axis=(1, 3)).ravel()
    return float(blocks.max()), float(np.percentile(blocks, 90))


def extract(video_path: Path, stride: int = 1) -> dict:
    if stride < 1:
        raise ValueError("stride must be at least one")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    data = {k: [] for k in ("mafd", "block_max", "block_p90", "edge_ratio", "luma_mean", "luma_std")}
    previous = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is None:
                diff = None
                data["mafd"].append(0.0)
                data["block_max"].append(0.0)
                data["block_p90"].append(0.0)
            else:
                diff = cv2.absdiff(gray, previous)
                mx, p90 = _block_stats(diff)
                data["mafd"].append(float(diff.mean()))
                data["block_max"].append(mx)
                data["block_p90"].append(p90)
            gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
            data["edge_ratio"].append(float((magnitude > EDGE_THR).mean()))
            data["luma_mean"].append(float(gray.mean()))
            data["luma_std"].append(float(gray.std()))
            previous = gray
        index += 1
    cap.release()
    n = len(data["mafd"])
    if n == 0:
        raise RuntimeError(f"no decoded frames in {video_path}")

    motion = np.asarray(data["mafd"], dtype=float)
    cuts = []
    for i in range(2, n):
        base = float(np.median(motion[max(0, i - 5):i]))
        if base > 0 and motion[i] > 3 * base and motion[i] > 3.0:
            cuts.append(i)

    def stats(name):
        values = np.asarray(data[name], dtype=float)
        return {"mean": round(float(values.mean()), 6),
                "p90": round(float(np.percentile(values, 90)), 6),
                "max": round(float(values.max()), 6)}

    return {"frames": n, "fps": fps, "duration": n / fps,
            "per_frame": {name: [round(float(v), 6) for v in values]
                          for name, values in data.items()},
            "stats": {name: stats(name) for name in data}, "scene_cuts": cuts}


def run_all(ids: list[str] | None = None, stride: int = 1,
            output_dir: Path | None = None) -> dict[str, dict]:
    ids = ids or list_ids()
    output_dir = output_dir or config.RUN_DIR / "features_v2" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for vid in ids:
        feat = extract(config.RAW_DIR / f"{vid}.mp4", stride)
        feat["video_id"] = vid
        (output_dir / f"{vid}.json").write_text(
            json.dumps(feat, ensure_ascii=False), encoding="utf-8")
        out[vid] = feat
        print(f"  {vid:<20} mafd={feat['stats']['mafd']['mean']:.3f} "
              f"edge={feat['stats']['edge_ratio']['mean']:.5f} "
              f"frames={feat['frames']}")
    return out


def load(video_id: str, feature_dir: Path | None = None) -> dict:
    path = (feature_dir or config.RUN_DIR / "features_v2" / "features") / f"{video_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
