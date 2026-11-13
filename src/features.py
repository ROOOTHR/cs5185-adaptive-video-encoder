"""内容特征分析（项目第 1 步）。

只做四件事，每件都能在报告里讲清"它衡量什么、为什么和码率分配有关"：

1. mafd        相邻帧灰度平均绝对差 —— 整体时间变化强度
2. block_max   16x9 分块后帧差最大的那块 —— 局部剧烈运动（足球、奔跑）
3. edge_ratio  强梯度像素占比 —— 细节/纹理密度（决定该不该保高频）
4. luma_std    灰度标准差 —— 对比度，低对比场景更容易看出块效应

不做光流、不做块匹配：本项目只有 150 帧，帧差已经够用，且实现透明可解释。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import config
from .metadata import list_ids

EDGE_THR = 100.0        # Sobel 幅值阈值（ksize=3 时幅值上限约 1442）
GRID_ROWS, GRID_COLS = 9, 16
SCENE_CUT_K = 3.0       # 帧差超过滑动中位数的 3 倍，判为场景切换


def _block_stats(diff: np.ndarray) -> tuple[float, float]:
    """把帧差图切成 9x16 块，返回（最大块均值, 90 分位块均值）。"""
    h, w = diff.shape
    bh, bw = h // GRID_ROWS, w // GRID_COLS
    d = diff[: bh * GRID_ROWS, : bw * GRID_COLS]
    d = d.reshape(GRID_ROWS, bh, GRID_COLS, bw).mean(axis=(1, 3))
    flat = d.ravel()
    return float(flat.max()), float(np.percentile(flat, 90))


def extract(video_path: Path, stride: int = 1) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mafd, bmax, bp90, edge, luma, lstd = [], [], [], [], [], []
    prev = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is None:
            mafd.append(0.0)
            bmax.append(0.0)
            bp90.append(0.0)
        else:
            diff = cv2.absdiff(g, prev)
            mx, p90 = _block_stats(diff)
            mafd.append(float(diff.mean()))
            bmax.append(mx)
            bp90.append(p90)

        gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        edge.append(float((mag > EDGE_THR).mean()))

        luma.append(float(g.mean()))
        lstd.append(float(g.std()))

        prev = g
        idx += 1

    cap.release()
    n = len(mafd)
    if n == 0:
        raise RuntimeError(f"{video_path} 读不到帧")

    arr = lambda a: np.asarray(a, dtype=float)
    m = arr(mafd)

    # 场景切换：帧差突增
    cuts = []
    for i in range(2, n):
        lo = max(0, i - 5)
        base = float(np.median(m[lo:i])) if i > lo else 0.0
        if base > 0 and m[i] > SCENE_CUT_K * base and m[i] > 3.0:
            cuts.append(i)

    def stat(a):
        v = arr(a)
        return {"mean": round(float(v.mean()), 3),
                "p90": round(float(np.percentile(v, 90)), 3),
                "max": round(float(v.max()), 3)}

    return {
        "frames": n,
        "fps": fps,
        "duration": round(n / fps, 2),
        "per_frame": {
            "mafd": [round(x, 3) for x in mafd],
            "block_max": [round(x, 3) for x in bmax],
            "edge_ratio": [round(x, 5) for x in edge],
            "luma_mean": [round(x, 2) for x in luma],
            "luma_std": [round(x, 2) for x in lstd],
        },
        "stats": {
            "mafd": stat(mafd),
            "block_max": stat(bmax),
            "edge_ratio": stat(edge),
            "luma_std": stat(lstd),
        },
        "scene_cuts": cuts,
    }


def run_all(ids: list[str] | None = None, stride: int = 1) -> dict[str, dict]:
    config.ensure_dirs()
    ids = ids or list_ids()
    out = {}
    for vid in ids:
        path = config.RAW_DIR / f"{vid}.mp4"
        feat = extract(path, stride=stride)
        feat["video_id"] = vid
        jf = config.FEATURE_DIR / f"{vid}.json"
        jf.write_text(json.dumps(feat, ensure_ascii=False), encoding="utf-8")
        out[vid] = feat
        s = feat["stats"]
        print(f"  {vid:<20} mafd={s['mafd']['mean']:6.2f} "
              f"blkmax={s['block_max']['mean']:6.2f} "
              f"edge={s['edge_ratio']['mean']:.4f} "
              f"cuts={len(feat['scene_cuts'])}")
    return out


def load(video_id: str) -> dict:
    jf = config.FEATURE_DIR / f"{video_id}.json"
    return json.loads(jf.read_text(encoding="utf-8"))
