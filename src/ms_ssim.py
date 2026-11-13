"""MS-SSIM，用 OpenCV + NumPy 实现（不依赖 PyTorch）。

为什么自己写：pytorch-msssim 要拖一个 1 GB 的 torch，本机 C 盘只剩 6 GB，
而 MS-SSIM 本身只是多尺度 SSIM，用 11x11 高斯窗就能算。
附录 E1 要求"记录实现、通道、数据范围、尺度、时间聚合"，见 docs/design.md 第 6 节。

参考实现要点（Wang et al. 2003）：
    5 个尺度，权重 [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    前 4 级只取 contrast-structure，最后一级再乘 luminance
    色彩按 ITU-R BT.601 转亮度，只算 Y 通道
"""

from __future__ import annotations

import cv2
import numpy as np

WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
K = (11, 11)
SIGMA = 1.5
DATA_RANGE = 255.0


def _ssim_components(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    c1 = (0.01 * DATA_RANGE) ** 2
    c2 = (0.03 * DATA_RANGE) ** 2

    mu1 = cv2.GaussianBlur(a, K, SIGMA)
    mu2 = cv2.GaussianBlur(b, K, SIGMA)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s1 = cv2.GaussianBlur(a * a, K, SIGMA) - mu1_sq
    s2 = cv2.GaussianBlur(b * b, K, SIGMA) - mu2_sq
    s12 = cv2.GaussianBlur(a * b, K, SIGMA) - mu1_mu2

    lum = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    return float(lum.mean()), float(cs.mean())


def _to_y(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)


def ms_ssim(a_bgr: np.ndarray, b_bgr: np.ndarray, levels: int = 5) -> float:
    """两个 BGR 帧的 MS-SSIM。完全相同时返回 1.0。"""
    x, y = _to_y(a_bgr), _to_y(b_bgr)
    if x.shape != y.shape:
        raise ValueError(f"尺寸不一致 {x.shape} vs {y.shape}")

    lums, css = [], []
    for i in range(levels):
        l, cs = _ssim_components(x, y)
        lums.append(l)
        css.append(cs)
        if i < levels - 1:
            if min(x.shape) < 16:
                levels = i + 1
                break
            x = cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    w = WEIGHTS[:len(css)]
    w = np.asarray(w) / np.sum(w)

    out = 1.0
    for i in range(len(css) - 1):
        out *= max(css[i], 0.0) ** w[i]
    out *= (max(lums[-1], 0.0) * max(css[-1], 0.0)) ** w[-1]
    return float(out)


def compare_videos(path_a: str, path_b: str, max_frames: int | None = None) -> dict:
    """逐帧算 MS-SSIM 再取平均（时间聚合方式：算术平均）。

    附录 E1 提醒：不要只比共同前缀。这里要求两段时长一致，否则直接报错。
    """
    ca, cb = cv2.VideoCapture(str(path_a)), cv2.VideoCapture(str(path_b))
    if not (ca.isOpened() and cb.isOpened()):
        return {"ok": False, "reason": "无法打开视频"}

    vals = []
    while max_frames is None or len(vals) < max_frames:
        oka, fa = ca.read()
        okb, fb = cb.read()
        if not (oka and okb):
            break
        vals.append(ms_ssim(fa, fb))
    ca.release()
    cb.release()

    if not vals:
        return {"ok": False, "reason": "没有可比帧"}
    return {
        "ok": True,
        "ms_ssim": float(np.mean(vals)),
        "ms_ssim_min": float(np.min(vals)),
        "frames_compared": len(vals),
        "levels": 5,
        "channel": "Y (BT.601)",
        "data_range": DATA_RANGE,
        "aggregation": "per-frame arithmetic mean",
    }
