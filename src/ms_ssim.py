"""Y-plane MS-SSIM matching pytorch-msssim's 2-D reference implementation."""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from .encode import _exe

WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
WINDOW = 11
SIGMA = 1.5
DATA_RANGE = 255.0
K1, K2 = 0.01, 0.03

_COORD = np.arange(WINDOW, dtype=np.float64) - WINDOW // 2
_GAUSS = np.exp(-(_COORD ** 2) / (2 * SIGMA ** 2))
_GAUSS /= _GAUSS.sum()
_RADIUS = WINDOW // 2


def _valid_filter(image: np.ndarray) -> np.ndarray:
    # The reference implementation uses unpadded separable convolutions.
    filtered = cv2.sepFilter2D(image, cv2.CV_64F, _GAUSS, _GAUSS,
                                borderType=cv2.BORDER_REFLECT)
    return filtered[_RADIUS:-_RADIUS, _RADIUS:-_RADIUS]


def _components(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mu1, mu2 = _valid_filter(a), _valid_filter(b)
    s1 = _valid_filter(a * a) - mu1 * mu1
    s2 = _valid_filter(b * b) - mu2 * mu2
    s12 = _valid_filter(a * b) - mu1 * mu2
    c1, c2 = (K1 * DATA_RANGE) ** 2, (K2 * DATA_RANGE) ** 2
    lum = (2 * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1)
    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    return float(lum.mean()), float(cs.mean())


def ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a single-frame 5-scale MS-SSIM from grayscale/Y or BGR images."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"frame shape mismatch: {a.shape} vs {b.shape}")
    if min(a.shape) <= (WINDOW - 1) * 16:
        raise ValueError("five-scale MS-SSIM requires each frame dimension > 160")
    x, y = a.astype(np.float64), b.astype(np.float64)
    factors = []
    for level in range(len(WEIGHTS)):
        lum, cs = _components(x, y)
        factors.append(max(lum * cs, 0.0) if level == len(WEIGHTS) - 1 else max(cs, 0.0))
        if level < len(WEIGHTS) - 1:
            h, w = x.shape
            x = x[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
            y = y[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return float(np.prod(np.asarray(factors) ** np.asarray(WEIGHTS)))


def _reader(path: str | Path, width: int, height: int):
    ff = _exe("ffmpeg")
    stderr = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [ff, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
         "-an", "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE, stderr=stderr)
    return process, stderr, width * height


def _finish(reader) -> tuple[int, str]:
    process, stderr, _ = reader
    if process.stdout:
        process.stdout.close()
    code = process.wait(timeout=120)
    stderr.seek(0)
    message = stderr.read().decode("utf-8", "replace")
    stderr.close()
    return code, message[-1000:]


def compare_videos(reference: str | Path, distorted: str | Path, width: int,
                   height: int, expected_frames: int) -> dict:
    """Stream exactly paired raw Y frames; fail on either short or extra stream."""
    ra = rb = None
    values, psnr_frames, squared_error = [], [], 0.0
    pixels = 0
    mismatch = None
    try:
        ra = _reader(reference, width, height)
        rb = _reader(distorted, width, height)
        size = width * height
        while True:
            a = ra[0].stdout.read(size)
            b = rb[0].stdout.read(size)
            if not a and not b:
                break
            if len(a) != size or len(b) != size:
                mismatch = f"partial/mismatched raw frame at index {len(values)}"
                # Drain both streams before waiting to avoid leaving FFmpeg running.
                while ra[0].stdout.read(size):
                    pass
                while rb[0].stdout.read(size):
                    pass
                break
            x = np.frombuffer(a, np.uint8).reshape(height, width)
            y = np.frombuffer(b, np.uint8).reshape(height, width)
            delta = x.astype(np.float64) - y.astype(np.float64)
            mse = float(np.mean(delta * delta))
            squared_error += float(np.sum(delta * delta))
            pixels += delta.size
            psnr_frames.append(float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse))
            values.append(ms_ssim(x, y))
        ca, ea = _finish(ra)
        cb, eb = _finish(rb)
        if ca or cb:
            mismatch = f"FFmpeg raw decode failed ({ca}, {cb}): {ea} {eb}".strip()
        if len(values) != expected_frames:
            mismatch = f"expected {expected_frames} paired frames, received {len(values)}"
        if mismatch:
            return {"ok": False, "reason": mismatch, "frames_compared": len(values)}
    except Exception as exc:
        for reader in (ra, rb):
            if reader:
                try:
                    reader[0].kill()
                    _finish(reader)
                except Exception:
                    pass
        return {"ok": False, "reason": str(exc), "frames_compared": len(values)}
    mean = float(np.mean(values))
    db = math.inf if mean >= 1.0 else -10 * math.log10(1 - mean)
    mse_all = squared_error / pixels
    psnr_all = math.inf if mse_all == 0 else 10 * math.log10(255 ** 2 / mse_all)
    return {"ok": True, "frames_compared": len(values), "ms_ssim": mean,
            "ms_ssim_db": db, "ms_ssim_min": float(np.min(values)),
            "ms_ssim_per_frame": values, "psnr_y_db": psnr_all,
            "psnr_y_per_frame": psnr_frames, "channel": "Y",
            "data_range": 255, "levels": 5, "window": WINDOW,
            "sigma": SIGMA, "weights": list(WEIGHTS),
            "temporal_aggregation": "arithmetic mean of per-frame scores"}


def sample_y_frames(path: str | Path, width: int, height: int,
                    expected_frames: int, count: int = 10) -> tuple[list[int], list[np.ndarray]]:
    """Sample evenly spaced Y frames while verifying the complete decoded stream."""
    reader = _reader(path, width, height)
    indices = sorted(set(np.linspace(0, expected_frames - 1, count, dtype=int).tolist()))
    selected, frame_index = [], 0
    try:
        size = width * height
        while True:
            raw = reader[0].stdout.read(size)
            if not raw:
                break
            if len(raw) != size:
                raise ValueError(f"partial raw frame at index {frame_index}")
            if frame_index in indices:
                selected.append(np.frombuffer(raw, np.uint8).reshape(height, width).copy())
            frame_index += 1
        code, message = _finish(reader)
        if code or frame_index != expected_frames:
            raise ValueError(f"expected {expected_frames} frames; read {frame_index}; ffmpeg={code} {message}")
        if len(selected) != len(indices):
            raise ValueError("could not sample requested frame indices")
        return indices, selected
    except Exception:
        try:
            reader[0].kill()
            _finish(reader)
        except Exception:
            pass
        raise
