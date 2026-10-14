"""Mastering helpers: loudness to target with a lookahead true-peak limiter, verified by ffmpeg ebur128.

pedalboard's JUCE Limiter adds make-up gain equal to -threshold (it drives peaks to 0 dBFS), so the
ceiling stage is done here: 4x-oversampled peak detection, lookahead min-hold, smoothed gain.
"""
import re
import subprocess

import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly


def true_peak_env(x: np.ndarray) -> np.ndarray:
    """Per-sample max |x| over the 4x-oversampled signal (inter-sample peaks), any channel."""
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(axis=1)
    n = len(x)
    return up[: n * 4].reshape(n, 4).max(axis=1)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.3, lookahead_ms: float = 8.0, release_ms: float = 60.0) -> np.ndarray:
    env = true_peak_env(x)
    ceiling = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    la = max(1, int(sr * lookahead_ms / 1000))
    held = minimum_filter1d(need, size=2 * la + 1, mode="nearest")  # reduction starts `la` before each peak
    smooth = uniform_filter1d(held, size=la, mode="nearest")        # ramps within the hold, never above `held` at peaks
    rel = max(1, int(sr * release_ms / 1000))
    released = minimum_filter1d(smooth, size=rel, mode="nearest", origin=-(rel // 2))  # hold reduction for the release time
    gain = np.minimum(smooth, uniform_filter1d(released, size=rel, mode="nearest"))
    return x * gain[:, None]


def measure(path: str) -> dict:
    """Integrated loudness, loudness range and true peak by ffmpeg's ebur128 filter."""
    p = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    s = p.stderr[p.stderr.rfind("Summary:"):]
    return {"lufs": float(re.search(r"I:\s+(-?[\d.]+) LUFS", s).group(1)),
            "lra": float(re.search(r"LRA:\s+(-?[\d.]+) LU", s).group(1)),
            "true_peak_dbtp": float(re.search(r"Peak:\s+(-?[\d.inf]+) dBFS", s).group(1))}


def to_target(x: np.ndarray, sr: int, lufs: float = -14.0, ceiling_db: float = -1.3, iters: int = 4) -> np.ndarray:
    """Gain + limiter, iterated so the limited result lands on the loudness target."""
    meter = pyln.Meter(sr)
    gain_db = lufs - meter.integrated_loudness(x)
    y = x
    for _ in range(iters):
        y = limit(x * 10 ** (gain_db / 20), sr, ceiling_db=ceiling_db)
        err = lufs - meter.integrated_loudness(y)
        if abs(err) < 0.05:
            break
        gain_db += err
    return y
