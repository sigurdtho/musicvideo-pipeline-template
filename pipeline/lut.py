"""Generated 3D LUTs (.cube, 33^3). One neutral grade until the master names more.

A saturated blue band is left alone so a single accent colour is not crushed. Overlays are composited
after grading.
"""
import colorsys
from pathlib import Path

import numpy as np

N = 33
BLUE = (190 / 360, 235 / 360)  # electric-blue band (hue)


def _blue_weight(h: np.ndarray, s: np.ndarray) -> np.ndarray:
    lo, hi = BLUE
    w = np.clip(np.minimum(h - lo, hi - h) / 0.03, 0, 1)
    return w * np.clip((s - 0.25) / 0.25, 0, 1)  # only a saturated accent counts


def _contrast(v, amount):
    return np.clip(0.5 + (v - 0.5) * amount, 0, 1)


PRESETS = {
    "cine": dict(sat=0.92, contrast=1.04, shadows=(-0.01, 0.004, 0.012), highs=(0.01, 0.008, -0.008), orange_keep=0.0),
}
GRADE_PREFIXES = []  # ("out_", "cine") shot-id prefix -> preset name


def build(preset: str) -> np.ndarray:
    p = PRESETS[preset]
    g = np.linspace(0, 1, N)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # .cube order: red fastest
    rgb = np.stack([r, gg, b], axis=-1).reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    wb = _blue_weight(h, s)
    orange = np.clip(1 - np.abs(h - 0.075) / 0.06, 0, 1) * p["orange_keep"]
    keep = np.maximum(wb, orange)
    s2 = s * (p["sat"] * (1 - keep) + np.maximum(1.0, p["sat"]) * keep)
    s2 = np.where(wb > 0, s * (1 + 0.15 * wb), s2)
    v2 = _contrast(v, p["contrast"])
    out = np.array([colorsys.hsv_to_rgb(*c) for c in np.stack([h, np.clip(s2, 0, 1), v2], axis=1)])
    lum = out.mean(axis=1, keepdims=True)
    out = out + np.array(p["shadows"]) * (1 - lum) ** 2 * (1 - wb[:, None]) + np.array(p["highs"]) * lum ** 2 * (1 - wb[:, None])
    return np.clip(out, 0, 1)


def write(preset: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build(preset)
    with open(path, "w") as f:
        f.write(f'TITLE "__SLUG__ {preset}"\nLUT_3D_SIZE {N}\n')
        for r, g, b in data:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
    return path


# per-shot grade: by shot id prefix / section
def grade_for(shot: dict) -> str:
    sid = shot.get("id", "").split("__")[0]
    for prefix, name in GRADE_PREFIXES:
        if sid.startswith(prefix):
            return name
    return "cine"


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "video" / "luts"
    for k in PRESETS:
        print(write(k, out / f"{k}.cube"))
