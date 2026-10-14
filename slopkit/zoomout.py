"""Climax composite: zoom out of the painting until it is just output in a terminal window on a desktop.

The desktop (templates/desktop.html) is rendered once at 3x as two layers: 'bg' (wallpaper + other windows) and
'chrome' (the main window's frame; its content area is transparent). Per frame the camera rect moves on a log scale
from the main window's CONTENT rect (the painting fills the frame) to the full desktop; the painting (a clip, already
graded and textured) is drawn into the content rect at its on-screen size, then the chrome goes on top. Pure CPU.

  python -m slopkit.zoomout render <desktop.json> <painting.mp4|png> <out.mp4> <seconds> [--hold-in 0.5] [--hold-out 1.0]
desktop.json: {"main": {x, y, w, h, title} (content rect, 16:9), "windows": [{x, y, w, h, title, img|lines}]}
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from slopkit import comp, htmlrender

SCALE = 3


def render_layers(desk: dict, workdir) -> tuple[str, str]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    bg = htmlrender.render_still("desktop.html", {**desk, "layer": "bg"}, workdir / "desk_bg.png", scale=SCALE)
    ch = htmlrender.render_still("desktop.html", {**desk, "layer": "chrome"}, workdir / "desk_chrome.png", scale=SCALE)
    return bg, ch


def camera(u: float, rect, size=(1920, 1080)):
    """Camera rect (x, y, w, h) in desktop coords: u=0 -> the content rect, u=1 -> the whole desktop (log zoom)."""
    W, H = size
    rx, ry, rw, rh = rect
    cw = math.exp(math.log(rw) + (math.log(W) - math.log(rw)) * u)
    f = (cw - rw) / (W - rw)                       # 0..1, keeps the content rect exactly framed at u=0
    cx = (rx + rw / 2) + ((W / 2) - (rx + rw / 2)) * f
    cy = (ry + rh / 2) + ((H / 2) - (ry + rh / 2)) * f
    ch = cw * H / W
    return cx - cw / 2, cy - ch / 2, cw, ch


def _crop(img_hi, cam, size):
    x, y, w, h = (v * SCALE for v in cam)
    M = np.float32([[size[0] / w, 0, -x * size[0] / w], [0, size[1] / h, -y * size[1] / h]])
    return cv2.warpAffine(img_hi, M, size, flags=cv2.INTER_AREA if w > size[0] else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def render(desk: dict, painting, out, seconds, fps=24, hold_in=0.5, hold_out=1.0, size=(1920, 1080), workdir=None,
           audio=None) -> dict:
    m = desk["main"]
    rect = (m["x"], m["y"], m["w"], m["h"])
    if abs(m["w"] / m["h"] - size[0] / size[1]) > 0.01:
        raise ValueError("main content rect must be 16:9 so the zoom starts on an exact full frame")
    workdir = workdir or tempfile.mkdtemp(prefix="zoomout_")
    bg_p, ch_p = render_layers(desk, workdir)
    bg, ch = comp.load_rgba(bg_p), comp.load_rgba(ch_p)
    still = str(painting).lower().endswith((".png", ".jpg", ".jpeg"))
    src = None if still else comp.read_frames(painting)
    pimg = cv2.imread(str(painting)) if still else None
    N = max(2, round(seconds * fps))
    n_in, n_out = round(hold_in * fps), round(hold_out * fps)
    with comp.Writer(out, *size, fps, audio=audio) as wr:
        for k in range(N):
            u = min(1.0, max(0.0, (k - n_in) / max(1, N - n_in - n_out - 1)))
            u = 0.5 - 0.5 * math.cos(math.pi * u)
            cam = camera(u, rect, size)
            frame = comp.over(np.zeros((size[1], size[0], 3), np.uint8), _crop(bg, cam, size))
            if still:  # placeholder only: keep it moving (slow drift) so the comp is never a frozen still
                s = 1.04 + 0.03 * k / N
                pf = cv2.warpAffine(pimg, cv2.getRotationMatrix2D((pimg.shape[1] / 2, pimg.shape[0] / 2), 0, s),
                                    (pimg.shape[1], pimg.shape[0]), borderMode=cv2.BORDER_REFLECT)
            else:
                pf = next(src, None)
                if pf is None:
                    raise ValueError("painting clip shorter than the zoom-out")
            sx, sy = size[0] / cam[2], size[1] / cam[3]
            x0, y0 = (rect[0] - cam[0]) * sx, (rect[1] - cam[1]) * sy
            w, h = rect[2] * sx, rect[3] * sy
            M = np.float32([[w / pf.shape[1], 0, x0], [0, h / pf.shape[0], y0]])
            mask = cv2.warpAffine(np.ones(pf.shape[:2], np.float32), M, size, flags=cv2.INTER_LINEAR)[..., None]
            pw = cv2.warpAffine(pf, M, size, flags=cv2.INTER_AREA if w < pf.shape[1] else cv2.INTER_LINEAR)
            frame = (frame * (1 - mask) + pw * mask).astype(np.uint8)
            frame = comp.over(frame, _crop(ch, cam, size))
            wr.write(frame)
    return {"out": str(out), "frames": N, "layers": [bg_p, ch_p]}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "render":
        hi = float(a[a.index("--hold-in") + 1]) if "--hold-in" in a else 0.5
        ho = float(a[a.index("--hold-out") + 1]) if "--hold-out" in a else 1.0
        print(json.dumps(render(json.load(open(a[1])), a[2], a[3], float(a[4]), hold_in=hi, hold_out=ho)))
