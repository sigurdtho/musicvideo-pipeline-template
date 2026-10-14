"""Still-based shots (rule 4 degrades where I2V kept failing, or where the MOTION is 'completely still').

The shot's picked still is centre-cropped to 16:9 and animated deterministically in the edit:
  push      slow push-in toward a point: {"effect": ..., "push": 0.04, "focus": [x, y]} (normalised coordinates)
  rows_off  ceiling lights go out row by row from the vanishing point (the far end) toward camera, in perspective
            steps, down to `floor` brightness: {"effect": "rows_off", "vp": [x, y], "steps": 10, "floor": 0.4}
  warm      the light warms slightly over the shot: {"effect": "warm", "amount": 1.0}
  flash_below  (on a clip, not a still) a warm explosion flash lighting the subject from below:
            {"effect": "flash_below", "at": 0.72, "rise": 0.08, "decay": 0.7, "strength": 0.9}
  dust      fine dust drifting slowly through the light (brightest where the still is brightest): {"effect": "dust",
            "count": 260, "seed": 7}
"""
import cv2
import numpy as np


def _crop_16x9(img):
    h, w = img.shape[:2]
    ch = round(w * 9 / 16)
    if ch <= h:
        y0 = (h - ch) // 2
        return img[y0:y0 + ch]
    cw = round(h * 16 / 9)
    x0 = (w - cw) // 2
    return img[:, x0:x0 + cw]


def _smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def _push(img, scale, focus, size):
    """Crop 1/scale of the frame, keeping `focus` at the same relative position, and resize to size."""
    h, w = img.shape[:2]
    cw, ch = w / scale, h / scale
    fx, fy = focus[0] * w, focus[1] * h
    x0 = min(max(fx - focus[0] * cw, 0), w - cw)
    y0 = min(max(fy - focus[1] * ch, 0), h - ch)
    m = np.float32([[size[0] / cw, 0, -x0 * size[0] / cw], [0, size[1] / ch, -y0 * size[1] / ch]])
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class _RowsOff:
    """Ceiling tubes switch off row by row from the vanishing point (far end) toward camera; the room dims a step per row.

    Tubes are the bright pixels of the still (with their glow); each is assigned a step by its distance from the
    vanishing point, spaced in perspective (rows bunch up near the far end)."""

    def __init__(self, img, vp, steps=10, floor=0.4, t_on=0.1, t_off=0.85):
        h, w = img.shape[:2]
        lum = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.emit = np.clip((lum - 165) / 60, 0, 1)
        self.glow = np.clip(cv2.GaussianBlur(self.emit, (0, 0), 14) * 3, 0, 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - vp[0] * w) / w) ** 2 + ((yy - vp[1] * h) / h) ** 2)
        r /= r.max()
        # step index per pixel: ring i covers r in [((i-1)/steps)^1.7, (i/steps)^1.7)
        self.step = np.clip(np.floor(steps * r ** (1 / 1.7)), 0, steps - 1).astype(np.int32)
        self.steps, self.floor, self.t_on, self.t_off = steps, floor, t_on, t_off

    def gain(self, k, n):
        p = (k / max(1, n - 1) - self.t_on) / (self.t_off - self.t_on) * self.steps  # steps completed (fractional)
        off = np.clip((p - self.step) * 3, 0, 1)  # each ring's tubes cut out over a third of a step
        room = 1 - (1 - self.floor) * min(1.0, max(0.0, p / self.steps))
        tube = 1 - off * (0.85 * self.emit + 0.45 * self.glow * (1 - self.emit))
        return np.clip(tube * room, 0, 1.2)


class _Dust:
    """Fine dust drifting through the light: tiny soft specks, lit by the still's own brightness (sunbeams)."""

    def __init__(self, img, n, count=260, seed=7):
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed)
        lum = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        self.light = cv2.GaussianBlur(np.clip((lum - 0.35) / 0.5, 0, 1), (0, 0), 25)
        self.xy = rng.uniform([0, 0], [w, h], (count, 2))
        self.v = rng.normal([3.0, -1.5], [6.0, 4.0], (count, 2)) / 24  # px per frame: a slow, mostly sideways drift
        self.r = rng.uniform(0.8, 2.2, count)
        self.a = rng.uniform(0.35, 0.9, count)
        self.phase = rng.uniform(0, 2 * np.pi, count)
        self.h, self.w = h, w

    def layer(self, k):
        wob = np.stack([np.sin(k / 18 + self.phase), np.cos(k / 23 + self.phase)], 1) * 2.0
        pts = self.xy + self.v * k + wob
        lay = np.zeros((self.h, self.w), np.float32)
        for (x, y), r, a in zip(pts, self.r, self.a):
            cv2.circle(lay, (int(x * 4), int(y * 4)), int(r * 4), float(a), -1, cv2.LINE_AA, shift=2)
        return cv2.GaussianBlur(lay, (0, 0), 0.8) * self.light


def still_frames(img_bgr, n: int, size=(1920, 1080), post: dict | None = None):
    """n frames of an animated still (generator). Effects apply to the still, then the push-in crops it."""
    post = post or {}
    base = _crop_16x9(img_bgr).astype(np.float32)
    effect = post.get("effect")
    push = float(post.get("push", 0.0))
    focus = post.get("focus") or post.get("vp") or [0.5, 0.5]
    rows = _RowsOff(base, post["vp"], post.get("steps", 10), post.get("floor", 0.4)) if effect == "rows_off" else None
    dust = _Dust(base, n, post.get("count", 260), post.get("seed", 7)) if effect == "dust" else None
    for k in range(n):
        p = k / max(1, n - 1)
        img = base
        if rows is not None:
            img = base * rows.gain(k, n)[..., None]
        elif dust is not None:
            d = dust.layer(k)[..., None]
            img = base + (255 - base) * d * np.float32([0.75, 0.88, 1.0])  # screen-like add in warm white (BGR)
        elif effect == "warm":
            a = float(post.get("amount", 1.0)) * _smooth(p)
            img = base * np.float32([1 - 0.10 * a, 1 + 0.02 * a, 1 + 0.06 * a])  # BGR: less blue, a little more red
        f = cv2.resize(img, size, interpolation=cv2.INTER_AREA) if push == 0 else _push(img, 1 + push * _smooth(p), focus, size)
        yield np.clip(f, 0, 255).astype(np.uint8)


def flash_below(frame, t, at, rise=0.08, decay=0.7, strength=0.9, cache={}):
    """Add a warm orange flash from the bottom of the frame at time `at` (seconds into the shot): a fast rise, then an
    exponential decay; strongest at the bottom edge, gone by 45% of the frame height."""
    if t < at:
        return frame
    u = (t - at) / rise
    g = min(1.0, u) if u < 1 else float(np.exp(-(t - at - rise) / max(1e-3, decay / 3)))
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in cache:
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        cache[key] = np.clip((y - 0.55) / 0.45, 0, 1) ** 1.4
    m = cache[key] * (g * strength)
    f = frame.astype(np.float32)
    glow = np.float32([40, 150, 255])  # BGR warm orange
    f = f + (glow - f * 0.35) * m[..., None] * 0.85
    return np.clip(f, 0, 255).astype(np.uint8)
