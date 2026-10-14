"""Global texture passes (numpy/OpenCV, CPU). Apply AFTER the SeedVR2 upscale (it would smear grain and cracks).

MythTexture  oil-painting surface for the MYTH layer: craquelure (two Voronoi crack scales, jittered), canvas weave,
             amber varnish (warm shadows, yellowed edges), vignette, per-frame film grain. The crack/canvas plate is
             locked to the painting: pass the edit's zoom per frame and it scales with it, wrapping every octave so an
             infinite zoom keeps a constant crack density.
UITexture    CRT look for the MODEL layer only (RGBA overlay frames): scanlines, chromatic offset, phosphor bloom,
             optional barrel curvature (full-frame terminal shots) and rare horizontal tears.
strength 1.0 = the 1080p final; the phone preview (~18 MB, 640x360) uses PREVIEW_STRENGTH so grain and fine cracks
don't eat the bitrate (or denoise before encoding).

  python -m slopkit.texture myth <in.mp4|png> <out.mp4|png> [--strength 1.0] [--zoom-track zoom.json]
  python -m slopkit.texture ui <in_png_dir> <out_png_dir> [--barrel 0.0]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PREVIEW_STRENGTH = 0.35


def _wrapped(fn, img, m):
    """Apply a filter as if the image tiled (OpenCV filters don't support BORDER_WRAP): pad by wrap, filter, crop."""
    h, w = img.shape[:2]
    return fn(cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_WRAP))[m:m + h, m:m + w]


def _periodic_noise(w, h, rng, kmax, n=24):
    """Smooth noise that tiles seamlessly (sum of sinusoids with integer periods over the plate)."""
    sw, sh = max(8, w // 4), max(8, h // 4)  # computed at quarter size, then upsampled (cheap, still near-periodic)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    out = np.zeros((sh, sw), np.float32)
    for _ in range(n):
        kx, ky = int(rng.integers(-kmax, kmax + 1)), int(rng.integers(-kmax, kmax + 1))
        out += np.sin(2 * np.pi * (kx * xx / sw + ky * yy / sh) + float(rng.uniform(0, 2 * np.pi))).astype(np.float32)
    out = (out / float(np.sqrt(n))).astype(np.float32)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def _voronoi_cracks(w, h, cells, seed, width=1.0):
    """Tileable Voronoi crack map (points wrap around the plate edges, so BORDER_WRAP sampling has no seams)."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((h, w), np.uint8) + 255
    xs, ys = rng.integers(0, w, cells), rng.integers(0, h, cells)
    pts[ys, xs] = 0
    m = 256
    padded = cv2.copyMakeBorder(pts, m, m, m, m, cv2.BORDER_WRAP)
    _, lab = cv2.distanceTransformWithLabels(padded, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    edge = np.zeros(lab.shape, np.uint8)
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8)
    edge = edge[m:m + h, m:m + w].astype(np.float32)
    # jitter the straight Voronoi edges into crack-like wiggles with a periodic displacement field
    dx = _periodic_noise(w, h, rng, 12) * 5 + _periodic_noise(w, h, rng, 60) * 1.2
    dy = _periodic_noise(w, h, rng, 12) * 5 + _periodic_noise(w, h, rng, 60) * 1.2
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    edge = cv2.remap(edge, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if width > 1:
        edge = _wrapped(lambda x: cv2.dilate(x, np.ones((2, 2), np.uint8)), edge, 4)
    edge *= np.clip(_periodic_noise(w, h, rng, 6) * 0.6 + 0.7, 0, 1)  # cracks fade in and out along their length
    return np.clip(edge, 0, 1)


def make_plate(w=1920, h=1080, seed=11, oversample=2) -> np.ndarray:
    """Float32 (H*os)x(W*os) plate in [-1, 1]: negative = crack (darken), positive = raised edge / canvas tooth.
    Cached in ~/.cache/slopkit (float16, ~33 MB at 1080p x2) because generation takes a few seconds."""
    from pathlib import Path as _P
    cache = _P.home() / ".cache" / "slopkit" / f"plate_{w}x{h}_s{seed}_o{oversample}.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32)
    plate = _make_plate(w, h, seed, oversample)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, plate.astype(np.float16))
    return plate


def _make_plate(w, h, seed, oversample):
    W, H = w * oversample, h * oversample
    big = _voronoi_cracks(W, H, cells=int(W * H / 9000), seed=seed, width=2)
    small = _voronoi_cracks(W, H, cells=int(W * H / 1400), seed=seed + 1)
    cracks = np.clip(big * 1.0 + small * 0.55, 0, 1)
    rim = np.roll(np.roll(cracks, 1, 0), 1, 1) * 0.35        # 1 px highlight on the lower-right lip of each crack
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p = 5.0 * oversample                                      # canvas weave period ~5 px at 1080p
    weave = (np.sin(gx * 2 * np.pi / p) * np.sin(gy * 2 * np.pi / p)) * 0.5
    rng = np.random.default_rng(seed + 2)
    tooth = _wrapped(lambda x: cv2.GaussianBlur(x, (0, 0), 0.8 * oversample), rng.standard_normal((H, W)).astype(np.float32), 16) * 0.6
    return np.clip(-cracks + rim + 0.18 * weave + 0.12 * tooth, -1, 1).astype(np.float32)


class MythTexture:
    def __init__(self, w=1920, h=1080, strength=1.0, seed=11, oversample=2):
        self.w, self.h, self.s, self.seed = w, h, strength, seed
        self.os = oversample
        self.plate = make_plate(w, h, seed, oversample)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        self.vig = (1 - 0.32 * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None]
        self.edge_yellow = (np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]

    def _plate_at(self, zoom: float) -> np.ndarray:
        """The plate scaled by `zoom` about the centre, wrapped per octave (blend of z and z/2) for infinite zooms."""
        lz = np.log2(max(zoom, 1e-6))
        frac = lz - np.floor(lz)
        out = None
        for z, wgt in ((2 ** frac, 1 - frac), (2 ** frac / 2, frac)):
            sc = z / self.os
            M = np.float32([[sc, 0, self.w / 2 - sc * self.plate.shape[1] / 2], [0, sc, self.h / 2 - sc * self.plate.shape[0] / 2]])
            p = cv2.warpAffine(self.plate, M, (self.w, self.h), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)
            out = p * wgt if out is None else out + p * wgt
        return out

    def apply(self, frame_bgr: np.ndarray, k: int, zoom: float = 1.0) -> np.ndarray:
        s = self.s
        f = frame_bgr.astype(np.float32) / 255.0
        lum = f.mean(2, keepdims=True)
        # varnish: warm the shadows, compress and yellow the highlights, yellow the edges
        warm = np.array([0.80, 0.93, 1.06], np.float32)          # BGR multipliers (amber)
        f = f * (1 - 0.35 * s) + (f * warm) * (0.35 * s)
        f = f - (0.06 * s) * np.clip(f - 0.75, 0, 1)
        f = f * (1 - 0.25 * s * self.edge_yellow) + np.array([0.20, 0.45, 0.62], np.float32) * (0.25 * s * self.edge_yellow)
        # cracks and canvas: darken in cracks, slight lift on lips; stronger in midtones than in deep shadow
        pl = self._plate_at(zoom)[..., None]
        amt = (0.28 * np.clip(lum * 1.6, 0.2, 1.0)) * s
        f = f + pl * amt * np.where(pl < 0, 1.0, 0.45)
        f = f * (self.vig ** s)
        # grain: per-frame seeded, luminance-weighted, slightly coarse (half-res, upsampled)
        rng = np.random.default_rng(self.seed * 100003 + k)
        g = rng.standard_normal((self.h // 2, self.w // 2)).astype(np.float32)
        g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_LINEAR)[..., None]
        f = f + g * (0.045 * s) * (0.35 + 0.65 * (1 - np.abs(lum - 0.5) * 2))
        return np.clip(f * 255, 0, 255).astype(np.uint8)


class UITexture:
    """UI-layer finish. mode="modern" (default, UI theme v2 2026-10-03, the user: no CRT): a faint warm glow on the gold
    accents and a fine per-frame grain matched to the paint grain, nothing else. mode="crt" keeps the v1 look
    (scanlines, chromatic offset, phosphor bloom, optional barrel) for reference only."""
    def __init__(self, w=1920, h=1080, strength=1.0, barrel=0.0, seed=5, mode="modern"):
        self.w, self.h, self.s, self.barrel, self.seed, self.mode = w, h, strength, barrel, seed, mode
        rows = np.ones((h, 1, 1), np.float32)
        rows[1::2] = 1 - 0.12 * strength                          # 2 px scanline period, 12 % darker
        self.rows = rows
        self.map = None
        if barrel:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
            r2 = nx * nx + ny * ny
            f = 1 + barrel * r2
            self.map = ((nx * f) * (w / 2) + w / 2).astype(np.float32), ((ny * f) * (h / 2) + h / 2).astype(np.float32)

    def apply(self, rgba: np.ndarray, k: int, tear: bool = False) -> np.ndarray:
        """rgba: HxWx4 uint8 (straight alpha, as rendered) -> same, with the UI finish (see the class doc)."""
        if self.mode == "modern":
            return self._modern(rgba, k)
        s = self.s
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        c = rgba[..., :3].astype(np.float32) / 255.0 * a          # premultiplied
        if self.map is not None:
            c = cv2.remap(c, *self.map, cv2.INTER_LINEAR, borderValue=0)
            a = cv2.remap(a, *self.map, cv2.INTER_LINEAR, borderValue=0)[..., None]
        # chromatic aberration: R right, B left by 1-2 px (alpha widened to cover both)
        sh = max(1, round(1.5 * s))
        b, g, r = c[..., 0], c[..., 1], c[..., 2]
        r = np.roll(r, sh, axis=1)
        b = np.roll(b, -sh, axis=1)
        c = np.stack([b, g, r], -1)
        a = np.maximum.reduce([a[..., 0], np.roll(a[..., 0], sh, 1), np.roll(a[..., 0], -sh, 1)])[..., None]
        # phosphor bloom on bright pixels
        bright = np.clip(c - 0.55, 0, 1)
        bloom = cv2.GaussianBlur(bright, (0, 0), 7)
        c = c + bloom * 0.9 * s
        a = np.clip(a + cv2.GaussianBlur(bright.max(2), (0, 0), 7)[..., None] * 0.9 * s, 0, 1)
        c = c * self.rows
        if tear:
            rng = np.random.default_rng(self.seed * 7919 + k)
            y0 = int(rng.integers(0, self.h - 80)); hh = int(rng.integers(12, 70)); dx = int(rng.integers(-60, 60))
            c[y0:y0 + hh] = np.roll(c[y0:y0 + hh], dx, 1)
            a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], dx, 1)
        rng = np.random.default_rng(self.seed * 31 + k)
        flick = 1 + (rng.random() - 0.5) * 0.04 * s             # faint phosphor flicker
        c = np.clip(c * flick, 0, 1)
        out_a = np.clip(a, 0, 1)
        out_c = np.where(out_a > 1e-4, c / np.maximum(out_a, 1e-4), 0)
        return np.concatenate([np.clip(out_c, 0, 1) * 255, out_a * 255], -1).astype(np.uint8)


def _ui_modern(self, rgba, k):
    s = self.s
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    c = rgba[..., :3].astype(np.float32) / 255.0 * a                       # premultiplied BGR
    # soft glow only from warm bright pixels (the gold accent and bone text), very low gain
    warm = np.clip(c[..., 2:3] - 0.55, 0, 1) * np.clip(c[..., 2:3] - c[..., 0:1] + 0.1, 0, 1)
    glow = cv2.GaussianBlur(warm, (0, 0), 9)[..., None] if warm.ndim == 2 else cv2.GaussianBlur(warm[..., 0], (0, 0), 9)[..., None]
    gcol = np.array([0.37, 0.70, 0.85], np.float32)                     # gold #D9B25F in BGR
    c = c + glow * gcol * 0.35 * s
    a = np.clip(a + glow * 0.35 * s, 0, 1)
    # fine luminance grain on UI pixels, new every frame (fixed seed per frame index)
    rng = np.random.default_rng(self.seed * 131 + k)
    g = rng.standard_normal(a.shape[:2]).astype(np.float32)[..., None] * 0.018 * s
    c = np.clip(c * (1 + g), 0, 1)
    out_a = np.clip(a, 0, 1)
    out_c = np.where(out_a > 1e-4, c / np.maximum(out_a, 1e-4), 0)
    return np.concatenate([np.clip(out_c, 0, 1) * 255, out_a * 255], -1).astype(np.uint8)


UITexture._modern = _ui_modern


if __name__ == "__main__":
    a = sys.argv[1:]
    st = float(a[a.index("--strength") + 1]) if "--strength" in a else 1.0
    if a[0] == "myth":
        from slopkit import comp
        src, dst = a[1], a[2]
        if src.lower().endswith((".png", ".jpg")):
            img = cv2.imread(src)
            cv2.imwrite(dst, MythTexture(img.shape[1], img.shape[0], st).apply(img, 0))
        else:
            zt = json.load(open(a[a.index("--zoom-track") + 1])) if "--zoom-track" in a else None
            info = comp.probe(src)
            tx = MythTexture(info["w"], info["h"], st)
            with comp.Writer(dst, info["w"], info["h"], info["fps"]) as wr:
                for k, fr in enumerate(comp.read_frames(src)):
                    wr.write(tx.apply(fr, k, zt[k] if zt else 1.0))
    elif a[0] == "ui":
        bar = float(a[a.index("--barrel") + 1]) if "--barrel" in a else 0.0
        out = Path(a[2]); out.mkdir(parents=True, exist_ok=True)
        tx = None
        for k, p in enumerate(sorted(Path(a[1]).glob("*.png"))):
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            tx = tx or UITexture(im.shape[1], im.shape[0], st, bar)
            cv2.imwrite(str(out / p.name), tx.apply(im, k))
