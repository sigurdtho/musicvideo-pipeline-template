"""Infinite zoom chains and match-cut morphs.

GPU side (written, not run): `plan` writes one Klein outpaint graph per level (each level's input is the previous
level's picked output) and a FLF2V graph per morph pair; `run` queues them through the project's pipeline.comfy
client and refuses to start without --gpu-ok (one GPU job at a time; check that nobody else needs the GPU).
CPU side: `compose` turns N nested levels (stills or Wan micro-motion clips, innermost first; level k+1 holds level k
scaled 1/factor in its centre) into one continuous log-scale zoom; `morph-preview` makes a draft optical-flow morph
between two stills for the 720p draft edit before the FLF2V renders exist.

  python -m slopkit.zoomchain plan <chain.json> <workflows_dir>
  python -m slopkit.zoomchain run <chain.json> --gpu-ok                (GPU; needs a bootstrapped project)
  python -m slopkit.zoomchain compose <out.mp4> <seconds> <in|out> <level0> <level1> ... [--factor 2] [--fps 24]
  python -m slopkit.zoomchain morph-preview <a.png> <b.png> <out.mp4> [--seconds 1.5]
chain.json: {"id": "in_wheel", "seed": 1, "levels": [{"prompt": ...}, ...], "factor": 2,
             "morphs": [{"id": "c_ages_1", "first": "x.png", "last": "y.png", "prompt": ...}]}
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from slopkit import comp, wf_slop


def plan(chain: dict, wf_dir) -> list[str]:
    out = []
    for k, lv in enumerate(chain.get("levels", [])[1:], start=1):  # level 0 is a normal Stage 6 still
        g = wf_slop.klein_outpaint(f"{chain['id']}_L{k}", f"{chain['id']}_L{k - 1}.png", lv["prompt"], chain.get("seed", 1) + k,
                                   factor=chain.get("factor", 2.0), prefix=f"__SLUG__/zoom/{chain['id']}_L{k}")
        out.append(str(g.save(wf_dir)))
    for m in chain.get("morphs", []):
        g = wf_slop.wan22_flf2v(f"{m['id']}_flf2v", m["first"], m["last"], m["prompt"], m.get("seed", 1),
                                prefix=f"__SLUG__/morph/{m['id']}")
        out.append(str(g.save(wf_dir)))
    return out


def run(chain: dict, gpu_ok=False):
    """Queue the chain level by level (each level needs the previous one) and then the morphs. GPU: comfy.run takes
    the pipeline's GPU file lock itself. chain["dir"] holds <id>_L0.png (the picked Stage 6 still) and the morph stills."""
    if not gpu_ok:
        raise SystemExit("zoomchain run uses the GPU: pass --gpu-ok only after confirming nobody else needs it")
    from pipeline import comfy  # needs the bootstrapped project (pipeline/ copied from the reference)
    d = Path(chain["dir"])
    pick = lambda res, ext: next(i for i in comfy.output_files(res["outputs"]) if i["filename"].endswith(ext))
    for k, lv in enumerate(chain["levels"][1:], start=1):
        name = comfy.upload(d / f"{chain['id']}_L{k - 1}.png", subfolder="__SLUG__")
        g = wf_slop.klein_outpaint(f"{chain['id']}_L{k}", name, lv["prompt"], chain.get("seed", 1) + k,
                                   factor=chain.get("factor", 2.0), prefix=f"__SLUG__/zoom/{chain['id']}_L{k}")
        res = comfy.run(g.api(), name=f"zoomchain:{chain['id']}:L{k}")
        comfy.download(pick(res, ".png"), d / f"{chain['id']}_L{k}.png")
    for m in chain.get("morphs", []):
        a, b = (comfy.upload(d / m[x], subfolder="__SLUG__") for x in ("first", "last"))
        g = wf_slop.wan22_flf2v(f"{m['id']}_flf2v", a, b, m["prompt"], m.get("seed", 1), prefix=f"__SLUG__/morph/{m['id']}")
        res = comfy.run(g.api(), name=f"morph:{m['id']}")
        comfy.download(pick(res, ".mkv"), d / f"{m['id']}_16fps.mkv")


# ---------------------------------------------------------------- CPU compositor
class _Level:
    def __init__(self, src, size):
        self.size = size
        self.video = None
        if str(src).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            self.img = cv2.resize(cv2.imread(str(src)), size, interpolation=cv2.INTER_AREA)
        else:
            self.frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in comp.read_frames(src)]
            self.video = True

    def at(self, k):
        if not self.video:
            return self.img
        n = len(self.frames)
        p = k % (2 * n - 2) if n > 1 else 0          # ping-pong so short micro-motion clips never freeze or jump
        return self.frames[p if p < n else 2 * n - 2 - p]


def _feather_mask(w, h, frac=0.08):
    m = np.ones((h, w), np.float32)
    bw, bh = max(1, int(w * frac)), max(1, int(h * frac))
    ramp = lambda n: (np.arange(n, dtype=np.float32) + 0.5) / n
    m[:, :bw] *= ramp(bw)[None, :]
    m[:, -bw:] *= ramp(bw)[::-1][None, :]
    m[:bh, :] *= ramp(bh)[:, None]
    m[-bh:, :] *= ramp(bh)[::-1][:, None]
    return m


def render_zoom(levels, k, z, factor, size):
    """Frame for continuous zoom z in [0, n-1] (z = n-1: the outermost level fills the frame; z = 0: the innermost)."""
    W, H = size
    n = len(levels)
    k0 = min(n - 1, math.ceil(z - 1e-9))
    rel = factor ** (z - k0)                       # visible part of level k0, relative width (<= 1)
    img = levels[k0].at(k)
    cw, ch = W * rel, H * rel
    M = np.float32([[1 / rel, 0, -(W - cw) / 2 / rel], [0, 1 / rel, -(H - ch) / 2 / rel]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    # paste finer levels (crisper detail) into the centre, feathered, down to a few pixels
    scale = 1 / rel
    for j in range(k0 - 1, -1, -1):
        scale /= factor
        w, h = int(round(W * scale)), int(round(H * scale))
        if w < 8:
            break
        sub = cv2.resize(levels[j].at(k), (w, h), interpolation=cv2.INTER_AREA if w < W else cv2.INTER_LINEAR).astype(np.float32)
        x0, y0 = (W - w) // 2, (H - h) // 2
        m = _feather_mask(w, h)[..., None]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        sa, sb = sub[ya - y0:yb - y0, xa - x0:xb - x0], m[ya - y0:yb - y0, xa - x0:xb - x0]
        out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - sb) + sa * sb
    return np.clip(out, 0, 255).astype(np.uint8)


def compose(sources, out, seconds, direction="in", factor=2.0, fps=24, size=(1920, 1080), ease=True) -> dict:
    """Log-scale zoom through the chain at constant perceived speed (z linear in time, optional ease at the ends)."""
    levels = [_Level(s, size) for s in sources]
    n, N = len(levels), max(2, round(seconds * fps))
    zs = []
    with comp.Writer(out, *size, fps) as wr:
        for k in range(N):
            u = k / (N - 1)
            u = 0.5 - 0.5 * math.cos(math.pi * u) if ease else u
            z = (n - 1) * (1 - u) if direction == "in" else (n - 1) * u
            zs.append(z)
            wr.write(render_zoom(levels, k, z, factor, size))
    return {"out": str(out), "frames": N, "levels": n, "zoom_range": factor ** (n - 1), "z_track": zs}


def zoom_track_for_texture(z_track, factor=2.0, direction="in"):
    """Per-frame zoom for MythTexture so the craquelure scales with the painting (wraps per octave there)."""
    zmax = max(z_track)
    return [factor ** (zmax - z) for z in z_track]


def morph_preview(a, b, out, seconds=1.5, fps=24, size=(1920, 1080)) -> dict:
    """Draft morph: DIS optical flow a->b, both warped toward the midpoint and cross-dissolved (no GPU)."""
    A = cv2.resize(cv2.imread(str(a)), size, interpolation=cv2.INTER_AREA)
    B = cv2.resize(cv2.imread(str(b)), size, interpolation=cv2.INTER_AREA)
    ga, gb = (cv2.cvtColor(cv2.resize(x, (size[0] // 2, size[1] // 2)), cv2.COLOR_BGR2GRAY) for x in (A, B))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fab = cv2.resize(dis.calc(ga, gb, None), size) * 2
    fba = cv2.resize(dis.calc(gb, ga, None), size) * 2
    gx, gy = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    N = max(2, round(seconds * fps))
    with comp.Writer(out, *size, fps) as wr:
        for k in range(N):
            u = k / (N - 1)
            s = u * u * (3 - 2 * u)
            wa = cv2.remap(A, gx - fab[..., 0] * s, gy - fab[..., 1] * s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wb = cv2.remap(B, gx - fba[..., 0] * (1 - s), gy - fba[..., 1] * (1 - s), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wr.write(cv2.addWeighted(wa, 1 - s, wb, s, 0))
    return {"out": str(out), "frames": N}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        print(json.dumps(plan(json.load(open(a[1])), a[2]), indent=1))
    elif a[0] == "run":
        run(json.load(open(a[1])), gpu_ok="--gpu-ok" in a)
    elif a[0] == "compose":
        fac = float(a[a.index("--factor") + 1]) if "--factor" in a else 2.0
        srcs = [x for x in a[4:] if not x.startswith("--") and not x.replace(".", "").isdigit()]
        r = compose(srcs, a[1], float(a[2]), a[3], fac)
        r.pop("z_track")
        print(json.dumps(r))
    elif a[0] == "morph-preview":
        sec = float(a[a.index("--seconds") + 1]) if "--seconds" in a else 1.5
        print(json.dumps(morph_preview(a[1], a[2], a[3], sec)))
