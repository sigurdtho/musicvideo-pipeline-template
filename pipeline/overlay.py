"""Render notice boxes (overlays/box.html) to transparent PNG sequences with headless Chrome, plus an alpha WebM.

  python -m pipeline.overlay render <box.json> <outdir>
box.json: see overlays/box.html (kind, title, name, body, x, y, w, t_in, t_out, type_cps, counter_*).
Frames cover [t_in, t_out + 150 ms] at 24 fps; frame 0 is t_in.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from pipeline.manifest import ROOT

CHROME = "/usr/bin/google-chrome-stable"
FPS = 24
POP_OUT = 0.150


def render(box: dict, outdir: str | Path, fps: int = FPS, webm: bool = True, template: str = "box.html",
           tail: float = POP_OUT) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("f*.png"):  # a re-render must not inherit frames from a longer earlier version
        old.unlink()
    n = math.ceil((box["t_out"] + tail - box["t_in"]) * fps) + 1
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--force-color-profile=srgb"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto((ROOT / "overlays" / template).as_uri())
        page.evaluate("document.fonts.ready.then(() => true)")
        page.evaluate("b => window.setBox(b)", box)
        page.evaluate("document.fonts.ready")
        for i in range(n):
            t = box["t_in"] + i / fps
            page.evaluate(f"window.renderAt({t})")
            page.screenshot(path=str(outdir / f"f{i:05d}.png"), omit_background=True)
        browser.close()
    res = {"frames": n, "fps": fps, "dir": str(outdir), "render_seconds": round(time.time() - t0, 1)}
    if webm:
        out = outdir.with_suffix(".webm")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(outdir / "f%05d.png"),
                        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "18", "-auto-alt-ref", "0",
                        str(out)], check=True)
        res["webm"] = str(out)
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["render"]:
        print(json.dumps(render(json.load(open(sys.argv[2])), sys.argv[3])))
