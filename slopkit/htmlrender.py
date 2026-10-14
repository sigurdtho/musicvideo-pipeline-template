"""Render an HTML template (slopkit/templates/*.html) to a transparent PNG sequence with headless Chrome (CPU only).

Template protocol: window.setSpec(spec) builds the DOM (may return a Promise); window.renderAt(t) poses time t.
Before rendering, SLOP.overflowAt(times) checks that every visible [data-box] element's text sits inside its box and
the box inside the frame; any hit raises TextOverflow (the render never ships clipped text).

  python -m slopkit.htmlrender <template.html> <spec.json> <outdir> <t0> <t1> [--fps 24] [--workers 4]
"""
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from slopkit import TEMPLATES

MIN_READABLE_PX = 36  # master.md: readable text >= 36 px at 1080p (phone preview legibility)


def chrome_path() -> str | None:
    for c in (os.environ.get("SLOP_CHROME"), "/usr/bin/google-chrome-stable", "/usr/bin/chromium"):
        if c and Path(c).exists():
            return c
    return None  # fall back to Playwright's bundled Chromium


class TextOverflow(RuntimeError):
    pass


def _open(p, template, size=(1920, 1080), scale=1):
    kw = {"headless": True, "args": ["--force-color-profile=srgb", "--disable-gpu", "--allow-file-access-from-files"]}
    if chrome_path():
        kw["executable_path"] = chrome_path()
    browser = p.chromium.launch(**kw)
    page = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=scale)
    tp = Path(template)
    page.goto((tp if tp.is_absolute() else TEMPLATES / tp).as_uri())
    page.evaluate("SLOP.fontsReady().then(() => true)")
    return browser, page


def _set(page, spec):
    page.evaluate("s => Promise.resolve(window.setSpec(s)).then(() => SLOP.fontsReady()).then(() => true)", spec)


def check(template, spec, times, size=(1920, 1080), min_px=MIN_READABLE_PX) -> dict:
    """Overflow hits (fail) and readable text under min_px (warn) at the sampled times."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = _open(p, template, size)
        _set(page, spec)
        over = page.evaluate("ts => SLOP.overflowAt(ts)", list(times))
        small = page.evaluate("([ts, m]) => SLOP.smallText(ts, m)", [list(times), min_px])
        browser.close()
    return {"overflow": over, "small_text": small}


def _render_chunk(args):
    template, spec, outdir, frames, size, scale, prefix = args
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = _open(p, template, size, scale)
        _set(page, spec)
        for i, t in frames:
            page.evaluate(f"window.renderAt({t!r})")
            page.screenshot(path=str(Path(outdir) / f"{prefix}{i:05d}.png"), omit_background=True)
        browser.close()
    return len(frames)


def render_sequence(template, spec, outdir, t0, t1, fps=24, size=(1920, 1080), scale=1, workers=1, check_overflow=True,
                    prefix="f", max_checks=480) -> dict:
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)  # a re-render never inherits frames from a longer earlier version
    outdir.mkdir(parents=True)
    n = max(1, round((t1 - t0) * fps))
    frames = [(i, t0 + i / fps) for i in range(n)]
    started = time.time()
    rep = {"overflow": [], "small_text": []}
    if check_overflow:
        step = max(1, math.ceil(n / max_checks))
        rep = check(template, spec, [t for _, t in frames[::step]], size)
        if rep["overflow"]:
            raise TextOverflow(json.dumps(rep["overflow"][:5]))
    workers = max(1, min(workers, n // 24 or 1))
    chunks = [frames[k::workers] for k in range(workers)]
    jobs = [(str(template), spec, str(outdir), c, size, scale, prefix) for c in chunks]
    if workers == 1:
        done = _render_chunk(jobs[0])
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            done = sum(ex.map(_render_chunk, jobs))
    return {"frames": done, "fps": fps, "t0": t0, "t1": t1, "dir": str(outdir), "seconds": round(time.time() - started, 1),
            "small_text": rep["small_text"]}


def render_still(template, spec, out, t=0.0, size=(1920, 1080), scale=1) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = _open(p, template, size, scale)
        _set(page, spec)
        page.evaluate(f"window.renderAt({t!r})")
        page.screenshot(path=str(out), omit_background=True)
        browser.close()
    return str(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    fps = int(a[a.index("--fps") + 1]) if "--fps" in a else 24
    wk = int(a[a.index("--workers") + 1]) if "--workers" in a else 1
    print(json.dumps(render_sequence(a[0], json.load(open(a[1])), a[2], float(a[3]), float(a[4]), fps=fps, workers=wk)))
