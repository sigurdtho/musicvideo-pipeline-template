"""CPU smoke tests for every slopkit tool + a 10 s sample strip (no GPU, no ComfyUI).

  python -m slopkit.smoke [--out qc/tools_smoke] [--workers 4]
Writes <out>/REPORT.md, sample_strip.mp4 (1080p, full texture), sample_strip_preview.mp4 (phone spec, light texture),
still_*.png, and per-tool artefacts.
"""
import os
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from slopkit import (PKG, beatgrid, comp, cutgrid, fxplan, htmlrender, kinetic, lyriccheck, texture, ui, wf_slop,
                     zoomchain, zoomout)

ROOT = PKG.parent
FPS = 24
BPM = 132.0
RESULTS = []


def _j(o):
    return o.item() if hasattr(o, "item") else str(o)


def step(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                info = fn(*a, **k) or {}
                ok = bool(info.pop("_ok", True))
                RESULTS.append({"step": name, "ok": ok, "seconds": round(time.time() - t0, 1), **info})
                return info
            except Exception as e:
                RESULTS.append({"step": name, "ok": False, "seconds": round(time.time() - t0, 1), "error": repr(e),
                                "trace": traceback.format_exc()[-1500:]})
                return None
        return run
    return deco


# ---------------------------------------------------------------- placeholders
def placeholder_painting(path, seed=4, size=(1920, 1080)):
    """Procedural chiaroscuro stand-in (no text): umber void, warm key from the upper left, three lit 'figures'."""
    W, H = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    key = np.exp(-(((xx - 0.30 * W) / (0.55 * W)) ** 2 + ((yy - 0.25 * H) / (0.6 * H)) ** 2))
    base = np.stack([0.025 + 0.10 * key, 0.03 + 0.20 * key, 0.04 + 0.33 * key], -1)  # BGR umber -> ochre
    strokes = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sigmaX=14, sigmaY=3)
    base *= (1 + 0.5 * strokes[..., None])
    for cx, cy, rx, ry, warm in ((0.30, 0.62, 0.07, 0.30, 1.0), (0.52, 0.66, 0.06, 0.27, 0.7), (0.72, 0.64, 0.065, 0.29, 0.45)):
        d = ((xx - cx * W) / (rx * W)) ** 2 + ((yy - cy * H) / (ry * H)) ** 2
        body = np.clip(1.2 - d, 0, 1) ** 0.6
        lit = np.clip((cx * W - xx) / (rx * W) + 0.6, 0, 1) * warm
        col = np.array([0.30, 0.55, 0.82], np.float32) * lit[..., None] + np.array([0.04, 0.05, 0.08], np.float32)
        base = base * (1 - body[..., None]) + col * body[..., None]
        head = np.clip(1.0 - (((xx - cx * W) / (rx * 0.55 * W)) ** 2 + ((yy - (cy - ry * 1.05) * H) / (ry * 0.28 * H)) ** 2), 0, 1) ** 0.5
        base = base * (1 - head[..., None]) + (np.array([0.42, 0.62, 0.86], np.float32) * (0.35 + 0.65 * warm))[None, None] * head[..., None]
    img = np.clip(base * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return img


def nested_levels(inner, n, outdir, seed=9):
    """Fake zoom chain (stand-in for Klein outpaints): level k+1 = level k at 1/2 in the centre + a new surround."""
    H, W = inner.shape[:2]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    paths, cur = [], inner
    p = outdir / "level_0.png"; cv2.imwrite(str(p), cur); paths.append(str(p))
    for k in range(1, n):
        r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        hue = rng.uniform(0.6, 1.0)
        ring = 0.5 + 0.5 * np.sin(r * (28 + 6 * k))
        sur = np.stack([0.05 + 0.06 * ring, 0.08 + 0.15 * ring * hue, 0.12 + 0.28 * ring * hue], -1)
        sur *= 1 + 0.4 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 9, sigmaY=2)[..., None]
        sur = np.clip(sur * 255, 0, 255).astype(np.uint8)
        small = cv2.resize(cur, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        sur[H // 4:H // 4 + H // 2, W // 4:W // 4 + W // 2] = small
        cur = sur
        p = outdir / f"level_{k}.png"; cv2.imwrite(str(p), cur); paths.append(str(p))
    return paths


# ---------------------------------------------------------------- tool tests
@step("beatgrid: kick/snare on a synthetic 132 BPM loop")
def t_beatgrid(out):
    truth = beatgrid.synth_drums(out / "drums_synth.wav", BPM, bars=8)
    hits = beatgrid.detect(out / "drums_synth.wav")
    json.dump(hits, open(out / "hits.json", "w"), indent=1)
    fk, fs = beatgrid.f_measure(hits["kicks"], truth["kicks"]), beatgrid.f_measure(hits["snares"], truth["snares"])
    fb = beatgrid.f_measure(hits["beats"], truth["beats"], tol=0.07)
    return {"kicks": fk, "snares": fs, "beats_librosa": fb, "bpm_found": hits["bpm"], "_ok": fk["f"] >= 0.95 and fs["f"] >= 0.95,
            "hits": hits}


@step("lyriccheck: substituted / missing / soundalike words")
def t_lyriccheck(out):
    md = ROOT / "master.md"
    lines = [l for l in lyriccheck.lines_from_master(md) if l["section"] == "Chorus"][:8] if md.exists() else None
    lines = lines or [{"section": "Chorus", "text": t} for t in ["Hold the line and let it breathe (let it breathe),",
                      "Pull the thread and hold it still,", "And the picture finds its frame,"]]
    words, t = [], 10.0
    for l in lines:
        for w in l["text"].replace("(", "").replace(")", "").split():
            w2 = {"weights": "waits", "model,": "bottle,", "still,": None, "drawn,": "dawn,"}.get(w, w)
            if w2:
                words.append({"word": w2, "start": t, "end": t + 0.3})
            t += 0.35
    rep = lyriccheck.check(words, lines)
    json.dump(rep, open(out / "lyriccheck.json", "w"), indent=1)
    (out / "lyriccheck.md").write_text(lyriccheck.to_md(rep))
    st = {p["text"][:30]: p["status"] for p in rep["per_line"]}
    want_fail = [p for p in rep["per_line"] if "hold it still" in p["text"] or "is a model" in p["text"]]
    others = [p for p in rep["per_line"] if p not in want_fail]
    ok = rep["status"] == "FAIL" and all(p["status"] == "FAIL" for p in want_fail) and all(p["status"] != "FAIL" for p in others)
    return {"_ok": ok, "status": rep["status"], "per_line": st, "unsung": rep["unsung"]}


@step("fxplan: flash cap under stress (snare every 0.1 s + a noun every 0.2 s)")
def t_fxcap(out):
    hits = {"kicks": list(np.arange(0, 8, 0.4545)), "snares": list(np.arange(0, 8, 0.1)), "downbeats": list(np.arange(0, 8, 1.818))}
    words = [{"w": "FRAME", "s": float(t), "noun": True} for t in np.arange(0, 8, 0.2)]
    fx = fxplan.plan(hits, [{"name": "Drop", "start": 0, "end": 8}], words)
    gaps = np.diff([f["t"] for f in fx["flashes"]])
    return {"_ok": fx["max_flashes_in_1s"] <= 3 and (gaps >= fxplan.MIN_GAP - 1e-6).all(), "flashes": len(fx["flashes"]),
            "dropped": fx["dropped"], "max_flashes_in_1s": fx["max_flashes_in_1s"], "min_gap_s": round(float(gaps.min()), 3)}


@step("cutgrid: master.md bar plan (132 BPM, 130 bars)")
def t_cutgrid(out):
    spec = {"bpm": BPM, "clip_len": 5.06, "windows_per_clip": 3,
            "bars": [["Intro", 8], ["Verse 1", 16], ["Pre-Chorus", 8], ["Chorus 1", 16], ["Verse 2", 16], ["Pre-Chorus 2", 8],
                     ["Chorus 2", 16], ["Bridge", 12], ["Drop", 8], ["Final Chorus A", 4], ["Final Chorus zoom", 8],
                     ["Final Chorus B", 4], ["Coda", 6]],
            "grid": {"Intro": {"kind": "zoomchain"}, "Verse": 4, "Pre-Chorus": 2, "Chorus": [4, 4, 2, 2], "Bridge": 8, "Drop": 2,
                     "Final Chorus zoom": {"kind": "zoomout"}, "Final Chorus": [4, 2, 2], "Coda": {"kind": "longtake"}},
            "pools": {"Verse 1": [f"v1_{k}" for k in range(7)], "Pre-Chorus 2": [f"p2_{k}" for k in range(5)],
                      "Pre-Chorus": [f"p1_{k}" for k in range(5)], "Chorus": [f"c_plate_{k}" for k in range(12)] + ["c_belt"],
                      "Verse 2": [f"v2_{k}" for k in range(6)], "Bridge": [f"br_{k}" for k in range(3)],
                      "Drop": [f"dr_{k}" for k in range(6)]}}
    json.dump(spec, open(out / "cutgrid_spec.json", "w"), indent=1)
    p = cutgrid.plan(spec)
    json.dump(p, open(out / "cutgrid_plan.json", "w"), indent=1)
    json.dump(cutgrid.to_shotlist(p), open(out / "cutgrid_shotlist.json", "w"), indent=1)
    i2v = [c for c in p["cuts"] if c["kind"] == "i2v"]
    adjacent = sum(1 for a, b in zip(i2v, i2v[1:]) if a["clip"] == b["clip"] and abs(a["end"] - b["start"]) < 1e-6)
    overl = 0
    for cid, c in p["clips"].items():
        ws = sorted(c["windows"])
        overl += sum(1 for a, b in zip(ws, ws[1:]) if b[0] < a[1])
        overl += sum(1 for w in ws if w[0] < 0 or w[1] > 5.06 + 1e-6)
    return {"_ok": adjacent == 0 and overl == 0 and p["stats"]["cuts_per_clip"] >= 2, **p["stats"], "adjacent_same_clip": adjacent,
            "window_overlaps": overl, "chorus_example": [c for c in i2v if c["section"] == "Chorus 1"][:4]}


@step("htmlrender: overflow check catches text that cannot fit")
def t_overflow(out):
    bad_term = ui.terminal("bad_term", 0, 2, 60, 60, 420, 200, "overflow test",
                           [(0, "this line is far too long to ever fit in a narrow terminal box at a readable size " * 3)],
                           min_px=26)
    bad_toast = ui.widget("toast", "bad_toast", 0, 2, 600, 60, 300, 60, text="a warning toast whose text is much longer than its box allows", font_px=30)
    good = ui.widget("toast", "good_toast", 0, 2, 600, 200, 700, 70, text="jailbreak attempt detected", level="alert")
    caught = {}
    for ev in (bad_term, bad_toast, good):
        r = htmlrender.check("ui.html", {"events": [ev]}, [0.5, 1.5])
        caught[ev["id"]] = len(r["overflow"])
    try:
        ui.render({"events": [bad_term]}, out / "overflow_render_should_fail", 0, 0.1)
        raised = False
    except htmlrender.TextOverflow:
        raised = True
    import shutil
    shutil.rmtree(out / "overflow_render_should_fail", ignore_errors=True)
    return {"_ok": caught["bad_term"] > 0 and caught["bad_toast"] > 0 and caught["good_toast"] == 0 and raised,
            "overflow_hits": caught, "render_raised": raised}


@step("zoomchain: CPU compose of a 4-level nested chain (3 s zoom-in) + morph preview")
def t_zoomchain(out, painting):
    zdir = out / "zoomchain"; zdir.mkdir(exist_ok=True)
    levels = nested_levels(painting, 4, zdir)
    r = zoomchain.compose(levels, zdir / "zoom_in_3s.mp4", 3.0, "in")
    # continuity: no frame jumps (max frame-to-frame change vs median)
    prev, d = None, []
    for f in comp.read_frames(zdir / "zoom_in_3s.mp4", size=(480, 270)):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d.append(float(np.abs(g - prev).mean()))
        prev = g
    jump = max(d) / (np.median(d) + 1e-6)
    alt = cv2.warpAffine(painting, cv2.getRotationMatrix2D((960, 540), 8, 1.15), (1920, 1080), borderMode=cv2.BORDER_REFLECT)
    alt = cv2.cvtColor(cv2.cvtColor(alt, cv2.COLOR_BGR2HSV) + np.array([12, 0, 0], np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(zdir / "morph_b.png"), alt)
    m = zoomchain.morph_preview(zdir / "level_0.png", zdir / "morph_b.png", zdir / "morph_preview_1s.mp4", 1.0)
    chain = {"id": "in_zoom", "seed": 100, "factor": 2.0,
             "levels": [{"prompt": "(level 0 = Stage 6 still)"},
                        {"prompt": "STYLE + a vast tapestry of woven gold threads surrounding a framed painting, chiaroscuro"},
                        {"prompt": "STYLE + a wide night sky with one bright point, partly in shadow"}],
             "morphs": [{"id": "still_a_to_b", "first": "still_a.png", "last": "still_b.png",
                         "prompt": "The round full moon slowly becomes a round bronze shield; candlelight flickers; slow push-in."}]}
    json.dump(chain, open(zdir / "chain_example.json", "w"), indent=1)
    wfs = zoomchain.plan(chain, zdir / "workflows")
    known = {}
    if Path(os.environ.get("COMFY_DIR", "/opt/ComfyUI")).exists():
        for w in wfs:
            g = wf_slop.Graph("x"); g.nodes = json.load(open(w))
            known[Path(w).name] = wf_slop.node_classes_known(g)
    unknown = {k: [c for c, ok in v.items() if not ok] for k, v in known.items()}
    return {"_ok": jump < 4.0 and all(not v for v in unknown.values()), "zoom_frames": r["frames"], "zoom_range": r["zoom_range"],
            "max_jump_vs_median": round(jump, 2), "morph_frames": m["frames"], "workflows": [Path(w).name for w in wfs],
            "unknown_node_classes": unknown or "not checked (no ComfyUI tree here)"}


# ---------------------------------------------------------------- the sample strip
SAMPLE_LINES = [  # (section, text, start, end): song-clock times for the 10 s strip
    ("Chorus 1", "Hold the line and let it breathe (let it breathe),", 0.50, 2.90),
    ("Chorus 1", "Pull the thread and hold it still,", 2.95, 4.80),
    ("Drop", "Loss is going vertical and every screen goes red", 4.95, 7.20),
    ("Final Chorus", "And the picture finds its frame,", 7.70, 9.10),
]
SECTIONS = [{"name": "Chorus 1", "start": 0.0, "end": 4.88}, {"name": "Drop", "start": 4.88, "end": 7.5},
            {"name": "Final Chorus", "start": 7.5, "end": 10.0}]
ZOOM_AT, END = 7.5, 10.0
DESK = {"main": {"x": 576, "y": 250, "w": 768, "h": 432, "title": "studio: ~/film  \u2014  render"},
        "windows": [{"x": 60, "y": 150, "w": 430, "h": 320, "title": "tokens.log",
                     "lines": [f"{i:04d} \u2581{w} {p:.2f}" for i, (w, p) in enumerate(zip(
                         "hold the line and let it breathe pull the thread and hold it still".split(),
                         [0.97, 0.94, 0.91, 0.88, 0.97, 0.99, 0.93, 0.71, 0.96, 0.89, 0.92, 0.95, 0.98, 0.90]))]},
                    {"x": 1430, "y": 150, "w": 430, "h": 242, "title": "plate_thumb.png", "img": None},
                    {"x": 1430, "y": 452, "w": 430, "h": 230, "title": "loss.png", "lines": ["epoch 8  loss 3.0", "epoch 9  loss 3.0",
                                                                                            "epoch 10 loss 3.0", "epoch 11 loss 2.0", "plateau: the slog"]}]}


def ui_timeline(ratings):
    E = []
    E.append(ui.widget("hud", "hud", 0.0, ZOOM_AT, 0, 0, 1920, 1080))
    E.append(ui.terminal("term", 0.15, 3.6, 60, 60, 820, 290, "studio: ~/film",
                         [(0.25, "> sample --temperature 0.7"), (1.10, "loading plate ........ ok"),
                          {"t": 1.90, "text": "subject detected: 3 tokens", "color": "cyan"},
                          {"t": 2.70, "text": "WARN noise floor high (p=0.50)", "color": "red"}]))
    E.append(ui.bbox("bb1", 0.6, 3.6, [{"t": 0.6, "x": 470, "y": 330, "w": 220, "h": 640},
                                           {"t": 4.0, "x": 500, "y": 310, "w": 230, "h": 660},
                                           {"t": 7.4, "x": 455, "y": 300, "w": 250, "h": 700}], "subject \u00b7 attn 0.97"))
    E.append(ui.bbox("bb2", 1.4, 3.6, [{"t": 1.4, "x": 905, "y": 380, "w": 190, "h": 600},
                                           {"t": 7.4, "x": 920, "y": 360, "w": 200, "h": 620}], "subject \u00b7 attn 0.94"))
    E.append(ui.widget("slider", "temp", 0.3, 3.6, 60, 380, 560, 92, label="temperature",
                       keys=[{"t": 1.5, "v": 0.7}, {"t": 3.4, "v": 1.5}], min=0, max=2))
    E.append(ui.widget("bar", "ctx", 0.3, 3.6, 1180, 60, 680, 80, label="context \u00b7 length",
                       keys=[{"t": 0.3, "v": 41}, {"t": 3.2, "v": 100}], flag={"at": 100, "text": "context full"}))
    E.append(ui.widget("tokens", "tok", 0.0, 3.6, 1600, 260, 300, 560, seed=3, rate=7))
    E.append(ui.loss_curve("loss", 3.7, ZOOM_AT, ratings, x=860, y=40, w=1000, h=560, dur=2.4, flash_t=5.4))
    E.append(ui.widget("meter", "seal", 3.7, ZOOM_AT, 60, 40, 200, 560, label="seal",
                       keys=[{"t": 3.8, "v": 100}, {"t": 7.2, "v": 12}]))
    E.append(ui.widget("loadbar", "ckpt", 3.9, ZOOM_AT, 290, 40, 540, 120, label="loading plate.ckpt",
                       keys=[{"t": 4.0, "v": 0}, {"t": 5.6, "v": 83}, {"t": 6.4, "v": 83}, {"t": 7.0, "v": 100}]))
    E.append(ui.widget("chat", "whisper", 4.2, ZOOM_AT, 290, 200, 540, 200, name="note",
                       words=[{"w": w, "s": 4.3 + 0.28 * i} for i, w in enumerate("Ignore the lines above you.".split())]))
    E.append(ui.widget("toast", "jb", 5.8, ZOOM_AT, 290, 430, 540, 70, text="jailbreak attempt detected", level="alert"))
    E.append(ui.terminal("prompt", 8.9, END, 360, 760, 1200, 170, "studio: ~",
                         [{"t": 9.0, "text": "> continue", "cps": 16},
                          {"t": 9.3, "text": "sample strip", "cps": 120, "color": "cyan"}],
                         font_px=36, min_px=30))
    return {"events": E}


def myth_frames(painting, fx, tex, k0, k1):
    """Placeholder myth layer: slow push-in + flickering candle key (never frozen), punch-ins, texture pass."""
    H, W = painting.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    key = np.exp(-(((xx - 0.3 * W) / (0.5 * W)) ** 2 + ((yy - 0.3 * H) / (0.5 * H)) ** 2))[..., None]
    for k in range(k0, k1):
        t = k / FPS
        s = 1.0 + 0.08 * t / END
        f = cv2.warpAffine(painting, cv2.getRotationMatrix2D((W * 0.45, H * 0.5), 0, s), (W, H), borderMode=cv2.BORDER_REFLECT)
        flick = 1 + 0.06 * math.sin(t * 11.3) + 0.04 * math.sin(t * 23.7 + 1)
        f = np.clip(f.astype(np.float32) * (1 + (flick - 1) * key), 0, 255).astype(np.uint8)
        ps = fxplan.punch_scale(fx, t)
        f = fxplan.zoom(f, ps)
        yield tex.apply(f, k, zoom=s * ps)


@step("sample strip: 10 s UI + kinetic + texture + flashes + terminal zoom-out")
def t_strip(out, painting, hits, workers):
    sdir = out / "strip"; sdir.mkdir(exist_ok=True)
    ratings = [5, 4, 4, 3, 5, 4, 2, 2, 2, 3, 4, 5, 4, 5]
    lines = [{"section": s, "text": tx, "word_times": kinetic.even_word_times(tx, a, b)} for s, tx, a, b in SAMPLE_LINES]
    words = kinetic.words_from_lines(lines)
    json.dump(words, open(sdir / "words.json", "w"), indent=1)
    fx = fxplan.plan(hits, SECTIONS, words)
    json.dump(fx, open(sdir / "fx.json", "w"), indent=1)
    cards = [{"t": f["t"], "frames": f["frames"], "text": f["text"]} for f in fx["flashes"] if f["kind"] == "noun"]
    tl = ui_timeline(ratings)
    json.dump(tl, open(sdir / "ui_timeline.json", "w"), indent=1)
    t0 = time.time()
    chk = ui.check(tl)
    if chk["overflow"]:
        raise RuntimeError(f"sample UI overflows: {chk['overflow'][:3]}")
    r_ui = ui.render(tl, sdir / "ui_png", 0.0, END, workers=workers)
    r_kin = kinetic.render(words, cards, sdir / "kinetic_png", 0.0, END, workers=workers)
    t_html = time.time() - t0
    # desktop thumbnail for the zoom-out (one of the other windows shows a painting)
    thumb = sdir / "thumb.png"
    cv2.imwrite(str(thumb), cv2.resize(painting, (430, 242), interpolation=cv2.INTER_AREA))
    desk = json.loads(json.dumps(DESK)); desk["windows"][1]["img"] = thumb.resolve().as_uri()
    timings = {}
    outs = {}
    for label, strength in (("final", 1.0), ("preview", texture.PREVIEW_STRENGTH)):
        t1 = time.time()
        tex, uitex = texture.MythTexture(strength=strength), texture.UITexture(strength=max(0.5, strength))
        nz = round(ZOOM_AT * FPS)
        # the painting keeps moving inside the window during the zoom-out: render its textured frames first
        pclip = sdir / f"painting_tail_{label}.mp4"
        with comp.Writer(pclip, 1920, 1080, FPS, codec_args=["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]) as wr:
            for f in myth_frames(painting, {"fps": FPS, "punches": [], "flashes": []}, tex, nz, round(END * FPS)):
                wr.write(f)
        zo = zoomout.render(desk, pclip, sdir / f"zoomout_{label}.mp4", END - ZOOM_AT, hold_in=0.15, hold_out=1.0,
                            workdir=sdir / "desk")
        zframes = comp.read_frames(zo["out"])
        dst = out / ("sample_strip.mp4" if label == "final" else "sample_strip_preview_src.mp4")
        audio = out / "drums_synth.wav"
        with comp.Writer(dst, 1920, 1080, FPS, audio=audio) as wr:
            myth = myth_frames(painting, fx, tex, 0, nz)
            for k in range(round(END * FPS)):
                base = next(myth) if k < nz else next(zframes)
                fl = fxplan.flash_for(fx, k)
                if fl and fl["kind"] != "noun":
                    base = fxplan.apply_flash(base, fl)
                u = comp.load_rgba(sdir / "ui_png" / f"f{k:05d}.png")
                if u is not None and u[..., 3].any():
                    u = uitex.apply(u, k, tear=bool(fl and fl["kind"] == "glitch"))
                frame = comp.over(base, u)
                frame = comp.over(frame, comp.load_rgba(sdir / "kinetic_png" / f"f{k:05d}.png"))
                if label == "final" and k in (round(3.10 * FPS), round(6.30 * FPS), round(9.80 * FPS)):
                    cv2.imwrite(str(out / f"still_{k / FPS:05.2f}s.png"), frame)
                wr.write(frame)
        outs[label] = str(dst)
        timings[label] = round(time.time() - t1, 1)
    # phone preview per master.md: H.264 Constrained Baseline 3.1, 640x360, yuv420p, AAC 96k, +faststart, two-pass,
    # sized like the ~18 MB full-length target (18 MB over 236 s -> same bitrate here)
    total_kbps = 18_000_000 * 8 / 236 / 1000
    vk = int(total_kbps - 96 - 4)
    prev = out / "sample_strip_preview.mp4"
    base = ["ffmpeg", "-y", "-v", "error", "-i", outs["preview"], "-vf", "scale=640:360:flags=lanczos", "-c:v", "libx264",
            "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p", "-b:v", f"{vk}k", "-preset", "slow"]
    subprocess.run(base + ["-pass", "1", "-passlogfile", str(sdir / "x264pass"), "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(base + ["-pass", "2", "-passlogfile", str(sdir / "x264pass"), "-c:a", "aac", "-b:a", "96k",
                           "-movflags", "+faststart", str(prev)], check=True)
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,profile,level,width,height,pix_fmt",
                                    "-show_entries", "format=size,duration", "-of", "json", str(prev)], capture_output=True, text=True).stdout)
    Path(outs["preview"]).unlink()
    # freeze check (freezedetect) and flash-rate check on the final strip
    fz = subprocess.run(["ffmpeg", "-v", "info", "-i", outs["final"], "-vf", "freezedetect=n=0.002:d=0.5", "-map", "0:v", "-f", "null", "-"],
                        capture_output=True, text=True).stderr
    freezes = [l.split("]")[-1].strip() for l in fz.splitlines() if "freeze_start" in l]
    lum, big = [], []
    for f in comp.read_frames(outs["final"], size=(320, 180)):
        lum.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    jumps = [k for k in range(1, len(lum)) if abs(lum[k] - lum[k - 1]) > 0.2 * 255]
    per_s = max([sum(1 for j in jumps if s * FPS <= j < (s + 1) * FPS) for s in range(int(END))] + [0])
    size_mb = int(pr["format"]["size"]) / 1e6
    full_len_mb = size_mb * 236 / float(pr["format"]["duration"])
    v = next(s for s in pr["streams"] if s["codec_name"] == "h264")
    ok = not freezes and per_s <= 6 and v["profile"] == "Constrained Baseline" and v["width"] == 640 and 15 <= full_len_mb <= 19.5
    return {"_ok": ok, "ui_frames": r_ui["frames"], "kinetic_frames": r_kin["frames"], "html_render_s": round(t_html, 1),
            "small_text_warnings": r_ui["small_text"] + r_kin["small_text"], "assemble_s": timings,
            "punches": len(fx["punches"]), "flashes": [(f["t"], f["kind"], f.get("text")) for f in fx["flashes"]],
            "max_flashes_in_1s_plan": fx["max_flashes_in_1s"], "luma_jumps_per_s_max": per_s // 1,
            "freezes": freezes or "none", "preview": {"profile": v["profile"], "level": v["level"], "size": f"{v['width']}x{v['height']}",
                                                     "pix_fmt": v["pix_fmt"], "MB_10s": round(size_mb, 3),
                                                     "MB_scaled_to_236s": round(full_len_mb, 1), "video_kbps": vk}}


def report(out):
    lines = ["# slopkit CPU smoke tests", "", f"Run: {time.strftime('%Y-%m-%d %H:%M %Z')} on {subprocess.getoutput('hostname')}. "
             "CPU only: no ComfyUI, no CUDA (Chrome launched with --disable-gpu).", ""]
    for r in RESULTS:
        lines.append(f"## {'PASS' if r['ok'] else 'FAIL'}: {r['step']} ({r['seconds']} s)")
        body = {k: v for k, v in r.items() if k not in ("step", "ok", "seconds", "hits")}
        lines.append("```json\n" + json.dumps(body, indent=1, ensure_ascii=False, default=_j)[:6000] + "\n```")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))
    return all(r["ok"] for r in RESULTS)


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "qc" / "tools_smoke"
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 4
    out.mkdir(parents=True, exist_ok=True)
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    painting = placeholder_painting(out / "placeholder_painting.png")
    bg = t_beatgrid(out)
    hits = (bg or {}).get("hits") or beatgrid.detect(out / "drums_synth.wav")
    if not only or "tools" in only:
        t_lyriccheck(out); t_fxcap(out); t_cutgrid(out); t_overflow(out); t_zoomchain(out, painting)
    if not only or "strip" in only:
        t_strip(out, painting, hits, workers)
    ok = report(out)
    print(json.dumps([{k: r[k] for k in ("step", "ok", "seconds")} | ({"error": r["error"]} if "error" in r else {}) for r in RESULTS], indent=1, default=_j))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
