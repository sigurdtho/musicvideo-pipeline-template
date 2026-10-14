"""UI theme v2 style frames (CPU only): UI + kinetic lyric over a placeholder painting, with the real comp order.

  python -m slopkit.styleframes <outdir>
Comp order (master.md Z-order): painting -> MythTexture -> glass blur under UI panels (comp.glass_over) -> UI
(UITexture modern) -> kinetic lyric (crisp). Placeholders are public-domain paintings, used only
as stand-ins for the generated paintings in these QC frames; they never enter the video.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from slopkit import comp, htmlrender, kinetic, texture, ui

ROOT = Path(__file__).resolve().parents[1]
PH = ROOT / "qc" / "ui_style" / "placeholder"


def crop169(path, y_frac=0.5, x_frac=0.5, zoom=1.0):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    cw = min(w, int(h * 16 / 9)) / zoom
    ch = cw * 9 / 16
    x0 = int((w - cw) * x_frac)
    y0 = int((h - ch) * y_frac)
    return cv2.resize(im[y0:y0 + int(ch), x0:x0 + int(cw)], (1920, 1080), interpolation=cv2.INTER_LANCZOS4)


def lyric(text, section, t):
    """One sung line, words spread over 0..2 s; returns kinetic words and the time to show the last word."""
    ws = text.split()
    wt = [{"word": w, "start": 0.2 + i * 0.35, "end": 0.2 + i * 0.35 + 0.3} for i, w in enumerate(ws)]
    return kinetic.words_from_lines([{"section": section, "text": text, "word_times": wt}])


def frame(name, bg, events, words, t, outdir, word_t=None):
    outdir.mkdir(parents=True, exist_ok=True)
    tl = {"events": events}
    errs = ui.validate(events)
    if errs:
        raise ValueError(errs)
    chk = ui.check(tl)
    if chk["overflow"]:
        raise RuntimeError(f"{name}: overflow {chk['overflow'][:3]}")
    ui_png = outdir / f"_{name}_ui.png"
    htmlrender.render_still("ui.html", tl, ui_png, t=t)
    myth = texture.MythTexture(1920, 1080, 1.0).apply(bg, 0)
    rgba = texture.UITexture(1920, 1080, 1.0).apply(comp.load_rgba(ui_png), 0)
    f = comp.glass_over(myth, rgba)
    if words:
        kin = outdir / f"_{name}_kin.png"
        htmlrender.render_still("kinetic.html", {"fps": 24, "hold_max": 0.8, "words": words, "cards": []}, kin,
                                t=word_t if word_t is not None else words[-1]["s"] + 0.05)
        f = comp.over(f, comp.load_rgba(kin))
    out = outdir / f"{name}.png"
    cv2.imwrite(str(out), f)
    cv2.imwrite(str(outdir / f"{name}_phone360.png"), cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA))
    return {"frame": str(out), "small_text": chk.get("small_text", [])[:5]}


def main(outdir, bgs=None):
    """bgs: optional 3 background paths ('-' = the default placeholder), e.g. generated stills for a readability check."""
    out = Path(outdir)
    bgs = bgs or ['-', '-', '-']
    pick = lambda i, default: default if bgs[i] in ('-', 'skip') else crop169(bgs[i])
    ratings = [5, 4, 4, 3, 5, 4, 2, 2, 2, 3, 4, 5, 4, 5]
    T = 5.0
    res = []
    # 1. Pre-Chorus: tracked boxes, context bar, attention trace terminal; lyric lower third
    bg = pick(0, crop169(PH / "The_Calling_of_Saint_Matthew_C.jpg", y_frac=0.62))
    ev = [ui.widget("hud", "hud", 0, 9, 0, 0, 1920, 1080),
          ui.bbox("t1", 1.0, 9, [{"t": 0, "x": 680, "y": 150, "w": 300, "h": 420}], "subject \u00b7 attn 0.97"),
          ui.bbox("t2", 1.3, 9, [{"t": 0, "x": 410, "y": 150, "w": 215, "h": 400}], "subject \u00b7 attn 0.94", label_side="below"),
          ui.bbox("t3", 1.6, 9, [{"t": 0, "x": 965, "y": 300, "w": 250, "h": 660}], "subject \u00b7 attn 0.91", label_side="below", label_align="right"),
          ui.widget("bar", "ctx", 0.5, 9, 1244, 96, 580, 92, label="context \u00b7 length", keys=[{"t": 0, "v": 41}], label_px=36),
          ui.terminal("trace", 0.3, 9, 1244, 724, 580, 260, "attention.trace",
                      [{"t": 0.4, "text": "layer 31 \u00b7 head 7", "instant": True, "color": "dim"},
                       {"t": 0.8, "text": "3 tokens above 0.90", "instant": True},
                       {"t": 1.2, "text": "> bend context: on", "instant": True, "color": "gold"}], font_px=36, min_px=36)]
    res.append(frame("01_boxes", bg, ev, lyric("hold the line", "Pre-Chorus 1", T), T, out))
    # 2. Pre-Chorus 2: the loss curve + temperature slider + an alert toast
    bg = pick(1, crop169(PH / "The_Taking_of_Christ_Car.jpg", y_frac=0.45))
    ev = [ui.widget("hud", "hud", 0, 9, 0, 0, 1920, 1080),
          ui.loss_curve("loss", 0.2, 9, ratings, x=900, y=96, w=924, h=560, dur=3.0, flash_t=None),
          ui.widget("slider", "temp", 0.2, 9, 96, 96, 560, 110, label="temperature", label_px=36, keys=[{"t": 0, "v": 0.7}, {"t": 4, "v": 1.42}],
                    min=0, max=2, warn=1.2),
          dict(ui.widget("toast", "alert", 3.0, 9, 900, 690, 924, 76, text="source \u00b7 integrity 0.50", level="alert", font_px=36),
               glitch=True)]
    if bgs[1] != "skip":
        res.append(frame("02_loss_curve", bg, ev, lyric("keep the frame still", "Pre-Chorus 2", T), T, out))
    # 3. Verse 2: system-prompt panel typing the oaths (paraphrased), bbox on the figure, verification ticks
    bg = pick(2, crop169(PH / "Rembrandt___The_Philosopher_in.jpg", y_frac=0.55))
    ev = [ui.widget("hud", "hud", 0, 9, 0, 0, 1920, 1080),
          ui.terminal("sys", 0.2, 9, 744, 96, 1080, 300, "notes.txt",
                      [{"t": 0.3, "text": "1. one figure, standing still", "cps": 60},
                       {"t": 1.3, "text": "2. the room stays empty behind them", "cps": 60},
                       {"t": 2.3, "text": "3. the light stays on the face,", "cps": 60},
                       {"t": 3.0, "text": "   and the hands stay in view", "cps": 60}], font_px=36, min_px=36),
          ui.bbox("fig", 1.0, 9, [{"t": 0, "x": 660, "y": 560, "w": 300, "h": 420}], "figure \u00b7 role=lead"),
          ui.terminal("verify", 2.0, 9, 1344, 430, 480, 240, "answer.verify",
                      [{"t": 2.1, "text": "claim 1   true  \u2713", "instant": True},
                       {"t": 2.4, "text": "claim 2   true  \u2713", "instant": True},
                       {"t": 2.7, "text": "claim 3   true  \u2713", "instant": True, "color": "gold"}], font_px=36, min_px=36)]
    res.append(frame("03_notes", bg, ev, lyric("one quiet room", "Verse 2", T), T, out))
    json.dump(res, open(out / "styleframes.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "qc" / "ui_style"), sys.argv[2:5] or None)
