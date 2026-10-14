"""r906c: layer-2 sync QA against the beat grid and the WhisperX word times -> qc/r906c_sync.md

  python -m pipeline.sync_check [--render N]
1. Every HUD event's t_in is classified: on a beat of the fitted 128.0 BPM grid (bar n = 0.02 + 1.875 (n - 1)), on a
   sung word start (audio/timing/lyrics.json), on a shot cut, or none (listed). Tolerance: 1 frame at 24 fps.
2. Every sung line is on screen (as a lyric line or as its hook slam), and every lyric word's highlight time equals its
   WhisperX start.
3. --render N: N sampled words are rendered in Chrome just before and just after their start; the word must switch
   from dim to the accent colour inside the lyric box (pixel test on the rendered frames).
"""
import json
import random
import sys

import numpy as np

from pipeline import hud
from pipeline.manifest import ROOT

TOL = 1 / 24 + 1e-6


def main(argv):
    spec = json.loads(hud.SPEC.read_text())
    shots = json.load(open(ROOT / "video" / "shotlist.json"))["shots"]
    cuts = [s["start"] for s in shots]
    L = hud.lines()
    words = [w["start"] for l in L for w in l["word_times"]]
    grid = [hud.BEAT0 + k * hud.BEAT for k in range(int(spec["dur"] / hud.BEAT) + 2)]

    def near(t, xs):
        return min(abs(t - x) for x in xs) <= TOL

    half = [g + hud.BEAT / 2 for g in grid] + [c - hud.BEAT / 2 for c in cuts]
    rows, cls = [], {"beat": 0, "word": 0, "cut": 0, "8th": 0, "none": 0}
    for e in spec["events"]:
        t = e["t_in"]
        c = "beat" if near(t, grid) else "word" if near(t, words) else "cut" if near(t, cuts) or near(t - 1 / 24, cuts) else "8th" if near(t, half) else "none"
        cls[c] += 1
        if c == "none":
            rows.append(f"| {e['kind']} | {t:.3f} | {e.get('text') or ' '.join(e.get('words', []))} |")
        for tk in e.get("ticks", []):  # counter ticks are hard swaps on the beat or on the sung word
            c2 = "beat" if near(tk["t"], grid) else "word" if near(tk["t"], words) else "cut" if near(tk["t"], cuts) else "none"
            cls[c2] += 1
            if c2 == "none":
                rows.append(f"| counter tick | {tk['t']:.3f} | {tk['text']} |")
    # lyrics coverage
    shown = {(l["section"], l["words"][0]["s"]) for l in spec["lyrics"]}
    missing, wt_bad = [], 0
    for l in L:
        if (l["section"], l["line"]) in hud.HOOK_LINES:
            t_here = l["word_times"][0]["start"]
            ok = any(e["kind"] == "slam" and abs(e["t_in"] - t_here) <= 0.5 for e in spec["events"])
            if not ok:
                missing.append(f"{l['section']} {l['line']} (hook slam)")
            continue
        if not any(abs(s - l["word_times"][0]["start"]) < 1e-3 for _, s in shown):
            missing.append(f"{l['section']} {l['line']}")
    for l in spec["lyrics"]:
        src = next(x for x in L if x["word_times"] and abs(x["word_times"][0]["start"] - l["words"][0]["s"]) < 1e-3)
        for w, ws in zip(l["words"], src["word_times"]):
            wt_bad += abs(w["s"] - ws["start"]) > 1e-3
    md = ["# r906c layer-2 sync check", "",
          f"Spec `overlays/hud_spec.json`: {len(spec['events'])} events, {len(spec['lyrics'])} lyric lines; "
          f"{len(L)} sung lines in `audio/timing/lyrics.json` (6 are carried by the hook slams).", "",
          "## Event and tick times (tolerance 1 frame)", "",
          f"- on the 128.0 BPM beat grid: {cls['beat']}", f"- on a sung word start (WhisperX): {cls['word']}",
          f"- on a shot cut (or its first frame after): {cls['cut']}",
          f"- on an 8th note (the swap slams appear half a beat before the downbeat they swap on): {cls['8th']}",
          f"- none of these: {cls['none']}", ""]
    if rows:
        md += ["| kind | t_in s | text |", "| --- | ---: | --- |", *rows, ""]
    md += ["## Lyrics", "", f"- sung lines missing on screen: {missing or 'none'}",
           f"- words whose highlight time differs from the WhisperX start: {wt_bad}", ""]
    if "--render" in argv:  # the rendered page itself: which word carries the 'cur' (accent) class at each time
        from playwright.sync_api import sync_playwright
        from slopkit import htmlrender
        n = int(argv[argv.index("--render") + 1])
        rnd = random.Random(7)
        allw = [(l, i, w) for l in spec["lyrics"] for i, w in enumerate(l["words"]) if l["t_in"] < w["s"] - 0.1]
        res = []
        with sync_playwright() as pw:
            browser, page = htmlrender._open(pw, hud.TEMPLATE)
            htmlrender._set(page, spec)
            for l, i, w in rnd.sample(allw, min(n, len(allw))):
                st = []
                for t in (w["s"] - 1 / 24, w["s"] + 1 / 48):
                    page.evaluate(f"window.renderAt({t!r})")
                    cls = page.evaluate("() => [...document.querySelectorAll('#lyr .w')].map(e => e.className)")
                    st.append("cur" in cls[i] if i < len(cls) else None)
                res.append((w["w"], w["s"], st[0], st[1]))
            browser.close()
        good = sum(1 for _, _, a, b in res if a is False and b is True)
        md += [f"## Rendered word probes ({len(res)} random words; the page's own state 1 frame before / half a frame after "
               f"the WhisperX start)", "", f"**{good}/{len(res)}** words turn to the accent colour exactly at their start.", "",
               "| word | start s | accent before | accent after |", "| --- | ---: | --- | --- |",
               *[f"| {a} | {b:.3f} | {c} | {d} |" for a, b, c, d in res], ""]
    (ROOT / "qc" / "r906c_sync.md").write_text("\n".join(md))
    print("\n".join(md[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
