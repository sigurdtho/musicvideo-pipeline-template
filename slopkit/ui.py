"""MODEL-layer UI timeline: event constructors, validation and rendering (templates/ui.html).

An event is a dict {type, id, t_in, t_out, x, y, w, h, ...}; see ui.html for each widget's fields. Paths for HUD
boxes are keyframes [{t, x, y, w, h}] (song clock); values for meters are keyframes [{t, v}].

  python -m slopkit.ui render <ui_timeline.json> <outdir> <t0> <t1> [--workers 4]
  python -m slopkit.ui check  <ui_timeline.json>                       overflow check over each event's lifetime
"""
import json
import re
import sys
from pathlib import Path

from slopkit import htmlrender

TYPES = {"stitch", "beam", "heat", "terminal", "tokens", "bbox", "slider", "bar", "loss", "meter", "chat", "loadbar", "toast", "hud",
         "weave", "card", "rings", "pathtext", "links", "brackets", "micro", "sil", "odo", "streaks", "flash", "shock", "invert"}  # v4
def terminal(id, t_in, t_out, x, y, w, h, title, lines, **kw):
    """lines: [(t, text) or {t, text, cps, color, instant}]"""
    ls = [l if isinstance(l, dict) else {"t": l[0], "text": l[1]} for l in lines]
    return {"type": "terminal", "id": id, "t_in": t_in, "t_out": t_out, "x": x, "y": y, "w": w, "h": h, "title": title,
            "lines": ls, **kw}


def bbox(id, t_in, t_out, keys, label, **kw):
    return {"type": "bbox", "id": id, "t_in": t_in, "t_out": t_out, "keys": keys, "label": label, **kw}


def widget(type, id, t_in, t_out, x, y, w, h, **kw):
    return {"type": type, "id": id, "t_in": t_in, "t_out": t_out, "x": x, "y": y, "w": w, "h": h, **kw}


def loss_curve(id, t_in, t_out, ratings, x=860, y=420, w=1000, h=560, t_draw=None, dur=3.6, flash_t=None,
               reader="@your_handle"):
    """A loss curve from a list of ratings (loss = 6 - rating)."""
    return widget("loss", id, t_in, t_out, x, y, w, h, ratings=ratings, t_draw=t_in + 0.2 if t_draw is None else t_draw,
                  dur=dur, flash_t=flash_t, title="eval loss", subtitle=reader,
                  bracket={"from": 8, "to": 10, "text": "plateau", "sub": ""},
                  marker={"at": 12, "text": "new low"})


def validate(events: list[dict]) -> list[str]:
    errs, ids = [], set()
    for e in events:
        if e.get("type") not in TYPES:
            errs.append(f"{e.get('id')}: unknown type {e.get('type')}")
        if e.get("id") in ids:
            errs.append(f"duplicate id {e.get('id')}")
        ids.add(e.get("id"))
        if not e.get("t_out", 0) > e.get("t_in", 0):
            errs.append(f"{e.get('id')}: t_out <= t_in")
        if e.get("type") not in ("bbox", "hud"):
            x, y, w, h = (e.get(k, 0) for k in "xywh")
            if x < 0 or y < 0 or x + w > 1920 or y + h > 1080:
                errs.append(f"{e.get('id')}: box {x},{y},{w}x{h} leaves the 1920x1080 frame")
    return errs


def check(timeline: dict, fps=24) -> dict:
    """Overflow check over every event's whole life (every 2nd frame), all events in one page."""
    ts = sorted({round(e["t_in"] + k / fps, 4) for e in timeline["events"]
                 for k in range(0, int((e["t_out"] - e["t_in"]) * fps) + 1, 2)})
    return htmlrender.check("ui.html", timeline, ts)


def render(timeline: dict, outdir, t0, t1, fps=24, workers=1) -> dict:
    errs = validate(timeline["events"])
    if errs:
        raise ValueError("; ".join(errs))
    return htmlrender.render_sequence("ui.html", timeline, outdir, t0, t1, fps=fps, workers=workers)


if __name__ == "__main__":
    a = sys.argv[1:]
    tl = json.load(open(a[1]))
    if a[0] == "check":
        r = check(tl)
        print(json.dumps(r, indent=1))
        sys.exit(1 if r["overflow"] else 0)
    wk = int(a[a.index("--workers") + 1]) if "--workers" in a else 1
    print(json.dumps(render(tl, a[2], float(a[3]), float(a[4]), workers=wk)))
