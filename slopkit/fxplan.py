"""Punch-ins and flash frames from the beat grid (slopcore mode), plus the numpy effects for the edit.

Per-section FX table (master.md): Choruses punch on every kick (x1.06), the Drop punches harder (x1.10) and flashes on
snares (white / invert / data card), Verses punch on downbeats; noun flash cards (rendered by kinetic.html) come from
sung nouns, at most N per bar. Every flash, nouns included, passes one cap: no 4 flashes inside any 1 s window (so
<= 3 full-screen flashes per second) and >= 0.25 s between flashes. White flashes are capped at 80 % luminance.

  python -m slopkit.fxplan plan <hits.json> <sections.json> <words.json|-> <out_fx.json>
sections.json: [{name, start, end}]
"""
import json
import sys

import numpy as np

MAX_PER_S = 3
MIN_GAP = 0.25
WHITE = 204  # 80 % luminance
TABLE = [  # first match by substring of the section name
    ("Drop", {"punch": "kick", "scale": 1.10, "flash": "snare", "kinds": ["white", "invert", "data"], "nouns_per_bar": 2}),
    ("Chorus", {"punch": "kick", "scale": 1.06, "flash": None, "nouns_per_bar": 1}),
    ("Pre", {"punch": "downbeat", "scale": 1.05, "flash": "glitch", "nouns_per_bar": 1}),
    ("Verse", {"punch": "downbeat", "scale": 1.04, "flash": None, "nouns_per_bar": 1}),
    ("Bridge", {"punch": None, "flash": None, "nouns_per_bar": 1}),
    ("", {"punch": None, "flash": None, "nouns_per_bar": 1}),
]
ATTACK_F, RELEASE_F = 1, 6


def rules(name: str) -> dict:
    return next(r for k, r in TABLE if k in name)


def _cap_ok(times: list[float], t: float) -> bool:
    ts = sorted(times + [t])
    i = ts.index(t)
    if any(abs(x - t) < MIN_GAP for x in times):
        return False
    for a in range(max(0, i - MAX_PER_S), i + 1):  # every window of MAX_PER_S+1 consecutive flashes spans >= 1 s
        b = a + MAX_PER_S
        if b < len(ts) and ts[b] - ts[a] < 1.0:
            return False
    return True


def plan(hits: dict, sections: list[dict], words: list[dict] | None = None, fps=24) -> dict:
    kicks, snares, downs = np.array(hits["kicks"]), np.array(hits["snares"]), np.array(hits["downbeats"])
    punches, cands = [], []
    for sec in sections:
        r = rules(sec["name"])
        inside = lambda ts: [float(t) for t in ts if sec["start"] <= t < sec["end"]]
        src = {"kick": kicks, "downbeat": downs}.get(r["punch"])
        if src is not None:
            punches += [{"t": round(t, 4), "scale": r["scale"], "section": sec["name"]} for t in inside(src)]
        if r["flash"] in ("snare", "glitch"):
            pool = inside(snares) if r["flash"] == "snare" else inside(downs)[1::2]
            for k, t in enumerate(pool):
                kind = r.get("kinds", ["glitch"])[k % len(r.get("kinds", ["glitch"]))]
                cands.append({"t": round(t, 4), "frames": 1 if kind != "data" else 2, "kind": kind, "prio": 1, "section": sec["name"]})
    # noun cards: the sung onset of a listed noun, at most nouns_per_bar per bar
    per_bar = {}
    for w in words or []:
        if not w.get("noun"):
            continue
        sec = next((s for s in sections if s["start"] <= w["s"] < s["end"]), None)
        if sec is None:
            continue
        bar = int(np.searchsorted(downs, w["s"], side="right"))
        key = (sec["name"], bar)
        if per_bar.get(key, 0) >= rules(sec["name"])["nouns_per_bar"]:
            continue
        per_bar[key] = per_bar.get(key, 0) + 1
        cands.append({"t": round(w["s"], 4), "frames": 2, "kind": "noun", "text": w["w"].upper(), "prio": 2, "section": sec["name"]})
    flashes, dropped = [], []
    for c in sorted(cands, key=lambda c: (-c["prio"], c["t"])):
        if _cap_ok([f["t"] for f in flashes], c["t"]):
            flashes.append(c)
        else:
            dropped.append(c)
    flashes.sort(key=lambda f: f["t"])
    for f in flashes:
        f["frame"] = int(round(f["t"] * fps))
    return {"fps": fps, "punches": punches, "flashes": flashes, "dropped": len(dropped),
            "max_flashes_in_1s": max_per_window([f["t"] for f in flashes])}


def max_per_window(ts, win=1.0) -> int:
    ts = sorted(ts)
    best, j = 0, 0
    for i in range(len(ts)):
        while ts[i] - ts[j] >= win:
            j += 1
        best = max(best, i - j + 1)
    return best


# ---------------------------------------------------------------- effects (numpy / OpenCV)
def punch_scale(fx: dict, t: float) -> float:
    fps, s = fx["fps"], 1.0
    for p in fx["punches"]:
        d = (t - p["t"]) * fps
        if 0 <= d < ATTACK_F + RELEASE_F:  # full scale on the hit frame, ease back over RELEASE_F frames
            env = 1.0 if d < ATTACK_F else (1 - (d - ATTACK_F) / RELEASE_F) ** 2
            s = max(s, 1 + (p["scale"] - 1) * env)
    return s


def zoom(frame, s: float):
    import cv2
    if s <= 1.0005:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flash_for(fx: dict, frame_index: int) -> dict | None:
    for f in fx["flashes"]:
        if f["frame"] <= frame_index < f["frame"] + f["frames"]:
            return f
    return None


def apply_flash(frame, f: dict):
    """white / invert / data / glitch; 'noun' cards come from the kinetic overlay, so the frame passes through."""
    import cv2
    k = f["kind"]
    if k == "white":
        return np.full_like(frame, WHITE)
    if k == "invert":
        return 255 - frame
    if k == "data":
        e = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
        out = np.zeros_like(frame)
        out[e > 0] = (95, 178, 217)  # gold accent #D9B25F in BGR (UI theme v2)
        return out
    if k == "glitch":
        out = frame.copy()
        h = frame.shape[0]
        rng = np.random.default_rng(f["frame"])
        for _ in range(6):
            y0 = int(rng.integers(0, h - 40)); hh = int(rng.integers(8, 60))
            out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(rng.integers(-80, 80)), axis=1)
        out[:, :, 2] = np.roll(out[:, :, 2], 6, axis=1)
        return out
    return frame


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        words = None if a[3] == "-" else json.load(open(a[3]))
        r = plan(json.load(open(a[1])), json.load(open(a[2])), words)
        json.dump(r, open(a[4], "w"), indent=1)
        print(json.dumps({"punches": len(r["punches"]), "flashes": len(r["flashes"]), "dropped": r["dropped"],
                          "max_flashes_in_1s": r["max_flashes_in_1s"]}))
