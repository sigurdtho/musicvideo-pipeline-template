"""Cut-grid shot planner: cuts on a per-section beat grid, and one rendered clip feeds 2-3 cuts.

The reference planner gives every lyric beat its own shot (79 shots, median 3.4 s). Slopcore needs 1-2 s cuts
(~160-180 in 4 min) without doubling the GPU time, so each Wan render (81 frames @ 16 fps = 5.06 s) is split into up to
`windows_per_clip` non-overlapping windows. Windows are taken most-active-first when a clip's activity curve is known
(stage10.activity on the 16 fps file), else from the end backwards (clips do their action late). A clip never feeds
two adjacent cuts.

spec.json:
  {"bpm": 132, "sections": [{name, start, end}] | "bars": [[name, n_bars], ...], "offset": 0.0,
   "grid": {"Verse 1": 4, "Chorus": [4, 4, 2, 2], "Intro": {"kind": "zoomchain"}, ...},   beats per cut (pattern cycles)
           {"kind": "lipsync", "clip": "v1_plate_a", "pattern": [4, 2], "every": 2}: a lip-sync render covering the
           whole section; every `every`-th cut (starting with the first) plays it at song time (window = cut time
           minus section start), the others are cutaways from the pool
   "pools": {"Verse 1": ["v1_plate_a", ...], "Chorus": [...]},                              clip ids per section (substring match)
   "clip_len": 5.06, "windows_per_clip": 3, "max_speed": 1.2}
  python -m slopkit.cutgrid plan <spec.json> <out_plan.json> [--beats audio/timing/beats.json] [--shotlist out_shots.json]
"""
import json
import sys

import numpy as np

GAP = 0.12  # seconds left between two windows of one clip, so repeated frames never touch


def sections_from_bars(bpm: float, bars: list, offset: float = 0.0) -> list[dict]:
    bar, t, out = 4 * 60.0 / bpm, offset, []
    for name, n in bars:
        out.append({"name": name, "start": round(t, 4), "end": round(t + n * bar, 4), "bars": n})
        t += n * bar
    return out


def grid_beats(bpm: float, end: float, offset: float = 0.0) -> list[float]:
    return [round(offset + k * 60.0 / bpm, 4) for k in range(int((end - offset) * bpm / 60.0) + 2)]


def _lookup(table: dict, name: str, default=None):
    if name in table:
        return table[name]
    for k, v in table.items():  # substring match: "Chorus" covers "Chorus 2" and "Final Chorus"
        if k in name:
            return v
    return default


class Clip:
    def __init__(self, cid, length, k, activity=None):
        self.id, self.len, self.k, self.act, self.used = cid, length, k, activity, []

    def free(self, need) -> tuple[float, float] | None:
        if len(self.used) >= self.k:
            return None
        cands = []
        step = 1 / 48
        for t0 in np.arange(0.0, self.len - need + 1e-9, step):
            t1 = t0 + need
            if all(t1 + GAP <= a or t0 >= b + GAP for a, b in self.used):
                cands.append(float(t0))
        if not cands:
            return None
        if self.act is not None:
            cum = np.concatenate([[0.0], np.cumsum(self.act)])
            e = lambda t: np.interp(t * 16, np.arange(len(cum)), cum)
            t0 = max(cands, key=lambda c: e(c + need) - e(c))
        else:
            t0 = cands[-1]  # latest free stretch first
        return (round(t0, 4), round(t0 + need, 4))


def plan(spec: dict, beats: list[float] | None = None) -> dict:
    bpm = spec.get("bpm", 132)
    secs = spec.get("sections") or sections_from_bars(bpm, spec["bars"], spec.get("offset", 0.0))
    beats = beats or grid_beats(bpm, secs[-1]["end"], spec.get("offset", 0.0))
    beats = np.array(beats)
    clip_len, K, vmax = spec.get("clip_len", 5.06), spec.get("windows_per_clip", 3), spec.get("max_speed", 1.2)
    acts = spec.get("activity", {})
    clips: dict[str, Clip] = {}
    cuts, prev, auto, lips = [], None, 0, set()
    for sec in secs:
        g = _lookup(spec["grid"], sec["name"], 4)
        ls = g if isinstance(g, dict) and g.get("kind") == "lipsync" else None
        if ls:
            g = ls.get("pattern", [4])
        if isinstance(g, dict):  # long take / comp / zoom chain: one cut for the whole section
            cuts.append({"id": f"{sec['name'].lower().replace(' ', '_')}__{g['kind']}", "section": sec["name"], "kind": g["kind"],
                         "start": sec["start"], "end": sec["end"], "dur": round(sec["end"] - sec["start"], 4)})
            prev = None
            continue
        pattern = g if isinstance(g, list) else [g]
        b0 = int(np.argmin(np.abs(beats - sec["start"])))
        bounds, k, i = [sec["start"]], 0, b0
        while True:
            i += pattern[k % len(pattern)]
            k += 1
            if i >= len(beats) or beats[i] >= sec["end"] - 0.25:
                break
            bounds.append(float(beats[i]))
        bounds.append(sec["end"])
        mb = _lookup(spec.get("min_beats", {}), sec["name"], spec.get("min_beats_default", 0))
        ibi_s = float(np.median(np.diff(beats)))
        while mb and len(bounds) > 2 and bounds[-1] - bounds[-2] < mb * ibi_s - 0.05:
            del bounds[-2]  # a too-short tail merges into the cut before it
        pool = list(_lookup(spec.get("pools", {}), sec["name"], []))
        rr = 0
        for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            dur = b - a
            if ls and n % ls.get("every", 2) == 0:  # lip-sync cut, anchored to song time
                w = (round(a - sec["start"], 4), round(b - sec["start"], 4))
                lsc = clips.setdefault(ls["clip"], Clip(ls["clip"], sec["end"] - sec["start"], 10 ** 6))
                lsc.used.append(w)
                lips.add(ls["clip"])
                prev = ls["clip"]
                cuts.append({"id": f"{ls['clip']}__c{len(lsc.used)}", "section": sec["name"], "kind": "lipsync",
                             "clip": ls["clip"], "start": round(a, 4), "end": round(b, 4), "dur": round(dur, 4),
                             "window": list(w), "speed": 1.0, "render_span": [sec["start"], sec["end"]]})
                continue
            pick = None
            for j in range(len(pool)):
                cid = pool[(rr + j) % len(pool)]
                if cid == prev:
                    continue
                c = clips.setdefault(cid, Clip(cid, clip_len, K, acts.get(cid)))
                for speed in (1.0, vmax):
                    w = c.free(min(dur * speed, clip_len))
                    if w:
                        pick = (c, w, speed)
                        break
                if pick:
                    rr = (rr + j + 1) % len(pool)
                    break
            if not pick:  # pool exhausted: a new clip has to be rendered for this section
                auto += 1
                cid = f"{sec['name'].lower().replace(' ', '_')}__new{auto}"
                pool.append(cid)
                c = clips.setdefault(cid, Clip(cid, clip_len, K))
                pick = (c, c.free(min(dur, clip_len)), 1.0)
            c, w, speed = pick
            c.used.append(w)
            prev = c.id
            cuts.append({"id": f"{c.id}__c{len(c.used)}", "section": sec["name"], "kind": "i2v", "clip": c.id,
                         "start": round(a, 4), "end": round(b, 4), "dur": round(dur, 4), "window": list(w), "speed": speed})
    used = {cid: c for cid, c in clips.items() if c.used and cid not in lips}
    durs = [c["dur"] for c in cuts if c["kind"] == "i2v"]
    return {"sections": secs, "cuts": cuts,
            "clips": {cid: {"windows": c.used, "uses": len(c.used)} for cid, c in used.items()},
            "stats": {"cuts": len(cuts), "i2v_cuts": len(durs), "unique_clips": len(used),
                      "cuts_per_clip": round(len(durs) / max(1, len(used)), 2),
                      "median_cut_s": round(float(np.median(durs)), 3) if durs else None,
                      "min_cut_s": round(min(durs), 3) if durs else None, "new_clips_needed": auto,
                      "lipsync_cuts": sum(c["kind"] == "lipsync" for c in cuts), "lipsync_renders": len(lips),
                      "long_takes": [c["id"] for c in cuts if c["kind"] not in ("i2v", "lipsync")]}}


def to_shotlist(p: dict) -> list[dict]:
    """Pipeline-style shot entries: the edit plays clip `reuse_clip` over `clip_range` (stage10.i2v_window)."""
    out = []
    for k, c in enumerate(p["cuts"]):
        sh = {"id": c["id"], "anchor": c["section"], "kind": c["kind"], "start": c["start"], "end": c["end"],
              "duration": c["dur"], "index": k}
        if c["kind"] in ("i2v", "lipsync"):
            sh.update({"reuse_clip": c["clip"], "clip_range": c["window"]})
        out.append(sh)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    spec = json.load(open(a[1]))
    beats = json.load(open(a[a.index("--beats") + 1]))["beats"] if "--beats" in a else None
    p = plan(spec, beats)
    json.dump(p, open(a[2], "w"), indent=1)
    if "--shotlist" in a:
        json.dump(to_shotlist(p), open(a[a.index("--shotlist") + 1], "w"), indent=1)
    print(json.dumps(p["stats"]))
