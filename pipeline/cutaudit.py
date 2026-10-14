"""Hard-cut audit. Every hard cut must land on a beat frame.

  python -m pipeline.cutaudit --shotlist video/shotlist.json --beats audio/timing/beats.json
  python -m pipeline.cutaudit --edit edit.json --beats beats.json --fps 24 --tol 1
  python -m pipeline.cutaudit --self-test

Frame index is int(round(t * fps)). A cut passes when its frame is within ±tol
of the nearest beat frame (default tol 1 at 24 fps). A cut whose frame is after
that beat is reported as late. The first shot is the start of the film, not a cut.
Dissolves, fades, crossfades and wipes are not hard cuts.

Exit 1 when any hard cut is outside the tolerance.
"""
import argparse
import json
import sys
from pathlib import Path

SOFT = ("dissolve", "fade", "crossfade", "wipe")


def frame_at(t, fps):
    return int(round(float(t) * fps))


def load_beats(path, downbeats=False):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        times = data
    else:
        key = "downbeats" if downbeats else "beats"
        if key not in data:
            raise SystemExit(f"{path} has no '{key}' list")
        times = data[key]
    return [float(t) for t in times]


def _soft(kind):
    text = (kind or "hard").lower()
    return any(word in text for word in SOFT)


def cuts_from_shotlist(data):
    shots = data["shots"] if isinstance(data, dict) else data
    out = []
    for i, shot in enumerate(shots):
        if i == 0:
            continue
        kind = str(shot.get("transition") or shot.get("cut") or "hard")
        if _soft(kind):
            continue
        out.append({"t": float(shot["start"]), "id": str(shot.get("id", i)), "kind": "hard"})
    return out


def cuts_from_edit(data):
    if isinstance(data, dict) and "shots" in data and "cuts" not in data:
        return cuts_from_shotlist(data)
    raw = data["cuts"] if isinstance(data, dict) else data
    out = []
    for i, cut in enumerate(raw):
        if isinstance(cut, (int, float)):
            out.append({"t": float(cut), "id": str(i), "kind": "hard"})
            continue
        kind = str(cut.get("kind") or cut.get("transition") or "hard")
        if _soft(kind):
            continue
        out.append({"t": float(cut.get("t", cut.get("start"))), "id": str(cut.get("id", i)), "kind": "hard"})
    return out


def audit(cuts, beats, fps, tol):
    beat_frames = [frame_at(b, fps) for b in beats]
    rows = []
    for cut in cuts:
        cf = frame_at(cut["t"], fps)
        if not beat_frames:
            rows.append({**cut, "frame": cf, "beat_frame": None, "delta": None, "late": True, "early": False, "ok": False})
            continue
        nearest = min(beat_frames, key=lambda bf: (abs(bf - cf), bf))
        delta = cf - nearest
        rows.append({**cut, "frame": cf, "beat_frame": nearest, "delta": delta,
                     "late": delta > tol, "early": delta < -tol, "ok": abs(delta) <= tol})
    return rows


def report(rows, fps, tol):
    lines = [f"hard cuts {len(rows)}  fps {fps}  tolerance ±{tol} frame(s)"]
    bad = 0
    for row in rows:
        if row["ok"]:
            lines.append(f"  ok    {row['id']}  t={row['t']:.3f}s  frame {row['frame']}  beat {row['beat_frame']}  delta {row['delta']:+d}")
            continue
        bad += 1
        which = "late" if row["late"] else "early"
        lines.append(f"  {which:5} {row['id']}  t={row['t']:.3f}s  frame {row['frame']}  beat {row['beat_frame']}  delta {row['delta']:+d}")
    lines.append(f"{'FAIL' if bad else 'PASS'}  {bad} outside tolerance")
    return "\n".join(lines), bad


def self_test():
    beats = [0.0, 0.5, 1.0]
    fps, tol = 24, 1
    exact = audit([{"t": 0.5, "id": "exact", "kind": "hard"}], beats, fps, tol)
    plus = audit([{"t": 0.5 + 1 / 24, "id": "plus", "kind": "hard"}], beats, fps, tol)
    late = audit([{"t": 0.5 + 3 / 24, "id": "late", "kind": "hard"}], beats, fps, tol)
    assert exact[0]["ok"] and exact[0]["delta"] == 0, exact
    assert plus[0]["ok"] and plus[0]["delta"] == 1, plus
    assert (not late[0]["ok"]) and late[0]["late"] and late[0]["delta"] == 3, late
    edit = {"cuts": [{"t": 0.5 + 3 / 24, "kind": "dissolve"}, {"t": 0.5, "kind": "hard"}]}
    kept = cuts_from_edit(edit)
    assert len(kept) == 1 and kept[0]["t"] == 0.5, kept
    shotlist = {"shots": [
        {"id": "a", "start": 0.0},
        {"id": "b", "start": 0.5, "transition": "hard"},
        {"id": "c", "start": 0.7, "transition": "crossfade"},
    ]}
    cuts = cuts_from_shotlist(shotlist)
    assert [c["id"] for c in cuts] == ["b"], cuts
    assert audit(cuts, beats, fps, tol)[0]["ok"]
    print("cutaudit self-test pass")
    return 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    p = argparse.ArgumentParser(description="Check hard cuts against the beat grid")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--shotlist")
    src.add_argument("--edit")
    p.add_argument("--beats", required=True)
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--tol", type=int, default=1)
    p.add_argument("--downbeats", action="store_true")
    args = p.parse_args(argv)
    data = json.loads(Path(args.shotlist or args.edit).read_text())
    cuts = cuts_from_shotlist(data) if args.shotlist else cuts_from_edit(data)
    rows = audit(cuts, load_beats(args.beats, args.downbeats), args.fps, args.tol)
    text, bad = report(rows, args.fps, args.tol)
    print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
