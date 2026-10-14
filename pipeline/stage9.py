"""Stage 9: overlays. Every notice, achievement, level-up and counter from the shot list, timed to the final
timeline, rendered from overlays/box.html to transparent PNG sequences + alpha WebM.

  python -m pipeline.stage9 plan        resolve box timing/layout -> overlays/boxes.json
  python -m pipeline.stage9 render      render every box (headless Chrome) -> overlays/render/<box_id>/
  python -m pipeline.stage9 samples     composite one held frame of each box over its shot's still for review
  python -m pipeline.stage9 retime      re-plan after a timing change: move boxes whose length is unchanged, re-render the rest
  python -m pipeline.stage9 text        character dialogue boxes (stage2.TEXT_LINES) + karaoke lyric lines -> overlays/text.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from pipeline import manifest, overlay
from pipeline.manifest import ROOT

OV = ROOT / "overlays"
QC = ROOT / "qc" / "stage9"
LAYOUT = {  # kind: (x, y, width, anchor) on the 1920x1080 frame
    "achievement": (960, 540, 1240, "center"),
    "notice": (960, 846, 1180, "center"),
    "levelup": (96, 96, None, "topleft"),
    "ding": (1824 - 300, 96, None, "topleft"),
    "tiny": (960, 540, None, "center"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def plan_boxes() -> list[dict]:
    shots = json.load(open(ROOT / "video" / "shotlist.json"))["shots"]
    edit = json.load(open(ROOT / "audio" / "timing" / "song_edit.json"))
    sp = edit["spoken"]
    boxes = []
    for sh in shots:
        a = sh.get("anchor", ())
        spoken_ids = [a[1]] if a and a[0] == "spoken" else (list(a[1]) if a and a[0] == "spoken_group" else [])
        for i, b in enumerate(sh.get("boxes", [])):
            kind = b["kind"]
            box = {"id": f"{sh['id']}__{i}", "shot": sh["id"], "kind": kind if kind != "ding" else "notice"}
            if kind == "ding":
                # a small corner notice. Title comes from the shot. It does not scan the lyric for a word.
                x, y, w, anchor = LAYOUT["ding"]
                box.update({k: v for k, v in b.items() if k not in ("kind", "at", "big", "split_out")})
                box.update({"title": b.get("title") or "notice", "x": b.get("x", x), "y": b.get("y", y),
                            "anchor": anchor, "t_in": round(sh["start"] + 0.05, 3),
                            "t_out": round(min(sh["end"] - 0.05, sh["start"] + 0.8), 3)})
                boxes.append(box)
                continue
            if spoken_ids and kind in ("notice", "achievement", "tiny"):
                t_in = sp[spoken_ids[0]]["start"] - 0.06
                t_out = max(sp[spoken_ids[-1]]["end"] + 1.1, t_in + 2.2)
            elif kind == "counter":
                t_in, t_out = sh["start"] + 0.25, sh["end"] - 0.15
            elif kind == "levelup":
                t_in, t_out = sh["start"] + 0.2, min(sh["end"] - 0.1, sh["start"] + 1.6)
            else:
                t_in = sh["start"] + b.get("at", 0.15) * sh["duration"]
                t_out = min(sh["end"] - 0.1, t_in + max(2.0, len(b.get("body", "")) / 14))
            t_out = min(t_out, sh["end"] - 0.05) if kind != "achievement" else t_out
            box.update({k: v for k, v in b.items() if k not in ("kind", "at", "big", "split_out")})
            if kind == "counter":
                box.update({"x": b.get("x", 1500), "y": b.get("y", 180), "anchor": "center", "w": None})
            else:
                x, y, w, anchor = LAYOUT.get(kind, LAYOUT["notice"])
                box.update({"x": b.get("x", x), "y": b.get("y", y), "anchor": anchor})
                if w:
                    box["w"] = w + (120 if b.get("big") else 0)
            box.update({"t_in": round(t_in, 3), "t_out": round(t_out, 3)})
            boxes.append(box)
    # a centre or lower box ends before the next one pops in. Corner notices may overlap.
    main = sorted((b for b in boxes if b["kind"] in ("achievement", "notice", "tiny") and b.get("anchor") != "topleft"),
                  key=lambda b: b["t_in"])
    for a, b in zip(main, main[1:]):
        if a["t_out"] > b["t_in"] - 0.15:
            a["t_out"] = round(max(a["t_in"] + 1.0, b["t_in"] - 0.15), 3)
    return boxes


def cmd_plan():
    boxes = plan_boxes()
    (OV / "boxes.json").write_text(json.dumps(boxes, indent=1, ensure_ascii=False))
    manifest.upsert("overlay:plan", stage="9", kind="overlay_plan", path="overlays/boxes.json", workflow="pipeline/stage9.py plan",
                    status="done", score={"boxes": len(boxes)})
    log(f"plan: {len(boxes)} boxes")


def cmd_render():
    boxes = json.load(open(OV / "boxes.json"))
    for b in boxes:
        aid = f"overlay:{b['id']}"
        if (manifest.get(aid) or {}).get("status") in ("done", "rejected"):
            continue
        # the HTML renders on a local clock starting at t_in
        local = dict(b, t_out=b["t_out"] - b["t_in"], t_in=0.0)
        local = {k: v for k, v in local.items() if v is not None}
        r = overlay.render(local, OV / "render" / b["id"])
        manifest.upsert(aid, stage="9", kind="overlay", path=rel(r["webm"]), prompt=None, seed=None,
                        model="headless Chrome + overlays/box.html", workflow="pipeline/overlay.py", score=None, status="done",
                        meta={"frames_dir": rel(r["dir"]), "frames": r["frames"], "t_in": b["t_in"], "t_out": b["t_out"],
                              "shot": b["shot"], "text": box_text(b), "sig": sig(b)})
        log(f"{aid}: {r['frames']} frames in {r['render_seconds']}s")


def sig(b: dict) -> str:
    """Everything that changes how a box looks, apart from when it plays."""
    return json.dumps({k: v for k, v in b.items() if k not in ("t_in", "t_out")}, sort_keys=True, ensure_ascii=False)


def box_text(b: dict) -> str:
    return " / ".join(str(b.get(k, "")) for k in ("title", "name", "body") if b.get(k))


def cmd_retime():
    """After a timing change: boxes keep their frames when their length is the
    same (they render on a local clock), so only t_in/t_out move; boxes whose length changed are re-rendered."""
    cmd_plan()
    boxes = {b["id"]: b for b in json.load(open(OV / "boxes.json"))}
    moved = rerender = 0
    for a in manifest.query(stage="9", kind="overlay"):
        bid = a["id"].split(":", 1)[1]
        if a["meta"].get("layer") in ("dialogue", "lyric"):
            continue
        b = boxes.get(bid)
        if b is None:
            manifest.upsert(a["id"], status="rejected", meta={"review": "no longer in the plan"})
            continue
        # older renders have no signature: their text must match (a moved position needs a manual 'stale')
        same_look = a["meta"]["sig"] == sig(b) if "sig" in a["meta"] else a["meta"].get("text") == box_text(b)
        if same_look and abs((b["t_out"] - b["t_in"]) - (a["meta"]["t_out"] - a["meta"]["t_in"])) < 1 / 48:
            if abs(b["t_in"] - a["meta"]["t_in"]) > 1e-3:
                manifest.upsert(a["id"], meta={"t_in": b["t_in"], "t_out": b["t_out"]})
                moved += 1
        else:
            manifest.upsert(a["id"], status="stale")
            rerender += 1
    log(f"retime: {moved} boxes moved, {rerender} to re-render")
    cmd_render()


DIALOGUE_STYLE = {}  # voice id -> (accent, fill, glow). Empty means no dialogue boxes.
LYRIC_Y = {"bottom": 990, "top": 78}


def plan_text() -> list[dict]:
    from pipeline import stage2, voice_spec
    edit = json.load(open(ROOT / "audio" / "timing" / "song_edit.json"))
    sp, edits = edit["spoken"], edit["edits"]
    lines = json.load(open(ROOT / "audio" / "timing" / "lyrics.json"))
    boxes = json.load(open(OV / "boxes.json"))

    def to_final(t):
        return t + sum(e["insert"] for e in edits if e["at"] <= t)

    out = []
    # dialogue boxes for speakers listed in DIALOGUE_STYLE
    for lid in sorted(stage2.TEXT_LINES, key=lambda k: sp[k]["start"]):
        line = next(l for l in voice_spec.LINES if l["id"] == lid)
        if line["voice"] not in DIALOGUE_STYLE:  # voices without a colour are the shots' own boxes, not dialogue cards
            continue
        p = sp[lid]
        t_out = max(p["end"], p["start"] + 2.4)
        accent, fill, glow = DIALOGUE_STYLE[line["voice"]]
        out.append({"id": f"dlg_{lid}", "layer": "dialogue", "template": "dialogue.html", "tail": 0.2,
                    "speaker": line["voice"].capitalize(), "text": line["text"], "accent": accent, "fill": fill, "glow": glow,
                    "x": 960, "y": 900, "type_cps": 30,
                    "t_in": round(p["start"] - 0.05, 3), "t_out": round(t_out, 3)})
    sung = [l for l in lines if l.get("id") not in stage2.TEXT_LINES]
    timed = []
    for l in sung:
        ws = [w for w in l["word_times"] if w.get("start") is not None]
        words = []
        for i, w in enumerate(ws):
            e = w.get("end") or (ws[i + 1]["start"] if i + 1 < len(ws) else w["start"] + 0.3)
            words.append((w["word"], to_final(w["start"]), to_final(max(e, w["start"] + 0.05))))
        timed.append((l, words))
    for k, (l, words) in enumerate(timed):
        t_in = words[0][1] - 0.18
        t_out = words[-1][2] + 0.15
        if k + 1 < len(timed):
            t_out = min(t_out, timed[k + 1][1][0][1] - 0.18 - 0.05)
        t_out = max(t_out, words[-1][1] + 0.1)
        # a notice or dialogue box in the lower half at the same time pushes the lyric to the top of the frame
        busy = any(b["y"] > 600 and b["t_in"] < t_out + 0.3 and b["t_out"] > t_in for b in boxes) or \
            any(d["t_in"] < t_out + 0.3 and d["t_out"] > t_in for d in out if d["layer"] == "dialogue")
        out.append({"id": f"lyr_{l['section'].replace(' ', '')}_{l['line']}", "layer": "lyric", "template": "lyric.html",
                    "tail": 0.32, "y": LYRIC_Y["top" if busy else "bottom"], "text": l["text"],
                    "words": [{"w": w, "s": round(s - t_in, 3), "e": round(e - t_in, 3)} for w, s, e in words],
                    "t_in": round(t_in, 3), "t_out": round(t_out, 3)})
    return out


def cmd_text(only: list[str] | None = None):
    items = plan_text()
    (OV / "text.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    for b in items:
        if only and b["id"] not in only:
            continue
        aid = f"overlay:{b['id']}"
        prev = manifest.get(aid) or {}
        if prev.get("status") == "done" and prev.get("meta", {}).get("spec") == b:
            continue
        local = dict(b, t_out=round(b["t_out"] - b["t_in"], 3), t_in=0.0)
        r = overlay.render(local, OV / "render" / b["id"], template=b["template"], tail=b["tail"])
        manifest.upsert(aid, stage="9", kind="overlay", path=rel(r["webm"]), prompt=None, seed=None,
                        model=f"headless Chrome + overlays/{b['template']}", workflow="pipeline/overlay.py", score=None,
                        status="done", meta={"frames_dir": rel(r["dir"]), "frames": r["frames"], "t_in": b["t_in"],
                                             "t_out": b["t_out"], "layer": b["layer"], "text": b["text"], "spec": b})
        log(f"{aid}: {r['frames']} frames in {r['render_seconds']}s")


def cmd_samples():
    """One composited frame per box (held state) over its shot's still -> qc/stage9/<box>.jpg for review."""
    QC.mkdir(parents=True, exist_ok=True)
    stills = {a["meta"]["shot"]: ROOT / a["path"] for a in manifest.query(stage="6", kind="still") if a["status"] == "picked"}
    for a in manifest.query(stage="9", kind="overlay"):
        frames = sorted((ROOT / a["meta"]["frames_dir"]).glob("f*.png"))
        if not frames:
            continue
        held = frames[min(len(frames) - 1, int(len(frames) * 0.6))]
        bg = stills.get(a["meta"]["shot"])
        out = QC / f"{a['id'].split(':', 1)[1]}.jpg"
        src = ["-f", "lavfi", "-i", "color=black:s=1920x1080"] if bg is None else ["-i", str(bg)]
        subprocess.run(["ffmpeg", "-y", "-v", "error", *src, "-i", str(held), "-filter_complex",
                        "[0]scale=1920:1080,setsar=1[b];[b][1]overlay=0:0", "-frames:v", "1", "-q:v", "3", str(out)], check=True)
    log("samples written to qc/stage9/")


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "plan":
        manifest.set_stage("9", status="running", started=manifest.now())
        cmd_plan()
    elif cmd == "render":
        cmd_render()
    elif cmd == "retime":
        cmd_retime()
    elif cmd == "text":
        cmd_text(argv[1:] or None)
    elif cmd == "samples":
        cmd_samples()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
