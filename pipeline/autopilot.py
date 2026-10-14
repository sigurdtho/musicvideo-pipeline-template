"""Autopilot for review-paused periods (session limit): keep the GPU busy with provisional picks.

Every provisional choice is flagged meta.provisional=True and is reviewed (and redone if rejected) once
subagent review resumes. Runs until --until HH:MM or until a cycle finds nothing to do twice in a row.
  python -m pipeline.autopilot --until 00:30
"""
import sys
import time
from datetime import datetime

from pipeline import characters, manifest, stage5, stage6, stage7, stage8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] autopilot: {msg}", flush=True)


def provisional_looks() -> int:
    n = 0
    for look in characters.LOOKS:
        cands = [a for a in manifest.query(stage="5", kind="look_candidate") if a["meta"]["look"] == look]
        if cands and not any(a["status"] == "picked" for a in cands):
            base = characters.CHARACTERS[look].get("base")
            fresh = [a for a in cands if a["status"] == "done" and (not base or a["meta"].get("edit_of"))]
            if len(fresh) >= stage5.N:
                pick = sorted(fresh, key=lambda a: a["id"])[0]
                manifest.upsert(pick["id"], status="picked", meta={"provisional": True, "review": "provisional, review pending"})
                log(f"provisional look {look}: {pick['id']}")
                n += 1
        sheets = [a for a in manifest.query(stage="5", kind="look_sheet") if a["meta"]["look"] == look]
        if len(sheets) >= 2 and not any(a["status"] == "picked" for a in sheets) and look not in stage6.NO_SHEET:
            letter = "B"  # sheet A put ties/lanyards on the back view three times out of three so far
            stage5.cmd_crops(look, letter)
            manifest.upsert(f"look:{look}:sheet:{letter}", meta={"provisional": True})
            log(f"provisional sheet {look}: {letter}")
            n += 1
    return n


def needs_sheet() -> list[str]:
    out = []
    for look in characters.LOOKS:
        if look in stage6.NO_SHEET:
            continue
        picked = any(a["meta"]["look"] == look and a["status"] == "picked" for a in manifest.query(stage="5", kind="look_candidate"))
        has = any(a["meta"]["look"] == look for a in manifest.query(stage="5", kind="look_sheet"))
        if picked and not has:
            out.append(look)
    return out


def variants_ready() -> list[str]:
    out = []
    for look in characters.LOOKS:
        base = characters.CHARACTERS[look].get("base")
        if not base or stage5.picked_candidate(base) is None:
            continue
        if not any(a["meta"]["look"] == look and a["meta"].get("edit_of") for a in manifest.query(stage="5", kind="look_candidate")):
            out.append(look)
    return out


def reset_stale(minutes: int = 20) -> None:
    """Assets left 'running' by a killed job would be skipped forever; mark old ones failed so they re-render."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for a in list(manifest.load()["assets"].values()):
        if a.get("status") == "running" and a.get("kind") in ("still", "clip", "lipsync", "look_candidate", "look_sheet"):
            age = (now - datetime.fromisoformat(a["updated"])).total_seconds() / 60
            if age > minutes:
                manifest.upsert(a["id"], status="failed", meta={"stale_running_reset": manifest.now()})
                log(f"reset stale {a['id']} ({age:.0f} min)")


def cycle() -> bool:
    reset_stale()
    work = provisional_looks() > 0
    v = variants_ready()
    if v:
        stage5.cmd_candidates(v)
        work = True
    s = needs_sheet()
    if s:
        stage5.cmd_sheets(s)
        work = True
    before = len(manifest.query(stage="6", kind="still"))
    stage6.cmd_render([])
    work |= len(manifest.query(stage="6", kind="still")) > before
    stage6.cmd_provisional()
    c0 = len(manifest.query(stage="7", kind="clip"))
    stage7.cmd_render([])
    work |= len(manifest.query(stage="7", kind="clip")) > c0
    l0 = len(manifest.query(stage="8", kind="lipsync"))
    stage8.cmd_render([])
    work |= len(manifest.query(stage="8", kind="lipsync")) > l0
    return work


def main(argv):
    until = argv[argv.index("--until") + 1] if "--until" in argv else None
    idle = 0
    while True:
        if until and datetime.now().strftime("%H:%M") == until:
            log("reached --until; stopping")
            return 0
        try:
            busy = cycle()
        except Exception as e:  # one failing asset must not stop the night shift
            log(f"cycle error: {e!r}")
            busy = True
        idle = 0 if busy else idle + 1
        if idle >= 2:
            log("nothing left to do")
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
