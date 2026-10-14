"""Stage 6: stills. 4 Klein 9B candidates per shot from the character sheets -> review -> pick.

  python -m pipeline.stage6 render [shot_id ...]     candidates for every shot that needs a still
  python -m pipeline.stage6 grids                    labelled 2x2 review grids in qc/stage6/
  python -m pipeline.stage6 pick <shot_id> <A-D> [reason]
  python -m pipeline.stage6 retry <shot_id> [new SHOT text]   new seeds (rule 4), or a rewritten prompt
  python -m pipeline.stage6 edit <shot_id> <letter> <change>   recreate one candidate with a single change
Prompt = STYLE + SHOT + CHARACTER line(s) (master.md). References are single-view crops (face, front) of each
character's picked sheet; Stage 0 showed whole four-view sheets make the model duplicate the character.
"""
import json
import sys
import time
from pathlib import Path

from pipeline import characters, comfy, manifest, review, wf
from pipeline.manifest import ROOT

IMG = ROOT / "images" / "stills"
QC = ROOT / "qc" / "stage6"
N = 4
W, H = 1920, 1088
NO_REF = set()    # character ids that are props and take no sheet crop
NO_SHEET = set()  # looks that use the picked candidate instead of a turnaround sheet
NOUN = {}        # optional short noun per character id


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def shots_needing_stills() -> list[dict]:
    sl = json.load(open(ROOT / "video" / "shotlist.json"))["shots"]
    return [s for s in sl if s["kind"] in ("i2v", "lipsync") and not s.get("reuse_still")]


def refs_for(chars: list[str]) -> tuple[list[str], str]:
    """Upload face + front crops per referenced character; return input names and the figure legend."""
    names, legend, fig = [], [], 1
    for c in chars:
        if c in NO_REF:
            continue
        sheet = next((a for a in manifest.query(stage="5", kind="look_sheet") if a["meta"]["look"] == c and a["status"] == "picked"), None)
        if sheet is not None:
            crops = sheet["meta"]["crops"]
            names += [comfy.upload(ROOT / crops["face"]), comfy.upload(ROOT / crops["front"])]
            legend.append(f"The {NOUN.get(c, 'character')} is the one shown in Figure {fig} (face) and Figure {fig + 1} (full body); "
                          f"keep the same face, build, hair and clothes.")
            fig += 2
        else:  # crowds and companions without a sheet: the picked look candidate itself
            cand = next((a for a in manifest.query(stage="5", kind="look_candidate") if a["meta"]["look"] == c and a["status"] == "picked"), None)
            if cand is not None:
                names.append(comfy.upload(ROOT / cand["path"]))
                legend.append(f"The {NOUN.get(c, 'character')} look like Figure {fig}.")
                fig += 1
    return names, " ".join(legend)


def prompt_for(shot: dict, shot_text: str | None = None) -> str:
    chars = [c for c in shot.get("chars", []) if c not in NO_REF]
    # close-ups may use the upper-body part of a CHARACTER line (shot_plan char_text) so the camera stays close
    char_lines = shot.get("char_text") or " ".join(characters.CHARACTERS[c]["full"] for c in chars if c in characters.CHARACTERS)
    people = [c for c in chars if NOUN.get(c) == "man"]
    alone = " Only one man in the frame, no duplicates, no reflections of him." if len(people) == 1 else ""
    return f"{characters.STYLE} {shot_text or shot['shot']}{alone} {char_lines}".strip()


def render(shot: dict, seeds, shot_text=None, attempt=0):
    # wide shots where the character is tiny render from text only (references force a portrait framing)
    refs, legend = ([], "") if shot.get("ref_mode") == "none" else refs_for(shot.get("chars", []))
    for x in shot.get("extra_refs", []):  # continuity: another shot's picked still (the same axe, the same face)
        st = next(a for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == x["still"] and a["status"] == "picked")
        refs.append(comfy.upload(ROOT / st["path"]))
        legend = f"{legend} {x['legend'].format(fig=len(refs))}".strip()
    prompt = prompt_for(shot, shot_text)
    if legend:
        prompt = f"{prompt} {legend}"
    for k, seed in enumerate(seeds):
        letter = chr(65 + k + attempt * N)
        aid = f"still:{shot['id']}:{letter}"
        if (manifest.get(aid) or {}).get("status") in ("done", "picked", "rejected", "running"):
            continue
        manifest.upsert(aid, stage="6", kind="still", status="running", meta={"shot": shot["id"], "attempt": attempt})
        g = wf.klein(f"still_{shot['id']}_{letter}", prompt, seed=seed, width=W, height=H, refs=refs,
                     ref_megapixels=1.0 if len(refs) <= 2 else 0.6, prefix=f"__SLUG__/stills/{shot['id']}")
        wfp = g.save(ROOT / "workflows" / "stage6")
        res = comfy.run(g.api(), name=aid, timeout=600)
        dest = comfy.download(comfy.output_files(res["outputs"])[0], IMG / shot["id"] / f"cand_{letter}.png")
        manifest.upsert(aid, stage="6", kind="still", path=rel(dest), prompt=prompt, seed=seed, model="flux-2-klein-9b-kv-fp8",
                        workflow=rel(wfp), score=None, status="done",
                        meta={"shot": shot["id"], "refs": refs, "attempt": attempt, "render_seconds": res["seconds"]})
        log(f"{aid}: {res['seconds']}s")


def look_ready(c: str) -> bool:
    if c in NO_REF:
        return True
    if any(a["meta"]["look"] == c and a["status"] == "picked" for a in manifest.query(stage="5", kind="look_sheet")):
        return True
    # looks without a sheet (the crowd, and animals: the people-trained QuadView LoRA anthropomorphises them)
    return c in NO_SHEET and any(a["meta"]["look"] == c and a["status"] == "picked" for a in manifest.query(stage="5", kind="look_candidate"))


def cmd_render(ids):
    for shot in shots_needing_stills():
        if ids and shot["id"] not in ids:
            continue
        missing = [c for c in shot.get("chars", []) if not look_ready(c)]
        if missing:
            log(f"{shot['id']}: waiting for looks {missing}")
            continue
        if any(a["status"] == "picked" for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == shot["id"]):
            continue
        base = 7000 + 131 * shot["index"]
        render(shot, [base + k for k in range(N)])
    comfy.free()
    cmd_grids(ids)


def cmd_grids(ids=None):
    by_shot = {}
    for a in manifest.query(stage="6", kind="still"):
        by_shot.setdefault(a["meta"]["shot"], []).append(a)
    for sid, cands in by_shot.items():
        if ids and sid not in ids:
            continue
        cands = sorted([c for c in cands if c["status"] in ("done", "picked")], key=lambda a: a["id"])
        if cands:
            review.grid([ROOT / c["path"] for c in cands], QC / f"{sid}.jpg", labels=[c["id"].rsplit(":", 1)[1] for c in cands],
                        cols=2, cell_w=760)


def cmd_pick(sid, letter, reason=""):
    for a in manifest.query(stage="6", kind="still"):
        if a["meta"]["shot"] == sid and a["status"] in ("done", "picked"):
            pick = a["id"].endswith(f":{letter}")
            manifest.upsert(a["id"], status="picked" if pick else "rejected", meta={"review": reason})
    log(f"pick {sid}: {letter}")


def faces(path) -> int:
    """Frontal faces found by OpenCV's Haar cascade: a cheap duplicate-character screen (no tokens)."""
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 headless dropped the Haar cascades
        return -1
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (960, 544))
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return len(det.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(28, 28)))


def cmd_provisional():
    """While review is paused (session limit), give unreviewed shots a provisional pick so clips keep rendering.
    Single-person shots prefer a candidate with exactly one detected face; everything is re-reviewed later."""
    by_shot = {}
    for a in manifest.query(stage="6", kind="still"):
        by_shot.setdefault(a["meta"]["shot"], []).append(a)
    shots = {s["id"]: s for s in shots_needing_stills()}
    for sid, cands in by_shot.items():
        if sid not in shots or any(c["status"] == "picked" for c in cands):
            continue
        done = sorted([c for c in cands if c["status"] == "done"], key=lambda a: a["id"])
        if len(done) < N:
            continue
        people = [c for c in shots[sid].get("chars", []) if NOUN.get(c) == "man"]
        best = done[0]
        if len(people) == 1:
            ones = [c for c in done if faces(ROOT / c["path"]) == 1]
            best = ones[0] if ones else best
        manifest.upsert(best["id"], status="picked", meta={"provisional": True, "review": "provisional (face-count screen), review pending"})
        log(f"provisional {sid}: {best['id'].rsplit(':', 1)[1]}")


def cmd_retry(sid, new_text=None):
    """Rule 4: up to 3 rounds of new seeds, then a rewritten prompt (new_text), then degrade the shot."""
    shot = next(s for s in shots_needing_stills() if s["id"] == sid)
    tries = max([a["meta"].get("attempt", 0) for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == sid] + [0]) + 1
    for a in manifest.query(stage="6", kind="still"):
        if a["meta"]["shot"] == sid and a["status"] == "done":
            manifest.upsert(a["id"], status="rejected")
    base = 7000 + 131 * shot["index"] + 1000 * tries
    render(shot, [base + k for k in range(N)], shot_text=new_text, attempt=tries)
    if new_text:
        manifest.decision(f"Still prompt rewritten for {sid}", f"After {tries - 1} rejected rounds the SHOT text became: {new_text}")
    comfy.free()
    cmd_grids([sid])


def cmd_edit(sid, letter, instruction):
    """Rule 4 rewrite by editing: Klein recreates one reviewed candidate (its only reference) with a single change, e.g.
    a detail the base prompt keeps losing. New candidates continue the shot's letters."""
    shot = next(s for s in shots_needing_stills() if s["id"] == sid)
    src = manifest.get(f"still:{sid}:{letter}")
    ref = comfy.upload(ROOT / src["path"])
    prompt = (f"{characters.STYLE} Recreate Figure 1 exactly: the same composition, people, poses, faces, lighting and "
              f"background. Change only this: {instruction}")
    tries = max([a["meta"].get("attempt", 0) for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == sid] + [0]) + 1
    for a in manifest.query(stage="6", kind="still"):
        if a["meta"]["shot"] == sid and a["status"] == "done" and a["id"] != src["id"]:
            manifest.upsert(a["id"], status="rejected")
    base = 7000 + 131 * shot["index"] + 1000 * tries
    for k in range(N):
        new_letter = chr(65 + k + tries * N)
        aid = f"still:{sid}:{new_letter}"
        manifest.upsert(aid, stage="6", kind="still", status="running", meta={"shot": sid, "attempt": tries})
        g = wf.klein(f"still_{sid}_{new_letter}", prompt, seed=base + k, width=W, height=H, refs=[ref], ref_megapixels=1.0,
                     prefix=f"__SLUG__/stills/{sid}")
        wfp = g.save(ROOT / "workflows" / "stage6")
        res = comfy.run(g.api(), name=aid, timeout=600)
        dest = comfy.download(comfy.output_files(res["outputs"])[0], IMG / sid / f"cand_{new_letter}.png")
        manifest.upsert(aid, stage="6", kind="still", path=rel(dest), prompt=prompt, seed=base + k, model="flux-2-klein-9b-kv-fp8",
                        workflow=rel(wfp), score=None, status="done",
                        meta={"shot": sid, "refs": [ref], "attempt": tries, "edit_of": src["id"], "render_seconds": res["seconds"]})
        log(f"{aid}: edit of {letter}, {res['seconds']}s")
    manifest.decision(f"Still edited for {sid}", f"Candidate {letter} recreated with one change: {instruction}")
    comfy.free()
    cmd_grids([sid])


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "render":
        manifest.set_stage("6", status="running", started=manifest.now())
        cmd_render(rest)
    elif cmd == "edit":
        cmd_edit(rest[0], rest[1], " ".join(rest[2:]))
    elif cmd == "grids":
        cmd_grids(rest)
    elif cmd == "pick":
        cmd_pick(rest[0], rest[1], " ".join(rest[2:]))
    elif cmd == "retry":
        cmd_retry(rest[0], " ".join(rest[1:]) or None)
    elif cmd == "provisional":
        cmd_provisional()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
