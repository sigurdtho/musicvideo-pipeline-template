"""Stage 5: looks. 8 candidates per look -> review -> pick -> QuadView multi-view sheet -> crops for Stage 6.

  python -m pipeline.stage5 candidates [look ...]   Klein 9B t2i, 8 seeds per look
  python -m pipeline.stage5 grids                   labelled review grids in qc/stage5/
  python -m pipeline.stage5 pick <look> <A-H>        record the reviewed pick (orchestrator, after subagent review)
  python -m pipeline.stage5 sheets [look ...]       QuadView sheets from the picks (2 seeds each)
  python -m pipeline.stage5 crops <look> <A|B>      pick a sheet; crop face + front views for stills
Gate `looks` is off in CLAUDE.md, so picks are made by review without waiting for the user.
"""
import sys
import time
from pathlib import Path

from PIL import Image

from pipeline import characters, comfy, manifest, review, wf
from pipeline.manifest import ROOT

IMG = ROOT / "images" / "looks"
QC = ROOT / "qc" / "stage5"
N = 8
SEEDS = [501 + 97 * i for i in range(N)]
BACKDROP = ("Full-body character portrait, standing, facing camera, relaxed neutral pose, plain dark-grey studio "
            "backdrop, soft even key light, the whole figure visible from head to feet.")
ANIMAL_BACKDROP = ("Full-body character portrait, the whole animal visible, facing camera at a slight three-quarter "
                   "angle, plain dark-grey studio backdrop, soft even key light.")
CROWD_BACKDROP = "Medium shot of a small group of distinct people side by side, facing the camera."
ANIMALS = set()  # non-human looks skip the people sheet
CROWD_LOOKS = set()  # looks that use the group backdrop and skip the turnaround
# Per-look clarifications after review (DECISIONS.md): framing and text-free props, never a changed CHARACTER line.
LOOK_EXTRA = {}  # look id -> extra prompt sentence. Never a changed CHARACTER line.
WIDE = {}  # look id -> (width, height) when the default portrait canvas is wrong


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def look_prompt(look: str) -> str:
    c = characters.CHARACTERS[look]
    frame = CROWD_BACKDROP if look in CROWD_LOOKS else ANIMAL_BACKDROP if look in ANIMALS else BACKDROP
    return f"{characters.STYLE} {frame} {c['full']}{LOOK_EXTRA.get(look, '')}"


def picked_candidate(look: str) -> dict | None:
    return next((a for a in manifest.query(stage="5", kind="look_candidate") if a["meta"]["look"] == look and a["status"] == "picked"), None)


def cmd_candidates(looks, rnd: int = 0):
    for look in looks:
        base = characters.CHARACTERS[look].get("base")
        refs = []
        if base:
            # "Same man, ..." variants are edits of the picked base look, so the face carries over (continuity)
            bp = picked_candidate(base)
            if bp is None:
                log(f"{look}: waiting for a pick of {base}")
                continue
            refs = [comfy.upload(ROOT / bp["path"])]
            prompt = (f"{characters.STYLE} {BACKDROP} The same man as in Figure 1, same face, hair and build. "
                      f"{characters.CHARACTERS[look]['line']}")
        else:
            prompt = look_prompt(look)
        w, h = WIDE.get(look, (832, 1216))
        for i0, seed in enumerate(SEEDS):
            i, seed = i0 + N * rnd, seed + 10007 * rnd
            aid = f"look:{look}:cand:{chr(65 + i)}"
            prior = manifest.get(aid) or {}
            if prior.get("status") in ("done", "picked", "rejected") and (not base or prior.get("meta", {}).get("edit_of")):
                continue
            g = wf.klein(f"look_{look}_{chr(65 + i)}", prompt, seed=seed, width=w, height=h, refs=refs, prefix=f"__SLUG__/looks/{look}")
            wfp = g.save(ROOT / "workflows" / "stage5")
            res = comfy.run(g.api(), name=aid, timeout=600)
            dest = comfy.download(comfy.output_files(res["outputs"])[0], IMG / look / f"cand_{chr(65 + i)}.png")
            manifest.upsert(aid, stage="5", kind="look_candidate", path=rel(dest), prompt=prompt, seed=seed,
                            model="flux-2-klein-9b-kv-fp8", workflow=rel(wfp), score=None, status="done",
                            meta={"look": look, "render_seconds": res["seconds"], "edit_of": base and picked_candidate(base)["id"]})
            log(f"{aid}: {res['seconds']}s")
    cmd_grids(looks)


def cmd_grids(looks):
    for look in looks:
        cands = sorted(manifest.query(stage="5", kind="look_candidate"), key=lambda a: a["id"])
        paths = [ROOT / a["path"] for a in cands if a["meta"]["look"] == look]
        if paths:
            out = review.grid(paths, QC / f"{look}_candidates.jpg", cols=4, cell_w=420)
            log(f"grid {look}: {rel(out)}")


def cmd_pick(look, letter, reason=""):
    for a in manifest.query(stage="5", kind="look_candidate"):
        if a["meta"]["look"] != look:
            continue
        picked = a["id"].endswith(f":{letter}")
        manifest.upsert(a["id"], status="picked" if picked else "rejected", meta={"review": reason} if picked else {})
    log(f"pick {look}: {letter}")


def _pad_to_sheet(src: Path, out: Path):
    im = Image.open(src).convert("RGB")
    canvas = Image.new("RGB", (1536, 1024), (255, 255, 255))
    im.thumbnail((1536, 1024), Image.LANCZOS)
    canvas.paste(im, ((1536 - im.width) // 2, (1024 - im.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def cmd_sheets(looks):
    for look in looks:
        if look in CROWD_LOOKS or look in ANIMALS:
            continue  # groups have no turnaround; non-human looks skip the people sheet
        pick = next((a for a in manifest.query(stage="5", kind="look_candidate") if a["meta"]["look"] == look and a["status"] == "picked"), None)
        if not pick:
            log(f"sheets {look}: no pick yet")
            continue
        ref = comfy.upload(_pad_to_sheet(ROOT / pick["path"], IMG / look / "sheet_ref.png"))
        for k, seed in enumerate((901, 902)):
            aid = f"look:{look}:sheet:{chr(65 + k)}"
            if (manifest.get(aid) or {}).get("status") in ("done", "picked", "rejected"):
                continue
            g = wf.charsheet(f"sheet_{look}_{chr(65 + k)}", ref, seed=seed, prefix=f"__SLUG__/looks/{look}_sheet")
            wfp = g.save(ROOT / "workflows" / "stage5")
            res = comfy.run(g.api(), name=aid, timeout=600)
            dest = comfy.download(comfy.output_files(res["outputs"])[0], IMG / look / f"sheet_{chr(65 + k)}.png")
            manifest.upsert(aid, stage="5", kind="look_sheet", path=rel(dest), prompt=wf.CHARSHEET_QUAD_PROMPT, seed=seed,
                            model="flux-2-klein-9b-kv-fp8+QuadView_klein9b_v1", workflow=rel(wfp), score=None,
                            status="done", meta={"look": look, "from": pick["id"], "render_seconds": res["seconds"]})
            log(f"{aid}: {res['seconds']}s")
        sheets = [ROOT / a["path"] for a in sorted(manifest.query(stage="5", kind="look_sheet"), key=lambda a: a["id"])
                  if a["meta"]["look"] == look]
        review.grid(sheets, QC / f"{look}_sheets.jpg", cols=2, cell_w=900)


def panels(im: Image.Image, white: float = 242.0, min_gap: int = 6, min_w: int = 80) -> list[tuple[int, int]]:
    """Split a character sheet into panels at near-white vertical gutters."""
    import numpy as np
    a = np.asarray(im.convert("L"), dtype=np.float32)
    col_white = a.mean(axis=0) > white
    spans, x, W = [], 0, a.shape[1]
    while x < W:
        if col_white[x]:
            x += 1
            continue
        x0 = x
        while x < W and not (col_white[x:x + min_gap].all() and x + min_gap <= W):
            x += 1
        if x - x0 >= min_w:
            spans.append((x0, x))
    return spans


def cmd_crops(look, letter):
    """QuadView layout: face close-up, then front / side / back full-body views; split at white gutters."""
    sheet = manifest.get(f"look:{look}:sheet:{letter}")
    im = Image.open(ROOT / sheet["path"]).convert("RGB")
    W, H = im.size
    sp = panels(im)
    if len(sp) >= 2:
        face, front = im.crop((sp[0][0], 0, sp[0][1], H)), im.crop((sp[1][0], 0, sp[1][1], H))
    else:  # fall back to the QuadView proportions
        face = im.crop((0, 0, int(W * 0.31), H))
        front = im.crop((int(W * 0.31), 0, int(W * 0.54), H))
    out = {}
    for name, c in (("face", face), ("front", front)):
        p = IMG / look / f"ref_{name}.png"
        c.save(p)
        out[name] = rel(p)
    for a in manifest.query(stage="5", kind="look_sheet"):
        if a["meta"]["look"] == look:
            manifest.upsert(a["id"], status="picked" if a["id"].endswith(f":{letter}") else "rejected",
                            meta={"crops": out} if a["id"].endswith(f":{letter}") else {})
    log(f"crops {look}: {out}")


def cmd_report():
    """qc/stage-5.md: picks with review reasons, sheets, and the LoRA decision."""
    md = ["# Stage 5 · Looks", "", f"Generated {manifest.now()}. Gate `looks` is off (CLAUDE.md), so picks come from review: "
          "subagents compared each 8-candidate grid against its CHARACTER line (details, anatomy, full figure, no text, originality).", "",
          "| Look | Picked candidate | Rounds | Sheet | Review |", "| --- | --- | ---: | --- | --- |"]
    for look in characters.LOOKS:
        cands = [a for a in manifest.query(stage="5", kind="look_candidate") if a["meta"]["look"] == look]
        pick = next((a for a in cands if a["status"] == "picked"), None)
        sheet = next((a for a in manifest.query(stage="5", kind="look_sheet") if a["meta"]["look"] == look and a["status"] == "picked"), None)
        rounds = len({a["id"].rsplit(":", 1)[1] for a in cands}) // N or 1
        md.append(f"| {look} | {pick['id'] if pick else 'pending'} | {rounds} | {sheet['id'] if sheet else 'pending'} | "
                  f"{(pick or {}).get('meta', {}).get('review', '')[:160]} |")
    md += ["", "Character LoRAs: master.md asks for them only where faces drift between shots. Stage 6 still reviews check identity "
           "per shot; a LoRA (ai-toolkit) is trained for a character only if its stills are rejected for identity drift in "
           "more than a third of its shots. Status is recorded in DECISIONS.md.", ""]
    (ROOT / "qc" / "stage-5.md").write_text("\n".join(md))
    log("report: qc/stage-5.md")


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    looks = rest or characters.LOOKS
    if cmd == "candidates":
        manifest.set_stage("5", status="running", started=manifest.now())
        rnd = 0
        if rest and rest[0].startswith("round="):
            rnd, rest = int(rest[0].split("=")[1]), rest[1:]
            looks = rest or characters.LOOKS
        cmd_candidates(looks, rnd)
    elif cmd == "grids":
        cmd_grids(looks)
    elif cmd == "pick":
        cmd_pick(rest[0], rest[1], " ".join(rest[2:]))
    elif cmd == "sheets":
        cmd_sheets(looks)
    elif cmd == "crops":
        cmd_crops(rest[0], rest[1])
    elif cmd == "report":
        cmd_report()
        return 0
    else:
        print(__doc__)
        return 2
    comfy.free()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
