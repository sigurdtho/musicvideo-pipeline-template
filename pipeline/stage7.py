"""Stage 7: clips. Wan 2.2 I2V (720p, lightx2v 4-step) per shot from its picked still; RIFE x3 in the same graph.

  python -m pipeline.stage7 render [shot_id ...]
  python -m pipeline.stage7 sheets [shot_id ...]     8-frame contact sheets + motion stats in qc/stage7/
  python -m pipeline.stage7 pick <shot_id> <attempt> [reason]
  python -m pipeline.stage7 retry <shot_id> [--n N] [new MOTION text]    N new seeds, reviewed together
  python -m pipeline.stage7 report                                      qc/stage-7.md
The prompt is the shot's MOTION line only (master.md: "MOTION describes movement only, for image-to-video").
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

from pipeline import comfy, manifest, vcheck, wf
from pipeline.manifest import ROOT

CLIPS = ROOT / "video" / "clips"
QC = ROOT / "qc" / "stage7"
W, H = 1280, 720


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def i2v_shots(include_still: bool = False) -> list[dict]:
    """i2v shots that need a clip; shots degraded to an animated still (source == "still") only with include_still."""
    return [s for s in json.load(open(ROOT / "video" / "shotlist.json"))["shots"]
            if s["kind"] == "i2v" and not s.get("reuse_clip") and (include_still or s.get("source") != "still")]


def picked_still(shot: dict) -> Path | None:
    sid = shot.get("reuse_still") or shot["id"]
    a = next((a for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == sid and a["status"] == "picked"), None)
    return ROOT / a["path"] if a else None


def render(shot: dict, attempt: int, motion: str | None = None):
    still = picked_still(shot)
    if still is None:
        log(f"{shot['id']}: no picked still yet")
        return
    src = CLIPS / shot["id"] / "input_720p.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    Image.open(still).convert("RGB").resize((W, H), Image.LANCZOS).save(src)
    name = comfy.upload(src)
    seed = 9000 + 37 * shot["index"] + 1000 * attempt
    prompt = motion or shot.get("motion") or "Subtle natural motion, slow cinematic camera drift."
    aid = f"clip:{shot['id']}:{attempt}"
    g = wf.wan22_i2v(f"clip_{shot['id']}_{attempt}", name, prompt, seed=seed, width=W, height=H, frames=81,
                     prefix=f"__SLUG__/clips/{shot['id']}_{attempt}")
    wfp = g.save(ROOT / "workflows" / "stage7")
    manifest.upsert(aid, stage="7", kind="clip", prompt=prompt, seed=seed, model="wan2.2_i2v_14B_fp8+lightx2v_4step+rife_v4.26",
                    workflow=rel(wfp), status="running", meta={"shot": shot["id"], "attempt": attempt, "still": rel(still)})
    res = comfy.run(g.api(), name=aid, timeout=1800)
    files = {}
    for item in comfy.output_files(res["outputs"]):
        key = "fps48" if "48fps" in item["filename"] else "fps16"
        files[key] = rel(comfy.download(item, CLIPS / shot["id"] / f"a{attempt}_{key}.mkv"))
    stats = vcheck.motion_stats(ROOT / files["fps16"])
    manifest.upsert(aid, path=files["fps48"], status="done", score={"flags": stats["flags"], "mean_flow": stats["mean_flow"]},
                    meta={"files": files, "render_seconds": res["seconds"], "motion": stats})
    vcheck.contact_sheet(ROOT / files["fps16"], QC / f"{shot['id']}_a{attempt}.jpg", n=8)
    log(f"{aid}: {res['seconds']}s flags {stats['flags']}")


def cmd_render(ids):
    for shot in i2v_shots():
        if ids and shot["id"] not in ids:
            continue
        if any(a["meta"]["shot"] == shot["id"] and a["status"] in ("done", "picked", "running") for a in manifest.query(stage="7", kind="clip")):
            continue
        render(shot, 0)
    comfy.free()


def cmd_pick(sid, attempt, reason=""):
    for a in manifest.query(stage="7", kind="clip"):
        if a["meta"]["shot"] == sid and a["status"] in ("done", "picked"):
            pick = a["meta"]["attempt"] == int(attempt)
            manifest.upsert(a["id"], status="picked" if pick else "rejected", meta={"review": reason})


def cmd_retry(sid, motion=None, n=1):
    shot = next(s for s in i2v_shots() if s["id"] == sid)
    prev = [a for a in manifest.query(stage="7", kind="clip") if a["meta"]["shot"] == sid]
    for a in prev:
        if a["status"] in ("done", "running"):
            manifest.upsert(a["id"], status="rejected")
    attempt = max([a["meta"]["attempt"] for a in prev] + [-1]) + 1
    for k in range(n):
        render(shot, attempt + k, motion)
    if motion:
        manifest.decision(f"Clip prompt rewritten for {sid}", f"After {attempt} rejected attempts the MOTION text became: {motion}")
    comfy.free()


def cmd_report():
    from pipeline import shot_overrides
    shots = i2v_shots()
    clips = manifest.query(stage="7", kind="clip")
    rows, ok_all, secs = [], True, []
    for sh in i2v_shots(include_still=True):
        if sh.get("source") == "still":
            still = picked_still(sh)
            ok_all &= still is not None
            rows.append(f"| {sh['id']} | still | {sum(1 for a in clips if a['meta']['shot'] == sh['id'])} | - | - | "
                        f"post: {sh['post']['effect']} | animated still in the edit (rule 4 degrade, see DECISIONS.md) |")
            continue
        mine = [a for a in clips if a["meta"]["shot"] == sh["id"]]
        pick = next((a for a in mine if a["status"] == "picked"), None)
        secs += [a["meta"]["render_seconds"] for a in mine if a["meta"].get("render_seconds")]
        still = picked_still(sh)
        match = vcheck.first_frame_match(ROOT / pick["meta"]["files"]["fps16"], still) if pick and still else -1
        fresh = bool(pick and still and pick["meta"].get("still") == rel(still) and match >= 0.98)
        ok_all &= fresh
        ov = shot_overrides.OVERRIDES.get(sh["id"], {})
        note = (pick or {}).get("meta", {}).get("review", "") or "no pick"
        rows.append(f"| {sh['id']} | {pick['meta']['attempt'] if pick else '-'} | {len(mine)} | "
                    f"{', '.join(pick['score']['flags']) if pick and pick['score']['flags'] else '-'} | {match:.3f} | "
                    f"{'yes' if 'motion' in ov else ''} | {note[:110]} |")
    n_pick = sum(1 for r in rows if "| no pick |" not in r and "| still |" not in r)
    md = [
        "# Stage 7 · Clips", "", f"Generated {manifest.now()}.", "",
        f"{len(shots)} i2v shots render a clip; {sum(1 for x in i2v_shots(True) if x.get('source') == 'still')} more are "
        f"animated stills in the edit (pipeline/post.py). Each clip is Wan 2.2 I2V 14B fp8 with the lightx2v 4-step LoRAs at {W}x{H}, 81 frames at "
        "16 fps (5.06 s), and RIFE v4.26 x3 in the same graph for a 48 fps master. The prompt is the shot's MOTION line "
        "only (master.md), and the input is the shot's picked still. A subagent reviewed every clip as an 8-frame contact "
        "sheet with motion statistics (morph, flicker and freeze heuristics). Criteria: no morphing or identity change, "
        "the MOTION happens (a natural partial version allowed), no text, no flicker or jumps.", "",
        f"Picked: {n_pick}/{len(shots)}. Attempts rendered: {len(clips)}. Render time per clip: median "
        f"{sorted(secs)[len(secs) // 2]:.0f} s" + (f", total {sum(secs) / 3600:.1f} h." if secs else "."), "",
        "Rule 4: shots whose MOTION was rewritten after review are marked below; each rewrite and its reason is in "
        "pipeline/shot_overrides.py and DECISIONS.md. Clips rendered from a provisional still that review later replaced "
        "were re-rendered from the final still.", "",
        "## Self-check", "",
        "Every pick passed subagent review, was rendered from the shot's final picked still (manifest meta.still), and "
        "its first frame matches that still (thumbnail correlation >= 0.98; uploads are content-addressed since 02:55, "
        "after shared input names were found to be raceable between concurrent jobs).", "",
        f"**{'PASS' if ok_all else 'FAIL'}**", "",
        "| Shot | Pick | Attempts | Flags | First frame vs still | MOTION rewritten | Review note |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |", *rows, ""]
    (ROOT / "qc" / "stage-7.md").write_text("\n".join(md))
    manifest.set_stage("7", status="done" if ok_all else "running", report="qc/stage-7.md")
    log(f"report: qc/stage-7.md ({'PASS' if ok_all else 'FAIL'})")


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "render":
        manifest.set_stage("7", status="running", started=manifest.now())
        cmd_render(rest)
    elif cmd == "pick":
        cmd_pick(rest[0], rest[1], " ".join(rest[2:]))
    elif cmd == "report":
        cmd_report()
    elif cmd == "retry":
        n = 1
        if len(rest) > 2 and rest[1] == "--n":
            n, rest = int(rest[2]), rest[:1] + rest[3:]
        cmd_retry(rest[0], " ".join(rest[1:]) or None, n)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
