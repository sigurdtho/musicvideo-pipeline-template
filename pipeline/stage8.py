"""Stage 8: lip-sync. InfiniteTalk (Wan 2.1 480p, native ComfyUI) per singing/speaking shot, driven by the
isolated voice stem on the final timeline (Stage 4 export). Primary engine: LTX-2.3 image+audio-to-video
(higher SyncNet, 720p, 3.7x faster in Stage 0); InfiniteTalk is the alternate engine for retries.

  python -m pipeline.stage8 render [shot_id ...]
  python -m pipeline.stage8 score [shot_id ...]          SyncNet confidence/offset + mouth-frame sheet
  python -m pipeline.stage8 pick <shot_id> <attempt> [reason]
  python -m pipeline.stage8 retry <shot_id> [ltx|infinitetalk]   (default: alternate engines)
Long shots are chunked inside the graph (81-frame windows with 9 motion frames).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from pipeline import comfy, manifest, toolrun, vcheck, wf
from pipeline.manifest import ROOT

OUTDIR = ROOT / "video" / "lipsync"
QC = ROOT / "qc" / "stage8"
STEM = {"lead": "stem_lead.wav"}


def _stem_name(voice: str) -> str:
    return STEM.get(voice or "lead", f"stem_voice_{voice}.wav")
PRE_ROLL = 0.25  # start the audio a little early so the mouth is already moving at the cut


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def lipsync_shots() -> list[dict]:
    return [s for s in json.load(open(ROOT / "video" / "shotlist.json"))["shots"] if s["kind"] == "lipsync"]


def audio_for(shot: dict) -> Path:
    stem = ROOT / "out" / "master" / _stem_name(shot.get("voice", "lead"))
    out = OUTDIR / shot["id"] / "voice.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = max(0.0, shot["start"] - PRE_ROLL)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{shot['duration'] + PRE_ROLL + 0.1:.3f}", "-i",
                    str(stem), "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out


PRIMARY = "ltx"  # Stage 0: LTX-2.3 ia2v SyncNet 6.11 @ 1280x720 in 45 s vs InfiniteTalk 5.55 @ 832x480 in 170 s


def render(shot: dict, attempt: int, engine: str = PRIMARY):
    still = next((a for a in manifest.query(stage="6", kind="still") if a["meta"]["shot"] == shot["id"] and a["status"] == "picked"), None)
    if still is None:
        log(f"{shot['id']}: no picked still")
        return
    w, h = (832, 480) if engine == "infinitetalk" else (1280, 720)
    src = OUTDIR / shot["id"] / f"input_{w}x{h}.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ROOT / still["path"]).convert("RGB").resize((w, h), Image.LANCZOS).save(src)
    img, aud = comfy.upload(src), comfy.upload(audio_for(shot))
    secs = shot["duration"] + PRE_ROLL
    seed = 11000 + 53 * shot["index"] + 1000 * attempt
    prompt = shot.get("prompt") or "A person faces the camera, mouth moving with the vocal."
    aid = f"lipsync:{shot['id']}:{attempt}"
    if engine == "infinitetalk":
        g = wf.infinitetalk(f"lipsync_{shot['id']}_{attempt}", img, aud, prompt, seed=seed, seconds=secs, width=w, height=h,
                            prefix=f"__SLUG__/lipsync/{shot['id']}_{attempt}")
        model = "wan2.1_i2v_480p_fp8+infinitetalk_single+lightx2v"
    else:
        g = wf.ltx23_ia2v(f"lipsync_ltx_{shot['id']}_{attempt}", img, aud, prompt, seed=seed, seconds=secs, width=w, height=h,
                          prefix=f"__SLUG__/lipsync/{shot['id']}_{attempt}_ltx")
        model = "ltx-2.3-22b-distilled-fp8 ia2v"
    wfp = g.save(ROOT / "workflows" / "stage8")
    manifest.upsert(aid, stage="8", kind="lipsync", prompt=prompt, seed=seed, model=model, workflow=rel(wfp), status="running",
                    meta={"shot": shot["id"], "attempt": attempt, "engine": engine, "pre_roll": PRE_ROLL, "still": still["path"]})
    res = comfy.run(g.api(), name=aid, timeout=3600)
    dest = comfy.download(comfy.output_files(res["outputs"])[0], OUTDIR / shot["id"] / f"a{attempt}_{engine}.mkv")
    manifest.upsert(aid, path=rel(dest), status="done", meta={"render_seconds": res["seconds"]})
    log(f"{aid}: {res['seconds']}s")


def score(aid: str):
    a = manifest.get(aid)
    r = toolrun.run("syncnet", "syncnet_score.py", [ROOT / a["path"], OUTDIR / a["meta"]["shot"] / f"syncnet_{a['meta']['attempt']}"],
                    name=f"stage8:syncnet:{aid}", log=ROOT / "qc" / "logs" / "stage8-tools.log")
    vcheck.contact_sheet(ROOT / a["path"], QC / f"{a['meta']['shot']}_a{a['meta']['attempt']}_mouth.jpg", n=8)
    manifest.upsert(aid, status="scored", score={"syncnet_confidence": r["confidence"], "syncnet_offset": r["offset"]})
    log(f"{aid}: SyncNet confidence {r['confidence']} offset {r['offset']}")


def cmd_render(ids):
    for shot in lipsync_shots():
        if ids and shot["id"] not in ids:
            continue
        if any(a["meta"]["shot"] == shot["id"] and a["status"] in ("done", "scored", "picked", "running") for a in manifest.query(stage="8", kind="lipsync")):
            continue
        render(shot, 0)
    comfy.free()
    for a in manifest.query(stage="8", kind="lipsync"):
        if a["status"] == "done" and (not ids or a["meta"]["shot"] in ids):
            score(a["id"])


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "render":
        manifest.set_stage("8", status="running", started=manifest.now())
        cmd_render(rest)
    elif cmd == "score":
        for a in manifest.query(stage="8", kind="lipsync"):
            if a["status"] == "done" and (not rest or a["meta"]["shot"] in rest):
                score(a["id"])
    elif cmd == "pick":
        for a in manifest.query(stage="8", kind="lipsync"):
            if a["meta"]["shot"] == rest[0] and a["status"] in ("done", "scored", "picked"):
                p = a["meta"]["attempt"] == int(rest[1])
                manifest.upsert(a["id"], status="picked" if p else "rejected", meta={"review": " ".join(rest[2:])})
    elif cmd == "retry":
        shot = next(s for s in lipsync_shots() if s["id"] == rest[0])
        prev = [a for a in manifest.query(stage="8", kind="lipsync") if a["meta"]["shot"] == shot["id"]]
        attempt = max([a["meta"]["attempt"] for a in prev] + [-1]) + 1
        other = "infinitetalk" if PRIMARY == "ltx" else "ltx"
        render(shot, attempt, engine=rest[1] if rest[1:2] else (other if attempt % 2 else PRIMARY))
        comfy.free()
        score(f"lipsync:{shot['id']}:{attempt}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
