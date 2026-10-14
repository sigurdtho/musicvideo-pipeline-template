"""Stage 10: edit and finish.

  python -m pipeline.stage10 upscale     SeedVR2 7B -> 1920x1080 for picked i2v 16 fps masters and lip-sync clips
  python -m pipeline.stage10 interp      RIFE x3 on the upscaled i2v masters (48 fps)
  python -m pipeline.stage10 final       assemble + composite + mux + check + report (stops at the first failure)
  python -m pipeline.stage10 report      qc/stage-10.md (sources, upscale, grades, overlays, checks)
  python -m pipeline.stage10 night       loop upscale (picked, else the newest unreviewed clip) + interp while clip jobs run
  python -m pipeline.stage10 assemble    per-shot segments: time-remapped to their slot, graded (generated LUTs), concatenated
  python -m pipeline.stage10 composite   notice boxes over the graded edit, frame-exact
  python -m pipeline.stage10 mux         + master audio -> out/final/__SLUG__.mp4
  python -m pipeline.stage10 check       black frames, duration, A/V sync, loudness, format -> qc/final.md
  python -m pipeline.stage10 draft       assemble + composite + mux + check from the 720p stage 7/8 renders (a still where no
                                         clip exists yet) -> out/draft/, qc/draft-check.md; tests the edit before the upscale
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import comfy, lut, manifest, master, post, toolrun, vcheck, wf
from pipeline.manifest import ROOT

FPS = 24
W, H = 1920, 1080
UP = ROOT / "video" / "up"
EDIT = ROOT / "video" / "edit"
FINAL = ROOT / "out" / "final" / "__SLUG__.mp4"
LUTS = ROOT / "video" / "luts"
PRE_ROLL = 0.25  # matches stage8
# Both lip-sync engines came out -2 frames on SyncNet (25 fps); delaying the picture 80 ms measured offset 0
# (v1_window: -2 -> 0, the opposite shift -> -3), so lip-sync segments sample their source 80 ms earlier.
LIP_DELAY = 0.08
DRAFT = False
FADE_OUT = 1.0  # review round 3: the video fades to black on the song's last hit


def FADE_START() -> float:
    return shotlist()["duration"] - FADE_OUT


def use_draft():
    global DRAFT, EDIT, FINAL
    DRAFT, EDIT, FINAL = True, ROOT / "video" / "edit_draft", ROOT / "out" / "draft" / "__SLUG___draft.mp4"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def shotlist():
    return json.load(open(ROOT / "video" / "shotlist.json"))


def picked(stage: str, kind: str, shot_id: str) -> dict | None:
    return next((a for a in manifest.query(stage=stage, kind=kind) if a["meta"]["shot"] == shot_id and a["status"] == "picked"), None)


# ------------------------------------------------------------------ upscale + interpolate
def source_for(sh: dict, provisional: bool = False) -> dict | None:
    """The picked clip / lip-sync render; provisionally the newest unreviewed clip (its upscale is reused if picked)."""
    if sh["kind"] != "i2v":
        return picked("8", "lipsync", sh["id"])
    if sh.get("source") == "still":
        return picked("6", "still", sh.get("reuse_still") or sh["id"])
    if sh.get("reuse_clip"):  # a montage cut: another shot's picked clip (and its upscale)
        return picked("7", "clip", sh["reuse_clip"])
    a = picked("7", "clip", sh["id"])
    if a is None and provisional:
        done = [c for c in manifest.query(stage="7", kind="clip") if c["meta"]["shot"] == sh["id"] and c["status"] == "done"]
        a = max(done, key=lambda c: c["meta"]["attempt"]) if done else None
    return a


def cmd_upscale(provisional: bool = False) -> int:
    UP.mkdir(parents=True, exist_ok=True)
    work = 0
    for sh in shotlist()["shots"]:
        if (sh["kind"] == "overlay" and sh.get("bg") == "black") or sh.get("source") == "still" or sh.get("reuse_clip"):
            continue
        a = source_for(sh, provisional)
        if a is None:
            continue
        src = ROOT / (a["meta"]["files"]["fps16"] if sh["kind"] == "i2v" else a["path"])
        aid = f"up:{sh['id']}"
        prev = manifest.get(aid) or {}
        if prev.get("status") == "done" and prev.get("meta", {}).get("from") == a["id"] and prev.get("meta", {}).get("verified"):
            continue
        reuse = finished_render("__SLUG__/up", sh["id"], src)
        if reuse is not None:  # resumable: a render that finished in ComfyUI but was never (correctly) downloaded
            dest = UP / f"{sh['id']}.mkv"
            dest.write_bytes(reuse.read_bytes())
            manifest.upsert(aid, stage="10", kind="upscaled", path=rel(dest), prompt=None, seed=1, model="seedvr2_7b_int8",
                            workflow=rel(ROOT / "workflows" / "stage10" / f"up_{sh['id']}.json"), status="done",
                            meta={"from": a["id"], "shot": sh["id"], "verified": True, "reused": str(reuse)})
            log(f"{aid}: reused {reuse.name}")
            work += 1
            continue
        g = wf.seedvr2_video(f"up_{sh['id']}", comfy.upload(src), width=W, height=H, model="7b", frames_per_chunk=21,
                             overlap=2, prefix=f"__SLUG__/up/{sh['id']}")
        wfp = g.save(ROOT / "workflows" / "stage10")
        res = comfy.run(g.api(), name=aid, timeout=3600)
        dest = comfy.download(comfy.output_files(res["outputs"])[0], UP / f"{sh['id']}.mkv")
        manifest.upsert(aid, stage="10", kind="upscaled", path=rel(dest), prompt=None, seed=1, model="seedvr2_7b_int8",
                        workflow=rel(wfp), status="done", meta={"from": a["id"], "shot": sh["id"], "render_seconds": res["seconds"],
                                                                 "verified": render_matches(dest, src, (W, H))})
        log(f"{aid}: {res['seconds']}s")
        work += 1
    comfy.free()
    return work


def render_matches(out, src, size=None) -> bool:
    """out is a render of src (same frame count, matching mid and last frames) and has the expected size."""
    po, ps = vcheck.probe(out), vcheck.probe(src)
    if po["frames"] != ps["frames"] or (size and (po["width"], po["height"]) != size):
        return False
    n = ps["frames"]
    for i in (n // 2, n - 1):
        a, b = vcheck.frame_at(out, i), vcheck.frame_at(src, i)
        if a is None or b is None or vcheck.similarity(a, center_fit(b, a.shape[1], a.shape[0])) < 0.9:
            return False
    return True


def center_fit(img, w, h):
    """Center-crop to w:h, then resize (what ResizeImageMaskNode 'scale dimensions' + crop=center does)."""
    ih, iw = img.shape[:2]
    if iw * h > ih * w:
        cw = round(ih * w / h)
        img = img[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
    elif iw * h < ih * w:
        ch = round(iw * h / w)
        img = img[(ih - ch) // 2:(ih - ch) // 2 + ch]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def finished_render(subfolder: str, sid: str, src) -> Path | None:
    """Newest file in ComfyUI's output/<subfolder> named <sid>_NNNNN_.mkv that is a render of src at W x H."""
    import re
    d = comfy.COMFY_DIR / "output" / subfolder
    pat = re.compile(rf"^{re.escape(sid)}_\d{{5}}_\.mkv$")
    src_mtime = Path(src).stat().st_mtime
    for f in sorted((f for f in d.glob("*.mkv") if pat.match(f.name)), key=lambda f: f.stat().st_mtime, reverse=True):
        # a render of src must be newer than src: frame matching alone cannot tell two lip-sync renders of one still apart
        if f.stat().st_mtime > src_mtime and render_matches(f, src, (W, H)):
            return f
    return None


def cmd_interp() -> int:
    work = 0
    for sh in shotlist()["shots"]:
        if sh["kind"] != "i2v" or sh.get("source") == "still" or sh.get("reuse_clip"):
            continue
        up = manifest.get(f"up:{sh['id']}")
        if not up or not up["meta"].get("verified"):
            continue
        aid = f"up48:{sh['id']}"
        if (manifest.get(aid) or {}).get("meta", {}).get("from") == up["meta"]["from"]:
            continue
        g = wf.interpolate(f"up48_{sh['id']}", comfy.upload(ROOT / up["path"]), fps_in=16.0, multiplier=3,
                           prefix=f"__SLUG__/up48/{sh['id']}")
        wfp = g.save(ROOT / "workflows" / "stage10")
        res = comfy.run(g.api(), name=aid, timeout=1800)
        dest = comfy.download(comfy.output_files(res["outputs"])[0], UP / f"{sh['id']}_48fps.mkv")
        pi, pu = vcheck.probe(dest), vcheck.probe(ROOT / up["path"])
        ok = (pi["width"], pi["height"]) == (W, H) and pi["frames"] == 3 * pu["frames"] - 2
        manifest.upsert(aid, stage="10", kind="upscaled48", path=rel(dest), prompt=None, seed=None, model="rife_v4.26",
                        workflow=rel(wfp), status="done" if ok else "failed",
                        meta={"from": up["meta"]["from"], "shot": sh["id"], "frames": pi["frames"]})
        log(f"{aid}: {res['seconds']}s {pi['width']}x{pi['height']} {pi['frames']} frames{'' if ok else ' UNEXPECTED'}")
        work += 1
    comfy.free()
    return work


def clip_jobs_active() -> bool:
    r = subprocess.run(["systemctl", "--user", "list-units", "__SLUG__-*", "--state=active", "--no-legend", "--plain"],
                       capture_output=True, text=True)
    units = [line.split()[0] for line in r.stdout.splitlines() if line.strip()]  # unit names only, not descriptions
    return any(u.startswith(("__SLUG__-clips", "__SLUG__-retry")) for u in units)


def cmd_night():
    """Keep the GPU busy while reviews are paused: upscale + interpolate everything renderable, then wait for clips."""
    while True:
        work = cmd_upscale(provisional=True) + cmd_interp()
        if work:
            continue
        if not clip_jobs_active():
            log("night: nothing left to upscale or interpolate")
            return
        time.sleep(120)


# ------------------------------------------------------------------ assemble
def iter_frames(path, idx):
    """Frames at the given non-decreasing indices, resized to W x H, streamed so only a few frames sit in RAM
    (ComfyUI keeps its models in system memory)."""
    cap = cv2.VideoCapture(str(path))
    cur, f, out_i, out = -1, None, None, None
    for i in idx:
        while cur < i:
            ok, nf = cap.read()
            if not ok:
                break
            f, cur = nf, cur + 1
        if out_i != cur:
            out = f if (f.shape[1], f.shape[0]) == (W, H) else cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
            out_i = cur
        yield out
    cap.release()


def iter_frames_frac(path, pos):
    """Frames at non-decreasing fractional positions, blending the two neighbouring frames, streamed."""
    cap = cv2.VideoCapture(str(path))
    buf, cur = {}, -1

    def get(i):
        nonlocal cur
        while cur < i:
            ok, f = cap.read()
            if not ok:
                break
            cur += 1
            buf[cur] = f
            for k in [k for k in buf if k < cur - 1]:
                del buf[k]
        return buf.get(i, buf[max(buf)])

    for p_ in pos:
        i = int(np.floor(p_))
        a = p_ - i
        f = get(i)
        if a > 0.01:
            f = cv2.addWeighted(f, 1 - a, get(i + 1), a, 0)
        yield f if (f.shape[1], f.shape[0]) == (W, H) else cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
    cap.release()


MAX_SPEEDUP = 1.2


def activity(path16) -> np.ndarray:
    """Motion energy per 16 fps frame interval: mean absolute luma change at thumbnail size."""
    cap = cv2.VideoCapture(str(path16))
    prev, out = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            out.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.array(out)


def i2v_window(n: int, last: int, act: np.ndarray, clip_range=None, loop=None) -> tuple[list[float], dict]:
    """48 fps source positions for an n-frame slot. Longer slots slow the clip down (never below half speed). Shorter
    ones take the stretch of the clip with the most motion, slightly favouring later stretches (actions tend to
    complete at the end), at up to MAX_SPEEDUP so more of the action fits.

    clip_range [t0, t1] (seconds) restricts the edit to a reviewed clean part of the clip; loop="pingpong" plays that
    part forward then back (for near-static shots whose slot is much longer than the clean part)."""
    if clip_range:
        lo, hi = round(clip_range[0] * 48), min(last, round(clip_range[1] * 48))
        if loop == "pingpong":
            span = 2 * (hi - lo)
            pos = [k * span / max(1, n - 1) for k in range(n)]
            return [lo + (q if q <= hi - lo else span - q) for q in pos], {
                "range": clip_range, "loop": loop, "speed": round(span / 2 / max(1, n - 1), 3)}
        pos, info = i2v_window(n, hi - lo, act[int(clip_range[0] * 16):int(np.ceil(clip_range[1] * 16))])
        return [lo + q for q in pos], {**info, "range": clip_range, "t0": round(info["t0"] + clip_range[0], 3)}
    length = last / 48
    if n - 1 >= last / 2:  # the slot is at least as long as the clip at real time
        return [k * last / max(1, n - 1) for k in range(n)], {"t0": 0.0, "speed": round(last / 2 / max(1, n - 1), 3)}
    speed = min(MAX_SPEEDUP, length / ((n - 1) / FPS))
    span = (n - 1) * speed / FPS
    cum = np.concatenate([[0.0], np.cumsum(act)])  # energy up to each 16 fps interval boundary

    def energy(t):  # integral of the step function act over [0, t] (t in seconds)
        j = min(float(len(act)), max(0.0, t * 16))
        k = int(np.floor(j))
        return cum[k] + (j - k) * (act[k] if k < len(act) else 0.0)

    best, t_best = -1.0, 0.0
    room = max(0.0, length - span)
    for t0 in np.arange(0.0, room + 1e-9, 1 / 48):
        sc = (energy(t0 + span) - energy(t0)) * (1 + 0.15 * (t0 / room if room > 0 else 0))
        if sc > best:
            best, t_best = sc, float(t0)
    return [min(last, (t_best + k * speed / FPS) * 48) for k in range(n)], {"t0": round(t_best, 3), "speed": round(speed, 3)}


WINDOWS = {}


def segment_frames(sh: dict, n: int):
    if sh["kind"] == "overlay" and sh.get("bg") == "black":
        return [np.zeros((H, W, 3), np.uint8)] * n
    if sh.get("source") == "still":
        st = source_for(sh)
        return post.still_frames(cv2.imread(str(ROOT / st["path"])), n, (W, H), sh.get("post"))
    if sh["kind"] == "i2v":
        a = source_for(sh, provisional=DRAFT)
        if a is None:  # draft only: no clip yet
            st = picked("6", "still", sh.get("reuse_still") or sh["id"])
            return [cv2.resize(post._crop_16x9(cv2.imread(str(ROOT / st["path"]))), (W, H), interpolation=cv2.INTER_AREA)] * n
        path = ROOT / (a["meta"]["files"]["fps48"] if DRAFT else manifest.get(f"up48:{sh.get('reuse_clip') or sh['id']}")["path"])
        last = vcheck.probe(path)["frames"] - 1  # 48 fps
        pos, WINDOWS[sh["id"]] = i2v_window(n, last, activity(ROOT / a["meta"]["files"]["fps16"]), sh.get("clip_range"),
                                            sh.get("clip_loop"))
        fx = sh.get("post_clip")
        if fx and fx.get("effect") == "flash_below":  # comp effect on a clip (e.g. the bomb's flash the model never drew)
            dur = n / FPS
            return (post.flash_below(f, k / FPS, fx["at"] * dur, fx.get("rise", 0.08), fx.get("decay", 0.7), fx.get("strength", 0.9))
                    for k, f in enumerate(iter_frames_frac(path, pos)))
        return iter_frames_frac(path, pos)
    clip = ROOT / (picked("8", "lipsync", sh["id"])["path"] if DRAFT else manifest.get(f"up:{sh['id']}")["path"])
    pr = vcheck.probe(clip)
    fps_src, last = pr["fps"] or 25.0, pr["frames"] - 1
    return iter_frames(clip, [min(last, max(0, round((PRE_ROLL + k / FPS - LIP_DELAY) * fps_src))) for k in range(n)])


def stale_sources() -> list[str]:
    """Shots whose upscaled (and, for i2v, 48 fps) render is missing, unverified or not from the current pick."""
    bad = []
    for sh in shotlist()["shots"]:
        if sh["kind"] == "overlay":
            continue
        a = source_for(sh)
        if sh.get("source") == "still":
            if a is None:
                bad.append(f"{sh['id']}: still")
            continue
        key = sh.get("reuse_clip") or sh["id"]
        up = manifest.get(f"up:{key}") or {}
        u48 = manifest.get(f"up48:{key}") or {}
        if a is None or up.get("meta", {}).get("from") != a["id"] or not up["meta"].get("verified"):
            bad.append(f"{sh['id']}: upscale")
        elif sh["kind"] == "i2v" and (u48.get("status") != "done" or u48["meta"].get("from") != a["id"]):
            bad.append(f"{sh['id']}: 48 fps")
    return bad


def cmd_assemble():
    if not DRAFT and (bad := stale_sources()):
        raise SystemExit(f"assemble: {len(bad)} shots not ready: {bad}")
    EDIT.mkdir(parents=True, exist_ok=True)
    for k in lut.PRESETS:
        lut.write(k, LUTS / f"{k}.cube")
    sl = shotlist()
    audio_len = master.measure  # noqa (kept for symmetry)
    total_frames = int(round(sf_duration(ROOT / "out" / "master" / "__SLUG___master.wav") * FPS))
    segs = []
    shots = sl["shots"]
    for j, sh in enumerate(shots):
        f0 = round(sh["start"] * FPS)
        f1 = round(shots[j + 1]["start"] * FPS) if j + 1 < len(shots) else total_frames
        n = max(1, f1 - f0)
        grade = lut.grade_for({**sh, "id": sh["reuse_clip"]}) if sh.get("reuse_clip") else lut.grade_for(sh)
        out = EDIT / f"seg_{j:03d}_{sh['id']}.mkv"
        frames = segment_frames(sh, n)
        p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                              "-i", "-", "-vf", f"lut3d={LUTS / (grade + '.cube')}", "-c:v", "ffv1", "-level", "3",
                              "-pix_fmt", "yuv444p", str(out)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        segs.append(out)
        if DRAFT:
            continue
        manifest.upsert(f"seg:{sh['id']}", stage="10", kind="segment", path=rel(out), prompt=None, seed=None, model=None,
                        workflow="pipeline/stage10.py assemble", score=None, status="done",
                        meta={"frames": n, "grade": grade, "start_frame": f0, "window": WINDOWS.get(sh["id"]),
                              "post": sh.get("post")})
        log(f"seg {j:03d} {sh['id']}: {n} frames, grade {grade}")
    lst = EDIT / "concat.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(EDIT / "graded.mkv")], check=True)
    log(f"assembled {len(segs)} segments -> {rel(EDIT / 'graded.mkv')}")


def sf_duration(path) -> float:
    import soundfile as sf
    return sf.info(str(path)).duration


# ------------------------------------------------------------------ overlays
def cmd_composite():
    boxes = []
    for a in manifest.query(stage="9", kind="overlay"):
        if a["status"] != "done":  # boxes dropped from the plan or waiting for a re-render
            continue
        frames = sorted((ROOT / a["meta"]["frames_dir"]).glob("f*.png"))
        boxes.append({"f0": round(a["meta"]["t_in"] * FPS), "frames": frames})
    src = EDIT / "graded.mkv"
    dst = EDIT / "composited.mkv"
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p", str(dst)], stdin=subprocess.PIPE)
    size = W * H * 3
    f = 0
    cache = {}
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
        for bi, b in enumerate(boxes):
            k = f - b["f0"]
            if 0 <= k < len(b["frames"]):
                key = (bi, k)
                if key not in cache:
                    rgba = cv2.imread(str(b["frames"][k]), cv2.IMREAD_UNCHANGED)
                    cache = {key: rgba}  # keep memory flat
                rgba = cache[key]
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                frame = frame * (1 - alpha) + rgba[:, :, :3].astype(np.float32) * alpha
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
        f += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    log(f"composited {f} frames, {len(boxes)} boxes -> {rel(dst)}")


def cmd_mux():
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(EDIT / "composited.mkv"), "-i",
                    str(ROOT / "out" / "master" / "__SLUG___master.wav"), "-map", "0:v", "-map", "1:a",
                    "-vf", f"fade=t=out:st={FADE_START():.3f}:d={FADE_OUT}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-r", str(FPS), "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
                    str(FINAL)], check=True)
    manifest.upsert("draft:video" if DRAFT else "final:video", stage="10", kind="draft" if DRAFT else "final", path=rel(FINAL),
                    prompt=None, seed=None, model=None, workflow="pipeline/stage10.py", score=None, status="done")
    log(f"muxed {rel(FINAL)}")


# ------------------------------------------------------------------ checks
def cmd_check() -> dict:
    sl = shotlist()
    info = vcheck.probe(FINAL)
    audio = ROOT / "out" / "master" / "__SLUG___master.wav"
    adur = sf_duration(audio)
    checks = {}
    checks["format"] = (info["width"] == W and info["height"] == H and abs(info["fps"] - FPS) < 0.01 and info["has_audio"],
                        f"{info['width']}x{info['height']} @ {info['fps']:.3f} fps, audio={info['has_audio']}")
    checks["duration"] = (abs(info["duration"] - adur) < 2 / FPS and abs(info["frames"] / FPS - adur) < 2 / FPS,
                          f"video {info['frames'] / FPS:.3f} s ({info['frames']} frames), container {info['duration']:.3f} s, master {adur:.3f} s")
    allowed = [(s["start"], s["end"]) for s in sl["shots"] if s["kind"] == "overlay" and s.get("bg") == "black"]
    allowed.append((FADE_START(), sl["duration"] + 1.0))  # the closing fade to black
    blacks = [b for b in vcheck.black_frames(FINAL)
              if not any(b["start"] >= a - 0.1 and b["end"] <= e + 0.1 for a, e in allowed)]
    checks["black_frames"] = (not blacks, f"unexpected black: {blacks}" if blacks else f"only intended black: {len(allowed) - 1} black-background shot(s) and the closing fade")
    lm = master.measure(FINAL)
    checks["loudness"] = (-14.5 <= lm["lufs"] <= -13.5 and lm["true_peak_dbtp"] < -1.0,
                          f"{lm['lufs']} LUFS, true peak {lm['true_peak_dbtp']} dBTP")
    # A/V sync 1: the muxed audio is the master, sample-aligned (cross-correlate the first 20 s)
    lag = av_lag(FINAL, audio)
    checks["av_sync_audio"] = (abs(lag) <= 1 / FPS, f"audio lag vs master {lag * 1000:.1f} ms")
    # A/V sync 2: SyncNet on lip-sync shots cut from the final file
    sync = []
    for s in [s for s in sl["shots"] if s["kind"] == "lipsync"][:4]:
        cut = EDIT / f"sync_{s['id']}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s['start']:.3f}", "-t", f"{min(s['duration'], 4.0):.3f}", "-i",
                        str(FINAL), "-c:v", "libx264", "-crf", "14", "-c:a", "aac", str(cut)], check=True)
        r = toolrun.run("syncnet", "syncnet_score.py", [cut, EDIT / f"syncnet_{s['id']}"], name=f"final:syncnet:{s['id']}",
                        log=ROOT / "qc" / "logs" / "stage10-tools.log")
        sync.append((s["id"], r.get("offset"), r.get("confidence")))
    good = [x for x in sync if x[1] is not None and abs(x[1]) <= 2 and (x[2] or 0) >= 3.0]
    checks["av_sync_lips"] = (len(good) == len(sync) and bool(sync), f"SyncNet (offset frames, confidence): {sync}")
    gates = manifest.load()["gates"]
    checks["gates"] = (all(g.get("status") == "approved" for g in gates.values() if g.get("enabled")),
                       {k: g.get("status") for k, g in gates.items()})
    write_final_md(checks, info, adur)
    return checks


def av_lag(video, audio) -> float:
    import soundfile as sf
    a, sr = sf.read(str(audio), frames=48000 * 20, always_2d=True)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-t", "20", "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                       capture_output=True)
    b = np.frombuffer(p.stdout, np.float32).reshape(-1, 2)
    x, y = a[:, 0][: len(b)], b[:, 0][: len(a)]
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)
    k = int(np.argmax(c))
    k = k - n if k > n // 2 else k
    return k / sr


def write_final_md(checks, info, adur):
    lines = ["# Draft checks" if DRAFT else "# Final checks", "",
             f"Generated {manifest.now()} by `pipeline/stage10.py {'draft' if DRAFT else 'check'}` for `{rel(FINAL)}`.", "",
             "| Check | Result | Detail |", "| --- | --- | --- |"]
    for k, (ok, detail) in checks.items():
        lines.append(f"| {k} | {'PASS' if ok else 'FAIL'} | {detail} |")
    allpass = all(v[0] for v in checks.values())
    lines += ["", f"**Overall: {'every check passing' if allpass else 'NOT all checks passing'}.**", ""]
    md = "qc/draft-check.md" if DRAFT else "qc/final.md"
    (ROOT / md).write_text("\n".join(lines))
    manifest.upsert("draft:checks" if DRAFT else "final:checks", stage="10", kind="qc", path=md, workflow="pipeline/stage10.py check",
                    status="pass" if allpass else "fail", score={k: bool(v[0]) for k, v in checks.items()})
    log(f"final checks: {'ALL PASS' if allpass else 'FAILURES'} {[(k, v[0]) for k, v in checks.items()]}")


def cmd_report():
    from collections import Counter
    sl = shotlist()["shots"]
    rows, ok_src = [], True
    secs = []
    for j, sh in enumerate(sl):
        if sh["kind"] == "overlay":
            rows.append(f"| {j:02d} | {sh['id']} | overlay | {sh['start']:.2f} | {sh['duration']:.2f} | black | - | - |")
            continue
        a = source_for(sh)
        if sh.get("source") == "still":
            ok_src &= a is not None
            rows.append(f"| {j:02d} | {sh['id']} | still | {sh['start']:.2f} | {sh['duration']:.2f} | {lut.grade_for(sh)} | "
                        f"{a['id'] if a else '-'} (animated still: {(sh.get('post') or {}).get('effect', 'code push')}) | {'yes' if a else 'NO'} |")
            continue
        key = sh.get("reuse_clip") or sh["id"]  # montage cuts use the reused shot's upscale
        up = manifest.get(f"up:{key}") or {}
        u48 = manifest.get(f"up48:{key}") or {}
        good = bool(a and up.get("meta", {}).get("from") == a["id"] and up["meta"].get("verified")
                    and (sh["kind"] != "i2v" or (u48.get("status") == "done" and u48["meta"].get("from") == a["id"])))
        ok_src &= good
        if up.get("meta", {}).get("render_seconds"):
            secs.append(up["meta"]["render_seconds"])
        win = ((manifest.get(f"seg:{sh['id']}") or {}).get("meta") or {}).get("window") or {}
        speed = ""
        if sh.get("source") == "still":
            speed = f"(animated still: {(sh.get('post') or {}).get('effect', 'code push')})"
        elif win:
            speed = (f"from {win.get('t0', 0)} s" if "t0" in win else "") + f" x{win['speed']}" + \
                    (f" range {win['range']}" if win.get("range") else "") + (f" {win['loop']}" if win.get("loop") else "")
        grade = lut.grade_for({**sh, "id": sh["reuse_clip"]}) if sh.get("reuse_clip") else lut.grade_for(sh)
        rows.append(f"| {j:02d} | {sh['id']} | {sh['kind']}{' (montage)' if sh.get('reuse_clip') else ''} | {sh['start']:.2f} | "
                    f"{sh['duration']:.2f} | {grade} | {a['id'] if a else '-'} {speed} | {'yes' if good else 'NO'} |")
    grades = Counter(lut.grade_for(s_) for s_ in sl if s_["kind"] != "overlay")
    checks = (ROOT / "qc" / "final.md").read_text().split("\n", 4)[-1] if (ROOT / "qc" / "final.md").exists() else "(not run yet)"
    md = [
        "# Stage 10 · Edit and finish", "", f"Generated {manifest.now()}.", "",
        "## Pipeline", "",
        f"- Upscale: SeedVR2 7B int8 (one-step restoration) from 720p to {W}x{H}, 21-frame temporal chunks with 2 overlapping "
        f"latent frames, LAB colour match to the resized input. Every output is verified against its source (same frame "
        f"count, mid and last frames correlate >= 0.9 after the node's centre crop)."
        + (f" Median {sorted(secs)[len(secs) // 2]:.0f} s per clip." if secs else ""),
        "- Interpolation: RIFE v4.26 x3 on the upscaled 16 fps i2v masters -> 48 fps (241 frames per 5.06 s clip).",
        f"- Assembly at {FPS} fps on the Stage 2 shot list, cutting on the beat grid. Shorter i2v shots take the stretch of "
        f"their 5.06 s clip with the most motion, favouring later stretches, at up to {MAX_SPEEDUP}x with blended "
        "in-between frames. Longer shots slow down to fill their slot, never below half speed. "
        "A shot can keep a reviewed clean range (clip_range, clip_loop). Animated stills use pipeline/post.py. Lip-sync "
        f"shots sample their render from {PRE_ROLL} s pre-roll with a {LIP_DELAY * 1000:.0f} ms picture delay.",
        "- Grade: generated 3D LUTs per sequence (" + ", ".join(f"{k} x{v}" for k, v in grades.items()) + ").",
        "- Overlays: the Stage 9 boxes composited frame-exact over the graded edit (alpha PNG sequences).",
        "- Mux: H.264 High CRF 16 yuv420p, AAC 384 kb/s 48 kHz from out/master/__SLUG___master.wav.", "",
        "Engineering notes: a 02:50 audit found that Stage 10 had been saving the LoadVideo input preview instead of the "
        "SeedVR2 result. Upload names were shared across shots, so concurrent jobs could overwrite each other's inputs. "
        "Fixes: uploads are now content-addressed, and history outputs of type 'input' are ignored. All 39 clips "
        "rendered by then matched their stills (first-frame correlation >= 0.999), and the finished SeedVR2 renders "
        "were recovered from ComfyUI's output folder after the same verification.", "",
        "## Self-check", "",
        f"Every shot's segment comes from its picked render through a verified upscale (and 48 fps interpolation for i2v): "
        f"**{'PASS' if ok_src else 'FAIL'}**", "",
        "## Final checks (qc/final.md)", "", checks, "",
        "## Edit decision list", "",
        "| # | Shot | Kind | Start s | Dur s | Grade | Source | Verified |", "| ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        *rows, ""]
    (ROOT / "qc" / "stage-10.md").write_text("\n".join(md))
    manifest.set_stage("10", report="qc/stage-10.md")
    log(f"report: qc/stage-10.md (sources {'PASS' if ok_src else 'FAIL'})")


def main(argv):
    cmd = argv[0] if argv else ""
    steps = {"upscale": cmd_upscale, "interp": cmd_interp, "assemble": cmd_assemble, "composite": cmd_composite,
             "mux": cmd_mux, "check": cmd_check, "night": cmd_night,
             "report": cmd_report}
    if cmd in steps:
        if cmd == "upscale":
            manifest.set_stage("10", status="running", started=manifest.now())
        steps[cmd]()
    elif cmd == "final":  # the finishing chain; stops at the first failure (assemble refuses stale sources)
        for k in ("assemble", "composite", "mux", "check", "report"):
            steps[k]()
    elif cmd == "draft":
        use_draft()
        for k in ("assemble", "composite", "mux", "check"):
            steps[k]()
    elif cmd == "all":
        for k in ("upscale", "interp", "assemble", "composite", "mux", "check"):
            steps[k]()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
