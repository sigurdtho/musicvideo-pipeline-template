"""Final 1080p of the approved r906c cut (the user 2026-10-14), ported from wot-mat's pipeline/final1080.py.

Picture per shot source (same timing, windows, grades and overlays as the r906c draft):
- Wan clips (18): SeedVR2 7B int8 720p -> 1920x1080, then RIFE x3 -> 48 fps (stage10 upscale / interp), then the same
  i2v_window code as the draft on the 1080p frames;
- stills: post.still_frames at 1920x1080 on the full-res Klein still (as the draft);
- code layers (HUD, lyrics, slams, end card): re-rendered by headless Chrome at native 1920x1080 (pipeline.hud).
Audio: the approved master, loudnorm (two-pass) to -14 LUFS integrated, true peak <= -1 dBTP, measured on the encode.

  python -m pipeline.final1080 gpu       SeedVR2 upscale + RIFE of every clip shot (one ComfyUI job at a time)
  python -m pipeline.final1080 overlays  HUD + end card re-render (CPU, Chrome)
  python -m pipeline.final1080 render    assemble + mattes + composite + mux -> out/final/launch-window.mp4 (pre-loudness)
  python -m pipeline.final1080 loud      loudnorm -> out/final/launch-window_final_1080p.mp4, measured
  python -m pipeline.final1080 qa        frames across the whole video -> qc/final1080/qa_*.jpg
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline import manifest, stage10 as S
from pipeline.manifest import ROOT

OUT = ROOT / "out" / "final" / "launch-window_final_1080p.mp4"
QA = ROOT / "qc" / "final1080"
TARGET_LUFS, TARGET_TP = -14.0, -1.5  # TP aims under -1 dBTP with margin for the AAC encode


def log(m):
    S.log(m)


def cmd_gpu():
    S.cmd_upscale()
    S.cmd_interp()
    bad = S.stale_sources()
    log(f"gpu: {len(bad)} shots not ready {bad}")
    if bad:
        raise SystemExit(1)


def cmd_overlays():
    from pipeline import hud
    hud.main(["render"])
    hud.main(["endcard"])


def cmd_render(start="assemble"):
    steps = [("assemble", S.cmd_assemble), ("mattes", S.cmd_mattes), ("composite", S.cmd_composite), ("mux", S.cmd_mux)]
    k = [n for n, _ in steps].index(start)
    for _, f in steps[k:]:  # resumable: `render composite` restarts after a finished assembly
        f()


def measure(path) -> dict:
    """ffmpeg ebur128 integrated loudness and true peak (dBTP, 4x oversampled)."""
    r = subprocess.run(["ffmpeg", "-nostats", "-v", "info", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    get = lambda key, after: float(tail.split(after, 1)[1].split(key, 1)[1].split()[0])  # noqa: E731
    return {"lufs": get("I:", "Integrated loudness:"), "lra": get("LRA:", "Loudness range:"),
            "true_peak_dbtp": get("Peak:", "True peak:")}


def loudnorm_filter(src) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-vn", "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    j = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    return (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
            f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")


def normalise(src, dst, extra_v=None) -> dict:
    """Copy (or re-encode) the picture, loudnorm the sound at 48 kHz; returns before/after measurements."""
    before = measure(src)
    af = loudnorm_filter(src) + ",aresample=48000"
    v = extra_v or ["-c:v", "copy"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v", "-map", "0:a", *v, "-af", af,
                    "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", str(dst)], check=True)
    after = measure(dst)
    return {"before": before, "after": after}


def cmd_loud():
    # from the lossless master (padded with silence under the end card), not from the mux's AAC
    tmp = QA / "loud_src.mkv"
    QA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(S.FINAL), "-i", str(ROOT / "out" / "master" / "launch-window_master.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", f"apad=pad_dur={S.endcard_s()}", "-c:a", "pcm_s24le",
                    "-t", str(S.vcheck.probe(S.FINAL)["frames"] / S.FPS), str(tmp)], check=True)
    res = normalise(tmp, OUT)
    tmp.unlink()
    a = res["after"]
    ok = -14.5 <= a["lufs"] <= -13.5 and a["true_peak_dbtp"] <= -1.0
    manifest.upsert("final:1080p", stage="10", kind="final", path=str(OUT.relative_to(ROOT)), prompt=None, seed=None,
                    model=None, workflow="pipeline/final1080.py loud", score=None, status="done" if ok else "failed",
                    meta={"loudness": res})
    (QA).mkdir(parents=True, exist_ok=True)
    (QA / "loudness.json").write_text(json.dumps(res, indent=1))
    log(f"loud: {res} {'OK' if ok else 'OUT OF SPEC'}")
    if not ok:
        raise SystemExit(1)


def contact(path, times, dest, cols=4, w=480):
    tiles = []
    for t in times:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
                            "-vcodec", "png", "-"], capture_output=True)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (w, round(w * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{int(t // 60)}:{t % 60:05.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(dest), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def cmd_qa(path=None):
    path = Path(path or OUT)
    QA.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True).stdout)
    times = list(np.arange(1.0, dur - 0.2, dur / 48))
    for k in range(0, len(times), 16):
        contact(path, times[k:k + 16], QA / f"qa_{path.stem}_{k // 16}.jpg")
    log(f"qa: {len(times)} frames -> {QA}")


def main(argv):
    cmds = {"gpu": cmd_gpu, "overlays": cmd_overlays, "render": cmd_render, "loud": cmd_loud, "qa": cmd_qa}
    if argv[:1] == ["all"]:
        for k in ("gpu", "render", "loud", "qa"):
            cmds[k]()
        return 0
    if argv and argv[0] in cmds:
        cmds[argv[0]](*argv[1:])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
