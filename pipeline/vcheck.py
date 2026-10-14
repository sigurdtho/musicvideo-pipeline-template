"""Video checks: contact sheets, motion statistics (morph/flicker/freeze flags), black frames, probing.

  python -m pipeline.vcheck sheet <video> <out.jpg> [n]
  python -m pipeline.vcheck stats <video>
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def probe(path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,width,height,avg_frame_rate,nb_read_frames:format=duration", "-of", "json",
                        str(path)], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {"width": v.get("width"), "height": v.get("height"), "fps": float(num) / float(den or 1),
            "frames": int(v.get("nb_read_frames") or 0), "duration": float(d.get("format", {}).get("duration") or 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def frames(path, step: int = 1, max_side: int = 0):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if max_side and max(f.shape[:2]) > max_side:
                s = max_side / max(f.shape[:2])
                f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
            yield i, f
        i += 1
    cap.release()


def contact_sheet(path, out, n: int = 8, cols: int = 4, tile_w: int = 480) -> str:
    """n evenly spaced frames, labelled with frame index and time."""
    info = probe(path)
    total = info["frames"] or 1
    picks = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, n)))
    got = {i: f for i, f in frames(path) if i in picks}
    tiles = []
    for i in picks:
        if i not in got:
            continue
        im = Image.fromarray(cv2.cvtColor(got[i], cv2.COLOR_BGR2RGB))
        im = im.resize((tile_w, int(tile_w * im.height / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        label = f"#{i}  {i / max(info['fps'], 1e-6):.2f}s"
        d.rectangle([0, 0, 8 + 9 * len(label), 22], fill=(0, 0, 0))
        d.text((4, 4), label, fill=(255, 255, 255))
        tiles.append(im)
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].height
    sheet = Image.new("RGB", (cols * tile_w, rows * th), (0, 0, 0))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * tile_w, (k // cols) * th))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return str(out)


def motion_stats(path, max_side: int = 480) -> dict:
    """Per-frame luma mean, mean abs diff and Farneback flow magnitude; flags from robust z-scores."""
    prev = None
    luma, diff, flow = [], [], []
    for _, f in frames(path, max_side=max_side):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        luma.append(float(g.mean()))
        if prev is not None:
            diff.append(float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()))
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow.append(float(np.linalg.norm(fl, axis=2).mean()))
        prev = g
    diff, flow, luma = np.array(diff), np.array(flow), np.array(luma)
    if len(diff) < 3:
        return {"frames": len(luma), "flags": ["too_short"]}

    def robust_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
        return (x - med) / mad

    # morph: diff spikes far above the clip's own motion level that optical flow does not explain
    resid = diff / (flow + 0.5)
    morph = np.where(robust_z(resid) > 6)[0].tolist()
    flick = np.where(np.abs(robust_z(np.diff(luma))) > 8)[0].tolist()
    flags = []
    if morph:
        flags.append("morph")
    if flick:
        flags.append("flicker")
    if np.median(flow) < 0.05:
        flags.append("frozen")
    return {"frames": int(len(luma)), "mean_diff": round(float(diff.mean()), 3), "max_diff": round(float(diff.max()), 3),
            "mean_flow": round(float(flow.mean()), 3), "max_flow": round(float(flow.max()), 3),
            "morph_frames": morph[:20], "flicker_frames": flick[:20], "flags": flags}


def black_frames(path, min_d: float = 0.05) -> list:
    p = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-vf", f"blackdetect=d={min_d}:pix_th=0.06", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    import re
    return [dict(zip(("start", "end", "duration"), map(float, m)))
            for m in re.findall(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)", p.stderr)]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sheet":
        print(contact_sheet(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8))
    elif cmd == "stats":
        print(json.dumps({"probe": probe(sys.argv[2]), "motion": motion_stats(sys.argv[2]), "black": black_frames(sys.argv[2])}))


def _small(img_bgr, size=(160, 90)) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def frame_at(path, index: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def similarity(a_bgr, b_bgr) -> float:
    """Correlation of two images at thumbnail size (1.0 = same picture); used to confirm a render came from its input."""
    return float((_small(a_bgr) * _small(b_bgr)).mean())


def first_frame_match(video, image_or_video) -> float:
    a = frame_at(video, 0)
    p = Path(image_or_video)
    b = cv2.imread(str(p)) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else frame_at(p, 0)
    if a is None or b is None:
        return -1.0
    if b.shape[1] / b.shape[0] > a.shape[1] / a.shape[0] + 0.01 or b.shape[1] / b.shape[0] < a.shape[1] / a.shape[0] - 0.01:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return similarity(a, b)
