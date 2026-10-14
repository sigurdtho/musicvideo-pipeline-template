"""Alpha compositing and ffmpeg frame IO (streamed: never load whole clips into RAM)."""
import json
import subprocess

import cv2
import numpy as np


def probe(path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"w": s["width"], "h": s["height"], "fps": num / den, "frames": int(s.get("nb_frames") or 0)}


def read_frames(path, size=None):
    info = probe(path)
    w, h = size or (info["w"], info["h"])
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + (["-vf", f"scale={w}:{h}"] if size else []) + \
          ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


class Writer:
    """BGR frames -> video. Default: H.264 CRF 16 yuv420p (smoke/drafts); pass codec_args for ffv1 intermediates."""
    def __init__(self, path, w, h, fps=24, codec_args=None, audio=None):
        self.args = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                     "-i", "-"] + (["-i", str(audio), "-shortest", "-c:a", "aac", "-b:a", "192k"] if audio else []) + \
                    (codec_args or ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]) + [str(path)]

    def __enter__(self):
        self.p = subprocess.Popen(self.args, stdin=subprocess.PIPE)
        return self

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __exit__(self, *exc):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc and not exc[0]:
            raise RuntimeError(f"ffmpeg exited {rc}")


def over(frame_bgr: np.ndarray, rgba: np.ndarray | None) -> np.ndarray:
    """Straight-alpha RGBA (as cv2 reads a PNG: BGRA) over a BGR frame."""
    if rgba is None:
        return frame_bgr
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (frame_bgr.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def load_rgba(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.shape[2] == 3:
        im = np.concatenate([im, np.full(im.shape[:2] + (1,), 255, np.uint8)], 2)
    return im


def glass_over(frame_bgr: np.ndarray, rgba: np.ndarray | None, sigma: float = 16.0, min_alpha: int = 60,
               dim: float = 0.86) -> np.ndarray:
    """UI theme v2 'gilded glass': blur (and slightly dim) the painting under the UI's panels, then put the UI on top.
    Panels are the UI pixels with alpha >= min_alpha (glass fills); hairlines and thin text are too small to matter.
    The mask is closed and feathered so the blur edge follows the rounded panel edge without a halo."""
    if rgba is None:
        return frame_bgr
    import cv2
    m = (rgba[..., 3] >= min_alpha).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
    blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigma).astype(np.float32) * dim
    base = frame_bgr.astype(np.float32) * (1 - m) + blur * m
    return over(np.clip(base, 0, 255).astype(np.uint8), rgba)
