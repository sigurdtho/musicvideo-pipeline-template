"""Report the time spans where two renders, or two audio files, differ.

  python -m pipeline.version_diff A B
  python -m pipeline.version_diff --self-test

Audio is compared as mono float in 50 ms windows (mean absolute difference).
Windows over the threshold merge into spans. A longer file adds a tail span.

Video is read with ffmpeg as gray frames at a low rate (default 4 fps), scaled
to about 96x54. A frame whose mean absolute difference exceeds 8/255 opens a
span. This tool does not use the GPU.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

WIN_S = 0.05
AUDIO_THR = 0.02
VIDEO_THR = 8 / 255
AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}


def spans_from_mask(mask, hop_s):
    spans = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((round(start * hop_s, 3), round(i * hop_s, 3)))
            start = None
    if start is not None:
        spans.append((round(start * hop_s, 3), round(len(mask) * hop_s, 3)))
    return spans


def read_mono(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=True)
        return x.mean(axis=1).astype(np.float64), int(sr)
    except Exception:
        with wave.open(str(path), "rb") as handle:
            sr = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
        if width == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise SystemExit(f"{path}: unsupported sample width {width}")
        return x.reshape(-1, channels).mean(axis=1), int(sr)


def diff_audio(path_a, path_b, thr=AUDIO_THR, win_s=WIN_S):
    a, sr_a = read_mono(path_a)
    b, sr_b = read_mono(path_b)
    if sr_a != sr_b:
        raise SystemExit(f"sample rates differ: {sr_a} vs {sr_b}")
    n = min(len(a), len(b))
    win = max(1, int(round(win_s * sr_a)))
    m = n // win
    if m:
        delta = np.abs(a[: m * win] - b[: m * win]).reshape(m, win).mean(1)
        spans = spans_from_mask(delta > thr, win / sr_a)
    else:
        spans = []
    if len(a) != len(b):
        tail0 = n / sr_a
        tail1 = max(len(a), len(b)) / sr_a
        if tail1 - tail0 > 1e-3:
            spans.append((round(tail0, 3), round(tail1, 3)))
    return spans


def read_video_gray(path, fps=4, width=96, height=54):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    raw = subprocess.check_output(cmd)
    frame = width * height
    count = len(raw) // frame
    arr = np.frombuffer(raw[: count * frame], dtype=np.uint8).reshape(count, height, width).astype(np.float64) / 255.0
    return arr


def diff_frames(a, b, fps, thr=VIDEO_THR):
    n = min(len(a), len(b))
    spans = []
    if n:
        delta = np.abs(a[:n] - b[:n]).mean(axis=(1, 2))
        spans = spans_from_mask(delta > thr, 1 / fps)
    if len(a) != len(b):
        spans.append((round(n / fps, 3), round(max(len(a), len(b)) / fps, 3)))
    return spans


def diff_video(path_a, path_b, fps=4, thr=VIDEO_THR):
    return diff_frames(read_video_gray(path_a, fps), read_video_gray(path_b, fps), fps, thr)


def is_audio(path):
    return Path(path).suffix.lower() in AUDIO_EXT


def format_spans(spans):
    if not spans:
        return "no change"
    return "\n".join(f"changed {a:.3f}–{b:.3f} s" for a, b in spans)


def self_test():
    sr = 16000
    n = sr
    a = np.zeros(n)
    b = np.zeros(n)
    b[int(0.40 * sr): int(0.60 * sr)] = 0.5

    def write(path, samples):
        with wave.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    tmp = tempfile.mkdtemp(prefix="vdiff-")
    try:
        pa, pb, pc = (os.path.join(tmp, name) for name in ("a.wav", "b.wav", "c.wav"))
        write(pa, a)
        write(pb, b)
        spans = diff_audio(pa, pb)
        assert spans and spans[0][0] <= 0.45 and spans[0][1] >= 0.55, spans
        write(pc, np.zeros(n + sr // 2))
        tail = diff_audio(pa, pc)
        assert tail and tail[-1][0] >= 0.95 and tail[-1][1] >= 1.4, tail
        va = np.zeros((8, 54, 96))
        vb = np.zeros((8, 54, 96))
        vb[3:5] = 20 / 255
        frames = diff_frames(va, vb, fps=4)
        assert frames == [(0.75, 1.25)], frames
    finally:
        shutil.rmtree(tmp)
    print("version_diff self-test pass")
    return 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    p = argparse.ArgumentParser(description="Changed time spans between two renders or two audio files")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--fps", type=float, default=4, help="video sample rate")
    args = p.parse_args(argv)
    if is_audio(args.a) or is_audio(args.b):
        spans = diff_audio(args.a, args.b)
    else:
        spans = diff_video(args.a, args.b, fps=args.fps)
    print(format_spans(spans))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
