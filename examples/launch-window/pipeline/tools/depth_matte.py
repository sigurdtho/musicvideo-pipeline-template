"""r906c: foreground mattes from monocular depth (Depth-Anything-V2-Small, CPU only, read-only shared weights).

  python depth_matte.py <video> <t0> <t1> <fps> <outdir>
Writes d00000.npy ... (float16, 960x540, relative inverse depth normalised to 0..1, larger = nearer) for every frame in
[t0, t1). Runs in the whisperx env (transformers + torch, no OpenCV); forced to CPU so it never touches the GPU
(CLAUDE.md rule 2). stage10.plate_matte turns the depth into a matte (pipeline env).
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # noqa: E402

MODEL = Path(os.environ.get("COMFY_DIR", "/opt/ComfyUI")) / "models/geometry_estimation/Depth-Anything-V2-Small-hf"


def frames(video, t0, t1, fps, W=1920, H=1080):
    n = int(round((t1 - t0) * fps))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(video), "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    for _ in range(n):
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def main(a):
    video, t0, t1, fps, out = a[0], float(a[1]), float(a[2]), float(a[3]), Path(a[4])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    for k, rgb in enumerate(frames(video, t0, t1, fps)):
        with torch.no_grad():
            inp = proc(images=rgb, return_tensors="pt")
            d = model(**inp).predicted_depth[0].numpy()  # relative inverse depth: larger = nearer
        d = torch.nn.functional.interpolate(torch.from_numpy(d)[None, None], size=(540, 960), mode="bicubic")[0, 0].numpy()
        d = (d - d.min()) / (np.ptp(d) + 1e-6)
        np.save(out / f"d{k:05d}.npy", d.astype(np.float16))
    print(f"{k + 1} depth maps -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
