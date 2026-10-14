"""Repaint through the official ACE-Step 1.5 API server (envs/acestep), the engine behind ACE-Step-ComfyUI.

The server is started only inside the GPU lock (after ComfyUI frees VRAM) and stopped afterwards.
"""
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path

import requests

from pipeline import comfy, gpu
from pipeline.manifest import ROOT

REPO = ROOT / "envs" / "acestep" / "ACE-Step-1.5"
PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
UNIT = "acestep"
DIT = "acestep-v15-xl-turbo"  # same family as the ComfyUI takes


def healthy() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=3).ok
    except requests.RequestException:
        return False


def start(timeout: int = 1800) -> None:
    if healthy():
        return
    env = {"ACESTEP_API_HOST": "127.0.0.1", "ACESTEP_API_PORT": str(PORT), "ACESTEP_CONFIG_PATH": DIT,
           "ACESTEP_INIT_LLM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"}
    subprocess.run(["systemd-run", "--user", f"--unit={UNIT}", "--collect", f"--working-directory={REPO}",
                    *sum([["-E", f"{k}={v}"] for k, v in env.items()], []),
                    "/bin/sh", "-c", f"exec {REPO}/.venv/bin/acestep-api >> {ROOT}/qc/logs/acestep-server.log 2>&1"],
                   check=True)
    t0 = time.time()
    while time.time() - t0 < timeout:  # first start downloads the checkpoints
        if healthy():
            return
        if subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True).stdout.strip() not in ("active", "activating"):
            raise RuntimeError("acestep-api exited; see qc/logs/acestep-server.log")
        time.sleep(5)
    raise RuntimeError("acestep-api did not become healthy")


def stop() -> None:
    subprocess.run(["systemctl", "--user", "stop", UNIT], capture_output=True)


@contextlib.contextmanager
def session(name: str = "acestep"):
    """GPU lock + ComfyUI freed + server up for the duration."""
    with gpu.gpu_job(name):
        if comfy.alive():
            comfy.free()
        start()
        try:
            yield
        finally:
            stop()


def _result(task_id: str, timeout: float = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.post(f"{BASE}/query_result", json={"task_id_list": [task_id]}, timeout=30).json()
        item = (r.get("data") or [{}])[0]
        if item.get("status") == 1:
            res = item["result"]
            return (json.loads(res) if isinstance(res, str) else res)[0]
        if item.get("status") == 2:
            raise RuntimeError(f"ACE-Step task failed: {json.dumps(item)[:2000]}")
        time.sleep(2)
    raise RuntimeError("ACE-Step task timed out")


def repaint(src: str | Path, start_s: float, end_s: float, caption: str, lyrics: str, seed: int, out: str | Path,
            bpm: int, key_scale: str, duration: float, steps: int = 8) -> dict:
    """Regenerate [start_s, end_s) of src; the rest of the song is kept. Must run inside session()."""
    # The server refuses absolute paths in src_audio_path, so the source goes up as a multipart file.
    body = {"task_type": "repaint", "repainting_start": float(start_s),
            "repainting_end": float(end_s), "chunk_mask_mode": "explicit", "prompt": caption, "lyrics": lyrics,
            "vocal_language": "en", "bpm": bpm, "key_scale": key_scale, "time_signature": "4",
            "audio_duration": float(duration), "use_random_seed": False, "seed": int(seed), "inference_steps": steps,
            "batch_size": 1, "audio_format": "flac", "model": DIT, "thinking": False}
    form = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in body.items()}
    t0 = time.time()
    with open(src, "rb") as f:
        r = requests.post(f"{BASE}/release_task", data=form, files={"src_audio": (Path(src).name, f, "audio/flac")},
                          timeout=300).json()
    task_id = (r.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"release_task failed: {r}")
    res = _result(task_id)
    url = res["file"] if res["file"].startswith("http") else BASE + res["file"]
    audio = requests.get(url, timeout=300)
    audio.raise_for_status()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(audio.content)
    return {"out": str(out), "seconds": round(time.time() - t0, 1), "dit_model": res.get("dit_model"),
            "seed_value": res.get("seed_value")}
