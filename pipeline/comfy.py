"""ComfyUI over HTTP: queue API-format workflows, follow /ws, collect /history outputs, free VRAM.

The base URL is the only thing that changes for a rented ComfyUI (fallback ladder rung 2).
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import requests
import websocket

from pipeline import gpu
from pipeline.manifest import ROOT

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/opt/ComfyUI"))
UNIT = "comfyui"


class ComfyError(RuntimeError):
    pass


def alive(timeout=3) -> bool:
    try:
        return requests.get(f"{BASE}/system_stats", timeout=timeout).ok
    except requests.RequestException:
        return False


def ensure_server(wait_s: int = 180) -> None:
    """Start ComfyUI headless on 127.0.0.1:8188 as a user unit if it isn't answering."""
    if alive():
        return
    subprocess.run(["systemd-run", "--user", f"--unit={UNIT}", "--collect", f"--working-directory={COMFY_DIR}",
                    "/bin/sh", "-c",
                    f"exec {COMFY_DIR}/.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch"
                    f" >> {ROOT}/qc/logs/comfyui.log 2>&1"], check=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if alive():
            return
        time.sleep(2)
    raise ComfyError("ComfyUI did not come up")


def upload(path: str | Path, subfolder: str = "__SLUG__") -> str:
    """Upload a local file into ComfyUI's input dir; returns the name LoadImage/LoadAudio/LoadVideo expect."""
    path = Path(path)
    # content-addressed name: many shots share basenames (input_720p.png, a0_fps16.mkv), and a job uploads before it
    # waits for the GPU lock, so a plain basename could be overwritten by another job in between
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    with path.open("rb") as f:
        r = requests.post(f"{BASE}/upload/image", files={"image": (f"{digest}_{path.name}", f)},
                          data={"subfolder": subfolder, "type": "input", "overwrite": "true"}, timeout=600)
    r.raise_for_status()
    d = r.json()
    return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]


def free(unload_models: bool = True, wait_s: float = 30.0) -> None:
    """Ask ComfyUI to drop its models, then wait until its process actually releases the VRAM (/free is async)."""
    requests.post(f"{BASE}/free", json={"unload_models": unload_models, "free_memory": True}, timeout=60)
    if not unload_models:
        return
    t0 = time.time()
    while time.time() - t0 < wait_s:
        held = sum(p["mib"] for p in gpu.compute_procs() if str(COMFY_DIR) in p["cmd"])
        if held < 1500:  # CUDA context + small caches
            return
        time.sleep(0.5)


def interrupt() -> None:
    requests.post(f"{BASE}/interrupt", timeout=10)


def queue(workflow: dict, client_id: str) -> str:
    r = requests.post(f"{BASE}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=60)
    if not r.ok:
        raise ComfyError(f"/prompt rejected: {r.status_code} {r.text[:2000]}")
    d = r.json()
    if d.get("node_errors"):
        raise ComfyError(f"node errors: {json.dumps(d['node_errors'])[:2000]}")
    return d["prompt_id"]


def history(prompt_id: str) -> dict | None:
    r = requests.get(f"{BASE}/history/{prompt_id}", timeout=30)
    r.raise_for_status()
    return r.json().get(prompt_id)


def _wait_ws(ws, prompt_id: str, timeout: float, on_progress=None) -> None:
    deadline = time.time() + timeout
    ws.settimeout(15)
    while time.time() < deadline:
        try:
            msg = ws.recv()
        except websocket.WebSocketTimeoutException:
            h = history(prompt_id)  # belt and braces: the ws can miss the final frame
            if h and h.get("status", {}).get("completed") is not None:
                return
            continue
        if not isinstance(msg, str):
            continue  # binary preview frames
        m = json.loads(msg)
        t, data = m.get("type"), m.get("data", {})
        if data.get("prompt_id") not in (None, prompt_id):
            continue
        if t == "progress" and on_progress:
            on_progress(data.get("node"), data.get("value"), data.get("max"))
        elif t == "execution_error":
            raise ComfyError(f"{data.get('node_type')}: {data.get('exception_message')}\n{''.join(data.get('traceback', [])[-6:])}")
        elif t == "execution_interrupted":
            raise ComfyError("interrupted")
        elif t == "execution_success" or (t == "executing" and data.get("node") is None):
            return
    raise ComfyError(f"timeout after {timeout}s")


def run(workflow: dict, timeout: float = 3600, name: str = "comfy", on_progress=None, lock: bool = True) -> dict:
    """Queue a workflow, wait for it, and return {'prompt_id', 'outputs', 'seconds', 'vram_peak_mib', 'vram_baseline_mib'}."""
    ensure_server()
    client_id = uuid.uuid4().hex
    ctx = gpu.gpu_job(name) if lock else _null()
    with ctx:
        ws = websocket.create_connection(f"{BASE.replace('http', 'ws', 1)}/ws?clientId={client_id}", timeout=30)
        try:
            with gpu.VramPeak() as vp:
                t0 = time.time()
                pid = queue(workflow, client_id)
                _wait_ws(ws, pid, timeout, on_progress)
                dt = time.time() - t0
        finally:
            ws.close()
    h = history(pid) or {}
    status = h.get("status", {})
    if status.get("status_str") == "error":
        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        raise ComfyError(f"execution error: {json.dumps(msgs)[:2000]}")
    return {"prompt_id": pid, "outputs": h.get("outputs", {}), "seconds": round(dt, 1),
            "vram_peak_mib": vp.peak, "vram_baseline_mib": vp.baseline}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def output_files(outputs: dict) -> list[dict]:
    """Flatten history outputs into [{'filename','subfolder','type','node','kind'}]."""
    files = []
    for node, out in outputs.items():
        for kind, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:  # skip input previews (LoadVideo echoes its source file as a UI output)
                if isinstance(it, dict) and "filename" in it and it.get("type") != "input":
                    files.append({**it, "node": node, "kind": kind})
    return files


def download(item: dict, dest: str | Path) -> Path:
    """Copy one output file to dest (local disk copy when possible, else /view)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = COMFY_DIR / (item.get("type") or "output") / item.get("subfolder", "") / item["filename"]
    if BASE.startswith("http://127.0.0.1") and local.exists():
        shutil.copy2(local, dest)
    else:
        r = requests.get(f"{BASE}/view", params={"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                                  "type": item.get("type", "output")}, timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest
