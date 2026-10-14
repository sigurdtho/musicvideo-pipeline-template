"""Run a tool script inside its own venv under the GPU lock, timing it and sampling the VRAM peak.

Tool scripts live in pipeline/tools/, import nothing from this package, and print one JSON object as
their last stdout line.
"""
import json
import subprocess
import time

from pipeline import comfy, gpu
from pipeline.manifest import ROOT

ACESTEP_PY = ROOT / "envs" / "acestep" / "ACE-Step-1.5" / ".venv" / "bin" / "python"


def python_for(tool: str) -> str:
    return str(ACESTEP_PY if tool == "acestep" else ROOT / "envs" / tool / "bin" / "python")


def run(tool: str, script: str, args: list[str], name: str | None = None, timeout: float = 7200,
        uses_gpu: bool = True, log=None) -> dict:
    cmd = [python_for(tool), str(ROOT / "pipeline" / "tools" / script), *map(str, args)]
    ctx = gpu.gpu_job(name or f"{tool}:{script}") if uses_gpu else _null()
    with ctx:
        if uses_gpu and comfy.alive():
            comfy.free()  # one GPU job at a time: ComfyUI must drop its weights first
        with gpu.VramPeak() as vp:
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
            dt = time.time() - t0
    if log:
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n{p.stdout[-20000:]}\n{p.stderr[-20000:]}\n")
    if p.returncode != 0:
        raise RuntimeError(f"{tool}/{script} exit {p.returncode}: {p.stderr[-3000:]}")
    lines = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
    out = json.loads(lines[-1]) if lines else {}
    out.update({"seconds": round(dt, 1), "vram_peak_mib": vp.peak, "vram_baseline_mib": vp.baseline})
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
