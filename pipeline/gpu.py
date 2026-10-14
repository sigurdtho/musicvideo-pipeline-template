"""One GPU queue: a cross-process lock, a guard against foreign VRAM holders, and a VRAM peak sampler."""
import contextlib
import os
import threading
import time

import pynvml
from filelock import FileLock

from pipeline.manifest import ROOT

_LOCK = FileLock(str(ROOT / ".gpu.lock"))
# Processes that may hold VRAM without blocking us: the desktop, ComfyUI and our own tool venvs.
_OURS = (os.environ.get("COMFY_DIR", "/opt/ComfyUI") + "/", str(ROOT / "envs") + "/")
_BLOCKERS = ("ollama", "hermes")


class GpuBlocked(RuntimeError):
    """Another heavy process (Ollama, Hermes Agent, ...) holds VRAM. Ask the user; never sudo."""


def _nvml():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def used_mib() -> int:
    return pynvml.nvmlDeviceGetMemoryInfo(_nvml()).used // (1 << 20)


def compute_procs() -> list[dict]:
    h = _nvml()
    out = []
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
        try:
            cmd = open(f"/proc/{p.pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        out.append({"pid": p.pid, "mib": (p.usedGpuMemory or 0) // (1 << 20), "cmd": cmd})
    return out


def foreign_holders(min_mib: int = 1024) -> list[dict]:
    """Big VRAM users that aren't ComfyUI or our venvs."""
    return [p for p in compute_procs()
            if p["mib"] >= min_mib and not any(o in p["cmd"] for o in _OURS)]


def check_free() -> None:
    blockers = [p for p in compute_procs() if any(b in p["cmd"].lower() for b in _BLOCKERS)]
    big = foreign_holders()
    if blockers or big:
        raise GpuBlocked(f"VRAM held by: {blockers + big}")


class VramPeak:
    """Samples device memory.used every `interval` seconds inside a `with` block."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.peak, self.baseline = interval, 0, 0
        self._stop = threading.Event()

    def __enter__(self):
        self.baseline = self.peak = used_mib()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, used_mib())
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, used_mib())


@contextlib.contextmanager
def gpu_job(name: str, check: bool = True):
    """Hold the GPU for one job. Raises GpuBlocked if Ollama/Hermes/other big users are resident."""
    with _LOCK:
        if check:
            check_free()
        (ROOT / ".gpu.owner").write_text(f"{name} pid={os.getpid()} since={time.strftime('%H:%M:%S')}\n")
        try:
            yield
        finally:
            (ROOT / ".gpu.owner").unlink(missing_ok=True)
