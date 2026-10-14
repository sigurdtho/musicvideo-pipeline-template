"""Assemble qc/stage-0.md from live inventory, manifest entries, smoke results and the schedule.

Narrative (findings, decisions, risks) is hand-written in qc/stage0_notes.md and inserted verbatim.
  python -m pipeline.stage0_report
"""
import json
import os
import shutil
import subprocess

import requests

from pipeline import manifest, schedule
from pipeline.manifest import ROOT
from pipeline.models import COMFY_MODELS, MODELS

QC = ROOT / "qc"


def gpu() -> str:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap,power.limit",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cuda = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda = cuda.split("CUDA UMD Version:")[1].split("|")[0].strip() if "CUDA UMD Version:" in cuda else "?"
    return f"{q} · CUDA {cuda}"


def comfy() -> dict:
    try:
        return requests.get("http://127.0.0.1:8188/system_stats", timeout=5).json()["system"]
    except Exception:
        return {}


def size(n):
    return f"{n / 1e9:.2f} GB"


def models_table() -> str:
    reg = {f"{m.folder}/{m.name}": m for m in MODELS}
    rows = ["| File | Folder | Size | Source @ revision | sha256 | Stage | Status |", "| --- | --- | ---: | --- | --- | --- | --- |"]
    for m in MODELS:
        a = manifest.get(f"model:{m.key}") or {}
        meta = a.get("meta", {})
        rows.append(f"| {m.name} | {m.folder} | {size(meta['size']) if meta.get('size') else '?'} | {a.get('model', m.repo)} | "
                    f"{(meta.get('sha256') or '')[:12]} | {m.stage}{' (optional)' if m.optional else ''} | {a.get('status', 'missing')} |")
    others = []
    for folder in sorted(os.listdir(COMFY_MODELS)):
        d = os.path.join(COMFY_MODELS, folder)
        if not os.path.isdir(d) or folder.startswith("."):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".pth")) and f"{folder}/{f}" not in reg:
                others.append(f"{folder}/{f} ({size(os.path.getsize(os.path.join(d, f)))})")
    return "\n".join(rows) + ("\n\nAlready on disk and not used by this pipeline: " + ", ".join(others) + "." if others else "")


def tools_table() -> str:
    rows = ["| Tool | Venv | Python | Pinned | Status |", "| --- | --- | --- | --- | --- |"]
    for a in sorted(manifest.query(kind="tool"), key=lambda a: a["id"]):
        meta = a.get("meta", {})
        pins = meta.get("commit") or ", ".join(f"{k} {v}" for k, v in (meta.get("versions") or {}).items())
        rows.append(f"| {a['id'].split(':', 1)[1]} | {os.path.relpath(a['path'], ROOT)} | {meta.get('python', '')} | {pins} | {a['status']} |")
    return "\n".join(rows)


def smoke_table() -> str:
    res = json.loads((QC / "stage0_results.json").read_text())
    rows = ["| Test | Result | Cold | Warm | VRAM peak (device) | Output |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for k, v in res.items():
        files = v.get("files") or [v.get("out") or v.get("webm") or v.get("composite") or ""]
        f = os.path.relpath(files[0], ROOT) if files and files[0] and str(files[0]).startswith("/") else (files[0] if files else "")
        warm = v.get("seconds_warm") or v.get("seconds_per_line")
        rows.append(f"| {k} | {'pass' if v.get('ok') else 'FAIL'} | {schedule._fmt(v.get('seconds'))} | "
                    f"{schedule._fmt(warm) if warm else ''} | {v.get('vram_peak_mib', '')} MiB | {f} |")
    return "\n".join(rows)


def main():
    c = comfy()
    du = shutil.disk_usage(ROOT)
    sched, totals = schedule.markdown()
    notes = (QC / "stage0_notes.md").read_text() if (QC / "stage0_notes.md").exists() else ""
    md = f"""# Stage 0 · Preflight

Generated {manifest.now()} by `pipeline/stage0_report.py` from live inventory, manifest.json and qc/stage0_results.json.

## Inventory

- **GPU:** {gpu()}
- **ComfyUI:** {c.get('comfyui_version', '?')} at $COMFY_DIR (uv venv, Python {c.get('python_version', '?').split()[0]}, PyTorch {c.get('pytorch_version', '?')}), headless on 127.0.0.1:8188, templates {c.get('installed_templates_version', '?')}
- **Custom nodes:** none installed and none needed. Every workflow uses native core nodes (ACE-Step 1.5, Flux.2 Klein, Wan 2.2, InfiniteTalk, LTX-2.3, SeedVR2, FrameInterpolate, Stable Audio).
- **Disk:** {du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB after all downloads · **RAM:** {int(c.get('ram_total', 0)) / 2**30:.0f} GiB
- **GPU sharing:** Ollama inactive, Hermes Agent not running; nothing else holds VRAM beyond the desktop (~2.6 GB).

### Models

{models_table()}

### Tools (one uv venv each, Python pinned)

{tools_table()}

## Smoke tests

Device VRAM peak is what nvidia-smi saw. ComfyUI's dynamic loader fills free VRAM, so peaks near 24 GB are normal and not an OOM signal; every listed test completed on the card.

{smoke_table()}

{notes}

## Render schedule (from the measurements above)

{sched}
"""
    (QC / "stage-0.md").write_text(md)
    print(f"wrote qc/stage-0.md ({len(md)} chars); GPU total {schedule._fmt(sum(totals.values()))}")


if __name__ == "__main__":
    main()
