"""Install each non-ComfyUI tool into its own uv venv (CLAUDE.md rule 6), pinned and idempotent.

Usage: envs/pipeline/bin/python -m pipeline.setup_tools [tool ...]
Each finished install writes envs/<tool>/.installed.json and a manifest entry (kind=tool).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline import manifest

ROOT = manifest.ROOT
ENVS = ROOT / "envs"
PY = "3.12.13"
TORCH = ["torch==2.14.0", "torchaudio"]  # same build ComfyUI uses; hardlinked from the uv cache

# Pinned source commits (git ls-remote HEAD on 2026-10-01)
ACESTEP_REPO, ACESTEP_SHA = "https://github.com/ace-step/ACE-Step-1.5", "ca1e85fe9430"
BEATTHIS_REPO, BEATTHIS_SHA = "https://github.com/CPJKU/beat_this", "b95c8ab0c58c"
SYNCNET_REPO, SYNCNET_SHA = "https://github.com/joonson/syncnet_python", "907c0b579c2e"
SYNCNET_WEIGHTS = {  # official files from the authors' VGG site (what the repo's download_model.sh fetches)
    "syncnet_v2.model": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    "sfd_face.pth": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, cwd=None, env=None):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True, shell=isinstance(cmd, str))


def venv(tool) -> Path:
    d = ENVS / tool
    if not (d / "bin" / "python").exists():
        sh(["uv", "venv", str(d), "--python", PY, "-q"])
    return d


def pip(d: Path, *pkgs, extra=()):
    sh(["uv", "pip", "install", "--python", str(d / "bin" / "python"), *extra, *pkgs])


def git_checkout(repo, sha, dest: Path):
    if not (dest / ".git").exists():
        sh(["git", "clone", "--quiet", repo, str(dest)])
    sh(["git", "-C", str(dest), "fetch", "--quiet", "origin"])
    sh(["git", "-C", str(dest), "checkout", "--quiet", sha])
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def freeze(d: Path, names):
    out = subprocess.run(["uv", "pip", "freeze", "--python", str(d / "bin" / "python")], capture_output=True, text=True).stdout
    return {l.split("==")[0].lower(): l.split("==")[1] for l in out.splitlines() if "==" in l and l.split("==")[0].lower() in names}


def install_separator():
    d = venv("separator")
    pip(d, "audio-separator[gpu]==0.47.0", *TORCH, "audioread", "librosa")
    return d, {"versions": freeze(d, {"audio-separator", "torch", "onnxruntime-gpu"})}


def install_whisperx():
    d = venv("whisperx")
    pip(d, "whisperx==3.8.6")  # pins its own torch 2.8 stack, hence its own venv
    return d, {"versions": freeze(d, {"whisperx", "torch", "ctranslate2", "faster-whisper", "transformers"})}


def install_beats():
    d = venv("beats")
    pip(d, f"git+{BEATTHIS_REPO}@{BEATTHIS_SHA}", *TORCH, "soxr", "soundfile", "numpy")
    return d, {"commit": BEATTHIS_SHA, "versions": freeze(d, {"beat-this", "torch"})}


def install_tts():
    d = venv("tts")
    pip(d, "qwen-tts==0.1.1", *TORCH, "soundfile", "numpy")
    return d, {"versions": freeze(d, {"qwen-tts", "torch", "transformers"})}


def install_aesthetics():
    d = venv("aesthetics")
    pip(d, "audiobox-aesthetics==0.0.4", *TORCH, "soundfile", "requests")  # requests is an undeclared dep
    return d, {"versions": freeze(d, {"audiobox-aesthetics", "torch"})}


def install_acestep():
    d = ENVS / "acestep"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "ACE-Step-1.5"
    head = git_checkout(ACESTEP_REPO, ACESTEP_SHA, src)
    # The repo pins its own Python (.python-version) and lockfile; respect both.
    sh(["uv", "sync", "--frozen"] if (src / "uv.lock").exists() else ["uv", "sync"], cwd=src,
       env={"UV_PROJECT_ENVIRONMENT": str(src / ".venv")})
    return d, {"commit": head}


def install_syncnet():
    d = venv("syncnet")
    src = d / "syncnet_python"
    head = git_checkout(SYNCNET_REPO, SYNCNET_SHA, src)
    pip(d, *TORCH, "torchvision", "opencv-python-headless", "python_speech_features", "scipy", "numpy", "scenedetect")
    wdir = src / "data"
    wdir.mkdir(exist_ok=True)
    for name, url in SYNCNET_WEIGHTS.items():
        dest = (wdir / name) if name.startswith("syncnet") else (src / "detectors" / "s3fd" / "weights" / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            sh(["curl", "-fsSL", "-o", str(dest), url])
    return d, {"commit": head}


TOOLS = {
    "separator": install_separator,
    "whisperx": install_whisperx,
    "beats": install_beats,
    "tts": install_tts,
    "aesthetics": install_aesthetics,
    "acestep": install_acestep,
    "syncnet": install_syncnet,
}


def main(argv):
    names = argv or list(TOOLS)
    failed = []
    for name in names:
        marker = ENVS / name / ".installed.json"
        if marker.exists():
            log(f"skip {name}: installed")
            continue
        try:
            t0 = time.time()
            d, info = TOOLS[name]()
            info.update({"python": PY, "installed": manifest.now(), "seconds": round(time.time() - t0)})
            marker.write_text(json.dumps(info, indent=1))
            manifest.upsert(f"tool:{name}", stage="0", kind="tool", path=str(d), prompt=None, seed=None,
                            model=None, workflow=None, score=None, status="ready", meta=info)
            log(f"ok   {name} in {info['seconds']}s: {info}")
        except Exception as e:
            failed.append(name)
            manifest.upsert(f"tool:{name}", stage="0", kind="tool", path=str(ENVS / name), status="failed", meta={"error": repr(e)})
            log(f"FAIL {name}: {e!r}")
    log(f"FINISHED tools: failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
