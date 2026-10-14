"""Download every registered model at a pinned revision, verify sha256, and record it in the manifest.

Idempotent: files already in place with the right size and hash are only recorded.
Usage: envs/pipeline/bin/python -m pipeline.fetch_models [--include-optional] [--keys a,b]
"""
import argparse
import hashlib
import os
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

from pipeline import manifest
from pipeline.models import COMFY_MODELS, MODELS, local_path

STAGING = f"{COMFY_MODELS}/.staging___SLUG__"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(api: HfApi, m, overwrite: bool = False) -> bool:
    dest = local_path(m)
    rev = api.model_info(m.repo).sha
    info = api.get_paths_info(m.repo, [m.path], revision=rev)[0]
    size, want = info.size, (info.lfs.sha256 if info.lfs else None)
    prior = manifest.get(f"model:{m.key}")
    if prior and prior.get("status") == "ready" and os.path.exists(dest) and os.path.getsize(dest) == size:
        log(f"skip {m.key}: already verified")
        return True
    if os.path.exists(dest) and os.path.getsize(dest) != size and not overwrite:
        # A file with this name already exists and differs: it belongs to the user, never replace it.
        log(f"CONFLICT {m.key}: {dest} exists with a different size; left untouched")
        manifest.upsert(f"model:{m.key}", stage="0", kind="model", path=dest, model=f"{m.repo}@{rev}",
                        status="conflict", meta={"repo_path": m.path, "size": size, "local_size": os.path.getsize(dest)})
        return False
    if not (os.path.exists(dest) and os.path.getsize(dest) == size):
        log(f"get  {m.key}: {m.repo}/{m.path} @ {rev[:10]} ({size / 1e9:.2f} GB)")
        t0 = time.time()
        p = hf_hub_download(m.repo, m.path, revision=rev, local_dir=STAGING)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(p, dest)
        dt = time.time() - t0
        log(f"done {m.key}: {dt:.0f}s, {size / dt / 1e6:.0f} MB/s")
    got = sha256(dest)
    ok = want is None or got == want
    manifest.upsert(
        f"model:{m.key}", stage="0", kind="model", path=dest, prompt=None, seed=None,
        model=f"{m.repo}@{rev}", workflow=None, score=None,
        status="ready" if ok else "failed",
        meta={"repo_path": m.path, "size": size, "sha256": got, "needed_by_stage": m.stage, "optional": m.optional},
    )
    log(f"{'ok  ' if ok else 'BAD '} {m.key}: sha256 {'matches' if ok else f'mismatch {got} != {want}'}")
    if not ok:
        os.rename(dest, dest + ".bad")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--keys", default="")
    ap.add_argument("--overwrite", action="store_true", help="replace differing files (only for files this pipeline owns)")
    args = ap.parse_args()
    keys = set(filter(None, args.keys.split(",")))
    todo = [m for m in sorted(MODELS, key=lambda m: m.priority)
            if (not keys or m.key in keys) and (args.include_optional or not m.optional or m.key in keys)]
    api = HfApi()
    failed = []
    for m in todo:
        for attempt in range(3):
            try:
                if fetch(api, m, overwrite=args.overwrite):
                    break
            except Exception as e:  # network hiccups: retry, then move on
                log(f"ERR  {m.key} attempt {attempt + 1}: {e!r}")
                time.sleep(10 * (attempt + 1))
        else:
            failed.append(m.key)
    log(f"FINISHED: {len(todo) - len(failed)}/{len(todo)} ok; failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
