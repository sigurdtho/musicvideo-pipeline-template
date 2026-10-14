"""manifest.json: the single source of pipeline state.

Every file the pipeline produces or depends on gets an asset entry with the fields CLAUDE.md
requires (stage, prompt, seed, model, workflow, score, status). Writes are guarded by a file
lock and replaced atomically, so background jobs and the orchestrator can update it safely.
"""
import contextlib
import json
import os
from datetime import datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.json"
DECISIONS = ROOT / "DECISIONS.md"
_LOCK = FileLock(str(ROOT / ".manifest.lock"), timeout=120)

REQUIRED = ("stage", "kind", "path", "prompt", "seed", "model", "workflow", "score", "status")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _empty() -> dict:
    return {
        "version": 1,
        "project": "__SLUG__",
        "created": now(),
        "updated": now(),
        "config": {},
        "stages": {},
        "gates": {
            "song_take": {"enabled": True, "status": "not_reached"},
            "looks": {"enabled": False, "status": "disabled"},
            "final_cut": {"enabled": True, "status": "not_reached"},
        },
        "assets": {},
    }


def load() -> dict:
    if not MANIFEST.exists():
        return _empty()
    return json.loads(MANIFEST.read_text())


def _write(m: dict) -> None:
    m["updated"] = now()
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    os.replace(tmp, MANIFEST)


@contextlib.contextmanager
def edit():
    """Lock, load, yield for mutation, then write atomically."""
    with _LOCK:
        m = load()
        yield m
        _write(m)


def upsert(asset_id: str, **fields) -> dict:
    with edit() as m:
        a = m["assets"].get(asset_id)
        if a is None:
            a = {"id": asset_id, **{k: None for k in REQUIRED}, "attempts": 0, "meta": {}, "created": now()}
            m["assets"][asset_id] = a
        meta = fields.pop("meta", None)
        a.update(fields)
        if meta:
            a["meta"].update(meta)
        a["updated"] = now()
        return dict(a)


def get(asset_id: str) -> dict | None:
    return load()["assets"].get(asset_id)


def query(stage: str | None = None, kind: str | None = None, status: str | None = None) -> list[dict]:
    out = []
    for a in load()["assets"].values():
        if stage is not None and a.get("stage") != stage:
            continue
        if kind is not None and a.get("kind") != kind:
            continue
        if status is not None and a.get("status") != status:
            continue
        out.append(a)
    return out


def set_stage(stage: str, **fields) -> None:
    with edit() as m:
        s = m["stages"].setdefault(stage, {})
        s.update(fields)
        s["updated"] = now()


def set_gate(gate: str, **fields) -> None:
    with edit() as m:
        g = m["gates"].setdefault(gate, {})
        g.update(fields)
        g["updated"] = now()


def set_config(**fields) -> None:
    with edit() as m:
        m["config"].update(fields)


def decision(title: str, body: str) -> None:
    """Append a dated entry to DECISIONS.md (creative deviations, tooling choices, ladder steps)."""
    with _LOCK:
        new = not DECISIONS.exists()
        with DECISIONS.open("a") as f:
            if new:
                f.write("# Decisions\n\nEvery creative deviation, tooling substitution and fallback-ladder step, newest last.\n")
            f.write(f"\n## {now()[:16].replace('T', ' ')} · {title}\n\n{body.strip()}\n")
