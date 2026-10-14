"""Gates (CLAUDE.md: song_take=on, looks=off, final_cut=on): prepare a decision package, record the answer.

  python -m pipeline.gate song prepare [N]      top-N takes -> out/gate/song_take/*.mp3 + qc/stage1/gate.md
  python -m pipeline.gate song approve <asset>  record the user's pick (asset id, e.g. song:take:03)
  python -m pipeline.gate final prepare         480p preview under 30 MiB of out/final/__SLUG__.mp4 -> out/gate/final_cut/
  python -m pipeline.gate final approve         record the final-cut approval
"""
import subprocess
import sys

from pipeline import manifest
from pipeline.manifest import ROOT

GATE_DIR = ROOT / "out" / "gate"


def song_candidates(n: int = 3) -> list[dict]:
    pool = [a for a in manifest.query(stage="1") if a["status"] == "scored"
            and (a["kind"] == "song_take" or (a["kind"] == "song_repair" and a["meta"].get("kept")))]
    return sorted(pool, key=lambda a: -a["score"]["total"])[:n]


def prepare_song(n: int = 3) -> list[dict]:
    cands = song_candidates(n)
    out = GATE_DIR / "song_take"
    out.mkdir(parents=True, exist_ok=True)
    rows = ["| Pick | Asset | Variant | Score | Lyric recall | Weakest section | Breaks ok | Enjoyment (CE) | Quality (PQ) | BPM |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    files = []
    for i, a in enumerate(cands, 1):
        name = a["id"].replace(":", "_")
        mp3 = out / f"{i}_{name}.mp3"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ROOT / a["path"]), "-codec:a", "libmp3lame", "-q:a", "0", str(mp3)],
                       check=True)
        files.append(str(mp3))
        s = a["score"]
        weakest = min((x for x in a["meta"]["sections"] if x["recall"] is not None), key=lambda x: x["recall"])
        variant = a["meta"].get("variant", "repair of " + a["meta"].get("base", "?"))
        rows.append(f"| {i} | {a['id']} | {variant} | {s['total']:.3f} | {s['lyric_recall']:.2f} | {weakest['section']} "
                    f"({weakest['recall']:.2f}) | {s['breaks_ok']}/4 | {s['aesthetics']['CE']:.2f} | {s['aesthetics']['PQ']:.2f} | {s['bpm']} |")
    (ROOT / "qc" / "stage1" / "gate.md").write_text("# Song gate\n\n" + "\n".join(rows) + "\n\nFiles: " +
                                                    ", ".join(f"out/gate/song_take/{f.split('/')[-1]}" for f in files) + "\n")
    manifest.set_gate("song_take", status="pending", candidates=[a["id"] for a in cands], files=files, asked=manifest.now())
    return [dict(a, mp3=f) for a, f in zip(cands, files)]


def prepare_final(max_mib: float = 27.0) -> str:
    """A 480p preview under the 30 MiB phone-upload limit (two-pass, size-targeted); the 1080p master stays at
    out/final/__SLUG__.mp4."""
    import soundfile as sf
    src = ROOT / "out" / "final" / "__SLUG__.mp4"
    out = GATE_DIR / "final_cut" / "__SLUG___preview_480p.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = sf.info(str(ROOT / "out" / "master" / "__SLUG___master.wav")).duration
    audio_k = 96
    video_k = int(max_mib * 8 * 1024 * 1024 / dur / 1000) - audio_k - 8  # container overhead
    common = ["-vf", "scale=854:480:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-b:v", f"{video_k}k",
              "-pix_fmt", "yuv420p", "-passlogfile", str(out.parent / "x264pass")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *common, "-pass", "1", "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *common, "-pass", "2", "-c:a", "aac", "-b:a", f"{audio_k}k",
                    "-movflags", "+faststart", str(out)], check=True)
    for f in out.parent.glob("x264pass*"):
        f.unlink()
    manifest.upsert("gate:final_cut:preview", stage="10", kind="gate_preview", path=str(out.relative_to(ROOT)), prompt=None,
                    seed=None, model=None, workflow="pipeline/gate.py final prepare", score=None, status="done",
                    meta={"video_kbps": video_k, "audio_kbps": audio_k})
    manifest.set_gate("final_cut", status="pending", files=[str(src.relative_to(ROOT)), str(out.relative_to(ROOT))],
                      asked=manifest.now())
    return str(out)


def approve(gate: str, asset: str | None = None, note: str = "") -> None:
    fields = {"status": "approved", "answered": manifest.now(), "note": note}
    if asset:
        fields["asset"] = asset
        manifest.upsert(asset, status="approved")
    manifest.set_gate(gate, **fields)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:2] == ["song", "prepare"]:
        for c in prepare_song(int(a[2]) if len(a) > 2 else 3):
            print(c["id"], c["score"]["total"], c["mp3"])
    elif a[:2] == ["song", "approve"]:
        approve("song_take", a[2], " ".join(a[3:]))
    elif a[:2] == ["final", "prepare"]:
        print(prepare_final())
    elif a[:2] == ["final", "approve"]:
        approve("final_cut", None, " ".join(a[2:]))
    else:
        print(__doc__)
