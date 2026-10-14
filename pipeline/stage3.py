"""Stage 3: voices and sound effects.

  python -m pipeline.stage3 design          3 VoiceDesign candidates per voice from its VOICE line; pick one by metrics
  python -m pipeline.stage3 lines           5 clone takes per spoken line from the picked designs (needs Stage 2 gaps)
  python -m pipeline.stage3 sfx             Stable Audio candidates + synthesized bells
Picks are measurable: exact-word transcript, pitch range that fits the description, production quality.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline import lyrics, manifest, toolrun, voice_spec
from pipeline.manifest import ROOT

VOX = ROOT / "audio" / "voices"
QC = ROOT / "qc" / "stage3"
DESIGN_SEEDS = (31, 32, 33)
# Reference passages for designing each voice: the character's own lines first, padded with a neutral in-character
# sentence so the clone reference is ~6-10 s. These passages are references only and never appear in the video.
# voice id -> a short reference sentence for VoiceDesign. It is not used on screen.
DESIGN_TEXT = {}
# voice id -> (low_hz, high_hz) expected median pitch.
F0_RANGE = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def f0_median(path) -> float:
    import librosa
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)
    f0, voiced, _ = librosa.pyin(y.astype(np.float32), fmin=60, fmax=500, sr=sr, frame_length=2048)
    v = f0[voiced & ~np.isnan(f0)]
    return float(np.median(v)) if len(v) else 0.0


def word_exact(text: str, transcript: str) -> tuple[bool, float]:
    a, b = lyrics.norm_words(text), lyrics.norm_words(transcript)
    import jiwer
    wer = jiwer.wer(" ".join(a), " ".join(b)) if a else 0.0
    return a == b, round(wer, 3)


CLARIFY = {}  # voice id -> extra VoiceDesign sentence. Fill it per project when a design misses the brief.


def cmd_design(only: list[str] | None = None, seeds=DESIGN_SEEDS, clarify: bool = False):
    VOX.mkdir(parents=True, exist_ok=True)
    jobs, ids = [], []
    for voice, spec in voice_spec.VOICES.items():
        if only and voice not in only:
            continue
        instruct = spec["description"] + (CLARIFY.get(voice, "") if clarify else "")
        for seed in seeds:
            aid = f"voice:design:{voice}:{seed}"
            out = VOX / "designs" / f"{voice}_{seed}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            if (manifest.get(aid) or {}).get("status") in ("done", "picked", "scored"):
                continue
            jobs.append({"mode": "design", "instruct": instruct, "text": DESIGN_TEXT.get(voice) or "Speak one calm sentence, clearly.", "out": str(out), "seed": seed})
            ids.append(aid)
            manifest.upsert(aid, stage="3", kind="voice_design", path=rel(out), prompt=instruct, seed=seed,
                            model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", workflow="pipeline/tools/tts.py design",
                            status="running", meta={"voice": voice, "text": DESIGN_TEXT.get(voice) or "Speak one calm sentence, clearly."})
    if jobs:
        jf = VOX / "designs" / "jobs.json"
        jf.write_text(json.dumps(jobs, indent=1))
        toolrun.run("tts", "tts.py", ["batch", jf], name="stage3:design", log=ROOT / "qc" / "logs" / "stage3-tools.log")
        for aid in ids:
            manifest.upsert(aid, status="done")
    _score_designs()


def _score_designs():
    designs = [a for a in manifest.query(stage="3", kind="voice_design") if a["status"] in ("done", "scored", "picked")]
    paths = [str(ROOT / a["path"]) for a in designs]
    tr = toolrun.run("whisperx", "align.py", ["transcribe", VOX / "designs" / "transcripts", *paths], name="stage3:design-asr",
                     log=ROOT / "qc" / "logs" / "stage3-tools.log")["results"]
    ae = toolrun.run("aesthetics", "aesthetics.py", paths, name="stage3:design-aes",
                     log=ROOT / "qc" / "logs" / "stage3-tools.log")["scores"]
    by_voice = {}
    for a, p in zip(designs, paths):
        voice = a["meta"]["voice"]
        exact, wer = word_exact(a["meta"]["text"], tr[p]["text"])
        f0 = f0_median(p)
        lo, hi = F0_RANGE.get(voice, (80, 220))
        pitch_ok = lo <= f0 <= hi
        dur = sf.info(p).duration
        score = (1 - min(wer, 1)) * 0.5 + pitch_ok * 0.25 + ae[p]["PQ"] / 10 * 0.25
        manifest.upsert(a["id"], status="scored", score={"total": round(score, 3), "wer": wer, "exact": exact,
                        "f0_median": round(f0, 1), "pitch_ok": pitch_ok, "PQ": ae[p]["PQ"], "CE": ae[p]["CE"],
                        "duration": round(dur, 2)}, meta={"transcript": tr[p]["text"]})
        by_voice.setdefault(voice, []).append((score, a["id"]))
    for voice, cands in by_voice.items():
        best = max(cands)[1]
        manifest.upsert(best, status="picked")
        log(f"design {voice}: picked {best} from {[(round(s, 3), i.rsplit(':', 1)[1]) for s, i in sorted(cands, reverse=True)]}")


# ------------------------------------------------------------------ line takes
TAKES_PER_LINE = 5


def picked_design(voice: str) -> dict:
    return next(a for a in manifest.query(stage="3", kind="voice_design") if a["meta"]["voice"] == voice and a["status"] == "picked")


def cmd_lines():
    jobs, ids = [], []
    for line in voice_spec.LINES:
        d = picked_design(line["voice"])
        for k in range(TAKES_PER_LINE):
            aid = f"voice:line:{line['id']}:{k}"
            if (manifest.get(aid) or {}).get("status") in ("done", "scored", "picked", "rejected"):
                continue
            out = VOX / "lines" / line["id"] / f"take_{k}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            seed = 1000 + 17 * k + 7 * voice_spec.LINES.index(line)
            jobs.append({"mode": "clone", "ref": str(ROOT / d["path"]), "ref_text": d["meta"]["text"], "text": line["text"],
                         "out": str(out), "seed": seed})
            ids.append(aid)
            manifest.upsert(aid, stage="3", kind="voice_line", path=rel(out), prompt=line["text"], seed=seed,
                            model="Qwen/Qwen3-TTS-12Hz-1.7B-Base (clone of " + d["id"] + ")", workflow="pipeline/tools/tts.py clone",
                            status="running", meta={"line": line["id"], "voice": line["voice"], "section": line["section"]})
    if jobs:
        jf = VOX / "lines" / "jobs.json"
        jf.write_text(json.dumps(jobs, indent=1))
        toolrun.run("tts", "tts.py", ["batch", jf], name="stage3:lines", log=ROOT / "qc" / "logs" / "stage3-tools.log")
        for aid in ids:
            manifest.upsert(aid, status="done")
    _score_lines()


def _slot_limits() -> dict:
    """Max duration per line from the Stage 2 plan: time until the next placed line (or slot end)."""
    se = ROOT / "audio" / "timing" / "song_edit.json"
    if not se.exists():
        return {}
    sp = json.load(open(se))["spoken"]
    order = sorted(sp.items(), key=lambda kv: kv[1]["start"])
    lim = {}
    for (k, v), nxt in zip(order, order[1:] + [(None, None)]):
        same = nxt[1] is not None and nxt[1].get("group") == v.get("group")
        lim[k] = (nxt[1]["start"] - v["start"] - 0.15) if same else 99.0
    return lim


def trim_silence(src, out, thresh_db=-45.0, pad=0.04):
    x, sr = sf.read(src, always_2d=True)
    env = 20 * np.log10(np.abs(x).max(axis=1) + 1e-9)
    on = np.where(env > thresh_db)[0]
    a, b = (max(0, on[0] - int(pad * sr)), min(len(x), on[-1] + int(pad * sr))) if len(on) else (0, len(x))
    y = x[a:b].copy()
    f = min(len(y) // 4, int(0.005 * sr))
    if f > 0:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr, subtype="PCM_24")
    return out


def _score_lines():
    takes = [a for a in manifest.query(stage="3", kind="voice_line") if a["status"] in ("done", "scored", "picked", "rejected")]
    paths = [str(ROOT / a["path"]) for a in takes]
    tr = toolrun.run("whisperx", "align.py", ["transcribe", VOX / "lines" / "transcripts", *paths], name="stage3:lines-asr",
                     log=ROOT / "qc" / "logs" / "stage3-tools.log")["results"]
    ae = toolrun.run("aesthetics", "aesthetics.py", paths, name="stage3:lines-aes",
                     log=ROOT / "qc" / "logs" / "stage3-tools.log")["scores"]
    limits = _slot_limits()
    by_line = {}
    for a, p in zip(takes, paths):
        line = next(l for l in voice_spec.LINES if l["id"] == a["meta"]["line"])
        exact, wer = word_exact(line["text"], tr[p]["text"])
        x, sr = sf.read(p, always_2d=True)
        peak = float(np.abs(x).max())
        env = 20 * np.log10(np.abs(x).max(axis=1) + 1e-9)
        on = np.where(env > -45.0)[0]
        dur = float(((on[-1] - on[0]) / sr + 0.08) if len(on) else len(x) / sr)  # speech length after silence trim
        fits = bool(dur <= limits.get(line["id"], 99.0))
        ok = bool(exact and fits and peak < 0.999)
        score = (2.0 if exact else 1 - min(wer, 1)) + (0.5 if fits else 0) + ae[p]["PQ"] / 10 + (0.2 if peak < 0.999 else 0)
        manifest.upsert(a["id"], status="scored", score={"total": round(score, 3), "exact": exact, "wer": wer, "fits": fits,
                        "duration": round(dur, 3), "peak": round(peak, 3), "PQ": ae[p]["PQ"], "ok": ok},
                        meta={"transcript": tr[p]["text"], "words_json": rel(tr[p]["out"])})
        by_line.setdefault(line["id"], []).append((ok, score, a["id"], p, dur))
    picked = {}
    for lid, c in by_line.items():
        ok_c = [x for x in c if x[0]] or c  # if none passes every check, take the best and flag it
        best = max(ok_c, key=lambda x: x[1])
        for x in c:
            manifest.upsert(x[2], status="picked" if x[2] == best[2] else "rejected")
        picked[lid] = {"asset": best[2], "path": rel(best[3]), "duration": round(best[4], 3), "all_checks": best[0]}
    for lid, v in picked.items():  # trim leading/trailing silence so slots and box timings match the speech
        v["path_raw"], v["duration_raw"] = v["path"], v["duration"]
        out = VOX / "final" / f"{lid}.wav"
        v["path"], v["duration"] = rel(trim_silence(ROOT / v["path_raw"], out)), round(sf.info(out).duration, 3)
    (VOX / "picked.json").write_text(json.dumps(picked, indent=1))
    bad = [k for k, v in picked.items() if not v["all_checks"]]
    log(f"lines: picked {len(picked)}; not fully passing: {bad}")


# ------------------------------------------------------------------ sound effects
SFX_DIR = ROOT / "audio" / "sfx"
SFX = {  # name: (prompt, seconds). Empty until master.md names a sound. One dry knock is the smoke default.
    "wood_knock": ("a single soft knock on hollow wood, close, dry, no music", 2.0),
}
SFX_SEEDS = (71, 72, 73, 74)


def synth_bell(path, freq=1567.98, seconds=2.5, sr=48000, bright=1.0):
    """A bright bell 'ding': inharmonic partials with exponential decays (master.md: bells synthesized in code)."""
    t = np.arange(int(sr * seconds)) / sr
    partials = [(1.0, 1.0, 1.6), (2.76, 0.45 * bright, 0.9), (5.40, 0.25 * bright, 0.55), (8.93, 0.12 * bright, 0.35),
                (0.5, 0.18, 2.2)]
    y = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / d) for r, a, d in partials)
    y *= 1 - np.exp(-t / 0.002)  # 2 ms strike
    y = 0.6 * y / np.max(np.abs(y))
    sf.write(path, np.stack([y, y], axis=1), sr, subtype="PCM_24")
    return path


def cmd_sfx():
    from pipeline import comfy, wf
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    bells = {"bell_high": dict(freq=1567.98), "bell_mid": dict(freq=1174.66, seconds=3.0, bright=1.2),
             "bell_low": dict(freq=880.0, seconds=4.0, bright=0.8), "bell_long": dict(freq=1318.51, seconds=5.0, bright=0.6),
             "chime_short": dict(freq=2093.0, seconds=1.8, bright=1.4)}
    for name, kw in bells.items():
        p = synth_bell(SFX_DIR / f"{name}.wav", **kw)
        manifest.upsert(f"sfx:{name}", stage="3", kind="sfx", path=rel(p), prompt=f"synthesized bell {kw}", seed=None,
                        model="numpy additive synthesis", workflow="pipeline/stage3.py synth_bell", status="picked")
    for name, (prompt, secs) in SFX.items():
        for k, seed in enumerate(SFX_SEEDS):
            aid = f"sfx:{name}:{k}"
            if (manifest.get(aid) or {}).get("status") in ("done", "scored", "picked", "rejected"):
                continue
            g = wf.stable_audio(f"sfx_{name}_{k}", prompt, seed=seed, seconds=secs, negative="music, singing, speech",
                                prefix=f"__SLUG__/sfx/{name}")
            wfp = g.save(ROOT / "workflows" / "stage3")
            res = comfy.run(g.api(), name=aid, timeout=600)
            dest = comfy.download(comfy.output_files(res["outputs"])[0], SFX_DIR / name / f"cand_{k}.flac")
            manifest.upsert(aid, stage="3", kind="sfx", path=rel(dest), prompt=prompt, seed=seed, model="stable-audio-open-1.0",
                            workflow=rel(wfp), status="done", meta={"name": name})
    comfy.free()
    # pick: highest production quality among candidates without clipping
    cands = [a for a in manifest.query(stage="3", kind="sfx") if a.get("meta", {}).get("name") and a["status"] in ("done", "scored")]
    if cands:
        ae = toolrun.run("aesthetics", "aesthetics.py", [str(ROOT / a["path"]) for a in cands], name="stage3:sfx-aes",
                         log=ROOT / "qc" / "logs" / "stage3-tools.log")["scores"]
        best = {}
        for a in cands:
            sc = ae[str(ROOT / a["path"])]
            manifest.upsert(a["id"], status="scored", score=sc)
            n = a["meta"]["name"]
            if sc["PQ"] > best.get(n, (None, -1))[1]:
                best[n] = (a["id"], sc["PQ"])
        for n, (aid, _) in best.items():
            manifest.upsert(aid, status="picked")
        log(f"sfx picked: {best}")


def cmd_report():
    """qc/stage-3.md with the master.md self-checks: exact-word transcript; fits its gap; no clipping."""
    picked = json.load(open(VOX / "picked.json"))
    md = ["# Stage 3 · Voices and SFX", "", f"Generated {manifest.now()}.", "", "## Voices (designed once, reused by clone)", "",
          "| Voice | Design | Median pitch | Transcript exact | PQ | Instruction |", "| --- | --- | ---: | --- | ---: | --- |"]
    for a in manifest.query(stage="3", kind="voice_design"):
        if a["status"] == "picked":
            sc = a["score"]
            md.append(f"| {a['meta']['voice']} | {a['id']} | {sc['f0_median']} Hz | {sc['exact']} | {sc['PQ']} | {a['prompt'][:90]}... |")
    md += ["", "## Lines (5 takes each; picked take)", "", "| Line | Voice | Text | Exact words | Fits slot | Peak | Duration |",
           "| --- | --- | --- | --- | --- | ---: | ---: |"]
    allok = True
    for l in voice_spec.LINES:
        v = picked[l["id"]]
        a = manifest.get(v["asset"])
        sc = a["score"]
        ok = sc["exact"] and sc["fits"] and sc["peak"] < 0.999
        allok &= ok
        md.append(f"| {l['id']} | {l['voice']} | {l['text']} | {sc['exact']} | {sc['fits']} | {sc['peak']} | {v['duration']} s |")
    md += ["", f"Self-check (exact-word transcript, fits its gap, no clipping): **{'PASS' if allok else 'FAIL'}** for all {len(picked)} lines.", "",
           "## Sound effects", "", "A few bells are synthesized in code. "
           "Named effects in SFX are Stable Audio Open candidates (4 each), picked by Audiobox production quality:", ""]
    for a in manifest.query(stage="3", kind="sfx"):
        if a["status"] == "picked":
            md.append(f"- `{a['id']}`: {a['prompt']} ({a['path']})")
    (ROOT / "qc" / "stage-3.md").write_text("\n".join(md) + "\n")
    manifest.set_stage("3", status="done" if allok else "check_failed", finished=manifest.now())
    log(f"report: lines all checks {'PASS' if allok else 'FAIL'}")


def main(argv):
    cmd = argv[0] if argv else ""
    rest = argv[1:]
    if cmd == "design":
        manifest.set_stage("3", status="running", started=manifest.now())
        cmd_design()
    elif cmd == "design-retry":  # design-retry <voice> : 3 new seeds verbatim + 3 seeds with the clarification
        for a in manifest.query(stage="3", kind="voice_design"):
            if a["meta"]["voice"] == rest[0] and a["status"] == "picked":
                manifest.upsert(a["id"], status="scored")
        cmd_design([rest[0]], seeds=(34, 35, 36))
        cmd_design([rest[0]], seeds=(41, 42, 43), clarify=True)
    elif cmd == "lines":
        cmd_lines()
    elif cmd == "repick":  # re-score and re-pick against the Stage 2 slot limits (CPU + ASR)
        _score_lines()
    elif cmd == "report":
        cmd_report()
    elif cmd == "sfx":
        cmd_sfx()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
