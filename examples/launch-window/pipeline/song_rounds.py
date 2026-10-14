"""New song take rounds aiming at stage-1 score >= 0.9 with a natural ending (the "r" series).

  python -m pipeline.song_rounds gen <R>       ~16 ComfyUI ACE-Step turbo takes for round R (settings in ROUNDS)
  python -m pipeline.song_rounds analyze <R>   stage1 analysis + last-20 s ending check -> qc/stage1/rounds/r<R>.json
  python -m pipeline.song_rounds listen [r5]   top 3 of rounds 1-4 (or of r5+): -14 LUFS / -1.3 dBTP -> lw_r{round}_take{NN}.mp3 (+ $REVIEW_DIR)
  python -m pipeline.song_rounds table <R>...  markdown per-take rows for the report
Job lyrics = master.md's generator lyrics + the instrumental-outro tag (song_regen.lyrics(); DECISIONS.md 2026-10-12).
From r5 (qc/brief_r5.md): a shorter, upbeat caption per BPM, the same lyric lines under structure tags that ask for a kick
in every drop and end on a generated instrumental fade (lyrics_r5()), ~158 s, and three more checks (kick, 6-10 s fade
ending, 150-170 s). GPU jobs hold $GPU_LOCK like the other projects' jobs.
Takes are new ids song:take:r<R>:NN; nothing existing is overwritten and the song gate is not touched.
"""
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from pipeline import comfy, gpu, lyrics, manifest, master, song_regen, song_spec, stage1, wf
from pipeline.manifest import ROOT
from pipeline.song_regen import NOTES, ending, features, gpu_busy_elsewhere, log, rel, wait_gpu

N = 16
DURATIONS = (205.0, 207.0, 208.0, 210.0)
QC = ROOT / "qc" / "stage1" / "rounds"
BAR = 0.9
MIN_TAIL = 4.0  # s of music after the last word
OUTRO_CAPTION = ", clear lead vocal, ends with a soft piano and synth outro fading out on a held final chord"
# R: caption, LM temperature, LM cfg, seed base. Take 04 / 04b = song_spec.CAPTION, T 0.85, cfg 2.0.
# Rounds 2+ are set from what scored best in the previous round (qc/rounds_report.md). A tuple = (takes 00-07, takes 08-15),
# an A/B inside the round; each half still cycles all four durations.
ROUNDS = {
    1: {"caption": song_spec.CAPTION, "temperature": 0.85, "lm_cfg": 2.0, "seed0": 210001},
    # r1: 5/16 dropped or rewrote the last line, Pre-Chorus 3 weak in 7/16 -> test lower LM temperature vs higher LM cfg
    2: {"caption": song_spec.CAPTION, "temperature": (0.70, 0.85), "lm_cfg": (2.0, 2.5), "seed0": 220001},
    # r2: no hit, LM knobs barely matter (T 0.70 a little better on recall / clean tails); endings fail on hard stops and
    # wordless vocals -> keep T 0.70 / cfg 2.0 and A/B the caption: + an ending phrase, and that without "vocal chops"
    3: {"caption": (song_spec.CAPTION + OUTRO_CAPTION, song_spec.CAPTION.replace(", vocal chops", "") + OUTRO_CAPTION),
        "temperature": 0.70, "lm_cfg": 2.0, "seed0": 230001},
    # r3: the ending phrase drops the sung Outro line (9/16 missing); without "vocal chops" no hard stops, recall up
    # -> no chops everywhere; without the ending phrase vs r3's B caption again (new seeds)
    4: {"caption": (song_spec.CAPTION.replace(", vocal chops", ""), song_spec.CAPTION.replace(", vocal chops", "") + OUTRO_CAPTION),
        "temperature": 0.70, "lm_cfg": 2.0, "seed0": 240001},
}
# r5+ (qc/brief_r5.md, the user: shorter, more upbeat, take 04's oomph, a kick from Drop 1 on, a generated fade ending).
# song_spec.GENRE_STYLE stays as it was; this caption is a round setting. {bpm} is the take's BPM. The brief's reference
# acts are described, not named (CLAUDE.md rule 8: no real person's name).
CAPTION_R5 = ("energetic festival progressive house, Scandinavian festival anthem, {bpm} bpm, D major, driving, upbeat, "
              "euphoric, hard-hitting four-on-the-floor kick drum in every drop starting with the first drop, punchy short "
              "snare roll and riser builds, big bright supersaw drops, side-chained synth bass, confident bright powerful male "
              "tenor lead vocal, strong chest voice, anthemic tight delivery, singalong hook, stadium energy, not whiny, "
              "not breathy, not soft")
# Job structure tags per master.md section (lyrics.sections() names); the lines under them are master.md's, verbatim.
TAGS_R5 = {"Intro": "Intro - short, 4 bars, kick and bright synth stabs", "Verse 1": "Verse 1 - driving bass, light beat",
           "Pre-Chorus 1": "Pre-Chorus - short snare roll build", "Chorus 1": "Chorus - Drop 1, full four-on-the-floor kick",
           "Verse 2": "Verse 2 - driving bass, light beat", "Pre-Chorus 2": "Pre-Chorus - short snare roll build",
           "Chorus 2": "Chorus - Drop 2, full four-on-the-floor kick", "Bridge": "Bridge - short breakdown",
           "Pre-Chorus 3": "Pre-Chorus - short riser build", "Chorus 3": "Chorus - final drop, full four-on-the-floor kick",
           "Outro": "Outro"}
OUTRO_TAG_R5 = "[Outro - instrumental, fade out]"
ROUNDS[5] = {"caption": CAPTION_R5, "lyrics": "r5", "duration": 158.0, "bpm": [128, 130, 132], "temperature": (0.70, 0.85),
             "lm_cfg": [2.0, 2.5], "seed0": 250001}
# r5: 0 hits. Stage-1 max 0.884 (recall: the sung Intro and Pre-Chorus 3 drop out); the last line ends ~123-146 s and
# 10-31 s of instrumental follow, often cut by the file end (6-10 s wanted); Drop 1 kick <= Verse 1 in 9/16.
# r6: an instrumental 8-bar intro before the sung Intro lines, verses without kick, a clear-diction caption, a 4-bar
# outro fade tag; A/B (takes 00-07 / 08-15): plus an 8-bar instrumental Drop 1 after Chorus 1 (pushes the lines later).
CAPTION_R6 = CAPTION_R5.replace("anthemic tight delivery,", "anthemic tight delivery, crisp clear diction, every word sung "
                                "clearly,")
TAGS_R6 = {**TAGS_R5, "Intro": "Intro - vocal, clear", "Verse 1": "Verse 1 - stripped back, no kick drum, bass and plucks",
           "Verse 2": "Verse 2 - stripped back, no kick drum, bass and plucks",
           "Pre-Chorus 3": "Pre-Chorus - sung clearly over a short riser"}
PRE_R6 = {"Intro": "[Intro - 8 bars instrumental, four-on-the-floor kick and bright synth stabs]"}
DROP_R6 = {"Chorus 1": "[Drop - 8 bars instrumental, full four-on-the-floor kick, supersaw lead]"}
OUTRO_TAG_R6 = "[Outro - short instrumental, 4 bars, fade out]"
ROUNDS[6] = {"caption": CAPTION_R6, "lyrics": ("r6a", "r6b"), "duration": (160.0, 166.0), "bpm": [128, 130, 132],
             "temperature": 0.85, "lm_cfg": [2.0, 2.5], "seed0": 260001}
# r6: 0 hits, stage-1 max 0.864 (median 0.70, below r5: Pre-Chorus 3 / Chorus 3 / Outro drop out, more with the 4-bar
# outro tag); kick now passes 12/16. The music ends 3-5 s before the file end in every take, so the instrumental after
# the last line = duration - ~4 s - the last line (r6 B at 166 s: last line 145-150 s, 13-15 s tails; the separate
# outro tag always gets padded out). r7: r6 B's structure (instrumental intro + Drop 1, no-kick verses), r5's
# Pre-Chorus 3 tag; A/B on the ending: A sings the last line under one outro tag that asks for a short fade, no separate
# instrumental tag; B keeps r5's separate fade tag at 160 s.
TAGS_R7 = {**TAGS_R6, "Pre-Chorus 3": TAGS_R5["Pre-Chorus 3"]}
TAGS_R7A = {**TAGS_R7, "Outro": "Outro - final line, then a short instrumental fade out"}
ROUNDS[7] = {"caption": CAPTION_R6, "lyrics": ("r7a", "r7b"), "duration": 160.0, "bpm": [128, 130, 132],
             "temperature": [0.70, 0.85], "lm_cfg": 2.5, "seed0": 270001}
# r7: 0 hits, stage-1 max 0.892. A (one outro tag): the last line lands 1-4 s before the music end (no instrumental tail);
# B (separate fade tag): 11-18 s tails (best 10.9 / 11.0). r8: the one outro tag with a bar count, A/B 4 vs 8 bars.
TAGS_R8A = {**TAGS_R7, "Outro": "Outro - final line, then 4 bars instrumental fade out"}
TAGS_R8B = {**TAGS_R7, "Outro": "Outro - final line, then 8 bars instrumental fade out"}
ROUNDS[8] = {"caption": CAPTION_R6, "lyrics": ("r8a", "r8b"), "duration": 160.0, "bpm": [128, 130, 132],
             "temperature": [0.70, 0.85], "lm_cfg": 2.5, "seed0": 280001}
# r8: 0 hits; first stage-1 >= 0.9 (r8:12, 0.906) but tails 1-5 s with either bar count (4 vs 8 bars hardly differ).
# r9 (the last round): ~4 s more tail. A: the one outro tag with 16 bars; B: r8 B's tag + a short separate fade tag.
TAGS_R9A = {**TAGS_R7, "Outro": "Outro - final line, then 16 bars instrumental fade out"}
ROUNDS[9] = {"caption": CAPTION_R6, "lyrics": ("r9a", "r9b"), "duration": 160.0, "bpm": [128, 130, 132],
             "temperature": [0.70, 0.85], "lm_cfg": 2.5, "seed0": 290001}
BATTLE_LOCK = Path(os.path.expanduser(os.environ.get("GPU_LOCK", "~/.cache/mvp/gpu.lock")))
KICK_BAND = (40.0, 120.0)  # Hz
KICK_MIN_RATIO = 1.15      # Drop 1 kick punch / Verse 1 (take 04, a full kick in both: 1.17)
KICK_MIN_OF_BEST = 0.6     # Drop 1 / the strongest drop (r3:14, "almost no kick" in Drop 1: 0.15; take 04: 1.0)
FADE_S = (6.0, 10.0)       # instrumental after the last line
LENGTH_S = (150.0, 170.0)  # music end


def take_id(r: int, i: int) -> str:
    return f"song:take:r{r}:{i:02d}"


def take_path(r: int, i: int) -> Path:
    return ROOT / "audio" / "takes" / f"r{r}" / f"take_r{r}_{i:02d}.flac"


def settings(r: int, i: int) -> dict:
    """A tuple = an A/B in blocks (takes 00-07, 08-15); a list = cycled per take."""
    s = ROUNDS[r]
    pick = lambda v: v[i * len(v) // N] if isinstance(v, tuple) else v[i % len(v)] if isinstance(v, list) else v
    bpm = pick(s.get("bpm", song_spec.BPM))
    return {"seed": s["seed0"] + 7919 * i, "duration": pick(s.get("duration", list(DURATIONS))), "temperature": pick(s["temperature"]),
            "lm_cfg": pick(s["lm_cfg"]), "caption": pick(s["caption"]).format(bpm=f"{bpm:g}"), "bpm": bpm,
            "lyrics": pick(s.get("lyrics", "outro"))}


def lyrics_r5(tags: dict = TAGS_R5, outro: str = OUTRO_TAG_R5, pre: dict = {}, post: dict = {}) -> str:
    """master.md's generator lyric lines, verbatim, under the round's structure tags, then the instrumental fade tag.
    pre / post: lineless instrumental tags before / after a section."""
    secs = lyrics.sections()
    out = "\n\n".join((pre[s.tag] + "\n\n" if s.tag in pre else "") + f"[{tags[s.tag]}]\n" + "\n".join(s.lines)
                       + ("\n\n" + post[s.tag] if s.tag in post else "") for s in secs) + (f"\n\n{outro}\n" if outro else "\n")
    assert [l for s in lyrics.sections(out) for l in s.lines] == [l for s in secs for l in s.lines]
    return out


JOB_LYRICS = {"r5": lyrics_r5,
              "r6a": lambda: lyrics_r5(TAGS_R6, OUTRO_TAG_R6, PRE_R6),
              "r6b": lambda: lyrics_r5(TAGS_R6, OUTRO_TAG_R6, PRE_R6, DROP_R6),
              "r7a": lambda: lyrics_r5(TAGS_R7A, "", PRE_R6, DROP_R6),
              "r7b": lambda: lyrics_r5(TAGS_R7, OUTRO_TAG_R5, PRE_R6, DROP_R6),
              "r8a": lambda: lyrics_r5(TAGS_R8A, "", PRE_R6, DROP_R6),
              "r8b": lambda: lyrics_r5(TAGS_R8B, "", PRE_R6, DROP_R6),
              "r9a": lambda: lyrics_r5(TAGS_R9A, "", PRE_R6, DROP_R6),
              "r9b": lambda: lyrics_r5(TAGS_R8B, "[Fade out - 4 bars instrumental]", PRE_R6, DROP_R6)}


def job_lyrics(kind: str) -> str:
    return JOB_LYRICS[kind]() if kind in JOB_LYRICS else song_regen.lyrics()


@contextlib.contextmanager
def battle_lock(purpose: str):
    """The machine-wide GPU lock (flock, exclusive) that wot-mat's and the wb-test jobs take. Waits, never breaks it.
    Also waits out other projects' units / ComfyUI queue (wait_gpu) and resident Ollama/Hermes VRAM (gpu.check_free)."""
    f = open(BATTLE_LOCK, "a+")
    waited = 0
    while True:
        wait_gpu()
        try:
            gpu.check_free()
        except gpu.GpuBlocked as e:
            if waited % 600 == 0:
                log(f"waiting for VRAM: {e}")
            time.sleep(60)
            waited += 60
            continue
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if waited % 600 == 0:
                log(f"waiting for {BATTLE_LOCK}: {open(BATTLE_LOCK).read().strip()}")
            time.sleep(60)
            waited += 60
            continue
        if not gpu_busy_elsewhere():
            break
        fcntl.flock(f, fcntl.LOCK_UN)
    f.seek(0)
    f.truncate()
    f.write(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "purpose": f"launch-window {purpose}",
                        "acquired": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    f.flush()
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


# ------------------------------------------------------------------ generation
def cmd_gen(r: int):
    log(f"round {r}: checking the GPU before the round")
    wait_gpu()
    for i in range(N):
        aid = take_id(r, i)
        a = manifest.get(aid)
        if a and a.get("status") not in (None, "running", "failed"):
            continue
        s = settings(r, i)
        lyr = job_lyrics(s["lyrics"])
        wait_gpu()
        g = wf.ace15_song(f"song_take_r{r}_{i:02d}", s["caption"], lyr, seed=s["seed"], duration=s["duration"], bpm=s["bpm"],
                          keyscale=song_spec.KEYSCALE, variant="turbo", lm_cfg=s["lm_cfg"], temperature=s["temperature"],
                          prefix=f"launch-window/song/take_r{r}_{i:02d}")
        wfp = g.save(ROOT / "workflows" / "stage1")
        manifest.upsert(aid, stage="1", kind="song_take", prompt=s["caption"], seed=s["seed"], model=stage1.MODEL_NAMES["turbo"],
                        workflow=rel(wfp), status="running", attempts=(a or {}).get("attempts", 0) + 1,
                        meta={"variant": "turbo", "bpm": s["bpm"], "key": song_spec.KEYSCALE, "duration": s["duration"],
                              "series": f"r{r}", "round": r, "lm_temperature": s["temperature"], "lm_cfg": s["lm_cfg"],
                              **({"lyrics_job": lyr, "lyrics_kind": s["lyrics"]} if s["lyrics"] in JOB_LYRICS else {"lyrics_extra": song_regen.OUTRO_TAG.strip()})})
        try:
            with battle_lock(aid):
                res = comfy.run(g.api(), name=aid, timeout=1800)
            dest = comfy.download(comfy.output_files(res["outputs"])[0], take_path(r, i))
            manifest.upsert(aid, path=rel(dest), status="done", meta={"render_seconds": res["seconds"]})
            log(f"{aid}: seed {s['seed']} {s['duration']:.0f}s {s['bpm']} BPM T={s['temperature']} cfg={s['lm_cfg']}: {res['seconds']}s")
        except Exception as e:
            manifest.upsert(aid, status="failed", meta={"error": repr(e)[:2000]})
            log(f"{aid}: FAILED {e!r}")
    comfy.free()


# ------------------------------------------------------------------ kick check (r5+)
def _kick_env(path, sr: int = 22050, hop: int = 128):
    """40-120 Hz magnitude envelope, ~5.8 ms frames."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return np.sqrt((S[(f >= KICK_BAND[0]) & (f <= KICK_BAND[1])] ** 2).sum(0)), sr / hop


def _punch(env, fps, beats, t0, t1):
    """Kick punch in [t0, t1): per beat, the low-band peak on the beat (-30..+60 ms) minus the median between beats,
    averaged per bar; the 75th percentile over bars. A sustained or side-chained bass scores ~0, a kick scores high."""
    bs = [b for b in beats if t0 <= b < t1]
    p = [env[int((b0 - 0.03) * fps):int((b0 + 0.06) * fps)].max() - np.median(env[int((b0 + 0.12) * fps):int((b1 - 0.05) * fps)])
         for b0, b1 in zip(bs, bs[1:]) if b1 - b0 > 0.2]
    bars = [float(np.mean(p[j:j + 4])) for j in range(0, len(p) - 3, 4)]
    return round(float(np.percentile(bars, 75)), 2) if bars else None


def kick_check(a: dict) -> dict:
    """Drop 1 (end of Pre-Chorus 1 to the start of Verse 2) vs Verse 1, and vs the strongest of the three drops."""
    secs = {s["section"]: s for s in a["meta"]["sections"]}
    beats = json.load(open(ROOT / a["meta"]["beats"]))["beats"]
    env, fps = _kick_env(ROOT / a["path"])
    st = lambda k: secs.get(k, {}).get("start")
    en = lambda k: secs.get(k, {}).get("end")

    def drop(k, nxt):
        c0, c1 = st(f"Chorus {k}"), en(f"Chorus {k}")
        if c0 is None:
            return None
        t0 = en(f"Pre-Chorus {k}") if en(f"Pre-Chorus {k}") and en(f"Pre-Chorus {k}") < c0 else c0 - 8
        return _punch(env, fps, beats, t0 + 1, st(nxt) if nxt and st(nxt) and st(nxt) > c0 else c1 + 4)

    verse = _punch(env, fps, beats, st("Verse 1"), st("Pre-Chorus 1") or en("Verse 1")) if st("Verse 1") is not None else None
    drops = [drop(1, "Verse 2"), drop(2, "Bridge"), drop(3, "Outro")]
    d1, best = drops[0], max((d for d in drops if d is not None), default=None)
    ratio = round(d1 / max(verse, 1.0), 2) if d1 is not None and verse is not None else None
    of_best = round(d1 / best, 2) if d1 is not None and best else None
    ok = ratio is not None and of_best is not None and ratio >= KICK_MIN_RATIO and of_best >= KICK_MIN_OF_BEST
    why = ("sections not found" if ratio is None else
           ", ".join(m for m, bad in (("Drop 1 no more kick than Verse 1", ratio < KICK_MIN_RATIO),
                                      ("weak Drop 1 kick vs the later drops", (of_best or 0) < KICK_MIN_OF_BEST)) if bad))
    return {"verse1_punch": verse, "drop_punch": drops, "kick_ratio": ratio, "drop1_of_best": of_best, "kick_ok": ok, "kick_fails": why}


def checks_r5(a: dict, e: dict) -> dict:
    """The r5 bars on top of stage 1: kick, a 6-10 s instrumental fade/ring-out (no hard stop), length 150-170 s."""
    k = kick_check(a)
    tail = e["tail_after_line_s"]
    fade_ok = (not e["hard_stop"] and e["last_line"] and e["clean_tail"] and e["kind"] in ("fade", "ring-out")
               and tail is not None and FADE_S[0] <= tail <= FADE_S[1])
    length = e["music_end_s"]
    return _plain({**k, "fade_ok": fade_ok, "tail_s": tail, "length_s": length,
                   "length_ok": LENGTH_S[0] <= length <= LENGTH_S[1]})


# ------------------------------------------------------------------ ending check
TRIADS = {"major": (0, 4, 7), "minor": (0, 3, 7)}


def tonic_triad(key: str) -> set[str]:
    n, mode = key.split()
    i = NOTES.index(n)
    return {NOTES[(i + d) % 12] for d in TRIADS[mode]}


def _plain(x):
    """numpy scalars (np.bool_ from the comparisons above) -> plain Python, so the manifest can store them."""
    return json.loads(json.dumps(x, default=lambda o: o.item()))


def ending_check(a: dict) -> dict:
    """The three ending rules on the last 20 s, on top of song_regen.ending's measurements."""
    words = json.load(open(ROOT / a["meta"]["transcript"]))["words"]
    e = ending(ROOT / a["path"], ROOT / a["meta"]["vocals"], words)
    secs = {s["section"]: s for s in a["meta"]["sections"]}
    out_sec = secs["Outro"]
    last_line = out_sec["recall"] is not None and out_sec["recall"] >= 0.75 and out_sec["end"] is not None \
        and out_sec["start"] >= max(s["end"] or 0 for k, s in secs.items() if k != "Outro") - 1
    line_end = out_sec["end"] if last_line else None
    # 1. no hard stop: no cliff in the last 3 s, the music is not still loud where the file ends, and it does not stop
    #    right under the last word (take 04: 1.3 s after "months", which the user heard as a dead stop)
    hard = (e["kind"] == "hard stop" or (e["music_end_s"] >= e["file_end_s"] - 0.3 and e["rms_curve_db_re_p95"][-1] > -25)
            or (line_end is not None and e["music_end_s"] - line_end < MIN_TAIL))
    # 2. nothing sung after the last line: no transcript words, and the vocal stem quiet (0.5 s windows) from 2 s
    #    after "months" (its held note) to the end
    music_end = e["music_end_s"]
    after = line_end if line_end is not None else music_end - 20
    extra_words = [w["word"] for w in words if w.get("start") is not None and after + 0.5 < w["start"] < music_end]
    extra_sung = [float(t) for t in e["sung_windows_s"] if t >= after + 2.0]
    clean_tail = not extra_words and len(extra_sung) < 2
    # 3. resolves on a sustained chord: tonic triad of D major (the brief) or of the take's own key, held stable
    key = features(ROOT / a["path"])["key"]
    tonics = tonic_triad(song_spec.KEYSCALE) | tonic_triad(key)
    top = set(e["final_chord_top3"])
    resolves = any(len(tonic_triad(k) & top) >= 2 for k in (song_spec.KEYSCALE, key))
    sustained = e["chord_stability"] >= 0.8 and e["decay_s"] >= 1.5
    ok = (not hard) and last_line and clean_tail and resolves and sustained
    why = [m for m, bad in (("hard stop", hard), ("last line missing", not last_line),
                            (f"extra vocals {extra_words or extra_sung[:6]}", not clean_tail),
                            ("does not resolve", not resolves), ("not sustained", not sustained)) if bad]
    return _plain({**e, "key_est": key, "tonic_notes": sorted(tonics), "hard_stop": hard, "last_line": last_line,
            "last_line_end_s": line_end, "tail_after_line_s": round(music_end - line_end, 1) if line_end else None,
            "extra_words": extra_words, "extra_sung_s": extra_sung, "clean_tail": clean_tail, "resolves": resolves,
            "sustained": sustained, "ending_ok": ok, "fails": why})


def describe(e: dict) -> str:
    tail = f"{e['tail_after_line_s']} s instrumental after the last line" if e["tail_after_line_s"] is not None else "last line missing"
    return (f"{e['kind']} (decay {e['decay_s']} s, slope {e['slope_db_per_s_last10']} dB/s), {tail}, "
            f"ends on {'-'.join(e['final_chord_top3'])} (stab {e['chord_stability']})"
            + ("" if e["ending_ok"] else f"; FAILS: {', '.join(e['fails'])}"))


def cmd_analyze(r: int):
    ids = [take_id(r, i) for i in range(N)]
    todo = [a for a in map(manifest.get, ids) if a and a["status"] == "done"]
    if todo:
        log(f"round {r}: checking the GPU before analysis")
        with battle_lock(f"song_rounds analyze {r}"):
            stage1._analyze_assets(todo)
    res = {}
    for a in (a for a in map(manifest.get, ids) if a and a["status"] == "scored"):
        e = ending_check(a)
        s = a["score"]
        c = checks_r5(a, e) if r >= 5 else None
        hit = s["total"] >= BAR and (c["kick_ok"] and c["fade_ok"] and c["length_ok"] if c else e["ending_ok"])
        res[a["id"]] = {"round": r, "seed": a["seed"], "duration": a["meta"]["duration"], "temperature": a["meta"]["lm_temperature"],
                        "lm_cfg": a["meta"]["lm_cfg"], "stage1_total": s["total"], "lyric_recall": s["lyric_recall"],
                        "min_section_recall": s["min_section_recall"], "weak_sections": a["meta"]["weak_sections"],
                        "section_recall": {x["section"]: x["recall"] for x in a["meta"]["sections"]},
                        "bpm": s["bpm"], "aesthetics": s["aesthetics"], "ending": e, "ending_text": describe(e), "hit": hit,
                        **({"bpm_asked": a["meta"]["bpm"], "r5": c} if c else {})}
        manifest.upsert(a["id"], meta={"ending": e, "ending_ok": e["ending_ok"], "hit": hit, **({"r5_checks": c} if c else {})})
        log(f"{a['id']}: stage1 {s['total']:.3f} recall {s['lyric_recall']:.2f} weak {a['meta']['weak_sections']} "
            f"ending {'OK' if e['ending_ok'] else 'no'} {describe(e)}"
            + (f" | kick {c['kick_ratio']} (of best {c['drop1_of_best']}) {'OK' if c['kick_ok'] else 'no'}, fade "
               f"{'OK' if c['fade_ok'] else 'no'} ({c['tail_s']} s), length {c['length_s']} s {'OK' if c['length_ok'] else 'no'}"
               f"{' HIT' if hit else ''}" if c else ""))
    QC.mkdir(parents=True, exist_ok=True)
    (QC / f"r{r}.json").write_text(json.dumps({"round": r, "settings": {k: v for k, v in ROUNDS[r].items()}, "takes": res}, indent=1))
    manifest.upsert(f"song:take:r{r}:analysis", stage="1", kind="qc", path=rel(QC / f"r{r}.json"), prompt=None, seed=None,
                    model="stage1 tools + librosa", workflow=f"pipeline/song_rounds.py analyze {r}", score=None, status="done")
    summary(r, res)
    return res


def summary(r, res):
    hits = [k for k, v in res.items() if v["hit"]]
    ok = [v for v in res.values() if v["ending"]["ending_ok"]]
    tot = sorted(v["stage1_total"] for v in res.values())
    weak = {}
    for v in res.values():
        for w in v["weak_sections"]:
            weak[w] = weak.get(w, 0) + 1
    log(f"round {r}: {len(res)} scored, {len(ok)} good endings, hits {hits}; stage1 max {tot[-1] if tot else None} "
        f"median {tot[len(tot) // 2] if tot else None}; weak sections {weak}")


# ------------------------------------------------------------------ overall top 3 -> listen
def all_results(r5: bool = False) -> dict:
    """Rounds 1-4, or the r5 series (rounds 5+)."""
    res = {}
    for f in sorted(QC.glob("r*.json")):
        d = json.load(open(f))
        if (d["round"] >= 5) == r5:
            res.update(d["takes"])
    return res


def passed(v: dict) -> int:
    c = v.get("r5")
    return (v["stage1_total"] >= BAR) + (sum((c["kick_ok"], c["fade_ok"], c["length_ok"])) if c else v["ending"]["ending_ok"])


def ranked(res: dict) -> list[str]:
    """Hits first, then most bars passed, by stage-1 score, ties by the natural-ending score."""
    return sorted(res, key=lambda k: (not res[k]["hit"], -passed(res[k]), -res[k]["stage1_total"], -res[k]["ending"]["natural"]))


def cmd_listen(n: int = 3, r5: bool = False):
    res = all_results(r5)
    top = ranked(res)[:n]
    tracks_f = song_regen.LISTEN[0] / "tracks.json"
    tracks = json.load(open(tracks_f))
    rounds = {v["round"] for v in res.values()}
    # replace this series' earlier picks only (the r1-r4 picks and the r3:14 oomph files stay)
    tracks["tracks"] = [t for t in tracks["tracks"] if not any(t["file"].startswith(f"lw_r{q}_take") for q in rounds)]
    for i, aid in enumerate(top, 1):
        a, v = manifest.get(aid), res[aid]
        r = v["round"]
        name = f"lw_r{r}_take{aid.split(':')[-1]}"  # the take number in its round (r1:10 -> lw_r1_take10); rank in the label
        x, sr = sf.read(ROOT / a["path"], always_2d=True, dtype="float64")
        y = master.to_target(x, sr, lufs=-14.0, ceiling_db=-1.3)  # stage 4 / smoke-mix target, no trim or fade
        flac = ROOT / "audio" / "takes" / "rounds_norm" / f"{name}.flac"
        flac.parent.mkdir(parents=True, exist_ok=True)
        sf.write(flac, y, sr, subtype="PCM_24")
        m = master.measure(flac)
        # own ids per series, so the r5 series does not overwrite the r1-r4 picks' entries
        manifest.upsert(f"song:take:{'r5series' if r5 else 'rounds'}:norm:{i}", stage="1", kind="song_norm", path=rel(flac), prompt=None, seed=a["seed"],
                        model="pipeline/master.py to_target", workflow="pipeline/song_rounds.py listen", status="done",
                        score={"lufs": m["lufs"], "true_peak": m["true_peak_dbtp"], "lra": m["lra"]}, meta={"source": aid, "rank": i})
        for d in song_regen.LISTEN:
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(flac), "-codec:a", "libmp3lame", "-b:a", "320k",
                            str(d / f"{name}.mp3")], check=True)
        manifest.upsert(f"listen:{name}", stage="1", kind="listen_mp3", path=rel(song_regen.LISTEN[0] / f"{name}.mp3"), prompt=None,
                        seed=a["seed"], model="ffmpeg libmp3lame 320k", workflow="pipeline/song_rounds.py listen", score=None,
                        status="done", meta={"source": aid, "rank": i, "copies": [str(d / f"{name}.mp3") for d in song_regen.LISTEN]})
        weakest = min(v["section_recall"].items(), key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 9))
        dur = len(y) / sr
        c = v.get("r5")
        label = (f"Round {r} #{i}: {aid} (seed {a['seed']}, {v['bpm_asked']} BPM asked; ending: {v['ending']['kind']}, "
                 f"{c['tail_s']} s tail; kick ratio {c['kick_ratio']}; {'all bars passed' if v['hit'] else f'{passed(v)}/4 bars'})"
                 if c else f"Round {r} #{i}: {aid} (seed {a['seed']}; ending: {v['ending']['kind']})")
        tracks["tracks"].append({
            "n": len(tracks["tracks"]) + 1, "id": aid, "label": label,
            "model": "ACE-Step 1.5 XL turbo", "licence": "MIT (commercial-safe)", "length": f"{int(dur // 60)}:{int(dur % 60):02d}",
            "file": f"{name}.mp3", "score": v["stage1_total"], "recall": v["lyric_recall"], "weakest": weakest[0],
            "weakest_recall": weakest[1], "ce": v["aesthetics"]["CE"], "pq": v["aesthetics"]["PQ"], "bpm": v["bpm"],
            "brief_bpm": v.get("bpm_asked", tracks.get("brief_bpm", 128))})
        log(f"{name}.mp3 <- {aid} {m}")
    for j, t in enumerate(tracks["tracks"], 1):
        t["n"] = j
    tracks_f.write_text(json.dumps(tracks, indent=1))
    return top


def table(rounds: list[int]) -> str:
    """Per-take markdown rows (r5 series) for qc/r5_report.md."""
    rows = ["| Take | Seed | BPM asked / measured | T | cfg | Stage-1 | Recall | Weak | Kick ratio | Drop 1 / best | Ending | Tail s "
            "| Length s | Bars |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for r in rounds:
        res = json.load(open(QC / f"r{r}.json"))["takes"]
        for k in sorted(res):
            v, c = res[k], res[k]["r5"]
            marks = " ".join(f"{n}{'✓' if ok else '✗'}" for n, ok in (("0.9", v["stage1_total"] >= BAR), ("kick", c["kick_ok"]),
                                                                     ("fade", c["fade_ok"]), ("len", c["length_ok"])))
            rows.append(f"| {'**' + k.split('take:')[1] + '**' if v['hit'] else k.split('take:')[1]} | {v['seed']} | "
                        f"{v['bpm_asked']} / {v['bpm']} | {v['temperature']} | {v['lm_cfg']} | {v['stage1_total']:.3f} | "
                        f"{v['lyric_recall']:.2f} | {', '.join(v['weak_sections']) or '-'} | {c['kick_ratio']} | {c['drop1_of_best']} | "
                        f"{v['ending']['kind']} | {c['tail_s']} | {c['length_s']} | {marks} |")
    return "\n".join(rows)


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "gen":
        cmd_gen(int(argv[1]))
    elif cmd == "analyze":
        cmd_analyze(int(argv[1]))
    elif cmd == "listen":
        cmd_listen(r5=argv[1:] == ["r5"])
    elif cmd == "table":
        print(table([int(x) for x in argv[1:]]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
