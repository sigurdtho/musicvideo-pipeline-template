"""Stage 2: stems, lyric timing, beat grid, song edit list and the shot list for the approved take.

  python -m pipeline.stage2 stems     vocals/instrumental, lead/backing, drums/bass/other -> audio/stems/
  python -m pipeline.stage2 align     WhisperX forced alignment of every lyric line on the lead stem
  python -m pipeline.stage2 beats     beat_this beats + downbeats on the full mix
  python -m pipeline.stage2 plan      spoken-line slots, break extensions (2-bar loops), shot list
  python -m pipeline.stage2 all
Outputs: audio/timing/{lyrics,beats,song_edit}.json, video/shotlist.json, qc/stage-2.md.
`plan` re-runs after Stage 3 with real voice durations (audio/voices/picked.json) to fix the final timing.
"""
import json
import shutil
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from pipeline import lyrics, manifest, shot_plan, song_spec, toolrun, voice_spec
from pipeline.manifest import ROOT

STEMS = ROOT / "audio" / "stems"
TIMING = ROOT / "audio" / "timing"
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEMUCS_MODEL = "htdemucs_ft.yaml"
WORDS_PER_S = 2.6          # spoken pace used until a real take is measured
LINE_PAUSE = 0.35
LOG = ROOT / "qc" / "logs" / "stage2-tools.log"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def approved_take() -> dict:
    gate = manifest.load()["gates"]["song_take"]
    if gate.get("status") != "approved":
        raise SystemExit("song gate not approved yet")
    return manifest.get(gate["asset"])


def song_path() -> Path:
    return ROOT / approved_take()["path"]


# ------------------------------------------------------------------ stems
def cmd_stems():
    take = str(song_path())
    STEMS.mkdir(parents=True, exist_ok=True)
    work = STEMS / "work"
    r1 = toolrun.run("separator", "separate.py", [work, VOCAL_MODEL, take], name="stage2:vocals", log=LOG)["stems"][take]
    vocals = next(f for f in r1 if "(Vocals)" in f)
    inst = next(f for f in r1 if "(Instrumental)" in f)
    r2 = toolrun.run("separator", "separate.py", [work, KARAOKE_MODEL, vocals], name="stage2:karaoke", log=LOG)["stems"][vocals]
    r3 = toolrun.run("separator", "separate.py", [work, DEMUCS_MODEL, inst], name="stage2:demucs", log=LOG)["stems"][inst]
    named = {"vocals": vocals, "instrumental": inst,
             "lead": next(f for f in r2 if "(Vocals)" in f), "backing": next(f for f in r2 if "(Instrumental)" in f),
             "drums": next(f for f in r3 if "(Drums)" in f), "bass": next(f for f in r3 if "(Bass)" in f),
             "other": next(f for f in r3 if "(Other)" in f)}
    model = {"vocals": VOCAL_MODEL, "instrumental": VOCAL_MODEL, "lead": KARAOKE_MODEL, "backing": KARAOKE_MODEL,
             "drums": DEMUCS_MODEL, "bass": DEMUCS_MODEL, "other": DEMUCS_MODEL}
    for name, src in named.items():
        dest = STEMS / f"{name}.flac"
        shutil.copy2(src, dest)
        manifest.upsert(f"stem:{name}", stage="2", kind="stem", path=rel(dest), prompt=None, seed=None, model=model[name],
                        workflow="pipeline/tools/separate.py", score=None, status="done",
                        meta={"source": approved_take()["id"]})
    log(f"stems: {list(named)}")


# ------------------------------------------------------------------ lyric timing
def _line_windows(words: list[dict]) -> list[dict]:
    """Rough window per sung lyric line from a free transcript (sequence-matched), gaps interpolated."""
    lines = []
    for s in lyrics.sections():
        for i, text in enumerate(s.lines):
            lines.append({"section": s.tag, "line": i, "text": text, "words": lyrics.norm_words(text)})
    ref, owner = [], []
    for k, l in enumerate(lines):
        ref += l["words"]
        owner += [k] * len(l["words"])
    hyp, times = [], []
    for w in words:
        for t in lyrics.norm_words(w.get("word", "")):
            hyp.append(t)
            times.append((w.get("start"), w.get("end")))
    hits = {}
    for a, b, n in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_matching_blocks():
        for j in range(n):
            if times[b + j][0] is not None:
                hits.setdefault(owner[a + j], []).append(times[b + j])
    for k, l in enumerate(lines):
        ts = hits.get(k, [])
        l["start"], l["end"] = (min(t[0] for t in ts), max(t[1] for t in ts)) if ts else (None, None)
        l["matched"] = round(len(ts) / max(1, len(l["words"])), 2)
    # interpolate missing windows between timed neighbours, proportional to word counts
    for k, l in enumerate(lines):
        if l["start"] is not None:
            continue
        prev = next((lines[j] for j in range(k - 1, -1, -1) if lines[j]["start"] is not None), None)
        nxt = next((lines[j] for j in range(k + 1, len(lines)) if lines[j]["start"] is not None), None)
        t0 = prev["end"] if prev else 0.0
        t1 = nxt["start"] if nxt else song_spec.DURATION
        l["start"], l["end"], l["interpolated"] = t0 + 0.1, max(t0 + 0.6, t1 - 0.1), True
    return lines


def _assign_forced_words(lines: list[dict], words: list[dict]) -> None:
    """WhisperX splits segments at sentence punctuation, so map aligned words back to lyric lines by word order."""
    k = 0
    for l in lines:
        n = len(l["text"].split())
        ws = words[k:k + n]
        k += n
        got = lyrics.norm_words(" ".join(w["word"] for w in ws))
        want = lyrics.norm_words(l["text"])
        l["word_times"] = [{"word": w["word"], "start": w.get("start"), "end": w.get("end")} for w in ws]
        l["forced_ok"] = got == want
        timed = [w for w in ws if w.get("start") is not None]
        if timed:
            l["start"], l["end"] = round(timed[0]["start"], 3), round(timed[-1]["end"], 3)
    if k != len(words):
        raise RuntimeError(f"forced alignment word count mismatch: lines need {k}, aligner returned {len(words)}")


def cmd_align():
    # The karaoke lead/backing split is unreliable for ASR (Whisper hallucinated on it), so timing comes from the
    # full BS-RoFormer vocal stem, the same stem that scored 0.92 lyric recall in Stage 1 (DECISIONS.md).
    lead = str(STEMS / "vocals.flac")
    TIMING.mkdir(parents=True, exist_ok=True)
    tr = toolrun.run("whisperx", "align.py", ["transcribe", TIMING / "asr", lead], name="stage2:asr", log=LOG)["results"][lead]
    lines = _line_windows(json.load(open(tr["out"]))["words"])
    segs = [{"text": l["text"], "start": max(0.0, l["start"] - 0.4), "end": l["end"] + 0.4} for l in lines]
    seg_file = TIMING / "force_segments.json"
    seg_file.write_text(json.dumps(segs, indent=1))
    out = TIMING / "forced.json"
    toolrun.run("whisperx", "align.py", ["force", lead, seg_file, out], name="stage2:force", log=LOG)
    forced = json.load(open(out))["segments"]
    _assign_forced_words(lines, [w for seg in forced for w in seg.get("words", [])])
    (TIMING / "lyrics.json").write_text(json.dumps(lines, indent=1))
    untimed = [l["text"] for l in lines if l["start"] is None]
    interp = [l["text"] for l in lines if l.get("interpolated")]
    manifest.upsert("timing:lyrics", stage="2", kind="timing", path=rel(TIMING / "lyrics.json"), model="WhisperX large-v3 + wav2vec2",
                    workflow="pipeline/tools/align.py force", status="done",
                    score={"lines": len(lines), "untimed": len(untimed), "interpolated": len(interp)})
    log(f"align: {len(lines)} lines, untimed {len(untimed)}, interpolated windows {len(interp)}")


def cmd_beats():
    take = str(song_path())
    r = toolrun.run("beats", "beats.py", [TIMING, take], name="stage2:beats", log=LOG)["results"][take]
    shutil.copy2(r["out"], TIMING / "beats.json")
    manifest.upsert("timing:beats", stage="2", kind="timing", path=rel(TIMING / "beats.json"), model="beat_this final0",
                    workflow="pipeline/tools/beats.py", status="done", score={"bpm": r["bpm"], "beats": r["beats"]})
    log(f"beats: {r}")


# ------------------------------------------------------------------ plan: spoken slots, song edit, shots
# Fill these from master.md when the song has spoken slots or on-screen lines.
# SPOKEN_GROUPS: (group, (section_before, section_after), line ids, "loop"|"hold"|"tail")
# TEXT_GROUPS: groups that are boxes, not voiced. TEXT_LINES: line ids shown as text.
SPOKEN_GROUPS = []
TEXT_LINES = set()
TEXT_GROUPS = set()


def natural_end(threshold_db: float = -45.0) -> float:
    """Where the song's own ending dies away: the last 50 ms window of the instrumental stem above threshold_db."""
    import soundfile as sf
    x, sr = sf.read(str(ROOT / "audio" / "stems" / "instrumental.flac"), always_2d=True)
    x = x.mean(1)
    win = int(0.05 * sr)
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(1))
    loud = np.where(20 * np.log10(rms + 1e-9) > threshold_db)[0]
    return float((loud[-1] + 1) * win / sr) if len(loud) else len(x) / sr
QUOTE_READ_HOLD = 0.6  # a dialogue box can stay up a little past the line so it can be read
MAX_LEN = {"i2v": 10.0, "lipsync": 9.0, "overlay": 6.0}  # i2v: a 5 s clip played down to half speed
MIN_FILL = 2.0  # shorter instrumental remainders extend the previous shot instead of getting a fill


def _est_duration(line_id: str, actual: dict) -> float:
    if line_id in actual:
        return actual[line_id]
    text = next(l["text"] for l in voice_spec.LINES if l["id"] == line_id)
    return len(lyrics.norm_words(text)) / WORDS_PER_S + 0.4


def _pool_key(tag: str) -> str:
    base = tag.rsplit(" ", 1)[0] if tag.split(" ")[-1].isdigit() else tag
    return {"Chorus": "Chorus", "Break": "Break"}.get(base, base)


def plan(actual_durations: dict | None = None) -> dict:
    actual = actual_durations or {}
    lines = json.load(open(TIMING / "lyrics.json"))
    beats = json.load(open(TIMING / "beats.json"))
    bt = np.array(beats["beats"])
    ibi = float(np.median(np.diff(bt)))
    bar, two_bars = 4 * ibi, 8 * ibi
    end_orig = song_spec.DURATION
    sec_span = {}
    for l in lines:
        sp = sec_span.setdefault(l["section"], [l["start"], l["end"]])
        sp[0], sp[1] = min(sp[0], l["start"]), max(sp[1], l["end"])

    # Spoken slots come from SPOKEN_GROUPS. Empty means the song runs straight through.
    edits, groups = [], {}
    for group, (a, b), ids, material in SPOKEN_GROUPS:
        t0 = 0.0 if a == "start" else sec_span[a][1] + 0.25
        t1 = end_orig if b == "end" else sec_span[b][0] - 0.35
        durs = [_est_duration(i, actual) for i in ids]
        need = sum(durs) + LINE_PAUSE * (len(ids) - 1) + (0.6 if material != "tail" else 0.8)
        avail = max(0.0, t1 - t0)
        ext, at = 0.0, None
        if need > avail and group not in TEXT_GROUPS:
            if material == "loop":
                ext = int(np.ceil((need - avail) / two_bars)) * two_bars
                at = float(bt[max(0, np.searchsorted(bt, t1) - 1)])
            elif material == "hold":
                ext = int(np.ceil((need - avail) / bar)) * bar
                at = t1
            else:  # tail: the music has already cut; just let the last bell ring out
                ext = need - avail + 1.5
                at = end_orig
            edits.append({"group": group, "at": round(at, 3), "insert": round(ext, 3), "kind": material,
                          "source": [round(max(0.0, at - two_bars), 3), round(at, 3)] if material == "loop" else None})
        groups[group] = {"ids": ids, "t0": t0, "t1": t1, "durs": durs, "need": round(need, 2), "avail": round(avail, 2),
                         "extended_by": round(ext, 3), "material": material}
    edits.sort(key=lambda e: e["at"])

    def to_final(t: float) -> float:
        return t + sum(e["insert"] for e in edits if e["at"] <= t)

    total = to_final(end_orig) + sum(e["insert"] for e in edits if e["at"] >= end_orig)
    fb = [to_final(x) for x in bt]
    for e in edits:
        start_final = to_final(e["at"]) - (e["insert"] if e["at"] < end_orig else 0.0)
        fb += [start_final + k * ibi for k in range(int(round(e["insert"] / ibi)))]
    fb = np.array(sorted(set(round(x, 3) for x in fb)))

    def on_beat(t: float, after: bool = False) -> float:
        cand = fb[fb >= t - 1e-6] if after else fb[fb <= t + 1e-6]
        if not len(cand):
            return t
        return float(cand[0] if after else cand[-1])

    # ---- 2. place spoken lines: spread over each slot, starting on a beat
    placements = {}
    for g, s in groups.items():
        t0f = to_final(s["t0"])
        t1f = to_final(s["t0"]) + (s["t1"] - s["t0"]) + s["extended_by"]
        n = len(s["ids"])
        if g in TEXT_GROUPS:  # text-only: fit the boxes into the break as it is
            room = (t1f - t0f) - 0.6 - LINE_PAUSE * (n - 1) - 0.4
            if sum(s["durs"]) > room:
                s["durs"] = [d * room / sum(s["durs"]) for d in s["durs"]]
        free = (t1f - t0f) - sum(s["durs"])
        lead = min(max(0.2, free * 0.12), two_bars / 2) if g != "intro" else min(max(0.5, free * 0.1), bar)
        gap = min(bar, max(LINE_PAUSE, (free - lead - min(free * 0.12, two_bars / 2)) / max(1, n - 1)))
        t = on_beat(t0f + lead, after=True)
        for i, d in zip(s["ids"], s["durs"]):
            placements[i] = {"start": round(t, 3), "end": round(t + d, 3), "group": g}
            t = t + d + gap
        s.update({"t0_final": round(t0f, 3), "t1_final": round(t1f, 3)})
    for k in TEXT_LINES:
        placements[k]["text"] = True

    # the video ends with the song's own ending (review round 3), plus a short tail for the fade to black
    total = min(total, to_final(natural_end()) + 0.4)

    # ---- 3. anchored shots -> cuts on beats
    def span(sh):
        a = sh["anchor"]
        if a[0] == "line":
            l = next(x for x in lines if x["section"] == a[1] and x["line"] == a[2])
            st, en = to_final(l["start"]), to_final(l["end"])
            f0, f1 = sh.get("frac", (0.0, 1.0))
            return st + f0 * (en - st), st + f1 * (en - st), l["section"]
        ids = [a[1]] if a[0] == "spoken" else a[1]
        sec = next(v["group"] for k, v in placements.items() if k == ids[0])
        return placements[ids[0]]["start"], placements[ids[-1]]["end"], sec

    spans = [span(sh) for sh in shot_plan.SHOTS]
    order = sorted(range(len(spans)), key=lambda k: spans[k][0])
    cuts = []
    for k in order:
        c = on_beat(spans[k][0] - 0.12)
        if cuts and c < cuts[-1][1] + 0.5 * ibi:
            c = on_beat(cuts[-1][1] + 0.5 * ibi, after=True)
        cuts.append((k, c))
    cuts[0] = (cuts[0][0], 0.0)

    # ---- 4. cap lengths; fill instrumental stretches from the section's pool
    shots, fill_used = [], {}
    for j, (k, c) in enumerate(cuts):
        t_next = cuts[j + 1][1] if j + 1 < len(cuts) else total
        sh = dict(shot_plan.SHOTS[k])
        content_end = spans[k][1]
        cap = MAX_LEN[sh["kind"]]
        stop = min(t_next, max(on_beat(content_end + ibi, after=True), c + 2 * ibi), c + cap)
        if t_next - stop < MIN_FILL or sh.get("run_on"):  # no fill: let the shot run (slow-mo / longer lip-sync / held)
            stop = t_next
        else:
            stop = on_beat(stop, after=True) if on_beat(stop, after=True) < t_next else stop
        sh.update({"start": round(c, 3), "end": round(stop, 3)})
        shots.append(sh)
        t, n = stop, 0
        pool_key = _pool_key(spans[k][2])
        pool = shot_plan.FILL.get(pool_key) or shot_plan.FILL["Chorus"]
        while t_next - t >= MIN_FILL:
            seg_end = on_beat(t + two_bars, after=True)
            if t_next - seg_end < MIN_FILL:
                seg_end = t_next
            i = fill_used.get(pool_key, 0)
            f = dict(pool[i % len(pool)])
            fill_used[pool_key] = i + 1
            f.update({"id": f"{sh['id']}__fill{n}", "fill": True, "anchor": ("fill", pool_key, i), "start": round(t, 3),
                      "end": round(seg_end, 3), "variant": i // len(pool)})
            shots.append(f)
            t, n = seg_end, n + 1
        if shots[-1]["end"] < t_next:
            shots[-1]["end"] = round(t_next, 3)
    # montage shots (review round 3): the span splits into beat-aligned cuts, each reusing another shot's picked clip;
    # the shot's boxes stay on the first cut (achievement boxes run across the cuts)
    expanded = []
    for sh in shots:
        srcs = sh.get("montage")
        if not srcs:
            expanded.append(sh)
            continue
        a, b = sh["start"], sh["end"]
        bi = [a] + [float(x) for x in fb if a + 0.05 < x < b - 0.05] + [b]  # beat boundaries inside the span
        bounds = [bi[round(k * (len(bi) - 1) / len(srcs))] for k in range(len(srcs) + 1)]
        for k, src in enumerate(srcs):
            m = {kk: vv for kk, vv in sh.items() if kk != "montage"}
            m.update({"id": sh["id"] if k == 0 else f"{sh['id']}__m{k}", "reuse_clip": src,
                      "start": round(bounds[k], 3), "end": round(bounds[k + 1], 3)})
            if k:
                m["boxes"] = []
            expanded.append(m)
    shots = expanded
    for i, sh in enumerate(shots):
        sh["index"] = i
        sh["duration"] = round(sh["end"] - sh["start"], 3)
    return {"bpm": round(60 / ibi, 2), "ibi": round(ibi, 4), "two_bars": round(two_bars, 3), "duration_final": round(total, 3),
            "edits": edits, "slots": {g: {k: v for k, v in s.items() if k != "durs"} for g, s in groups.items()},
            "spoken": placements, "shots": shots, "beats_final": fb.round(3).tolist()}


def cmd_plan():
    picked = ROOT / "audio" / "voices" / "picked.json"
    actual = {k: v["duration"] for k, v in json.load(open(picked)).items()} if picked.exists() else {}
    p = plan(actual)
    (TIMING / "song_edit.json").write_text(json.dumps({k: p[k] for k in ("bpm", "ibi", "two_bars", "duration_final",
                                                                          "edits", "slots", "spoken")}, indent=1))
    (ROOT / "video").mkdir(exist_ok=True)
    (ROOT / "video" / "shotlist.json").write_text(json.dumps({"duration": p["duration_final"], "bpm": p["bpm"],
                                                              "beats": p["beats_final"], "shots": p["shots"]}, indent=1))
    long_i2v = [s["id"] for s in p["shots"] if s["kind"] == "i2v" and s["duration"] > 5.0]
    manifest.upsert("plan:shotlist", stage="2", kind="shotlist", path="video/shotlist.json", workflow="pipeline/stage2.py plan",
                    status="done", score={"shots": len(p["shots"]), "duration": p["duration_final"],
                                          "i2v_over_5s": len(long_i2v)},
                    meta={"actual_voice_durations": bool(actual), "edits": p["edits"]})
    log(f"plan: {len(p['shots'])} shots, {p['duration_final']:.1f}s, edits {p['edits']}, i2v>5s {long_i2v}")
    return p


def cmd_report():
    """qc/stage-2.md with the master.md self-checks: every lyric line timed; every cut on a beat."""
    lines = json.load(open(TIMING / "lyrics.json"))
    se = json.load(open(TIMING / "song_edit.json"))
    sl = json.load(open(ROOT / "video" / "shotlist.json"))
    beats = np.array(sl["beats"])
    untimed = [l["text"] for l in lines if l["start"] is None or l["end"] is None]
    off_beat = [s["id"] for s in sl["shots"][1:] if np.min(np.abs(beats - s["start"])) > 0.002]
    kinds = {}
    for s in sl["shots"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    d = [s["duration"] for s in sl["shots"]]
    md = ["# Stage 2 · Stems and timing", "", f"Generated {manifest.now()} for `{approved_take()['id']}`.", "",
          "## Self-checks", "", f"- Every lyric line timed: **{'PASS' if not untimed else 'FAIL'}** ({len(lines)} lines, untimed: {untimed or 'none'})",
          f"- Every cut on a beat: **{'PASS' if not off_beat else 'FAIL'}** ({len(sl['shots'])} shots; off-beat: {off_beat or 'none'})", "",
          "## Stems", "", ", ".join(f"`audio/stems/{k}.flac`" for k in ("vocals", "instrumental", "lead", "backing", "drums", "bass", "other")), "",
          "## Timing", "", f"- Tempo {se['bpm']} BPM (beat_this, median inter-beat {se['ibi']} s); final length {se['duration_final']} s",
          f"- Song edits (room for spoken lines): " + ("; ".join(f"{e['group']}: +{e['insert']} s {e['kind']} at {e['at']} s" for e in se["edits"]) or "none"), "",
          "| Spoken group | Slot (final time) | Needed | Room in take | Extended by |", "| --- | --- | ---: | ---: | ---: |"]
    for g, v in se["slots"].items():
        md.append(f"| {g} | {v['t0_final']}–{v['t1_final']} | {v['need']} s | {v['avail']} s | {v['extended_by']} s |")
    md += ["", "## Shot list", "", f"`video/shotlist.json`: {len(sl['shots'])} shots ({', '.join(f'{k} {v}' for k, v in kinds.items())}), "
           f"durations {min(d):.2f}–{max(d):.2f} s (mean {np.mean(d):.2f} s), {sum(1 for s in sl['shots'] if s.get('fill'))} instrumental fill shots.", ""]
    (ROOT / "qc" / "stage-2.md").write_text("\n".join(md))
    manifest.set_stage("2", status="done" if not untimed and not off_beat else "check_failed", finished=manifest.now())
    log(f"report: untimed {len(untimed)}, off-beat {len(off_beat)}")


def main(argv):
    cmd = argv[0] if argv else ""
    steps = {"stems": cmd_stems, "align": cmd_align, "beats": cmd_beats, "plan": cmd_plan, "report": cmd_report}
    if cmd == "all":
        manifest.set_stage("2", status="running", started=manifest.now())
        for k in ("stems", "align", "beats", "plan", "report"):
            steps[k]()
    elif cmd in steps:
        steps[cmd]()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
