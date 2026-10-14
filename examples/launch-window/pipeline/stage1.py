"""Stage 1: the song. 16 ACE-Step takes -> per-take checks -> ranking -> repaint weak sections -> song gate.

  python -m pipeline.stage1 takes [--mix 16:0] [--n 16]   (T:S = turbo:SFT takes; SFT weights removed, so 16:0)
  python -m pipeline.stage1 analyze            vocal stems, transcripts, beats, aesthetics, loudness, scores
  python -m pipeline.stage1 rank               writes qc/stage1/ranking.md
  python -m pipeline.stage1 repaint <take_no>  repaint that take's weak sections (3 seeds each), keep improvements
  python -m pipeline.stage1 all                takes + analyze + rank
Idempotent: finished assets are skipped. State: manifest assets song:take:NN, song:repair:NN:*.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline import comfy, lyrics, manifest, master, song_spec, toolrun, wf
from pipeline.manifest import ROOT

TAKES = ROOT / "audio" / "takes"
QC = ROOT / "qc" / "stage1"
N_TAKES = 16
SEEDS = [10007 + 7919 * i for i in range(N_TAKES)]
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_NAMES = {"turbo": "acestep_v1.5_xl_turbo_bf16", "sft": "acestep_v1.5_xl_sft_bf16"}
# Room the spoken parts need (words / 2.6 w/s + 1 s), from master.md's spoken lines.
BREAK_NEED = {}  # launch-window: the [Intro] is sung and there are no Break sections
TWO_BARS = 8 * 60 / song_spec.BPM
WEAK = 0.7  # sections below this lyric recall are repaint candidates


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def take_id(i: int) -> str:
    return f"song:take:{i:02d}"


# ------------------------------------------------------------------ generation
def variant_for(i: int, mix: str) -> str:
    """mix 'T:S' = first T takes turbo, next S takes SFT (e.g. 12:4)."""
    t, _s = (int(x) for x in mix.split(":"))
    return "turbo" if i < t else "sft"


def cmd_takes(mix: str = "16:0", n: int = N_TAKES):
    lyr = song_spec.generator_lyrics()
    for i in range(n):
        variant = variant_for(i, mix)
        aid = take_id(i)
        a = manifest.get(aid)
        if a and a.get("status") not in (None, "running", "failed"):
            continue
        seed = SEEDS[i]
        g = wf.ace15_song(f"song_take_{i:02d}", song_spec.CAPTION, lyr, seed=seed, duration=song_spec.DURATION,
                          bpm=song_spec.BPM, keyscale=song_spec.KEYSCALE, variant=variant, prefix=f"launch-window/song/take_{i:02d}")
        wfp = g.save(ROOT / "workflows" / "stage1")
        manifest.upsert(aid, stage="1", kind="song_take", prompt=song_spec.CAPTION, seed=seed, model=MODEL_NAMES[variant],
                        workflow=rel(wfp), status="running", attempts=(a or {}).get("attempts", 0) + 1,
                        meta={"variant": variant, "bpm": song_spec.BPM, "key": song_spec.KEYSCALE, "duration": song_spec.DURATION})
        try:
            res = comfy.run(g.api(), name=aid, timeout=1800)
            item = comfy.output_files(res["outputs"])[0]
            dest = comfy.download(item, TAKES / f"take_{i:02d}.flac")
            manifest.upsert(aid, path=rel(dest), status="done", meta={"render_seconds": res["seconds"]})
            log(f"{aid}: {res['seconds']}s -> {rel(dest)}")
        except Exception as e:
            manifest.upsert(aid, status="failed", meta={"error": repr(e)[:2000]})
            log(f"{aid}: FAILED {e!r}")
    comfy.free()


# ------------------------------------------------------------------ analysis
def vocal_activity(path, hop: float = 0.1):
    """RMS envelope (dB) of a vocal stem in `hop` s frames."""
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    n = int(sr * hop)
    frames = len(x) // n
    rms = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms), hop


def breaks_check(vocals_path, sec_times: dict) -> dict:
    """For each spoken gap, measure how long the vocal stem stays quiet where the gap should be."""
    db, hop = vocal_activity(vocals_path)
    active = db > (np.percentile(db, 95) - 30)  # within 30 dB of loud singing
    order = [s.tag for s in lyrics.sections()]

    def quiet_run(t0, t1):
        a, b = max(0, int(t0 / hop)), min(len(active), int(t1 / hop))
        if b <= a:
            return 0.0
        best = cur = 0
        for v in active[a:b]:
            cur = 0 if v else cur + 1
            best = max(best, cur)
        return best * hop

    out = {}
    for tag, need in BREAK_NEED.items():
        i = order.index(tag)
        prev_end = 0.0 if tag == "Intro" else next((sec_times[t]["end"] for t in reversed(order[:i]) if sec_times.get(t, {}).get("end")), None)
        nxt = next((sec_times[t]["start"] for t in order[i + 1:] if sec_times.get(t, {}).get("start")), None)
        if prev_end is None or nxt is None or nxt <= prev_end:
            out[tag] = {"gap": None, "quiet": 0.0, "need": need, "ok": False}
            continue
        q = quiet_run(prev_end, nxt)
        out[tag] = {"gap": round(nxt - prev_end, 2), "quiet": round(q, 2), "need": need,
                    "ok": q >= TWO_BARS, "extend_by": round(max(0.0, need - q), 2)}
    return out


def cmd_analyze():
    todo = [a for a in manifest.query(stage="1", kind="song_take") if a["status"] == "done"]
    if not todo:
        log("analyze: nothing new")
        return
    _analyze_assets(todo)


def cmd_rescore():
    """Recompute scores from saved stems/transcripts/beats (CPU only) after a scoring change."""
    for a in manifest.query(stage="1"):
        if a["kind"] not in ("song_take", "song_repair") or a["status"] != "scored":
            continue
        _score(a, str(ROOT / a["path"]), str(ROOT / a["meta"]["vocals"]), json.load(open(ROOT / a["meta"]["transcript"]))["words"],
               json.load(open(ROOT / a["meta"]["beats"]))["bpm"], a["score"]["aesthetics"], rel(ROOT / a["meta"]["transcript"]),
               rel(ROOT / a["meta"]["beats"]), a["meta"].get("transcript_text", ""))


def _analyze_assets(todo: list[dict]):
    """Vocal stem, transcript, beats, aesthetics, loudness and score for each asset (batched tool runs)."""
    paths = [str(ROOT / a["path"]) for a in todo]
    stems_dir = TAKES / "stems"
    log(f"analyze: {len(paths)} takes: vocal stems")
    sep = toolrun.run("separator", "separate.py", [stems_dir, VOCAL_MODEL, *paths], name="stage1:separate",
                      log=ROOT / "qc" / "logs" / "stage1-tools.log")
    vocals = {p: next(f for f in sep["stems"][p] if "(Vocals)" in f) for p in paths}
    log("analyze: transcripts")
    tr = toolrun.run("whisperx", "align.py", ["transcribe", TAKES / "transcripts", *vocals.values()], name="stage1:whisperx",
                     log=ROOT / "qc" / "logs" / "stage1-tools.log")["results"]
    log("analyze: beats")
    bt = toolrun.run("beats", "beats.py", [TAKES / "beats", *paths], name="stage1:beats",
                     log=ROOT / "qc" / "logs" / "stage1-tools.log")["results"]
    log("analyze: aesthetics")
    ae = toolrun.run("aesthetics", "aesthetics.py", paths, name="stage1:aesthetics",
                     log=ROOT / "qc" / "logs" / "stage1-tools.log")["scores"]
    for a, p in zip(todo, paths):
        v = vocals[p]
        _score(a, p, v, json.load(open(tr[v]["out"]))["words"], bt[p]["bpm"], ae[p], rel(tr[v]["out"]), rel(bt[p]["out"]),
               tr[v]["text"])


def _score(a, p, v, words, bpm, aes, transcript_rel, beats_rel, text):
    m = lyrics.match(words)
    sec_times = {s["section"]: s for s in m["sections"]}
    brk = breaks_check(v, sec_times)
    loud = master.measure(p)
    tempo_ok = any(abs(bpm - b) / b < 0.08 for b in (song_spec.BPM, song_spec.BPM / 2))
    sung = [s for s in m["sections"] if s["recall"] is not None]
    min_rec = min(s["recall"] for s in sung)
    # Missing lyrics can only be fixed by repaint; short breaks are fixed in the mix (2-bar loops / holds),
    # so the weakest section weighs more than break room (DECISIONS.md, Stage 1 scoring).
    score = (0.45 * m["recall"] + 0.20 * min_rec + 0.08 * ((sum(b["ok"] for b in brk.values()) / len(brk)) if brk else 1.0)
             + 0.25 * (aes["CE"] + aes["PQ"]) / 20 + 0.02 * tempo_ok)
    weak = [s["section"] for s in sung if s["recall"] < WEAK]
    manifest.upsert(a["id"], status="scored", score={
        "total": round(score, 4), "lyric_recall": m["recall"], "min_section_recall": min_rec,
        "breaks_ok": sum(b["ok"] for b in brk.values()), "aesthetics": aes, "bpm": bpm, "tempo_ok": tempo_ok,
        "lufs": loud["lufs"], "true_peak": loud["true_peak_dbtp"]},
        meta={"vocals": rel(v), "transcript": transcript_rel, "beats": beats_rel, "sections": m["sections"],
              "breaks": brk, "weak_sections": weak, "transcript_text": text[:3000]})
    log(f"{a['id']}: score {score:.3f} recall {m['recall']:.2f} breaks {sum(b['ok'] for b in brk.values())}/{len(brk)} "
        f"CE {aes['CE']} PQ {aes['PQ']} bpm {bpm} weak {weak}")


# ------------------------------------------------------------------ repaint
REPAINT_SEEDS = 3


def _window(a, section):
    """Repaint window for a section: its sung span padded by half a bar, bounded by its neighbours."""
    secs = a["meta"]["sections"]
    tags = [s["section"] for s in secs]
    i = tags.index(section)
    s = secs[i]
    start = s["start"] if s["start"] is not None else next((x["end"] for x in reversed(secs[:i]) if x["end"]), 0.0)
    end = s["end"] if s["end"] is not None else next((x["start"] for x in secs[i + 1:] if x["start"]), song_spec.DURATION)
    pad = TWO_BARS / 4
    return max(0.0, start - pad), min(song_spec.DURATION, end + pad)


def cmd_repaint(take_no: int):
    from pipeline import acestep
    base = manifest.get(take_id(take_no))
    weak = base["meta"].get("weak_sections", [])
    if not weak:
        log(f"repaint: {base['id']} has no weak sections")
        return
    src = ROOT / base["path"]
    lyr = song_spec.generator_lyrics()
    made = []
    with acestep.session(f"stage1:repaint:{take_no:02d}"):
        for section in weak:
            t0, t1 = _window(base, section)
            for k in range(REPAINT_SEEDS):
                aid = f"song:repair:{take_no:02d}:{section.replace(' ', '_')}:{k}"
                if (manifest.get(aid) or {}).get("status") in ("done", "scored"):
                    continue
                seed = base["seed"] * 10 + 101 * (k + 1) + len(made)
                out = TAKES / "repairs" / f"take_{take_no:02d}_{section.replace(' ', '_')}_{k}.flac"
                manifest.upsert(aid, stage="1", kind="song_repair", prompt=song_spec.CAPTION, seed=seed,
                                model="ACE-Step 1.5 " + acestep.DIT, workflow="pipeline/acestep.py repaint", status="running",
                                meta={"base": base["id"], "section": section, "window": [round(t0, 2), round(t1, 2)]})
                r = acestep.repaint(src, t0, t1, song_spec.CAPTION, lyr, seed, out, bpm=song_spec.BPM,
                                    key_scale=song_spec.KEYSCALE, duration=song_spec.DURATION)
                manifest.upsert(aid, path=rel(out), status="done", meta={"render_seconds": r["seconds"]})
                made.append(aid)
                log(f"{aid}: {r['seconds']}s window {t0:.1f}-{t1:.1f}")
    cmd_analyze_repairs(take_no)


def cmd_analyze_repairs(take_no: int):
    """Score repairs with the same checks; keep the best per section only if it beats the base take."""
    reps = [a for a in manifest.query(stage="1", kind="song_repair") if a["meta"]["base"] == take_id(take_no) and a["status"] == "done"]
    if reps:
        _analyze_assets(reps)
    base = manifest.get(take_id(take_no))
    best = {}
    for a in manifest.query(stage="1", kind="song_repair"):
        if a["meta"]["base"] != base["id"] or a["status"] != "scored":
            continue
        sec = a["meta"]["section"]
        rec = next(s["recall"] for s in a["meta"]["sections"] if s["section"] == sec)
        base_rec = next(s["recall"] for s in base["meta"]["sections"] if s["section"] == sec)
        gain = rec - base_rec
        ok = gain > 0.1 and a["score"]["total"] >= base["score"]["total"] - 0.01
        if ok and gain > best.get(sec, (None, 0))[1]:
            best[sec] = (a["id"], gain)
    for sec, (aid, gain) in best.items():
        manifest.upsert(aid, status="scored", meta={"kept": True, "gain": round(gain, 3)})
        log(f"repair kept for {sec}: {aid} (+{gain:.2f} recall)")
    if not best:
        log("repaint: no repair beat the base take")


# ------------------------------------------------------------------ ranking
def cmd_rank():
    takes = sorted([a for a in manifest.query(stage="1") if a["kind"] in ("song_take", "song_repair") and a["status"] in ("scored", "approved")],
                   key=lambda a: -a["score"]["total"])
    QC.mkdir(parents=True, exist_ok=True)
    lines = ["| Rank | Take | Seed | Score | Lyric recall | Min section | Breaks ok | CE | PQ | BPM | LUFS | Weak sections |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r, a in enumerate(takes, 1):
        s = a["score"]
        lines.append(f"| {r} | {a['id']} | {a['seed']} | {s['total']:.3f} | {s['lyric_recall']:.2f} | {s['min_section_recall']:.2f} | "
                     f"{s['breaks_ok']}/4 | {s['aesthetics']['CE']:.2f} | {s['aesthetics']['PQ']:.2f} | {s['bpm']} | {s['lufs']} | "
                     f"{', '.join(a['meta'].get('weak_sections', [])) or '-'} |")
    (QC / "ranking.md").write_text("\n".join(lines) + "\n")
    log(f"rank: {len(takes)} takes -> qc/stage1/ranking.md; best {takes[0]['id'] if takes else None}")
    return takes


def cmd_report():
    """qc/stage-1.md: ranking, the leading take's per-section checks, repaints, gate status."""
    takes = cmd_rank()
    best = takes[0]
    gate = manifest.load()["gates"]["song_take"]
    reps = [a for a in manifest.query(stage="1", kind="song_repair")]
    lines = ["# Stage 1 · Song", "", f"Generated {manifest.now()}.", "",
             f"16 ACE-Step 1.5 takes (12 XL turbo + 4 XL SFT) from master.md's generator lyrics and caption "
             f"(\"{song_spec.CAPTION}\"), duration {song_spec.DURATION:.0f} s. Each take was checked by: BS-RoFormer vocal stem, "
             "WhisperX transcript matched section by section against the lyrics, vocal-free room in each spoken break, "
             "beat_this tempo, ffmpeg loudness, Audiobox Aesthetics.", "", "## Ranking", "",
             (QC / "ranking.md").read_text(), "", f"## Leading take: {best['id']} (seed {best['seed']})", "",
             "| Section | Words | Lyric recall | Sung from | to |", "| --- | ---: | ---: | ---: | ---: |"]
    for s in best["meta"]["sections"]:
        if s["recall"] is not None:
            lines.append(f"| {s['section']} | {s['words']} | {s['recall']:.2f} | {s['start']} | {s['end']} |")
    lines += ["", "| Spoken break | Vocal-free room | Needed for its lines | Short by (mix extends with 2-bar loops / holds) |",
              "| --- | ---: | ---: | ---: |"]
    for k, b in best["meta"]["breaks"].items():
        lines.append(f"| {k} | {b['quiet']} s | {b['need']} s | {b.get('extend_by', b['need'])} s |")
    if reps:
        lines += ["", "## Repaints", "", "| Repair | Section | Window | Seed | Score | Section recall | Kept |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
        for a in sorted(reps, key=lambda a: a["id"]):
            sec = a["meta"]["section"]
            rec = next((s["recall"] for s in a["meta"].get("sections", []) if s["section"] == sec), None)
            lines.append(f"| {a['id']} | {sec} | {a['meta']['window']} | {a['seed']} | "
                         f"{a.get('score', {}) and a['score'].get('total')} | {rec} | {'yes' if a['meta'].get('kept') else 'no'} |")
    lines += ["", "## Song gate", "", f"Status: **{gate.get('status')}**" + (f", approved asset `{gate.get('asset')}`" if gate.get("asset") else ""),
              f"Candidates sent: {gate.get('candidates')}", ""]
    (ROOT / "qc" / "stage-1.md").write_text("\n".join(lines))
    log("report: qc/stage-1.md")


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "report":
        cmd_report()
        return 0
    mix = argv[argv.index("--mix") + 1] if "--mix" in argv else "16:0"
    if cmd == "takes":
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else N_TAKES
        cmd_takes(mix, n)
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "repaint":
        cmd_repaint(int(argv[1]))
    elif cmd == "rescore":
        cmd_rescore()
        cmd_rank()
    elif cmd == "rank":
        cmd_rank()
    elif cmd == "all":
        manifest.set_stage("1", status="running", started=manifest.now())
        cmd_takes(mix)
        cmd_analyze()
        cmd_rank()
        manifest.set_stage("1", status="takes_scored")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
