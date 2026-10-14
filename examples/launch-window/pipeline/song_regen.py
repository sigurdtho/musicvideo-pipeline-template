"""Take 04 regenerated at ~208 s with an instrumental outro section (the "04b" series), scored against take 04.

  python -m pipeline.song_regen gen        6 ComfyUI ACE-Step takes: take 04's seed/caption/settings, longer, + outro tag
  python -m pipeline.song_regen analyze    stage1 analysis (stems, WhisperX, beats, aesthetics, score) + similarity + ending
  python -m pipeline.song_regen listen     top 3: -14 LUFS / -1.3 dBTP -> qc/listen/take04b_{1,2,3}.mp3 (+ $REVIEW_DIR)
The extra lyric tag is job conditioning only; master.md's generator lyrics are unchanged (DECISIONS.md 2026-10-12).
Nothing here touches the song gate.
"""
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from pipeline import comfy, manifest, master, song_spec, stage1, wf
from pipeline.manifest import ROOT

BASE = "song:take:04"
DIR = ROOT / "audio" / "takes" / "04b"
QC = ROOT / "qc" / "stage1" / "04b"
OUTRO_TAG = "\n[Outro - instrumental, piano and synths fade]\n"
SEED = 41683  # take 04
# k: (seed, duration, lm temperature, lm cfg). Take 04 = (41683, 190, 0.85, 2.0); DiT sampler unchanged (turbo, 8 steps, cfg 1).
VARIANTS = {
    0: (SEED, 208.0, 0.85, 2.0),
    1: (SEED, 205.0, 0.85, 2.0),
    2: (SEED, 210.0, 0.85, 2.0),
    3: (SEED, 208.0, 0.70, 2.0),
    4: (SEED, 208.0, 0.85, 2.5),
    5: (SEED + 1, 208.0, 0.85, 2.0),
}
LISTEN = [ROOT / "qc" / "listen", Path(os.environ.get("REVIEW_DIR", str(ROOT / "qc" / "review_sync"))) / "launch-window" / "listen"]
FOREIGN_UNITS = ("wotmat-*", "wb-test-*", "dtc-*", "grok-*")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def take_id(k: int) -> str:
    return f"song:take:04b:{k}"


def take_path(k: int) -> Path:
    return DIR / f"take_04b_{k}.flac"


def lyrics() -> str:
    return song_spec.generator_lyrics().rstrip("\n") + "\n" + OUTRO_TAG


# ------------------------------------------------------------------ GPU etiquette
def gpu_busy_elsewhere() -> list[str]:
    """Held .gpu.lock in other projects, active foreign job units, or a non-empty ComfyUI queue."""
    busy = []
    for f in ROOT.parent.glob("*/.gpu.lock"):
        if f.parent.resolve() == ROOT.resolve():
            continue
        if subprocess.run(["flock", "-n", "-s", str(f), "true"]).returncode != 0:
            busy.append(f"lock {f}")
    units = subprocess.run(["systemctl", "--user", "list-units", "--state=active,activating", "--no-legend", "--plain",
                            *FOREIGN_UNITS], capture_output=True, text=True).stdout
    # dtc-review-server is a CPU web server for reviews, not a GPU job
    busy += [f"unit {l.split()[0]}" for l in units.splitlines() if l.strip() and "review-server" not in l]
    if comfy.alive():
        q = comfy.requests.get(f"{comfy.BASE}/queue", timeout=5).json()
        if q.get("queue_running") or q.get("queue_pending"):
            busy.append("comfyui queue")
    return busy


def wait_gpu(poll: float = 60.0) -> None:
    """Wait (never stop anything) until no other project's GPU job is running."""
    while busy := gpu_busy_elsewhere():
        log(f"waiting for GPU: {busy}")
        time.sleep(poll)


# ------------------------------------------------------------------ generation
def cmd_gen():
    base = manifest.get(BASE)
    lyr = lyrics()
    for k, (seed, dur, temp, lm_cfg) in VARIANTS.items():
        aid = take_id(k)
        a = manifest.get(aid)
        if a and a.get("status") not in (None, "running", "failed"):
            continue
        wait_gpu()
        g = wf.ace15_song(f"song_take_04b_{k}", song_spec.CAPTION, lyr, seed=seed, duration=dur, bpm=song_spec.BPM,
                          keyscale=song_spec.KEYSCALE, variant="turbo", lm_cfg=lm_cfg, temperature=temp,
                          prefix=f"launch-window/song/take_04b_{k}")
        wfp = g.save(ROOT / "workflows" / "stage1")
        manifest.upsert(aid, stage="1", kind="song_take", prompt=song_spec.CAPTION, seed=seed, model=base["model"],
                        workflow=rel(wfp), status="running", attempts=(a or {}).get("attempts", 0) + 1,
                        meta={"variant": "turbo", "bpm": song_spec.BPM, "key": song_spec.KEYSCALE, "duration": dur,
                              "series": "04b", "base": BASE, "lm_temperature": temp, "lm_cfg": lm_cfg,
                              "lyrics_extra": OUTRO_TAG.strip()})
        try:
            res = comfy.run(g.api(), name=aid, timeout=1800)
            dest = comfy.download(comfy.output_files(res["outputs"])[0], take_path(k))
            manifest.upsert(aid, path=rel(dest), status="done", meta={"render_seconds": res["seconds"]})
            log(f"{aid}: seed {seed} {dur:.0f}s T={temp} cfg={lm_cfg}: {res['seconds']}s")
        except Exception as e:
            manifest.upsert(aid, status="failed", meta={"error": repr(e)[:2000]})
            log(f"{aid}: FAILED {e!r}")
    comfy.free()


# ------------------------------------------------------------------ similarity to take 04
SR = 22050
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def features(path) -> dict:
    """Timbre/harmony statistics (librosa): MFCC 1-20 mean+std, chroma mean, spectral-contrast mean, key, log-mel."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    ch = librosa.feature.chroma_cqt(y=y, sr=SR)
    sc = librosa.feature.spectral_contrast(y=y, sr=SR)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, hop_length=2205))  # 10 frames/s
    c = ch.mean(1)
    scores = [(np.corrcoef(np.roll(prof, i), c)[0, 1], f"{NOTES[i]} {mode}") for mode, prof in (("major", KK_MAJ), ("minor", KK_MIN))
              for i in range(12)]
    return {"timbre": np.concatenate([mf.mean(1), mf.std(1), sc.mean(1)]), "chroma": c / c.sum(),
            "key": max(scores)[1], "key_r": round(float(max(scores)[0]), 3), "mel": mel}


def same_scale(k1: str, k2: str) -> bool:
    """Relative keys share one pitch set (D major = B minor); Krumhansl profiles often confuse them."""
    def tonic_major(k):
        n, mode = k.split()
        i = NOTES.index(n)
        return i if mode == "major" else (i + 3) % 12
    return tonic_major(k1) == tonic_major(k2)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def aligned_track(mel_a, mel_b, win: int = 50) -> dict:
    """Frame-wise correlation of log-mel (5 s windows, no time warping): how long the new take follows take 04."""
    n = min(mel_a.shape[1], mel_b.shape[1])
    r = [float(np.corrcoef(mel_a[:, i:i + win].ravel(), mel_b[:, i:i + win].ravel())[0, 1]) for i in range(0, n - win, win)]
    follow = next((i * 5 for i, v in enumerate(r) if v < 0.8), len(r) * 5)
    return {"first_30s_r": round(float(np.mean(r[:6])), 3), "mean_r": round(float(np.mean(r)), 3), "follows_04_until_s": follow}


# ------------------------------------------------------------------ ending
def _frames_db(x, sr, hop):
    n = int(sr * hop)
    f = len(x) // n
    return 20 * np.log10(np.sqrt((x[:f * n].reshape(f, n, -1) ** 2).mean((1, 2))) + 1e-9)


def ending(path, vocals_path, words) -> dict:
    """Last 20 s of music: RMS curve, sung windows and words, final chord and how the sound stops."""
    x, sr = sf.read(str(path), always_2d=True)
    db = _frames_db(x, sr, 0.05)
    loud = np.percentile(db, 95)
    live = np.where(db > loud - 45)[0]
    music_end = (live[-1] + 1) * 0.05
    file_end = len(x) / sr
    t0 = max(0.0, music_end - 20)
    curve = [round(float(_frames_db(x[int(t * sr):int((t + 1) * sr)], sr, 1.0)[0] - loud), 1) for t in np.arange(t0, music_end - 0.5, 1.0)]
    # how the music stops: time for the level to fall from -12 dB (re the last-20 s peak) to -45 dB re the song
    seg = db[int(t0 / 0.05):int(music_end / 0.05) + 1]
    peak_last = float(seg.max())
    last_loud = int(t0 / 0.05) + int(np.where(seg > peak_last - 12)[0][-1])
    decay_s = round(music_end - last_loud * 0.05, 2)
    slope = float(np.polyfit(np.arange(len(curve[-10:])), curve[-10:], 1)[0]) if len(curve) >= 4 else 0.0
    max_drop_100ms = float(max(db[i] - db[i + 2] for i in range(int((music_end - 3) / 0.05), int(music_end / 0.05) - 2)))
    # vocals: separator stem vs the rest, 0.5 s windows (as song_extend), and WhisperX words inside the window
    v, vsr = sf.read(str(vocals_path), always_2d=True)
    n = min(len(v), len(x))
    i_db = _frames_db((x[:n] - v[:n])[int(t0 * sr):int(music_end * sr)], sr, 0.5)
    v_db = _frames_db(v[int(t0 * vsr):int(music_end * vsr)], vsr, 0.5)
    m = min(len(i_db), len(v_db))
    sung = [round(t0 + j * 0.5, 1) for j in range(m) if v_db[j] > loud - 40 and v_db[j] > i_db[j] - 12]
    words = [w for w in words if w.get("start") is not None and w["start"] < music_end]  # WhisperX invents words in silence
    tail_words = [w["word"] for w in words if w["start"] >= t0]
    last_word = max((w for w in words if w.get("end") is not None), key=lambda w: w["end"], default=None)
    # final chord: chroma over the last 3 s above -30 dB re peak; stable = high frame-to-mean similarity
    y, _ = librosa.load(str(path), sr=SR, mono=True, offset=max(0, music_end - 6), duration=6)
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    e = librosa.feature.rms(y=y, hop_length=512)[0]
    keep = (20 * np.log10(e + 1e-9) > 20 * np.log10(e.max() + 1e-9) - 30)
    chk = ch[:, keep] if keep.sum() > 4 else ch
    mean = chk.mean(1)
    stability = float(np.mean([_cos(chk[:, j], mean) for j in range(chk.shape[1])]))
    top = [NOTES[i] for i in np.argsort(mean)[::-1][:3]]
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time")
    held = round(float(6 - (onsets[-1] if len(onsets) else 0)), 1)  # seconds from the last attack to the end of the window
    tonic = {"D", "F#", "A"}
    natural = 0.0
    natural += 0.35 * min(1.0, decay_s / 4.0)                     # rings out / fades rather than stopping
    natural += 0.20 * (max_drop_100ms < 15)                       # no hard cut in the last 3 s
    natural += 0.15 * (music_end < file_end - 0.3)                 # ends before the file does
    natural += 0.15 * (len(tail_words) == 0 or (last_word and last_word["end"] < music_end - 8))  # instrumental tail
    natural += 0.15 * (len(tonic & set(top)) >= 2 and stability > 0.8)  # resolves on a held tonic chord
    kind = ("hard stop" if decay_s < 0.6 or max_drop_100ms >= 20 else
            "fade" if slope < -1.5 and decay_s >= 3 else "ring-out" if decay_s >= 1.5 else "short release")
    return {"music_end_s": round(music_end, 2), "after_last_word_s": round(music_end - last_word["end"], 1) if last_word else None, "file_end_s": round(file_end, 2), "decay_s": decay_s,
            "slope_db_per_s_last10": round(slope, 2), "max_drop_100ms_last3_db": round(max_drop_100ms, 1), "kind": kind,
            "rms_curve_db_re_p95": curve, "sung_windows_s": sung, "tail_words": " ".join(tail_words),
            "last_word": last_word and {"word": last_word["word"], "end": round(last_word["end"], 2)},
            "final_chord_top3": top, "chord_stability": round(stability, 3), "held_after_last_attack_s": held,
            "natural": round(natural, 3)}


def cmd_analyze():
    todo = [manifest.get(take_id(k)) for k in VARIANTS]
    todo = [a for a in todo if a and a["status"] == "done"]
    if todo:
        wait_gpu()
        stage1._analyze_assets(todo)  # the normal stage-1 checks and score
    takes = [manifest.get(take_id(k)) for k in VARIANTS]
    takes = [a for a in takes if a and a["status"] == "scored"]
    base = manifest.get(BASE)
    originals = [a for a in manifest.query(stage="1", kind="song_take") if a["id"].count(":") == 2 and a.get("path")]
    feats = {a["id"]: features(ROOT / a["path"]) for a in originals + takes}
    pop = np.stack([f["timbre"] for f in feats.values()])
    mu, sd = pop.mean(0), pop.std(0) + 1e-9
    z = {k: (f["timbre"] - mu) / sd for k, f in feats.items()}
    f04 = feats[BASE]
    ae04 = base["score"]["aesthetics"]

    def sim(aid, a):
        f = feats[aid]
        ae = a["score"]["aesthetics"]
        return {"timbre_cos": round(_cos(z[aid], z[BASE]), 3), "chroma_cos": round(_cos(f["chroma"], f04["chroma"]), 3),
                "aesthetics_dist": round(float(np.sqrt(sum((ae[k] - ae04[k]) ** 2 for k in ("CE", "CU", "PC", "PQ")))), 3),
                "key": f["key"], "key_r": f["key_r"], "same_scale_as_04": same_scale(f["key"], f04["key"]), "bpm": a["score"]["bpm"], **aligned_track(f04["mel"], f["mel"])}

    baseline = {a["id"]: sim(a["id"], a) for a in originals if a["id"] != BASE}
    res = {}
    for a in takes:
        s = sim(a["id"], a)
        words = json.load(open(ROOT / a["meta"]["transcript"]))["words"]
        e = ending(ROOT / a["path"], ROOT / a["meta"]["vocals"], words)
        # similarity 0..1: timbre cos (z-scored vs the 16 original takes), chroma, aesthetics, tempo and key match
        s["similarity"] = round(0.45 * max(0.0, s["timbre_cos"]) + 0.25 * max(0.0, (s["chroma_cos"] - 0.8) / 0.2)
                                + 0.15 * max(0.0, 1 - s["aesthetics_dist"] / 2)
                                + 0.10 * (abs(s["bpm"] - base["score"]["bpm"]) < 2) + 0.05 * s["same_scale_as_04"], 3)
        rank = round(0.4 * s["similarity"] + 0.4 * e["natural"] + 0.2 * a["score"]["total"], 4)
        outro = next(x["recall"] for x in a["meta"]["sections"] if x["section"] == "Outro")
        eligible = bool(outro is not None and outro >= 0.75)  # the last line must be sung before the instrumental outro
        res[a["id"]] = {"similarity": s, "ending": e, "stage1_total": a["score"]["total"], "rank_score": rank,
                        "outro_line_recall": outro, "eligible": eligible}
        manifest.upsert(a["id"], meta={"vs_04": s, "ending": e, "rank_score": rank, "eligible": eligible})
        log(f"{a['id']}: stage1 {a['score']['total']:.3f} sim {s['similarity']} ending {e['natural']} ({e['kind']}) rank {rank}")
    e04 = ending(ROOT / base["path"], ROOT / base["meta"]["vocals"], json.load(open(ROOT / base["meta"]["transcript"]))["words"])
    QC.mkdir(parents=True, exist_ok=True)
    out = {"take04": {"key": f04["key"], "bpm": base["score"]["bpm"], "aesthetics": ae04, "ending": e04},
           "baseline_other_seeds": baseline, "takes": res}
    (QC / "analysis.json").write_text(json.dumps(out, indent=1))
    manifest.upsert("song:take:04b:analysis", stage="1", kind="qc", path=rel(QC / "analysis.json"), prompt=None, seed=None,
                    model="librosa + stage1 tools", workflow="pipeline/song_regen.py analyze", score=None, status="done")
    return out


# ------------------------------------------------------------------ listen copies
def cmd_listen(n: int = 3):
    res = json.load(open(QC / "analysis.json"))["takes"]
    top = sorted(res, key=lambda k: (not res[k]["eligible"], -res[k]["rank_score"]))[:n]
    norm = DIR / "norm"
    norm.mkdir(parents=True, exist_ok=True)
    for i, aid in enumerate(top, 1):
        a = manifest.get(aid)
        x, sr = sf.read(ROOT / a["path"], always_2d=True, dtype="float64")
        y = master.to_target(x, sr, lufs=-14.0, ceiling_db=-1.3)  # stage 4 / smoke-mix target
        flac = norm / f"take04b_{i}.flac"
        sf.write(flac, y, sr, subtype="PCM_24")
        m = master.measure(flac)
        manifest.upsert(f"song:take:04b:norm:{i}", stage="1", kind="song_norm", path=rel(flac), prompt=None, seed=a["seed"],
                        model="pipeline/master.py to_target", workflow="pipeline/song_regen.py listen", status="done",
                        score={"lufs": m["lufs"], "true_peak": m["true_peak_dbtp"], "lra": m["lra"]}, meta={"source": aid, "rank": i})
        for d in LISTEN:
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(flac), "-codec:a", "libmp3lame", "-b:a", "320k",
                            str(d / f"take04b_{i}.mp3")], check=True)
        manifest.upsert(f"listen:take04b_{i}", stage="1", kind="listen_mp3", path=rel(LISTEN[0] / f"take04b_{i}.mp3"), prompt=None,
                        seed=a["seed"], model="ffmpeg libmp3lame 320k", workflow="pipeline/song_regen.py listen", score=None,
                        status="done", meta={"source": aid, "rank": i, "copies": [str(d / f"take04b_{i}.mp3") for d in LISTEN]})
        log(f"take04b_{i}.mp3 <- {aid} {m}")
    return top


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "gen":
        cmd_gen()
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "listen":
        cmd_listen()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
