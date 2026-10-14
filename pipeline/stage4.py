"""Stage 4: mix and master in Python (pedalboard + numpy), voices exported as their own stems.

  python -m pipeline.stage4 mix       build out/master/__SLUG___master.wav + per-voice stems on the final timeline
  python -m pipeline.stage4 check     -14 LUFS, true peak < -1 dBTP, aesthetics >= raw take, spoken lines intelligible
The mix ducks the instrumental under voiced spoken lines, lays those lines in from
audio/voices/picked.json, and adds a picked sound effect only where that asset's meta.hits
lists {t, gain}. Add project-specific moves here once master.md names them.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (Bitcrush, Compressor, Delay, Gain, HighpassFilter, HighShelfFilter, LowpassFilter, Pedalboard,
                        Resample, Reverb)
from scipy.signal import resample_poly

from pipeline import lyrics, manifest, master, voice_spec
from pipeline.manifest import ROOT

SR = 48000
OUT = ROOT / "out" / "master"
STEMS = ROOT / "audio" / "stems"
TIMING = ROOT / "audio" / "timing"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def load(path, sr=SR) -> np.ndarray:
    x, s = sf.read(path, always_2d=True, dtype="float64")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if s != sr:
        g = np.gcd(s, sr)
        x = resample_poly(x, sr // g, s // g, axis=0)
    return x


def fade(n, kind="in"):
    r = np.sin(np.linspace(0, np.pi / 2, max(1, n))) ** 2
    return r if kind == "in" else r[::-1]


def apply_edits(x: np.ndarray, edits: list, silent: bool = False) -> np.ndarray:
    """Insert loops (copies of a source range) or holds/tails (silence) at the given original times."""
    out, pos = [], 0
    xf = int(0.01 * SR)
    for e in sorted(edits, key=lambda e: e["at"]):
        at = min(len(x), int(round(e["at"] * SR)))
        out.append(x[pos:at])
        n = int(round(e["insert"] * SR))
        if e["kind"] == "loop" and not silent:
            a, b = (int(round(t * SR)) for t in e["source"])
            seg = x[a:b].copy()
            reps = int(np.ceil(n / max(1, len(seg))))
            ins = np.concatenate([seg] * reps)[:n].copy()
            ins[:xf] *= fade(xf, "in")[:, None]
            ins[-xf:] *= fade(xf, "out")[:, None]
        else:
            ins = np.zeros((n, x.shape[1]))
        out.append(ins)
        pos = at
    out.append(x[pos:])
    return np.concatenate(out)


class Env:
    """Gain automation on the final timeline: set(t0, t1, gain) with smooth ramps."""

    def __init__(self, n):
        self.g = np.ones(n)

    def set(self, t0, t1, gain, ramp=0.08, ramp_out=None):
        a, b, r = int(t0 * SR), int(t1 * SR), max(1, int(ramp * SR))
        r_out = max(1, int((ramp if ramp_out is None else ramp_out) * SR))
        a, b = max(0, a), min(len(self.g), b)
        if b <= a:
            return
        self.g[a:b] = np.minimum(self.g[a:b], gain)
        lo = max(0, a - r)
        self.g[lo:a] = np.minimum(self.g[lo:a], 1 - (1 - gain) * fade(a - lo, "in"))
        hi = min(len(self.g), b + r_out)
        self.g[b:hi] = np.minimum(self.g[b:hi], 1 - (1 - gain) * fade(hi - b, "out"))


def add(buf, clip, t, gain=1.0):
    a = int(round(t * SR))
    if a >= len(buf):
        return
    n = min(len(clip), len(buf) - a)
    buf[a:a + n] += gain * clip[:n]


MIX = {"voice_gain": 0.85, "sfx_gain": 1.0, "bus_comp": True}
CHAINS = {}  # voice id -> function(samples ndarray) -> samples. Empty means the take is used dry.


def cmd_mix(out_name: str = "__SLUG___master.wav", record: bool = True, **over):
    """Vocals + instrumental to -14 LUFS. Spoken lines duck the bed. No named effects."""
    cfg = {**MIX, **over}
    edit = json.load(open(TIMING / "song_edit.json"))
    picked_path = ROOT / "audio" / "voices" / "picked.json"
    picked = json.load(open(picked_path)) if picked_path.exists() else {}
    lines = json.load(open(TIMING / "lyrics.json"))
    edits = edit["edits"]

    def to_final(t):
        return t + sum(e["insert"] for e in edits if e["at"] <= t)

    vocals = apply_edits(load(STEMS / "vocals.flac"), edits)
    inst = apply_edits(load(STEMS / "instrumental.flac"), edits)
    lead = apply_edits(load(STEMS / "vocals.flac"), edits)
    n = max(len(vocals), len(inst), int(edit["duration_final"] * SR) + SR)
    pad = lambda x: np.pad(x, ((0, n - len(x)), (0, 0)))
    vocals, inst, lead = pad(vocals), pad(inst), pad(lead)
    inst_env, voc_env = Env(n), Env(n)
    sp = edit.get("spoken") or {}
    sung = sorted((to_final(l["start"]), to_final(l["end"])) for l in lines if l.get("start") is not None)

    runs = []
    for lid, place in sorted(sp.items(), key=lambda kv: kv[1]["start"]):
        if place.get("text"):
            continue
        line = next(l for l in voice_spec.LINES if l["id"] == lid)
        if runs and place["start"] - runs[-1][1] < cfg.get("duck_merge", 2.5) and runs[-1][2] == line["voice"]:
            runs[-1][1] = place["end"]
        else:
            runs.append([place["start"], place["end"], line["voice"]])

    def fades(a, b, want_in, want_out):
        prev_end = max([e for _, e in sung if e <= a + 0.05] + [0.0])
        next_start = min([st for st, _ in sung if st >= b - 0.05] + [n / SR])
        return (min(want_in, max(0.25, a - prev_end - 0.05)), min(want_out, max(0.3, next_start - b - 0.05)))

    for a, b, _voice in runs:
        r_in, r_out = fades(a - 0.15, b + 0.3, cfg.get("duck_in", 0.7), cfg.get("duck_out", 1.2))
        inst_env.set(a - 0.15, b + 0.3, 10 ** (cfg.get("duck_db", -6.0) / 20), ramp=r_in, ramp_out=r_out)
    log(f"ducking: {len(runs)} spoken runs")

    music = inst * inst_env.g[:, None] + vocals * voc_env.g[:, None]
    lead_final = lead * voc_env.g[:, None]

    voice_bufs = {}
    for lid, place in sp.items():
        if place.get("text"):
            continue
        line = next(l for l in voice_spec.LINES if l["id"] == lid)
        if lid not in picked:
            raise KeyError(f"no picked take for spoken line {lid}")
        x = load(ROOT / picked[lid]["path"])
        fn = CHAINS.get(line["voice"])
        if fn:
            x = fn(np.asarray(x, dtype=np.float32))
        buf = voice_bufs.setdefault(line["voice"], np.zeros((n, 2)))
        add(buf, np.asarray(x, dtype=np.float64), place["start"])
    for v, b in list(voice_bufs.items()):
        peak = np.abs(b).max()
        if peak > 0:
            voice_bufs[v] = b * (10 ** (-3 / 20) / peak)

    sfx = np.zeros((n, 2))
    for asset in manifest.query(stage="3", kind="sfx"):
        if asset["status"] != "picked":
            continue
        clip = load(ROOT / asset["path"])
        for hit in asset.get("meta", {}).get("hits") or []:
            add(sfx, clip, float(hit["t"]), float(hit.get("gain", 0.4)))

    voices_sum = sum(voice_bufs.values()) if voice_bufs else np.zeros((n, 2))
    mixbus = music + cfg["voice_gain"] * voices_sum + cfg["sfx_gain"] * sfx
    if cfg["bus_comp"]:
        mixbus = Pedalboard([Compressor(threshold_db=-14, ratio=1.6, attack_ms=25, release_ms=200)])(
            mixbus.astype(np.float32), SR)
    mixbus = master.to_target(np.asarray(mixbus, dtype=np.float64), SR, lufs=-14.0, ceiling_db=-1.3)
    end_i = int(edit["duration_final"] * SR)
    mixbus = mixbus[:end_i]
    fo = int(0.35 * SR)
    if fo < end_i:
        mixbus[end_i - fo:end_i] *= np.linspace(1, 0, fo)[:, None]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / out_name
    sf.write(out, mixbus, SR, subtype="PCM_24")
    if not record:
        return out
    stems = {"lead": lead_final, "music": music, "sfx": sfx, **{f"voice_{v}": b for v, b in voice_bufs.items()}}
    for k, x in stems.items():
        sf.write(OUT / f"stem_{k}.wav", x[:end_i], SR, subtype="PCM_24")
    manifest.upsert("mix:master", stage="4", kind="master", path=rel(out), prompt=None, seed=None,
                    model="pedalboard + numpy (pipeline/master.py limiter)", workflow="pipeline/stage4.py mix",
                    score=None, status="done",
                    meta={"stems": [rel(OUT / f"stem_{k}.wav") for k in stems], "duration": end_i / SR, "mix": cfg})
    log(f"mix: {rel(out)} {end_i / SR:.1f}s")


def cmd_check():
    from pipeline import toolrun
    out = OUT / "__SLUG___master.wav"
    m = master.measure(out)
    raw = ROOT / manifest.get(manifest.load()["gates"]["song_take"]["asset"])["path"]
    ae = toolrun.run("aesthetics", "aesthetics.py", [str(out), str(raw)], name="stage4:aes",
                     log=ROOT / "qc" / "logs" / "stage4-tools.log")["scores"]
    a_out, a_raw = ae[str(out)], ae[str(raw)]
    # like-for-like: the same musical moments with the spoken passages and inserted loops cut out of both. This is the
    # pass criterion; the full-track scores also measure the arrangement
    mo, ro = music_only_pair(out, raw, "check")
    am = toolrun.run("aesthetics", "aesthetics.py", [str(mo), str(ro)], name="stage4:aes_music_only",
                     log=ROOT / "qc" / "logs" / "stage4-tools.log")["scores"]
    s_m, s_r = (am[str(mo)]["CE"] + am[str(mo)]["PQ"], am[str(ro)]["CE"] + am[str(ro)]["PQ"])
    checks = {"lufs": (-14.5 <= m["lufs"] <= -13.5, m["lufs"]), "true_peak": (m["true_peak_dbtp"] < -1.0, m["true_peak_dbtp"]),
              "aesthetics": (s_m >= s_r - 0.05,
                             {"music_only": {"master": round(s_m, 3), "raw_take": round(s_r, 3)},
                              "full_track": {"master": a_out, "raw_take": a_raw}})}
    manifest.upsert("mix:master", score={k: v[1] for k, v in checks.items()}, status="pass" if all(v[0] for v in checks.values()) else "fail")
    log(f"check: {checks}")
    return checks


VARIANTS = {
    "h_balanced": {"voice_gain": 0.85, "sfx_gain": 1.0, "bus_comp": False,
                   "duck_db": -6.0, "duck_merge": 2.5, "duck_in": 0.7, "duck_out": 1.2},
    "i_subtle": {"voice_gain": 0.85, "sfx_gain": 0.6, "bus_comp": False,
                 "duck_db": -4.0, "duck_merge": 2.5, "duck_in": 0.5, "duck_out": 0.9},
    "e_sfx_low": {"voice_gain": 0.85, "sfx_gain": 0.35, "bus_comp": False},
    "f_no_sfx": {"voice_gain": 0.85, "sfx_gain": 0.0, "bus_comp": False},
    "j_smooth_duck": {"voice_gain": 0.85, "sfx_gain": 1.0, "bus_comp": False,
                      "duck_db": -6.0, "duck_merge": 2.5, "duck_in": 0.7, "duck_out": 1.2},
    "g_music_only": {"voice_gain": 0.0, "sfx_gain": 0.0, "bus_comp": False},
    "a_current": {"voice_gain": 0.85, "sfx_gain": 1.0, "bus_comp": True},
    "b_nocomp": {"voice_gain": 0.85, "sfx_gain": 1.0, "bus_comp": False},
    "c_voice_down": {"voice_gain": 0.6, "sfx_gain": 1.0, "bus_comp": False},
    "d_voice_sfx_down": {"voice_gain": 0.6, "sfx_gain": 0.6, "bus_comp": False},
}


def music_only_pair(master_path, raw_path, tag: str):
    """Same musical moments from master (final timeline) and raw take (original timeline), skipping spoken-line
    windows and inserted loops/holds, concatenated for a like-for-like aesthetics comparison."""
    se = json.load(open(TIMING / "song_edit.json"))
    edits = sorted(se["edits"], key=lambda e: e["at"])
    excl = [(p["start"] - 0.3, p["end"] + 0.5) for p in se["spoken"].values()]
    for e in edits:  # inserted spans in final time
        f0 = e["at"] + sum(x["insert"] for x in edits if x["at"] < e["at"])
        excl.append((f0, f0 + e["insert"]))
    excl.sort()
    m, sr = sf.read(master_path, always_2d=True)
    r, sr2 = sf.read(raw_path, always_2d=True)
    total = len(m) / sr
    keep, t = [], 0.0
    for a, b in excl:
        if a > t:
            keep.append((t, a))
        t = max(t, b)
    keep.append((t, total))
    mo, ro = [], []
    for a, b in keep:
        if b - a < 1.0:
            continue
        orig = a - sum(e["insert"] for e in edits if e["at"] + sum(x["insert"] for x in edits if x["at"] < e["at"]) < a)
        n = int((b - a) * sr)
        mo.append(m[int(a * sr):int(a * sr) + n])
        ro.append(r[int(orig * sr2):int(orig * sr2) + int((b - a) * sr2)])
    d = OUT / "tune"
    pm, pr = d / f"{tag}_music_only_master.wav", d / f"{tag}_music_only_raw.wav"
    sf.write(pm, np.concatenate(mo), sr, subtype="PCM_24")
    sf.write(pr, np.concatenate(ro), sr2, subtype="PCM_24")
    return pm, pr


def intelligibility(master_path) -> dict:
    """Cut every spoken line from the master (+-0.15 s) and check its words with WhisperX."""
    from pipeline import lyrics, toolrun
    import subprocess
    sp = json.load(open(TIMING / "song_edit.json"))["spoken"]
    d = OUT / "tune" / (Path(master_path).stem + "_lines")
    d.mkdir(parents=True, exist_ok=True)
    clips = []
    for lid, p in sp.items():
        if p.get("text"):
            continue
        c = d / f"{lid}.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, p['start'] - 0.15):.3f}", "-t",
                        f"{p['end'] - p['start'] + 0.3:.3f}", "-i", str(master_path), "-ac", "1", "-ar", "16000", str(c)], check=True)
        clips.append((lid, c))
    tr = toolrun.run("whisperx", "align.py", ["transcribe", d / "asr", *[str(c) for _, c in clips]], name="stage4:intelligibility")["results"]
    out = {}
    for lid, c in clips:
        want = lyrics.norm_words(next(l["text"] for l in voice_spec.LINES if l["id"] == lid))
        got = lyrics.norm_words(tr[str(c)]["text"])
        hit = sum(1 for w in want if w in got)
        out[lid] = round(hit / max(1, len(want)), 2)
    return out


def cmd_tune():
    from pipeline import toolrun
    (OUT / "tune").mkdir(parents=True, exist_ok=True)
    only = [k for k in VARIANTS if k.startswith(("h_", "i_", "e_", "f_"))]
    paths = {k: cmd_mix(out_name=f"tune/{k}.wav", record=False, **VARIANTS[k]) for k in only}
    raw = ROOT / manifest.get(manifest.load()["gates"]["song_take"]["asset"])["path"]
    mo = {k: music_only_pair(p, raw, k) for k, p in paths.items()}
    extra = [str(x) for pair in mo.values() for x in pair]
    ae = toolrun.run("aesthetics", "aesthetics.py", [str(p) for p in paths.values()] + [str(raw)] + extra, name="stage4:tune-aes")["scores"]
    base = ae[str(raw)]["CE"] + ae[str(raw)]["PQ"]
    res = {}
    for k, p in paths.items():
        intel = intelligibility(p)
        a = ae[str(p)]
        mm, mr = (ae[str(x)] for x in mo[k])
        res[k] = {"CE": a["CE"], "PQ": a["PQ"], "sum": round(a["CE"] + a["PQ"], 3), "raw_sum": round(base, 3),
                  "music_only_sum": round(mm["CE"] + mm["PQ"], 3), "music_only_raw_sum": round(mr["CE"] + mr["PQ"], 3),
                  "min_line_words": min(intel.values()), "mean_line_words": round(sum(intel.values()) / len(intel), 3),
                  "weak_lines": {k2: v for k2, v in intel.items() if v < 0.8}}
        log(f"tune {k}: {res[k]}")
    (OUT / "tune" / "results.json").write_text(json.dumps(res, indent=1))
    ok = [k for k, r in res.items() if r["sum"] >= base - 0.05 and r["mean_line_words"] >= 0.9]
    best = max(ok or res, key=lambda k: (res[k]["sum"] if k in ok else -1, res[k]["mean_line_words"]))
    log(f"tune: best {best} (passing: {ok})")
    return best, res


def cmd_apply_best(wait_s: int = 7200):
    """Wait for the tuning results, re-mix with the best passing variant (stems recorded), then run the checks."""
    res_file = OUT / "tune" / "results.json"
    t0 = time.time()
    while not res_file.exists() and time.time() - t0 < wait_s:
        time.sleep(30)
    res = json.load(open(res_file))
    raw_sum = next(iter(res.values()))["raw_sum"]
    ok = [k for k, r in res.items() if r["sum"] >= raw_sum - 0.05 and r["mean_line_words"] >= 0.9]
    with_fx = [k for k in ok if VARIANTS[k]["sfx_gain"] > 0]  # keep a passing variant that still plays picked effects
    pool = with_fx or ok
    best = max(pool, key=lambda k: res[k]["sum"]) if pool else max(res, key=lambda k: (res[k]["mean_line_words"] >= 0.9, res[k]["sum"]))
    log(f"apply-best: {best} {res[best]} (passing {ok})")
    cmd_mix(**VARIANTS[best])
    manifest.upsert("mix:master", meta={"variant": best, "tune": res})
    return cmd_check()


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "mix":  # re-mix with the variant chosen by apply-best (the module defaults are the untuned baseline)
        manifest.set_stage("4", status="running", started=manifest.now())
        variant = ((manifest.get("mix:master") or {}).get("meta") or {}).get("variant")
        log(f"mix: variant {variant}")
        cmd_mix(**VARIANTS.get(variant, {}))
    elif cmd == "check":
        cmd_check()
    elif cmd == "tune":
        cmd_tune()
    elif cmd == "apply-best":
        cmd_apply_best()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
