"""Kick/snare onset detection and a beat grid for the edit (CPU, librosa).

Kicks come from a low-band (30-150 Hz) mel onset envelope, snares from a mid-band (1.2-5 kHz) one, each peak-picked
with an adaptive threshold. Beats/downbeats come from the pipeline's beat_this output (audio/timing/beats.json) when given;
otherwise librosa's beat tracker plus a downbeat phase chosen so snares land on beats 2 and 4. Run it on the drum stem
(audio/stems/drums.flac) for clean hits; the full mix works for drafts.

  python -m slopkit.beatgrid detect <audio> <out_hits.json> [--beats audio/timing/beats.json]
Output: {bpm, beats, downbeats, kicks, snares, kick_strength, snare_strength, source}
"""
import json
import sys

import numpy as np

HOP = 256
N_FFT = 2048


def _pick(env, sr, delta, min_gap=0.09):
    import librosa
    wait = max(1, int(min_gap * sr / HOP))
    return librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=14, post_avg=14, delta=delta, wait=wait)


def _band_onset(y, sr, fmin, fmax):
    """librosa onset envelope restricted to one mel band (librosa centres it, so peaks sit on the attacks)."""
    import librosa
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=8 if fmax <= 400 else 24, fmin=fmin, fmax=fmax)
    e = librosa.onset.onset_strength(S=librosa.power_to_db(M, ref=np.max), sr=sr, hop_length=HOP)
    return e / (np.percentile(e, 99.5) + 1e-9)


def _refine(y, sr, times, band, before=0.06, after=0.02, frac=0.3):
    """Move each onset to where the band-passed envelope first reaches frac x its local peak (attack start); the
    onset envelope peaks lag slow attacks (filtered snare noise) by 20-30 ms."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = max(20.0, band[0]), min(band[1], sr / 2 - 100)
    env = np.abs(sosfiltfilt(butter(4, [lo, hi], "band", fs=sr, output="sos"), y))
    k = max(1, int(0.002 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    out = []
    for x in times:
        a, b = max(0, int((x - before) * sr)), min(len(env), int((x + after) * sr))
        seg = env[a:b]
        if len(seg) < 8:
            out.append(x)
            continue
        pk = int(np.argmax(seg))
        floor = np.min(seg[: pk + 1]) if pk > 0 else 0.0
        hit = np.nonzero(seg[: pk + 1] >= floor + frac * (seg[pk] - floor))[0]
        out.append((a + (hit[0] if len(hit) else pk)) / sr)
    return np.array(out)


def detect(audio, beats_json=None, sr=22050, kick_band=(30, 150), snare_band=(1200, 5000), delta=0.14,
           rel_strength=0.45) -> dict:
    import librosa
    y, sr = librosa.load(str(audio), sr=sr, mono=True)
    ek, es = _band_onset(y, sr, *kick_band), _band_onset(y, sr, *snare_band)
    def keep(env, idx):  # adaptive: a hit must reach rel_strength x the typical strong hit (90th pct of picks)
        idx = np.asarray(idx, int)
        if not len(idx):
            return idx
        ref = np.percentile(env[idx], 90)
        return idx[env[idx] >= rel_strength * ref]
    ki = keep(ek, _pick(ek, sr, delta))
    si = keep(es, _pick(es, sr, delta))
    # (no kick/snare exclusion: in four-on-the-floor every snare sits on a kick; the relative threshold keeps the
    # kick's weak mid-band click out)
    t = lambda idx: librosa.frames_to_time(np.asarray(idx, int), sr=sr, hop_length=HOP)
    kicks, snares = _refine(y, sr, t(ki), kick_band), _refine(y, sr, t(si), snare_band)
    if beats_json:
        b = json.load(open(beats_json))
        beats, downbeats, bpm, src = np.array(b["beats"]), np.array(b["downbeats"]), b.get("bpm"), "beat_this"
    else:
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        tempo, bf = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP, units="frames", start_bpm=126)
        beats = t(bf)
        bpm = float(np.atleast_1d(tempo)[0])
        if bpm < 95 and len(beats) > 1:  # octave error (half tempo): put a beat between each pair
            beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
            bpm *= 2
        # downbeat phase: snares on beats 2 and 4 (backbeat), so pick the phase whose beats 2/4 hold the most snares
        def near(ts, x, tol=0.07):
            return len(ts) and np.min(np.abs(ts - x)) < tol
        scores = [sum(near(snares, beats[j]) for j in range(len(beats)) if (j - ph) % 4 in (1, 3)) for ph in range(4)]
        ph = int(np.argmax(scores))
        downbeats = beats[ph::4]
        src = "librosa"
    return {"bpm": round(float(bpm), 2) if bpm else None, "beats": np.round(beats, 4).tolist(),
            "downbeats": np.round(downbeats, 4).tolist(), "kicks": np.round(kicks, 4).tolist(),
            "snares": np.round(snares, 4).tolist(), "kick_strength": np.round(ek[ki], 3).tolist(),
            "snare_strength": np.round(es[si], 3).tolist(), "source": src, "audio": str(audio)}


def f_measure(found, truth, tol=0.03) -> dict:
    found, truth = list(found), list(truth)
    used, tp = set(), 0
    for x in truth:
        best = min(((abs(f - x), j) for j, f in enumerate(found) if j not in used), default=(9, None))
        if best[0] <= tol:
            used.add(best[1])
            tp += 1
    p = tp / max(1, len(found))
    r = tp / max(1, len(truth))
    return {"precision": round(p, 3), "recall": round(r, 3), "f": round(2 * p * r / max(1e-9, p + r), 3), "n_found": len(found),
            "n_truth": len(truth)}


def synth_drums(path, bpm=132.0, bars=8, sr=44100, seed=3) -> dict:
    """Test loop: kick on every beat, snare on 2 and 4, closed hats on 8ths, a saw bass. Returns the true hit times."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int((bars * 4 * beat + 1.0) * sr)
    y = np.zeros(n)
    kicks, snares = [], []

    def add(sig, t):
        i = int(t * sr)
        j = min(n, i + len(sig))
        y[i:j] += sig[: j - i]

    tk = np.arange(int(0.35 * sr)) / sr
    kick = np.sin(2 * np.pi * (48 * tk + (110 / 18) * (1 - np.exp(-18 * tk)))) * np.exp(-tk * 9) * 0.9
    ts = np.arange(int(0.2 * sr)) / sr
    from scipy.signal import butter, lfilter
    bb, ab = butter(2, [1500 / (sr / 2), 7000 / (sr / 2)], "band")
    snare = (lfilter(bb, ab, rng.standard_normal(len(ts))) * 0.9 + 0.35 * np.sin(2 * np.pi * 190 * ts)) * np.exp(-ts * 22)
    th = np.arange(int(0.05 * sr)) / sr
    bh, ah = butter(2, 8000 / (sr / 2), "high")
    hat = lfilter(bh, ah, rng.standard_normal(len(th))) * np.exp(-th * 90) * 0.25
    for k in range(bars * 4):
        t = 0.5 + k * beat
        add(kick, t)
        kicks.append(t)
        if k % 4 in (1, 3):
            add(snare, t)
            snares.append(t)
        add(hat, t + beat / 2)
    tt = np.arange(n) / sr
    bass = 0.12 * (2 * ((tt * 55) % 1) - 1) * (tt > 0.5)
    y = np.tanh((y + bass) * 1.2) * 0.8
    sf.write(str(path), y, sr)
    return {"bpm": bpm, "kicks": kicks, "snares": snares, "beats": kicks, "downbeats": kicks[::4]}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "detect":
        bj = a[a.index("--beats") + 1] if "--beats" in a else None
        r = detect(a[1], bj)
        json.dump(r, open(a[2], "w"), indent=1)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}))
