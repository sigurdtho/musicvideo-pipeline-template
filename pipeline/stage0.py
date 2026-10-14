"""Stage 0 preflight: smoke-test every workflow on this card and log time and VRAM peak.

Usage: envs/pipeline/bin/python -m pipeline.stage0 smoke <test> [<test> ...] | all
Results: manifest assets `smoke:<test>` and qc/stage0_results.json (read by the schedule writer).
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from pipeline import comfy, manifest, song_spec, wf
from pipeline.manifest import ROOT

OUT = ROOT / "qc" / "stage0"
RESULTS = ROOT / "qc" / "stage0_results.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(test, **res):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[test] = {**data.get(test, {}), **res, "at": manifest.now()}
    RESULTS.write_text(json.dumps(data, indent=1))


def run_graph(test, g, timeout=3600, outdir=None, prompt=None, seed=None, model=None, warm_g=None):
    """Run g cold; if warm_g (same graph, new seed) is given, run it right after and record the warm time too."""
    wf_path = g.save(ROOT / "workflows" / "smoke")
    log(f"{test}: queue {g.name}")
    res = comfy.run(g.api(), timeout=timeout, name=f"smoke:{test}")
    warm = comfy.run(warm_g.api(), timeout=timeout, name=f"smoke:{test}:warm") if warm_g is not None else None
    files = []
    for i, item in enumerate(comfy.output_files(res["outputs"])):
        dest = (outdir or OUT / test) / item["filename"]
        files.append(str(comfy.download(item, dest)))
    out = {"ok": True, "seconds": res["seconds"], "vram_peak_mib": res["vram_peak_mib"],
           "vram_baseline_mib": res["vram_baseline_mib"], "files": files, "workflow": str(wf_path.relative_to(ROOT))}
    if warm:
        out["seconds_warm"] = warm["seconds"]
    manifest.upsert(f"smoke:{test}", stage="0", kind="smoke", path=files[0] if files else None, prompt=prompt, seed=seed,
                    model=model, workflow=out["workflow"], score=None, status="pass",
                    meta={k: out[k] for k in ("seconds", "vram_peak_mib", "vram_baseline_mib", "files")})
    warm_note = f" (warm {warm['seconds']}s)" if warm else ""
    log(f"{test}: ok {res['seconds']}s{warm_note}, VRAM peak {res['vram_peak_mib']} MiB (baseline {res['vram_baseline_mib']})")
    return out


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames",
                        "-of", "json", str(path)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


# ------------------------------------------------------------------ tests
def t_ace_turbo():
    g = wf.ace15_song("smoke_ace15_xl_turbo", song_spec.CAPTION, song_spec.generator_lyrics(), seed=1001,
                      duration=song_spec.DURATION, bpm=song_spec.BPM, keyscale=song_spec.KEYSCALE, variant="turbo",
                      prefix="__SLUG__/smoke/ace_turbo")
    r = run_graph("ace_turbo", g, prompt=song_spec.CAPTION, seed=1001, model="acestep_v1.5_xl_turbo_bf16")
    r["probe"] = ffprobe(r["files"][0])
    return r


def t_ace_sft():
    g = wf.ace15_song("smoke_ace15_xl_sft", song_spec.CAPTION, song_spec.generator_lyrics(), seed=1001,
                      duration=song_spec.DURATION, bpm=song_spec.BPM, keyscale=song_spec.KEYSCALE, variant="sft",
                      prefix="__SLUG__/smoke/ace_sft")
    r = run_graph("ace_sft", g, prompt=song_spec.CAPTION, seed=1001, model="acestep_v1.5_xl_sft_bf16")
    r["probe"] = ffprobe(r["files"][0])
    return r


SFX_PROMPT = "a single soft knock on hollow wood, close, dry, no music"


def t_stable_audio():
    g = wf.stable_audio("smoke_stable_audio", SFX_PROMPT, seed=7, seconds=6.0, negative="music, speech, noise",
                        prefix="__SLUG__/smoke/sfx")
    return run_graph("stable_audio", g, prompt=SFX_PROMPT, seed=7, model="stable-audio-open-1.0")


# A nameless look test for the stills workflow. Not a character from any film.
STYLE = ("Cinematic music-video still, soft window light, muted grey and warm wood, subtle film grain, "
         "shallow depth of field, highly detailed.")
PERSON = ("A person in their thirties, average build, calm face, short dark hair, a plain grey shirt, standing.")
ROOM_SHOT = ("Standing in a plain room with one tall window, grey walls, a wooden floor, and soft daylight.")


def t_klein_t2i():
    prompt = f"{STYLE} {ROOM_SHOT} {PERSON}"
    g = wf.klein("smoke_klein9b_t2i", prompt, seed=42, width=1920, height=1088, prefix="__SLUG__/smoke/klein_t2i")
    return run_graph("klein_t2i", g, prompt=prompt, seed=42, model="flux-2-klein-9b-kv-fp8")


def _first_image(test):
    data = json.loads(RESULTS.read_text())
    return [f for f in data[test]["files"] if f.endswith(".png")][0]


def t_klein_portrait():
    prompt = (f"{STYLE} Full-body portrait on a plain light-grey studio background, standing, facing camera, "
              f"neutral pose. {PERSON}")
    g = wf.klein("smoke_klein9b_portrait", prompt, seed=43, width=832, height=1216, prefix="__SLUG__/smoke/klein_portrait")
    return run_graph("klein_portrait", g, prompt=prompt, seed=43, model="flux-2-klein-9b-kv-fp8")


def t_charsheet():
    from PIL import Image
    src = Image.open(_first_image("klein_portrait")).convert("RGB")
    canvas = Image.new("RGB", (1536, 1024), (255, 255, 255))
    src.thumbnail((1536, 1024), Image.LANCZOS)
    canvas.paste(src, ((1536 - src.width) // 2, (1024 - src.height) // 2))
    p = OUT / "charsheet" / "ref_padded.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(p)
    ref = comfy.upload(p)
    g = wf.charsheet("smoke_charsheet_quad", ref, seed=44, prefix="__SLUG__/smoke/charsheet")
    return run_graph("charsheet", g, prompt=wf.CHARSHEET_QUAD_PROMPT, seed=44, model="flux-2-klein-9b-kv-fp8+QuadView_klein9b_v1")


def t_klein_edit():
    sheet = comfy.upload(_first_image("charsheet"))
    prompt = (f"{STYLE} The man from the character sheet, same face, hair and clothes. {OFFICE_SHOT} "
              "Medium shot, he turns from the window toward camera.")
    g = wf.klein("smoke_klein9b_edit", prompt, seed=45, width=1920, height=1088, refs=[sheet], prefix="__SLUG__/smoke/klein_edit")
    return run_graph("klein_edit", g, prompt=prompt, seed=45, model="flux-2-klein-9b-kv-fp8")


MOTION = ("He turns slowly from the window toward camera, singing; papers lift off the desks in a rising wind; "
          "fluorescent tubes flicker and burst one by one.")


def t_wan22_i2v():
    img = comfy.upload(_first_image("klein_edit"))
    g = wf.wan22_i2v("smoke_wan22_i2v_720p", img, MOTION, seed=46, width=1280, height=720, frames=81,
                     prefix="__SLUG__/smoke/wan22")
    r = run_graph("wan22_i2v", g, timeout=3600, prompt=MOTION, seed=46, model="wan2.2_i2v_14B_fp8+lightx2v_4step")
    r["probe"] = {Path(f).name: ffprobe(f) for f in r["files"]}
    return r


def t_wan22_i2v_480():
    img = comfy.upload(_first_image("klein_edit"))
    g = wf.wan22_i2v("smoke_wan22_i2v_480p", img, MOTION, seed=46, width=832, height=480, frames=81,
                     prefix="__SLUG__/smoke/wan22_480")
    return run_graph("wan22_i2v_480", g, timeout=3600, prompt=MOTION, seed=46, model="wan2.2_i2v_14B_fp8+lightx2v_4step")


def t_interpolate():
    src = [f for f in json.loads(RESULTS.read_text())["wan22_i2v"]["files"] if "16fps" in f][0]
    v = comfy.upload(src)
    g = wf.interpolate("smoke_interp_rife_x3", v, fps_in=16.0, multiplier=3, prefix="__SLUG__/smoke/interp")
    return run_graph("interpolate", g, model="rife_v4.26")


def _sing_clip():
    """A 6 s sung excerpt for lip-sync tests: the isolated vocal stem if Stage 0 separation ran, else the full mix."""
    data = json.loads(RESULTS.read_text())
    if "separator" in data and data["separator"].get("ok"):
        src = data["separator"]["vocals"]
    else:
        src = data["ace_turbo"]["files"][0]
    dst = OUT / "lipsync_audio_6s.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "40", "-t", "6", "-i", src, "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def t_infinitetalk():
    img = comfy.upload(_first_image("klein_edit"))
    aud = comfy.upload(_sing_clip())
    prompt = "A tired man in an office shirt sings passionately to camera, natural head motion."
    g = wf.infinitetalk("smoke_infinitetalk_6s", img, aud, prompt, seed=47, seconds=6.0, width=832, height=480,
                        prefix="__SLUG__/smoke/infinitetalk")
    r = run_graph("infinitetalk", g, timeout=3600, prompt=prompt, seed=47, model="wan2.1_i2v_480p_fp8+infinitetalk_single")
    r["probe"] = ffprobe(r["files"][0])
    return r


def t_ltx_ia2v():
    img = comfy.upload(_first_image("klein_edit"))
    aud = comfy.upload(_sing_clip())
    prompt = "A tired man in an office shirt sings passionately to camera, natural head motion, static camera."
    g = wf.ltx23_ia2v("smoke_ltx23_ia2v_6s", img, aud, prompt, seed=48, seconds=6.0, width=1280, height=720,
                      prefix="__SLUG__/smoke/ltx")
    r = run_graph("ltx_ia2v", g, timeout=3600, prompt=prompt, seed=48, model="ltx-2.3-22b-distilled-fp8")
    r["probe"] = ffprobe(r["files"][0])
    return r


def t_seedvr2():
    src = [f for f in json.loads(RESULTS.read_text())["wan22_i2v"]["files"] if "16fps" in f][0]
    v = comfy.upload(src)
    g = wf.seedvr2_video("smoke_seedvr2_3b_1080p", v, width=1920, height=1080, model="3b", frames_per_chunk=21,
                         overlap=2, prefix="__SLUG__/smoke/seedvr2")
    r = run_graph("seedvr2", g, timeout=7200, model="seedvr2_3b_int8")
    r["probe"] = ffprobe(r["files"][0])
    return r


def t_seedvr2_7b():
    src = [f for f in json.loads(RESULTS.read_text())["wan22_i2v"]["files"] if "16fps" in f][0]
    v = comfy.upload(src)
    g = wf.seedvr2_video("smoke_seedvr2_7b_1080p", v, width=1920, height=1080, model="7b", frames_per_chunk=21,
                         overlap=2, prefix="__SLUG__/smoke/seedvr2_7b")
    return run_graph("seedvr2_7b", g, timeout=7200, model="seedvr2_7b_int8")


# ------------------------------------------------------------------ non-ComfyUI tools (own venvs)
def _tool(test, tool, script, args, **kw):
    from pipeline import toolrun
    log(f"{test}: {tool}/{script}")
    r = toolrun.run(tool, script, args, name=f"smoke:{test}", log=ROOT / "qc" / "logs" / f"smoke-{test}.tool.log", **kw)
    log(f"{test}: ok {r['seconds']}s, VRAM peak {r['vram_peak_mib']} MiB (baseline {r['vram_baseline_mib']})")
    return r


VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEMUCS_MODEL = "htdemucs_ft.yaml"


def t_separator():
    take = json.loads(RESULTS.read_text())["ace_turbo"]["files"][0]
    out = OUT / "separator"
    r1 = _tool("separator", "separator", "separate.py", [out, VOCAL_MODEL, take])
    files = r1["stems"][take]
    vocals = [f for f in files if "(Vocals)" in f][0]
    inst = [f for f in files if "(Instrumental)" in f][0]
    r2 = _tool("separator", "separator", "separate.py", [out, KARAOKE_MODEL, vocals])
    r3 = _tool("separator", "separator", "separate.py", [out, DEMUCS_MODEL, inst])
    res = {"ok": True, "vocals": vocals, "instrumental": inst, "lead_backing": r2["stems"][vocals],
           "demucs": r3["stems"][inst], "seconds": round(r1["seconds"] + r2["seconds"] + r3["seconds"], 1),
           "seconds_parts": [r1["seconds"], r2["seconds"], r3["seconds"]],
           "vram_peak_mib": max(r1["vram_peak_mib"], r2["vram_peak_mib"], r3["vram_peak_mib"]),
           "note": "first run includes model downloads"}
    manifest.upsert("smoke:separator", stage="0", kind="smoke", path=vocals, model=f"{VOCAL_MODEL}+{KARAOKE_MODEL}+{DEMUCS_MODEL}",
                    workflow="pipeline/tools/separate.py", status="pass", meta={k: res[k] for k in ("seconds", "vram_peak_mib")})
    return res


def _lyric_words(text):
    import re
    words = []
    for line in text.splitlines():
        if line.strip().startswith("["):
            continue
        words += re.findall(r"[a-z0-9']+", line.lower().replace("’", "'"))
    return words


def t_whisperx():
    import jiwer
    data = json.loads(RESULTS.read_text())
    vocals = data["separator"]["vocals"]
    from pipeline import lyrics
    r = _tool("whisperx", "whisperx", "align.py", ["transcribe", OUT / "whisperx", vocals])
    res = r["results"][vocals]
    out = res["out"]
    ref = " ".join(_lyric_words(song_spec.generator_lyrics()))
    hyp = " ".join(_lyric_words(res.get("text", "")))
    r["wer_vs_lyrics"] = round(jiwer.wer(ref, hyp), 3)
    m = lyrics.match(json.load(open(out))["words"])
    r["lyric_recall"] = m["recall"]
    r["sections"] = m["sections"]
    r["ok"] = True
    manifest.upsert("smoke:whisperx", stage="0", kind="smoke", path=str(out), model="whisperx large-v3 + wav2vec2 align",
                    workflow="pipeline/tools/align.py", score={"wer": r["wer_vs_lyrics"]}, status="pass",
                    meta={"seconds": r["seconds"], "vram_peak_mib": r["vram_peak_mib"]})
    return r


def t_beats():
    take = json.loads(RESULTS.read_text())["ace_turbo"]["files"][0]
    r = _tool("beats", "beats", "beats.py", [OUT / "beats", take])
    r.update(r["results"][take])
    out = r["out"]
    r["ok"] = True
    manifest.upsert("smoke:beats", stage="0", kind="smoke", path=str(out), model="beat_this final0",
                    workflow="pipeline/tools/beats.py", score={"bpm": r["bpm"]}, status="pass",
                    meta={"seconds": r["seconds"], "vram_peak_mib": r["vram_peak_mib"]})
    return r


def t_aesthetics():
    data = json.loads(RESULTS.read_text())
    takes = [data[k]["files"][0] for k in ("ace_turbo", "ace_sft") if k in data]
    r = _tool("aesthetics", "aesthetics", "aesthetics.py", takes)
    r["ok"] = True
    manifest.upsert("smoke:aesthetics", stage="0", kind="smoke", path=takes[0], model="audiobox-aesthetics",
                    workflow="pipeline/tools/aesthetics.py", score=r["scores"], status="pass",
                    meta={"seconds": r["seconds"], "vram_peak_mib": r["vram_peak_mib"]})
    return r


SYSTEM_VOICE = ("A calm adult voice, even pace, clear consonants, mid pitch, quiet and unhurried, close to the mic.")


def t_tts():
    out = OUT / "tts"
    out.mkdir(parents=True, exist_ok=True)
    design_text = "This is a short test of the voice. The room is quiet."
    jobs = [
        {"mode": "design", "instruct": SYSTEM_VOICE, "text": design_text, "out": str(out / "voice_design.wav"), "seed": 11},
        {"mode": "clone", "ref": str(out / "voice_design.wav"), "ref_text": design_text,
         "text": "The same voice says one more calm sentence.", "out": str(out / "voice_clone.wav"), "seed": 12},
    ]
    jf = out / "jobs.json"
    jf.write_text(json.dumps(jobs, indent=1))
    r = _tool("tts", "tts", "tts.py", ["batch", jf])
    r["ok"] = True
    manifest.upsert("smoke:tts", stage="0", kind="smoke", path=str(out / "voice_clone.wav"),
                    prompt=SYSTEM_VOICE, seed=11, model="Qwen3-TTS-12Hz-1.7B VoiceDesign+Base",
                    workflow="pipeline/tools/tts.py", status="pass", meta={"seconds": r["seconds"], "vram_peak_mib": r["vram_peak_mib"]})
    return r


def t_overlay():
    from pipeline import overlay
    box = {"kind": "notice", "title": "Notice", "name": "Smoke test",
           "body": "One short line of overlay text, to check alpha and size.",
           "x": 960, "y": 540, "w": 1180, "t_in": 0.0, "t_out": 3.0, "type_cps": 60}
    out = OUT / "overlay" / "achievement1"
    r = overlay.render(box, out)
    # composite one held frame over a still to check alpha and legibility
    still = _first_image("klein_edit")
    comp = OUT / "overlay" / "composite_check.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", still, "-i", str(out / "f00040.png"), "-filter_complex",
                    "[0]scale=1920:1080[b];[b][1]overlay=0:0", "-frames:v", "1", str(comp)], check=True)
    from PIL import Image
    a = Image.open(out / "f00040.png").getchannel("A")
    r.update({"ok": True, "composite": str(comp), "alpha_min_max": a.getextrema(), "seconds": r["render_seconds"]})
    manifest.upsert("smoke:overlay", stage="0", kind="smoke", path=r["webm"], prompt=None, seed=None,
                    model="headless Chrome " + subprocess.run([overlay.CHROME, "--version"], capture_output=True, text=True).stdout.strip(),
                    workflow="overlays/box.html + pipeline/overlay.py", status="pass", meta={"seconds": r["seconds"], "frames": r["frames"]})
    return r


def t_acestep_repaint():
    """Repaint 20 seconds of the turbo take through the official ACE-Step server."""
    from pipeline import acestep
    take = json.loads(RESULTS.read_text())["ace_turbo"]["files"][0]
    out = OUT / "acestep" / "repaint_60-80.flac"
    t0 = time.time()
    with acestep.session("smoke:acestep_repaint"):
        startup = round(time.time() - t0, 1)
        with gpu_peak() as vp:
            r = acestep.repaint(take, 60.0, 80.0, song_spec.CAPTION, song_spec.generator_lyrics(), seed=2001, out=out,
                                bpm=song_spec.BPM, key_scale=song_spec.KEYSCALE, duration=song_spec.DURATION)
    r.update({"ok": True, "server_startup_seconds": startup, "vram_peak_mib": vp.peak, "vram_baseline_mib": vp.baseline,
              "probe": ffprobe(out)})
    manifest.upsert("smoke:acestep_repaint", stage="0", kind="smoke", path=str(out), prompt=song_spec.CAPTION, seed=2001,
                    model=r.get("dit_model"), workflow="pipeline/acestep.py (ACE-Step 1.5 API /release_task repaint)",
                    status="pass", meta={"seconds": r["seconds"], "vram_peak_mib": vp.peak, "startup": startup})
    return r


def gpu_peak():
    from pipeline import gpu
    return gpu.VramPeak()


def t_mix():
    """pedalboard chain + pyloudnorm: master the turbo take to -14 LUFS / -1 dBTP and verify."""
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf
    from pedalboard import Compressor, HighpassFilter, Pedalboard
    from pipeline import master
    take = json.loads(RESULTS.read_text())["ace_turbo"]["files"][0]
    x, sr = sf.read(take, always_2d=True)
    t0 = time.time()
    pre = pyln.Meter(sr).integrated_loudness(x)
    y = Pedalboard([HighpassFilter(30), Compressor(threshold_db=-18, ratio=2.0, attack_ms=10, release_ms=120)])(
        x.T.astype(np.float32), sr).T.astype(np.float64)
    y = master.to_target(y, sr, lufs=-14.0, ceiling_db=-1.3)
    out = OUT / "mix" / "master_check.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr, subtype="PCM_24")
    m = master.measure(out)
    lufs, tp = m["lufs"], m["true_peak_dbtp"]
    r = {"ok": -14.5 < lufs < -13.5 and tp < -1.0, "input_lufs": round(pre, 2), **m,
         "seconds": round(time.time() - t0, 1), "files": [str(out)]}
    manifest.upsert("smoke:mix", stage="0", kind="smoke", path=str(out), model="pedalboard+pyloudnorm+ffmpeg ebur128",
                    workflow="pipeline/stage0.py:t_mix", score={"lufs": lufs, "true_peak": tp},
                    status="pass" if r["ok"] else "fail", meta={"seconds": r["seconds"]})
    return r


# ------------------------------------------------------------------ warm timings for batch planning
def t_ace_turbo_warm():
    mk = lambda s: wf.ace15_song(f"smoke_ace15_xl_turbo_s{s}", song_spec.CAPTION, song_spec.generator_lyrics(), seed=s,
                                 duration=song_spec.DURATION, bpm=song_spec.BPM, keyscale=song_spec.KEYSCALE,
                                 variant="turbo", prefix="__SLUG__/smoke/ace_turbo_warm")
    return run_graph("ace_turbo_warm", mk(1002), warm_g=mk(1003), prompt=song_spec.CAPTION, seed=1002,
                     model="acestep_v1.5_xl_turbo_bf16")


def t_klein_warm():
    prompt = f"{STYLE} {ROOM_SHOT} {PERSON}"
    mk = lambda s: wf.klein(f"smoke_klein9b_t2i_s{s}", prompt, seed=s, width=1920, height=1088, prefix="__SLUG__/smoke/klein_warm")
    return run_graph("klein_warm", mk(52), warm_g=mk(53), prompt=prompt, seed=52, model="flux-2-klein-9b-kv-fp8")


def t_klein_edit_warm():
    sheet = comfy.upload(_first_image("charsheet"))
    prompt = (f"{STYLE} One person alone, the person from the character sheet, same face, hair and clothes. {ROOM_SHOT}")
    mk = lambda s: wf.klein(f"smoke_klein9b_edit_s{s}", prompt, seed=s, width=1920, height=1088, refs=[sheet],
                            prefix="__SLUG__/smoke/klein_edit_warm")
    return run_graph("klein_edit_warm", mk(54), warm_g=mk(55), prompt=prompt, seed=54, model="flux-2-klein-9b-kv-fp8")


def t_tts_warm():
    """Five clone lines with cached weights: the per-line cost for Stage 3's 5 takes per line."""
    out = OUT / "tts_warm"
    out.mkdir(parents=True, exist_ok=True)
    ref = OUT / "tts" / "voice_design.wav"
    ref_text = "This is a short test of the voice. The room is quiet."
    lines = ["The first line is calm.", "The second line is calm.", "The third line is calm.",
             "The fourth line is calm.", "The fifth line is calm."]
    jobs = [{"mode": "clone", "ref": str(ref), "ref_text": ref_text, "text": t, "out": str(out / f"line{i}.wav"), "seed": 20 + i}
            for i, t in enumerate(lines)]
    jf = out / "jobs.json"
    jf.write_text(json.dumps(jobs, indent=1))
    r = _tool("tts_warm", "tts", "tts.py", ["batch", jf])
    audio_s = sum(x["seconds_audio"] for x in r["results"])
    r.update({"ok": True, "lines": len(lines), "audio_seconds": round(audio_s, 2),
              "seconds_per_line": round(r["seconds"] / len(lines), 1)})
    return r


def t_syncnet():
    """Score the InfiniteTalk smoke clip with SyncNet (lip-sync check for Stage 8)."""
    clip = json.loads(RESULTS.read_text())["infinitetalk"]["files"][0]
    r = _tool("syncnet", "syncnet", "syncnet_score.py", [clip, OUT / "syncnet_work"])
    r["ok"] = r.get("confidence") is not None
    manifest.upsert("smoke:syncnet", stage="0", kind="smoke", path=clip, model="SyncNet v2 (joonson/syncnet_python)",
                    workflow="pipeline/tools/syncnet_score.py", score={"confidence": r.get("confidence"), "offset": r.get("offset")},
                    status="pass" if r["ok"] else "fail", meta={"seconds": r["seconds"], "vram_peak_mib": r["vram_peak_mib"]})
    return r


def t_syncnet_ltx():
    clip = json.loads(RESULTS.read_text())["ltx_ia2v"]["files"][0]
    r = _tool("syncnet_ltx", "syncnet", "syncnet_score.py", [clip, OUT / "syncnet_ltx_work"])
    r["ok"] = r.get("confidence") is not None
    return r


TESTS = {
    "syncnet": t_syncnet, "syncnet_ltx": t_syncnet_ltx,
    "ace_turbo_warm": t_ace_turbo_warm, "klein_warm": t_klein_warm, "klein_edit_warm": t_klein_edit_warm,
    "tts_warm": t_tts_warm,
    "acestep_repaint": t_acestep_repaint, "mix": t_mix,
    "overlay": t_overlay,
    "separator": t_separator, "whisperx": t_whisperx, "beats": t_beats, "aesthetics": t_aesthetics, "tts": t_tts,
    "ace_turbo": t_ace_turbo,  # ace_sft, seedvr2 (3B): models removed 2026-10-02
    "stable_audio": t_stable_audio,
    "klein_t2i": t_klein_t2i, "klein_portrait": t_klein_portrait, "charsheet": t_charsheet, "klein_edit": t_klein_edit,
    "wan22_i2v": t_wan22_i2v, "wan22_i2v_480": t_wan22_i2v_480, "interpolate": t_interpolate,
    "infinitetalk": t_infinitetalk, "ltx_ia2v": t_ltx_ia2v, "seedvr2_7b": t_seedvr2_7b,
}


def main(argv):
    if not argv or argv[0] != "smoke":
        print(__doc__)
        return 2
    names = list(TESTS) if argv[1:] == ["all"] else argv[1:]
    failed = []
    for name in names:
        try:
            res = TESTS[name]()
            record(name, **res)
            if res.get("ok") is False:
                raise RuntimeError(f"{name} ran but failed its check: { {k: v for k, v in res.items() if k != 'files'} }")
        except Exception as e:
            failed.append(name)
            tb = traceback.format_exc()
            log(f"{name}: FAIL {e!r}\n{tb}")
            record(name, ok=False, error=repr(e)[:3000])
            manifest.upsert(f"smoke:{name}", stage="0", kind="smoke", status="fail", meta={"error": repr(e)[:3000]})
        finally:
            try:
                comfy.free()  # free VRAM between tests so each peak is measured cold
            except Exception:
                pass
    log(f"FINISHED smoke: failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
