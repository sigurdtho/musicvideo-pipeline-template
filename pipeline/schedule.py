"""Render schedule from Stage 0 measurements (qc/stage0_results.json) and master.md job counts.

  python -m pipeline.schedule        prints the markdown table used in qc/stage-0.md
"""
import json

from pipeline.manifest import ROOT

_results_path = ROOT / "qc" / "stage0_results.json"
R = json.loads(_results_path.read_text()) if _results_path.exists() else {}

# Job counts from master.md (stage table, characters, beats); shots assumed at the middle of 60-90.
SHOTS = 0
LIPSYNC_SHOTS = 0
LIPSYNC_SECONDS = 0
I2V_SHOTS = 0
AVG_SHOT_S = 4.0
RETRY = 1.5                 # rule 4 retries, budgeted as +50 %
LOOKS = 0
TTS_LINES = 0
VOICES = 0
SFX = 0
BOXES = 0


# Batch measurements from Stage 1 (16 takes): warm per-item costs replace Stage 0's cold first runs.
ACTUAL = {"repaint_warm": 8.2, "repaint_server_start": 106.3, "whisperx_per_take": 43.0 / 16, "separator_per_take": (19 * 60 + 47 - (19 * 60 + 32) - 0.0) * 60 / 16}


def t(test, key="seconds", default=None):
    d = R.get(test, {})
    return d.get(key, default) if d.get("ok", True) else default


def warm(test):
    return t(test, "seconds_warm") or t(test)


def rows():
    take = warm("ace_turbo_warm") or t("ace_turbo")
    sep = t("separator", "seconds_parts", [None, None, None])
    clip_s = AVG_SHOT_S
    wan = t("wan22_i2v")  # 81 frames = 5.06 s of 16 fps footage, already covers the average shot
    it_rate = (t("infinitetalk") or 0) / 6.0 if t("infinitetalk") else None
    ltx_rate = (t("ltx_ia2v") or 0) / 6.0 if t("ltx_ia2v") else None  # 1280x720 output
    svr = t("seedvr2")
    svr_rate = svr / 81 if svr else None  # seconds per 1080p output frame (16 fps source frames)
    out = [
        # stage, task, jobs, unit seconds, note
        ("1", "ACE-Step XL turbo takes (4:30)", 16, take, "SFT costs " + _fmt(t("ace_sft")) + " per take"),
        ("1", "vocal stem per take (lyric check)", 16, ACTUAL["separator_per_take"], "BS-RoFormer, measured in the Stage 1 batch"),
        ("1", "WhisperX transcript per take", 16, ACTUAL["whisperx_per_take"], "large-v3 warm, measured in the Stage 1 batch"),
        ("1", "aesthetics per take", 16, (t("aesthetics") or 0) / 2 or None, "Audiobox"),
        ("1", "repaint weak sections (3 seeds each)", 3, ACTUAL["repaint_warm"],
         "warm, measured in Stage 1; plus one server start " + _fmt(ACTUAL["repaint_server_start"])),
        ("2", "stems: vocals + lead/backing + drums/bass/other", 1, sum(x for x in sep if x) if all(sep) else None, ""),
        ("2", "forced alignment + beat grid", 1, (t("whisperx") or 0) + (t("beats") or 0) or None, ""),
        ("3", "voice designs (3 per voice)", VOICES * 3, t("tts_warm", "seconds_per_line"), ""),
        ("3", "line takes (5 per line)", TTS_LINES * 5, t("tts_warm", "seconds_per_line"), ""),
        ("3", "sound effects (4 candidates each)", SFX * 4, warm("stable_audio"), ""),
        ("4", "mix + master + checks", 1, (t("mix") or 0) + (t("aesthetics") or 0) or None, "CPU mostly"),
        ("5", "look candidates (8 per look)", LOOKS * 8, warm("klein_warm"), "Klein 9B t2i 1920x1088"),
        ("5", "multi-view sheets (2 per look)", LOOKS * 2, t("charsheet"), "QuadView LoRA"),
        ("6", "stills (4 per shot)", SHOTS * 4, warm("klein_edit_warm"), "Klein 9B multi-reference"),
        ("7", "image-to-video 720p, 81 f + RIFE x3", round(I2V_SHOTS * RETRY), wan, "480p costs " + _fmt(t("wan22_i2v_480"))),
        ("8", "lip-sync seconds (LTX-2.3 ia2v 720p, primary)", round(LIPSYNC_SECONDS * RETRY), ltx_rate,
         "InfiniteTalk 480p alternate costs " + _fmt(it_rate) + "/s"),
        ("8", "SyncNet scoring per shot", LIPSYNC_SHOTS, t("syncnet"), ""),
        ("9", "overlay boxes (~3 s each)", BOXES, t("overlay"), "headless Chrome, CPU"),
        ("10", "SeedVR2 7B upscale to 1080p, frames", int(270 * 16), (t("seedvr2_7b") or 0) / 81 or svr_rate,
         "on 16 fps masters before RIFE; 3B costs " + _fmt(svr_rate) + "/frame, so 7B is used"),
    ]
    return out


def _fmt(s):
    if s is None:
        return "n/a"
    s = float(s)
    return f"{s:.1f} s" if s < 120 else (f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h")


def markdown():
    lines = ["| Stage | Task | Jobs | Measured per job | GPU time | Note |", "| --- | --- | ---: | ---: | ---: | --- |"]
    totals = {}
    for st, task, n, unit, note in rows():
        total = n * unit if unit else None
        if total:
            totals[st] = totals.get(st, 0) + total
        lines.append(f"| {st} | {task} | {n} | {_fmt(unit)} | {_fmt(total)} | {note} |")
    lines.append("")
    lines.append("| Stage | GPU time |")
    lines.append("| --- | ---: |")
    for st in sorted(totals, key=int):
        lines.append(f"| {st} | {_fmt(totals[st])} |")
    lines.append(f"| **all** | **{_fmt(sum(totals.values()))}** |")
    return "\n".join(lines), totals


if __name__ == "__main__":
    print(markdown()[0])
