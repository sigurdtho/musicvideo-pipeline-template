"""SyncNet lip-sync score (envs/syncnet, joonson/syncnet_python at a pinned commit).

  syncnet_score.py <video> <workdir>
Runs the authors' face-track pipeline then SyncNet; prints {"tracks": [{"offset", "min_dist", "confidence"}],
"confidence": best, "offset": of best}. Typical good sync: confidence > 5, |offset| <= 1 frame (25 fps).
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "envs", "syncnet", "syncnet_python"))


def main():
    video, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    ref = "clip"
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    # SyncNet expects 25 fps; re-time the clip losslessly enough for scoring
    v25 = os.path.join(work, "in25.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-r", "25", "-c:v", "libx264", "-crf", "12",
                    "-c:a", "aac", "-ar", "16000", "-ac", "1", v25], check=True)
    py = sys.executable
    # music-video shots are 2-6 s with medium framing: keep tracks >= 1 s and faces >= 40 px (defaults: 4 s, 100 px)
    subprocess.run([py, "run_pipeline.py", "--videofile", v25, "--reference", ref, "--data_dir", work,
                    "--min_track", "25", "--min_face_size", "40", "--facedet_scale", "0.5"],
                   cwd=REPO, check=True, capture_output=True, text=True)
    p = subprocess.run([py, "run_syncnet.py", "--videofile", v25, "--reference", ref, "--data_dir", work],
                       cwd=REPO, check=True, capture_output=True, text=True)
    out = p.stdout + p.stderr
    offs = [int(x) for x in re.findall(r"AV offset:\s+(-?\d+)", out)]
    dists = [float(x) for x in re.findall(r"Min dist:\s+([\d.]+)", out)]
    confs = [float(x) for x in re.findall(r"Confidence:\s+([\d.]+)", out)]
    tracks = [{"offset": o, "min_dist": d, "confidence": c} for o, d, c in zip(offs, dists, confs)]
    best = max(tracks, key=lambda t: t["confidence"]) if tracks else None
    print(json.dumps({"tracks": tracks, "confidence": best and best["confidence"], "offset": best and best["offset"]}))


if __name__ == "__main__":
    main()
