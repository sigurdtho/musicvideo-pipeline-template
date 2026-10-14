"""Beat and downbeat tracking with beat_this (envs/beats); one model load per batch.

  beats.py <out_dir> <audio> [<audio> ...]
Writes <out_dir>/<stem>.beats.json; prints {"results": {"<audio>": {"beats", "downbeats", "bpm", "out"}}}.
"""
import json
import os
import sys

import numpy as np
from beat_this.inference import File2Beats


def main():
    out_dir, audios = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    f2b = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    results = {}
    for audio in audios:
        beats, downbeats = f2b(audio)
        beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
        ibi = np.diff(beats)
        bpm = float(60.0 / np.median(ibi)) if len(ibi) else 0.0
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(audio))[0] + ".beats.json")
        json.dump({"beats": beats.round(4).tolist(), "downbeats": downbeats.round(4).tolist(), "bpm": round(bpm, 2)},
                  open(out, "w"), indent=1)
        results[audio] = {"beats": len(beats), "downbeats": len(downbeats), "bpm": round(bpm, 2), "out": out}
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
