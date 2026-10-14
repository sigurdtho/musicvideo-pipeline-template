"""Audiobox Aesthetics scores (envs/aesthetics): CE content enjoyment, CU content usefulness,
PC production complexity, PQ production quality.

  aesthetics.py <audio> [<audio> ...]
Prints {"scores": {"<path>": {"CE":..,"CU":..,"PC":..,"PQ":..}}}.
Audio is read with soundfile and passed as tensors (torchaudio.load now needs torchcodec).
"""
import json
import sys

import soundfile as sf
import torch
from audiobox_aesthetics.infer import initialize_predictor


def main():
    paths = sys.argv[1:]
    predictor = initialize_predictor()
    scores = {}
    for p in paths:  # one at a time keeps long songs within memory
        x, sr = sf.read(p, always_2d=True, dtype="float32")
        out = predictor.forward([{"path": torch.from_numpy(x.T.copy()), "sample_rate": sr}])[0]
        scores[p] = {k: round(float(v), 3) for k, v in out.items()}
    print(json.dumps({"scores": scores}))


if __name__ == "__main__":
    main()
