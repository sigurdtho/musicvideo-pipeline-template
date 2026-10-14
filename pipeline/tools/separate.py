"""Stem separation with python-audio-separator (envs/separator); one model load per batch.

  separate.py <outdir> <model_filename> <input> [<input> ...]
Prints {"model": m, "stems": {"<input>": [files...]}}.
"""
import json
import os
import sys

from audio_separator.separator import Separator


def main():
    outdir, model, inputs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "separator", "models")
    sep = Separator(output_dir=outdir, model_file_dir=os.path.abspath(model_dir), output_format="FLAC")
    sep.load_model(model_filename=model)
    stems = {}
    for src in inputs:
        files = sep.separate(src)
        stems[src] = [f if os.path.isabs(f) else os.path.join(outdir, f) for f in files]
    print(json.dumps({"model": model, "stems": stems}))


if __name__ == "__main__":
    main()
