"""WhisperX transcription + wav2vec2 forced alignment (envs/whisperx); one model load per batch.

  align.py transcribe <out_dir> <audio> [<audio> ...] [--model large-v3]
      free transcription with word timestamps -> <out_dir>/<stem>.json
  align.py force <audio> <segments.json> <out.json>
      forced alignment of known text: segments.json = [{"text", "start", "end"}...] with rough windows
Prints {"results": {"<audio>": {"out", "segments", "words", "text"}}}.
"""
import json
import os
import sys

import whisperx

DEVICE = "cuda"


def main():
    args = [a for a in sys.argv[1:]]
    model_name = "large-v3"
    if "--model" in args:
        i = args.index("--model")
        model_name = args[i + 1]
        del args[i:i + 2]
    mode = args[0]
    align_model, meta = whisperx.load_align_model(language_code="en", device=DEVICE)
    results = {}
    if mode == "transcribe":
        out_dir, audios = args[1], args[2:]
        os.makedirs(out_dir, exist_ok=True)
        model = whisperx.load_model(model_name, DEVICE, compute_type="float16", language="en")
        for path in audios:
            audio = whisperx.load_audio(path)
            segs = model.transcribe(audio, batch_size=16, language="en")["segments"]
            aligned = whisperx.align(segs, align_model, meta, audio, DEVICE, return_char_alignments=False)
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
            words = [w for s in aligned["segments"] for w in s.get("words", [])]
            json.dump({"segments": aligned["segments"], "words": words}, open(out, "w"), indent=1)
            results[path] = {"out": out, "segments": len(aligned["segments"]), "words": len(words),
                             "text": " ".join(s["text"].strip() for s in aligned["segments"])[:6000]}
    else:
        path, seg_file, out = args[1], args[2], args[3]
        audio = whisperx.load_audio(path)
        aligned = whisperx.align(json.load(open(seg_file)), align_model, meta, audio, DEVICE, return_char_alignments=False)
        words = [w for s in aligned["segments"] for w in s.get("words", [])]
        json.dump({"segments": aligned["segments"], "words": words}, open(out, "w"), indent=1)
        results[path] = {"out": out, "segments": len(aligned["segments"]), "words": len(words)}
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
