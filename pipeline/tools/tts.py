"""Qwen3-TTS (official qwen-tts package) in envs/tts.

  tts.py batch jobs.json      jobs: [{"mode": "design", "instruct", "text", "out", "seed"},
                                     {"mode": "clone", "ref", "ref_text", "text", "out", "seed"}]
Designs a voice with VoiceDesign, reuses it through the Base model's ICL voice-clone prompt.
Prints {"results": [{"out", "seconds_audio", "sr"}...]}.
"""
import json
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_models = {}


def model(name):
    if name not in _models:
        for other in list(_models):  # one model resident at a time keeps VRAM low
            del _models[other]
            torch.cuda.empty_cache()
        _models[name] = Qwen3TTSModel.from_pretrained(name, device_map="cuda:0", dtype=torch.bfloat16,
                                                      attn_implementation="sdpa")
    return _models[name]


def main():
    jobs = json.load(open(sys.argv[2]))
    results = []
    clone_prompts = {}
    for job in sorted(jobs, key=lambda j: j["mode"] != "design"):  # all designs first, then clones
        torch.manual_seed(int(job.get("seed", 0)))
        np.random.seed(int(job.get("seed", 0)) % (2 ** 32))
        if job["mode"] == "design":
            wavs, sr = model(DESIGN).generate_voice_design(text=job["text"], instruct=job["instruct"],
                                                            language=job.get("language", "English"))
        else:
            m = model(BASE)
            key = (job["ref"], job["ref_text"])
            if key not in clone_prompts:
                clone_prompts[key] = m.create_voice_clone_prompt(ref_audio=job["ref"], ref_text=job["ref_text"])
            wavs, sr = m.generate_voice_clone(text=job["text"], language=job.get("language", "English"),
                                              voice_clone_prompt=clone_prompts[key])
        wav = np.asarray(wavs[0], dtype=np.float32)
        sf.write(job["out"], wav, sr, subtype="PCM_24")
        results.append({"out": job["out"], "seconds_audio": round(len(wav) / sr, 3), "sr": sr})
        print(f"wrote {job['out']} ({len(wav) / sr:.2f}s)", flush=True)
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
