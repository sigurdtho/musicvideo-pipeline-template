"""ComfyUI API-format graph builders for slopcore mode (written, NOT run here: these need the GPU).

klein_outpaint  one zoom-chain level: the previous level shrunk to 1/factor in the centre of a same-size canvas, the
                border repainted by Flux.2 Klein 9B (latent noise mask; the inner picture is pasted back unchanged).
wan22_flf2v     match-cut morph between two picked stills: Wan 2.2 I2V A14B with WanFirstLastFrameToVideo (official
                template video_wan2_2_14B_flf2v; same high/low fp8 models + lightx2v 4-step LoRAs as Stage 7).
Both use only core ComfyUI nodes and models already in $COMFY_DIR/models (no downloads).
"""
import os
import json
from pathlib import Path

WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
           "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
           "杂乱的背景，三条腿，背景人很多，倒着走, text, letters, inscription, signature, watermark, frame")


class Graph:  # same format as pipeline/graph.py (kept local so slopkit has no pipeline import)
    def __init__(self, name):
        self.name, self.nodes = name, {}

    def node(self, cls, **inputs):
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": cls, "inputs": {k.replace("__", "."): v for k, v in inputs.items() if v is not None}}
        return _Ref(nid)

    def api(self):
        return self.nodes

    def save(self, folder) -> Path:
        p = Path(folder) / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.nodes, indent=1, ensure_ascii=False))
        return p


class _Ref:
    def __init__(self, nid):
        self.id = nid

    def __getitem__(self, i):
        return [self.id, i]


def klein_outpaint(name, image, prompt, seed, width=1280, height=720, factor=2.0, feather=48, steps=4,
                   prefix="__SLUG__/zoom/level"):
    """image: ComfyUI input name of the previous (inner) level. Output: the next (outer) level, same size."""
    g = Graph(name)
    iw, ih = int(round(width / factor / 16) * 16), int(round(height / factor / 16) * 16)
    padx, pady = (width - iw) // 2, (height - ih) // 2
    padx, pady = padx - padx % 8, pady - pady % 8
    img = g.node("LoadImage", image=image)
    small = g.node("ImageScale", image=img[0], upscale_method="lanczos", width=iw, height=ih, crop="disabled")
    pad = g.node("ImagePadForOutpaint", image=small[0], left=padx, top=pady, right=width - iw - padx,
                 bottom=height - ih - pady, feathering=feather)
    model = g.node("UNETLoader", unet_name="flux-2-klein-9b-kv-fp8.safetensors", weight_dtype="default")[0]
    model = g.node("FluxKVCache", model=model)[0]
    clip = g.node("CLIPLoader", clip_name="qwen_3_8b_fp8mixed.safetensors", type="flux2", device="default")
    vae = g.node("VAELoader", vae_name="flux2-vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)[0]
    neg = g.node("ConditioningZeroOut", conditioning=pos)[0]
    ref = g.node("VAEEncode", pixels=pad[0], vae=vae[0])          # the padded canvas is also the reference (context)
    pos = g.node("ReferenceLatent", conditioning=pos, latent=ref[0])[0]
    neg = g.node("ReferenceLatent", conditioning=neg, latent=ref[0])[0]
    masked = g.node("SetLatentNoiseMask", samples=ref[0], mask=pad[1])
    sigmas = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    guider = g.node("CFGGuider", model=model, positive=pos, negative=neg, cfg=1.0)
    noise = g.node("RandomNoise", noise_seed=seed)
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    out = g.node("SamplerCustomAdvanced", noise=noise[0], guider=guider[0], sampler=sampler[0], sigmas=sigmas[0],
                 latent_image=masked[0])
    dec = g.node("VAEDecode", samples=out[0], vae=vae[0])
    keep = g.node("InvertMask", mask=pad[1])                       # 1 = original pixels (the inner level)
    final = g.node("ImageCompositeMasked", destination=dec[0], source=pad[0], x=0, y=0, resize_source=False, mask=keep[0])
    g.node("SaveImage", images=final[0], filename_prefix=prefix)
    return g


def wan22_flf2v(name, first, last, prompt, seed, width=1280, height=720, frames=81, lightning=True, negative=WAN_NEG,
                prefix="__SLUG__/morph/morph"):
    """first/last: ComfyUI input names of the two stills. Saves the 16 fps clip (RIFE runs later as in Stage 7)."""
    steps, split, cfg = (4, 2, 1.0) if lightning else (20, 10, 3.5)
    g = Graph(name)
    clip = g.node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = g.node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip[0], text=negative)
    a = g.node("LoadImage", image=first)
    b = g.node("LoadImage", image=last)
    fl = g.node("WanFirstLastFrameToVideo", positive=pos[0], negative=neg[0], vae=vae[0], width=width, height=height,
                length=frames, batch_size=1, start_image=a[0], end_image=b[0])
    hi = g.node("UNETLoader", unet_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", weight_dtype="default")[0]
    lo = g.node("UNETLoader", unet_name="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", weight_dtype="default")[0]
    if lightning:
        hi = g.node("LoraLoaderModelOnly", model=hi, lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", strength_model=1.0)[0]
        lo = g.node("LoraLoaderModelOnly", model=lo, lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", strength_model=1.0)[0]
    hi = g.node("ModelSamplingSD3", model=hi, shift=5.0)[0]
    lo = g.node("ModelSamplingSD3", model=lo, shift=5.0)[0]
    k1 = g.node("KSamplerAdvanced", model=hi, add_noise="enable", noise_seed=seed, steps=steps, cfg=cfg, sampler_name="euler",
                scheduler="simple", positive=fl[0], negative=fl[1], latent_image=fl[2], start_at_step=0, end_at_step=split,
                return_with_leftover_noise="enable")
    k2 = g.node("KSamplerAdvanced", model=lo, add_noise="disable", noise_seed=seed, steps=steps, cfg=cfg, sampler_name="euler",
                scheduler="simple", positive=fl[0], negative=fl[1], latent_image=k1[0], start_at_step=split, end_at_step=10000,
                return_with_leftover_noise="disable")
    img = g.node("VAEDecode", samples=k2[0], vae=vae[0])
    vid = g.node("CreateVideo", images=img[0], fps=16.0)
    g.node("SaveVideo", video=vid[0], filename_prefix=prefix + "_16fps", format="mkv", format__codec="h264",
           format__codec__encoding="re-encode", format__codec__encoding__crf=8.0)
    return g


def node_classes_known(graph: Graph, comfy_root=os.environ.get("COMFY_DIR", "/opt/ComfyUI")) -> dict:
    """CPU-only sanity check: every class_type is defined somewhere in the ComfyUI source tree (no server needed)."""
    import subprocess
    out = {}
    for cls in sorted({n["class_type"] for n in graph.nodes.values()}):
        r = subprocess.run(["grep", "-rlE", "--include=*.py", f'(class {cls}\\b|node_id="{cls}"|"{cls}":)', comfy_root],
                           capture_output=True, text=True)
        out[cls] = bool(r.stdout.strip())
    return out
