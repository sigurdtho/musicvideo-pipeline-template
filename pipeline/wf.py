"""API-format workflow builders. Settings follow ComfyUI's official templates (noted per builder).

Every builder returns a Graph; callers queue graph.api() via pipeline.comfy.run and record graph.name
as the manifest's `workflow`. All outputs go under ComfyUI/output/__SLUG__/ to stay apart from other work.
"""
from pipeline.graph import Graph

# master.md NEGATIVE line (for models that take one)
NEG_STYLE = "text, letters, subtitles, watermark, logo, extra fingers, deformed hands, blurry, low quality"
# Wan's standard negative from the official template, plus the master.md NEGATIVE line
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
           "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
           "杂乱的背景，三条腿，背景人很多，倒着走, " + NEG_STYLE)

ACE = {  # template audio_ace_step1_5_xl_{turbo,sft}
    "turbo": dict(unet="acestep_v1.5_xl_turbo_bf16.safetensors", steps=8, cfg=1.0, top_p=0.9),
    "sft": dict(unet="acestep_v1.5_xl_sft_bf16.safetensors", steps=50, cfg=7.0, top_p=1.0),
}


def save_video(g, video, prefix, crf=8):
    """Near-lossless intermediate (MKV, H.264 CRF 8) so later stages don't stack compression."""
    return g.node("SaveVideo", video=video, filename_prefix=prefix, format="mkv",
                  format__codec="h264", format__codec__encoding="re-encode", format__codec__encoding__crf=float(crf))


# ---------------------------------------------------------------- Stage 1: song
def ace15_song(name, tags, lyrics, seed, duration, bpm, keyscale, variant="turbo", timesig="4", language="en",
               lm_cfg=2.0, temperature=0.85, codes=True, steps=None, cfg=None, prefix="__SLUG__/song/take"):
    v = ACE[variant]
    g = Graph(name)
    unet = g.node("UNETLoader", unet_name=v["unet"], weight_dtype="default")
    clip = g.node("DualCLIPLoader", clip_name1="qwen_0.6b_ace15.safetensors", clip_name2="qwen_4b_ace15.safetensors",
                  type="ace", device="default")
    vae = g.node("VAELoader", vae_name="ace_1.5_vae.safetensors")
    model = g.node("ModelSamplingAuraFlow", model=unet[0], shift=3.0)
    pos = g.node("TextEncodeAceStepAudio1.5", clip=clip[0], tags=tags, lyrics=lyrics, seed=seed, bpm=bpm,
                 duration=float(duration), timesignature=timesig, language=language, keyscale=keyscale,
                 generate_audio_codes=codes, cfg_scale=lm_cfg, temperature=temperature, top_p=v["top_p"], top_k=0, min_p=0.0)
    neg = g.node("ConditioningZeroOut", conditioning=pos[0])
    lat = g.node("EmptyAceStep1.5LatentAudio", seconds=float(duration), batch_size=1)
    ks = g.node("KSampler", model=model[0], positive=pos[0], negative=neg[0], latent_image=lat[0], seed=seed,
                steps=steps or v["steps"], cfg=cfg if cfg is not None else v["cfg"], sampler_name="euler",
                scheduler="simple", denoise=1.0)
    audio = g.node("VAEDecodeAudio", samples=ks[0], vae=vae[0])
    g.node("SaveAudioAdvanced", audio=audio[0], filename_prefix=prefix, format="flac")
    return g


# ---------------------------------------------------------------- Stage 3: sound effects
def stable_audio(name, prompt, seed, seconds, negative="", steps=50, cfg=4.98, prefix="__SLUG__/sfx/sfx"):
    """Template audio_stable_audio_example."""
    g = Graph(name)
    ckpt = g.node("CheckpointLoaderSimple", ckpt_name="stable-audio-open-1.0.safetensors")
    clip = g.node("CLIPLoader", clip_name="t5-base.safetensors", type="stable_audio", device="default")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip[0], text=negative)
    lat = g.node("EmptyLatentAudio", seconds=float(seconds), batch_size=1)
    ks = g.node("KSampler", model=ckpt[0], positive=pos[0], negative=neg[0], latent_image=lat[0], seed=seed,
                steps=steps, cfg=cfg, sampler_name="dpmpp_3m_sde_gpu", scheduler="exponential", denoise=1.0)
    audio = g.node("VAEDecodeAudio", samples=ks[0], vae=ckpt[2])
    g.node("SaveAudioAdvanced", audio=audio[0], filename_prefix=prefix, format="flac")
    return g


# ---------------------------------------------------------------- Stages 5-6: looks, sheets, stills
def klein(name, prompt, seed, width, height, refs=(), steps=4, lora=None, lora_strength=1.0, scheduler="flux2",
          ref_megapixels=1.0, prefix="__SLUG__/img/klein"):
    """Flux.2 Klein 9B (KV, distilled). Templates image_flux2_klein_9b_kv_image_edit / CharacterSheet QuadView.

    refs: ComfyUI input names (see comfy.upload). scheduler 'flux2' (template, 4 steps) or 'simple'
    (CharacterSheet LoRA recipe: euler/simple, 8 steps, CFG 1).
    """
    g = Graph(name)
    model = g.node("UNETLoader", unet_name="flux-2-klein-9b-kv-fp8.safetensors", weight_dtype="default")[0]
    if lora:
        model = g.node("LoraLoaderModelOnly", model=model, lora_name=lora, strength_model=lora_strength)[0]
    if refs:
        model = g.node("FluxKVCache", model=model)[0]
    clip = g.node("CLIPLoader", clip_name="qwen_3_8b_fp8mixed.safetensors", type="flux2", device="default")
    vae = g.node("VAELoader", vae_name="flux2-vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)[0]
    neg = g.node("ConditioningZeroOut", conditioning=pos)[0]
    for r in refs:
        img = g.node("LoadImage", image=r)
        img = g.node("ImageScaleToTotalPixels", image=img[0], upscale_method="lanczos", megapixels=ref_megapixels,
                     resolution_steps=1)
        lat = g.node("VAEEncode", pixels=img[0], vae=vae[0])
        pos = g.node("ReferenceLatent", conditioning=pos, latent=lat[0])[0]
        neg = g.node("ReferenceLatent", conditioning=neg, latent=lat[0])[0]
    empty = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    if scheduler == "flux2":
        sigmas = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    else:
        sigmas = g.node("BasicScheduler", model=model, scheduler=scheduler, steps=steps, denoise=1.0)
    guider = g.node("CFGGuider", model=model, positive=pos, negative=neg, cfg=1.0)
    noise = g.node("RandomNoise", noise_seed=seed)
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    out = g.node("SamplerCustomAdvanced", noise=noise[0], guider=guider[0], sampler=sampler[0], sigmas=sigmas[0],
                 latent_image=empty[0])
    img = g.node("VAEDecode", samples=out[0], vae=vae[0])
    g.node("SaveImage", images=img[0], filename_prefix=prefix)
    return g


CHARSHEET_QUAD_PROMPT = ("Convert the character in the image to a Character Sheet showing a face close-up, "
                         "front, side and back full body views")


def charsheet(name, ref, seed, prefix="__SLUG__/img/sheet"):
    """Alissonerdx/CharacterSheet QuadView recipe: ref padded onto 1536x1024 white, LoRA 1.0, euler/simple 8 steps, CFG 1."""
    return klein(name, CHARSHEET_QUAD_PROMPT, seed, 1536, 1024, refs=[ref], steps=8, lora="QuadView_klein9b_v1.safetensors",
                 lora_strength=1.0, scheduler="simple", ref_megapixels=1.57, prefix=prefix)


# ---------------------------------------------------------------- Stage 7: image-to-video (+ interpolation)
def wan22_i2v(name, image, prompt, seed, width=1280, height=720, frames=81, negative=WAN_NEG, lightning=True,
              interp_multiplier=3, interp_model="rife_v4.26.safetensors", prefix="__SLUG__/clips/clip"):
    """Template video_wan2_2_14B_i2v (4-step lightx2v path: shift 5, cfg 1, split 2/4; else 20 steps, cfg 3.5, split 10).

    Saves the raw 16 fps clip and a RIFE x3 (48 fps) copy; the edit keeps every other frame for an exact 24 fps grid.
    """
    steps, split, cfg = (4, 2, 1.0) if lightning else (20, 10, 3.5)
    g = Graph(name)
    clip = g.node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = g.node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip[0], text=negative)
    img = g.node("LoadImage", image=image)
    i2v = g.node("WanImageToVideo", positive=pos[0], negative=neg[0], vae=vae[0], width=width, height=height,
                 length=frames, batch_size=1, start_image=img[0])
    hi = g.node("UNETLoader", unet_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", weight_dtype="default")[0]
    lo = g.node("UNETLoader", unet_name="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", weight_dtype="default")[0]
    if lightning:
        hi = g.node("LoraLoaderModelOnly", model=hi, lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                    strength_model=1.0)[0]
        lo = g.node("LoraLoaderModelOnly", model=lo, lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                    strength_model=1.0)[0]
    hi = g.node("ModelSamplingSD3", model=hi, shift=5.0)[0]
    lo = g.node("ModelSamplingSD3", model=lo, shift=5.0)[0]
    k1 = g.node("KSamplerAdvanced", model=hi, add_noise="enable", noise_seed=seed, steps=steps, cfg=cfg,
                sampler_name="euler", scheduler="simple", positive=i2v[0], negative=i2v[1], latent_image=i2v[2],
                start_at_step=0, end_at_step=split, return_with_leftover_noise="enable")
    k2 = g.node("KSamplerAdvanced", model=lo, add_noise="disable", noise_seed=seed, steps=steps, cfg=cfg,
                sampler_name="euler", scheduler="simple", positive=i2v[0], negative=i2v[1], latent_image=k1[0],
                start_at_step=split, end_at_step=10000, return_with_leftover_noise="disable")
    frames_img = g.node("VAEDecode", samples=k2[0], vae=vae[0])
    save_video(g, g.node("CreateVideo", images=frames_img[0], fps=16.0)[0], prefix + "_16fps")
    if interp_multiplier:
        im = g.node("FrameInterpolationModelLoader", model_name=interp_model)
        fi = g.node("FrameInterpolate", interp_model=im[0], images=frames_img[0], multiplier=interp_multiplier)
        save_video(g, g.node("CreateVideo", images=fi[0], fps=16.0 * interp_multiplier)[0],
                   prefix + f"_{16 * interp_multiplier}fps")
    return g


def interpolate(name, video, fps_in=16.0, multiplier=3, model="rife_v4.26.safetensors", prefix="__SLUG__/clips/interp"):
    """Template utility_video_frame_interpolation (FILM there; RIFE v4.26 here, per master.md)."""
    g = Graph(name)
    v = g.node("LoadVideo", file=video)
    comp = g.node("GetVideoComponents", video=v[0])
    im = g.node("FrameInterpolationModelLoader", model_name=model)
    fi = g.node("FrameInterpolate", interp_model=im[0], images=comp[0], multiplier=multiplier)
    save_video(g, g.node("CreateVideo", images=fi[0], fps=float(fps_in * multiplier))[0], prefix)
    return g


# ---------------------------------------------------------------- Stage 8: lip-sync
def infinitetalk(name, image, audio, prompt, seed, seconds, width=832, height=480, steps=6, audio_scale=1.0,
                 use_clip_vision=True, prefix="__SLUG__/lipsync/talk"):
    """Template video_wan2_1_infinitetalk (single speaker): Wan 2.1 I2V 480p + lightx2v, 6 steps, cfg 1, 25 fps.

    Long shots chain 81-frame chunks inside one graph: each chunk sees all previous frames (the node derives the
    audio offset from them) and its first 9 motion frames are trimmed before concatenation.
    """
    fps, chunk, motion = 25, 81, 9
    total = max(chunk, int(round(seconds * fps)))
    n_chunks = 1 + max(0, -(-(total - chunk) // (chunk - motion)))
    g = Graph(name)
    unet = g.node("UNETLoader", unet_name="wan2.1_i2v_480p_14B_fp8_scaled.safetensors", weight_dtype="default")
    model = g.node("LoraLoaderModelOnly", model=unet[0], lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                   strength_model=1.0)
    patch = g.node("ModelPatchLoader", name="wan2.1_infiniteTalk_single_fp16.safetensors")
    clip = g.node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = g.node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=clip[0], text=prompt)
    neg = g.node("ConditioningZeroOut", conditioning=pos[0])
    img = g.node("LoadImage", image=image)
    aud = g.node("LoadAudio", audio=audio)
    enc = g.node("AudioEncoderLoader", audio_encoder_name="wav2vec2-chinese-base_fp16.safetensors")
    feats = g.node("AudioEncoderEncode", audio_encoder=enc[0], audio=aud[0])
    cv_out = None
    if use_clip_vision:
        cv = g.node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
        cv_out = g.node("CLIPVisionEncode", clip_vision=cv[0], image=img[0], crop="none")[0]
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    acc = None
    for k in range(n_chunks):
        it = g.node("WanInfiniteTalkToVideo", mode="single_speaker", model=model[0], model_patch=patch[0],
                    positive=pos[0], negative=neg[0], vae=vae[0], width=width, height=height, length=chunk,
                    audio_encoder_output_1=feats[0], motion_frame_count=motion, audio_scale=audio_scale,
                    start_image=img[0], previous_frames=acc, clip_vision_output=cv_out)
        guider = g.node("CFGGuider", model=it[0], positive=it[1], negative=it[2], cfg=1.0)
        sigmas = g.node("BasicScheduler", model=it[0], scheduler="normal", steps=steps, denoise=1.0)
        noise = g.node("RandomNoise", noise_seed=seed + k)
        s = g.node("SamplerCustomAdvanced", noise=noise[0], guider=guider[0], sampler=sampler[0], sigmas=sigmas[0],
                   latent_image=it[3])
        frames = g.node("VAEDecode", samples=s[0], vae=vae[0])[0]
        if acc is None:
            acc = frames
        else:
            new = g.node("ImageFromBatch", image=frames, batch_index=motion, length=chunk - motion)[0]
            acc = g.node("ImageBatch", image1=acc, image2=new)[0]
    acc = g.node("ImageFromBatch", image=acc, batch_index=0, length=total)[0]
    save_video(g, g.node("CreateVideo", images=acc, fps=float(fps), audio=aud[0])[0], prefix)
    return g


def ltx23_ia2v(name, image, audio, prompt, seed, seconds, width=1280, height=720, fps=24,
               negative="pc game, console game, video game, cartoon, childish, ugly", prefix="__SLUG__/lipsync/ltx"):
    """Template video_ltx2_3_ia2v on the distilled fp8 checkpoint (no distill LoRA, no prompt-enhancer LLM).

    Half-res pass (8 manual sigmas) with the audio latent masked as fixed, x2 latent upsample, 3-sigma refine.
    """
    frames = int(round(seconds * fps / 8)) * 8 + 1
    ckpt_name = "ltx-2.3-22b-distilled-fp8.safetensors"
    g = Graph(name)
    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=ckpt_name)
    te = g.node("LTXAVTextEncoderLoader", text_encoder="gemma_3_12B_it_fp4_mixed.safetensors", ckpt_name=ckpt_name,
                device="default")
    avae = g.node("LTXVAudioVAELoader", ckpt_name=ckpt_name)
    pos = g.node("CLIPTextEncode", clip=te[0], text=prompt)
    neg = g.node("CLIPTextEncode", clip=te[0], text=negative)
    cond = g.node("LTXVConditioning", positive=pos[0], negative=neg[0], frame_rate=float(fps))
    img = g.node("LoadImage", image=image)
    img = g.node("ResizeImageMaskNode", input=img[0], resize_type="scale dimensions", resize_type__width=width,
                 resize_type__height=height, resize_type__crop="center", scale_method="lanczos")
    img = g.node("ResizeImageMaskNode", input=img[0], resize_type="scale longer dimension",
                 resize_type__longer_size=1536, scale_method="lanczos")
    pre = g.node("LTXVPreprocess", image=img[0], img_compression=18)
    empty = g.node("EmptyLTXVLatentVideo", width=width // 2, height=height // 2, length=frames, batch_size=1)
    vlat = g.node("LTXVImgToVideoInplace", vae=ckpt[2], image=pre[0], latent=empty[0], strength=0.7, bypass=False)
    aud = g.node("LoadAudio", audio=audio)
    aud = g.node("TrimAudioDuration", audio=aud[0], start_index=0.0, duration=float(seconds))
    alat = g.node("LTXVAudioVAEEncode", audio=aud[0], audio_vae=avae[0])
    mask = g.node("SolidMask", value=0.0, width=width, height=height)
    alat = g.node("SetLatentNoiseMask", samples=alat[0], mask=mask[0])
    av = g.node("LTXVConcatAVLatent", video_latent=vlat[0], audio_latent=alat[0])
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    g1 = g.node("CFGGuider", model=ckpt[0], positive=cond[0], negative=cond[1], cfg=1.0)
    sig1 = g.node("ManualSigmas", sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    s1 = g.node("SamplerCustomAdvanced", noise=g.node("RandomNoise", noise_seed=seed)[0], guider=g1[0], sampler=sampler[0],
                sigmas=sig1[0], latent_image=av[0])
    sep1 = g.node("LTXVSeparateAVLatent", av_latent=s1[0])
    crop = g.node("LTXVCropGuides", positive=cond[0], negative=cond[1], latent=sep1[0])
    upm = g.node("LatentUpscaleModelLoader", model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    up = g.node("LTXVLatentUpsampler", samples=sep1[0], upscale_model=upm[0], vae=ckpt[2])
    vlat2 = g.node("LTXVImgToVideoInplace", vae=ckpt[2], image=pre[0], latent=up[0], strength=1.0, bypass=False)
    av2 = g.node("LTXVConcatAVLatent", video_latent=vlat2[0], audio_latent=sep1[1])
    g2 = g.node("CFGGuider", model=ckpt[0], positive=crop[0], negative=crop[1], cfg=1.0)
    sig2 = g.node("ManualSigmas", sigmas="0.85, 0.7250, 0.4219, 0.0")
    s2 = g.node("SamplerCustomAdvanced", noise=g.node("RandomNoise", noise_seed=seed + 1)[0], guider=g2[0],
                sampler=sampler[0], sigmas=sig2[0], latent_image=av2[0])
    sep2 = g.node("LTXVSeparateAVLatent", av_latent=s2[0])
    imgs = g.node("VAEDecodeTiled", samples=sep2[0], vae=ckpt[2], tile_size=768, overlap=64, temporal_size=4096,
                  temporal_overlap=4)
    out_audio = g.node("LTXVAudioVAEDecode", samples=sep2[1], audio_vae=avae[0])
    save_video(g, g.node("CreateVideo", images=imgs[0], fps=float(fps), audio=out_audio[0])[0], prefix)
    return g


# ---------------------------------------------------------------- Stage 10: upscale
def seedvr2_video(name, video, width=1920, height=1080, model="3b", frames_per_chunk=0, overlap=0, color="lab",
                  seed=1, prefix="__SLUG__/upscale/up"):
    """Template utility_seedvr2_3b_int8_upscale_video: resize to target, 1-step restore, LAB color match.

    frames_per_chunk=0 runs the clip in one pass; otherwise manual temporal chunks (4n+1) crossfaded by `overlap`.
    """
    unet = {"3b": "seedvr2_3b_int8_convrot.safetensors", "7b": "seedvr2_7b_int8_convrot.safetensors"}[model]
    g = Graph(name)
    v = g.node("LoadVideo", file=video)
    comp = g.node("GetVideoComponents", video=v[0])
    resized = g.node("ResizeImageMaskNode", input=comp[0], resize_type="scale dimensions", resize_type__width=width,
                     resize_type__height=height, resize_type__crop="center", scale_method="lanczos")
    pre = g.node("SeedVR2Preprocess", resized_images=resized[0])
    vae = g.node("VAELoader", vae_name="seedvr2_ema_vae_fp16.safetensors")
    m = g.node("UNETLoader", unet_name=unet, weight_dtype="default")
    lat = g.node("VAEEncodeTiled", pixels=pre[0], vae=vae[0], tile_size=512, overlap=128, temporal_size=64,
                 temporal_overlap=8)[0]
    chunk = None
    if frames_per_chunk:
        chunk = g.node("SeedVR2TemporalChunk", latent=lat, temporal_overlap=overlap, chunking_mode="manual",
                       chunking_mode__frames_per_chunk=frames_per_chunk)
        lat = chunk[0]
    cond = g.node("SeedVR2Conditioning", model=m[0], vae_conditioning=lat)
    ks = g.node("KSampler", model=m[0], positive=cond[0], negative=cond[1], latent_image=lat, seed=seed, steps=1,
                cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)[0]
    if chunk is not None:
        ks = g.node("SeedVR2TemporalMerge", latents=ks, temporal_overlap=chunk[1])[0]
    dec = g.node("VAEDecodeTiled", samples=ks, vae=vae[0], tile_size=512, overlap=128, temporal_size=64,
                 temporal_overlap=8)
    post = g.node("SeedVR2PostProcessing", images=dec[0], original_resized_images=resized[0],
                  color_correction_method=color)
    save_video(g, g.node("CreateVideo", images=post[0], fps=comp[2], audio=comp[1])[0], prefix, crf=10)
    return g
