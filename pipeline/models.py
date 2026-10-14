"""Model registry: every ComfyUI weight the pipeline uses, where it comes from and where it lives.

Sources are official org repos or the repos ComfyUI's own workflow templates point at.
Revisions are resolved to commit SHAs at download time and recorded in the manifest.
"""
import os
from dataclasses import dataclass

COMFY_MODELS = os.path.join(os.environ.get("COMFY_DIR", "/opt/ComfyUI"), "models")


@dataclass(frozen=True)
class Model:
    key: str          # stable id used in manifest and workflows
    repo: str         # Hugging Face repo id
    path: str         # file path inside the repo
    folder: str       # ComfyUI models/ subfolder
    name: str         # final file name in that folder
    stage: str        # first stage that needs it
    priority: int     # download order (lower first)
    optional: bool = False


MODELS = [
    # Stage 1: song (turbo + VAE + text encoders already present; SFT added for a quality comparison)
    Model("ace15_xl_turbo", "Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors", "diffusion_models", "acestep_v1.5_xl_turbo_bf16.safetensors", "1", 0),
    Model("ace15_xl_sft", "Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors", "diffusion_models", "acestep_v1.5_xl_sft_bf16.safetensors", "1", 1, optional=True),
    Model("ace15_te_06b", "Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/text_encoders/qwen_0.6b_ace15.safetensors", "text_encoders", "qwen_0.6b_ace15.safetensors", "1", 0),
    Model("ace15_lm_4b", "Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/text_encoders/qwen_4b_ace15.safetensors", "text_encoders", "qwen_4b_ace15.safetensors", "1", 0),
    Model("ace15_vae", "Comfy-Org/ace_step_1.5_ComfyUI_files", "split_files/vae/ace_1.5_vae.safetensors", "vae", "ace_1.5_vae.safetensors", "1", 0),
    # Stage 3: sound effects
    Model("stable_audio_open", "Comfy-Org/stable-audio-open-1.0_repackaged", "stable-audio-open-1.0.safetensors", "checkpoints", "stable-audio-open-1.0.safetensors", "3", 2),
    Model("t5_base", "google-t5/t5-base", "model.safetensors", "text_encoders", "t5-base.safetensors", "3", 2),
    # Stages 5-6: looks, sheets, stills (Flux.2 Klein 9B; the KV build is the ungated distilled 9B)
    Model("klein9b_kv", "black-forest-labs/FLUX.2-klein-9b-kv-fp8", "flux-2-klein-9b-kv-fp8.safetensors", "diffusion_models", "flux-2-klein-9b-kv-fp8.safetensors", "5", 1),
    Model("qwen3_8b_te", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders", "qwen_3_8b_fp8mixed.safetensors", "5", 1),
    # The pre-existing user file came from flux2-dev (the Klein 4B templates' source); same VAE weights as the klein-9b repo copy.
    Model("flux2_vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", "vae", "flux2-vae.safetensors", "5", 0),
    Model("charsheet_quad", "Alissonerdx/CharacterSheet", "QuadView_klein9b_v1.safetensors", "loras", "QuadView_klein9b_v1.safetensors", "5", 1),
    Model("charsheet_triple", "Alissonerdx/CharacterSheet", "TripleView_klein9b_v1.safetensors", "loras", "TripleView_klein9b_v1.safetensors", "5", 1, optional=True),
    # Stage 7: image-to-video (Wan 2.2 I2V fp8 + lightx2v 4-step distill) and interpolation
    Model("wan22_i2v_high", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "7", 3),
    Model("wan22_i2v_low", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "7", 3),
    Model("wan22_lx2v_high", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "7", 3),
    Model("wan22_lx2v_low", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "7", 3),
    Model("umt5_xxl", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "7", 3),
    Model("wan21_vae", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae", "wan_2.1_vae.safetensors", "7", 3),
    Model("rife426", "Comfy-Org/frame_interpolation", "frame_interpolation/rife_v4.26.safetensors", "frame_interpolation", "rife_v4.26.safetensors", "7", 3),
    Model("film", "Comfy-Org/frame_interpolation", "frame_interpolation/film_net_fp16.safetensors", "frame_interpolation", "film_net_fp16.safetensors", "7", 3, optional=True),
    # Stage 8: lip-sync (InfiniteTalk on Wan 2.1 I2V 480p, all native nodes)
    Model("wan21_i2v_480", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors", "diffusion_models", "wan2.1_i2v_480p_14B_fp8_scaled.safetensors", "8", 4),
    Model("infinitetalk_single", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors", "model_patches", "wan2.1_infiniteTalk_single_fp16.safetensors", "8", 4),
    Model("wav2vec2_zh_base", "Kijai/wav2vec2_safetensors", "wav2vec2-chinese-base_fp16.safetensors", "audio_encoders", "wav2vec2-chinese-base_fp16.safetensors", "8", 4),
    Model("lx2v_i2v_480", "Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "8", 4),
    Model("clip_vision_h", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision", "clip_vision_h.safetensors", "8", 4),
    # Stage 10: upscale
    Model("seedvr2_3b", "Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", "diffusion_models", "seedvr2_3b_int8_convrot.safetensors", "10", 5, optional=True),
    Model("seedvr2_7b", "Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_int8_convrot.safetensors", "diffusion_models", "seedvr2_7b_int8_convrot.safetensors", "10", 5),
    Model("seedvr2_vae", "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", "vae", "seedvr2_ema_vae_fp16.safetensors", "10", 5),
    # Ladder rung 1 alternative: LTX-2.3 (faster/longer shots, audio-driven i2v as a lip-sync fallback)
    Model("ltx23_distilled", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", "checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors", "7", 6, optional=True),
    Model("gemma3_12b_fp4", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", "text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors", "7", 6, optional=True),
    Model("ltx23_upscaler", "Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "7", 6, optional=True),
]

BY_KEY = {m.key: m for m in MODELS}


def local_path(m: Model) -> str:
    return f"{COMFY_MODELS}/{m.folder}/{m.name}"
