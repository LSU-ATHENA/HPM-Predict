import torch

from .generation_protocol import (
    apply_generation_scheduler,
    get_generation_protocol,
)


def freeze_pipeline_modules(pipe):
    for name in ('text_encoder', 'text_encoder_2', 'unet', 'transformer', 'vae'):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, 'parameters'):
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    return pipe


_SDXL_PROTOCOL = get_generation_protocol('sdxl')
SDXL_CONFIG = {
    'model_id': 'stabilityai/stable-diffusion-xl-base-1.0',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'embed_dim': 2048,
    'pooled_dim': 1280,
    'max_seq_len': 77,
    'guidance_scale': _SDXL_PROTOCOL['guidance_scale'],
    'num_inference_steps': _SDXL_PROTOCOL['num_inference_steps'],
    'scheduler': _SDXL_PROTOCOL['scheduler'],
    'protocol_note': _SDXL_PROTOCOL['protocol_note'],
}


def load_sdxl_pipeline(device='cuda'):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    apply_generation_scheduler(pipe, SDXL_CONFIG.get('scheduler'))
    freeze_pipeline_modules(pipe)
    pipe.to(device)
    # Keep SDXL-family VAE upcast unless explicitly revisited; fp16 VAE decode
    # has caused corrupted/noisy images in prior diffusion workflows.
    pipe.upcast_vae()
    freeze_pipeline_modules(pipe)
    return pipe


_HUNYUAN_PROTOCOL = get_generation_protocol('hunyuan_dit')
HUNYUAN_CONFIG = {
    'model_id': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'clip_dim': 1024,
    'clip_seq_len': 77,
    't5_dim': 2048,
    't5_seq_len': 256,
    'guidance_scale': _HUNYUAN_PROTOCOL['guidance_scale'],
    'num_inference_steps': _HUNYUAN_PROTOCOL['num_inference_steps'],
    'scheduler': _HUNYUAN_PROTOCOL['scheduler'],
    'protocol_note': _HUNYUAN_PROTOCOL['protocol_note'],
}


def load_hunyuan_pipeline(device='cuda'):
    from diffusers import HunyuanDiTPipeline
    pipe = HunyuanDiTPipeline.from_pretrained(
        HUNYUAN_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    apply_generation_scheduler(pipe, HUNYUAN_CONFIG.get('scheduler'))
    freeze_pipeline_modules(pipe)
    pipe = pipe.to(device)
    freeze_pipeline_modules(pipe)
    return pipe


_DREAMSHAPER_PROTOCOL = get_generation_protocol('dreamshaper')
DREAMSHAPER_CONFIG = {
    'model_id': 'Lykon/dreamshaper-xl-v2-turbo',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'embed_dim': 2048,
    'pooled_dim': 1280,
    'max_seq_len': 77,
    'guidance_scale': _DREAMSHAPER_PROTOCOL['guidance_scale'],
    'num_inference_steps': _DREAMSHAPER_PROTOCOL['num_inference_steps'],
    'scheduler': _DREAMSHAPER_PROTOCOL['scheduler'],
    'protocol_note': _DREAMSHAPER_PROTOCOL['protocol_note'],
}


def load_dreamshaper_pipeline(device='cuda'):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        DREAMSHAPER_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    apply_generation_scheduler(pipe, DREAMSHAPER_CONFIG.get('scheduler'))
    freeze_pipeline_modules(pipe)
    pipe.to(device)
    # Keep SDXL-family VAE upcast unless explicitly revisited; fp16 VAE decode
    # has caused corrupted/noisy images in prior diffusion workflows.
    pipe.upcast_vae()
    freeze_pipeline_modules(pipe)
    return pipe


_PIXART_SIGMA_PROTOCOL = get_generation_protocol('pixart_sigma')
PIXART_SIGMA_CONFIG = {
    'model_id': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    't5_dim': 4096,
    't5_seq_len': 300,
    'guidance_scale': _PIXART_SIGMA_PROTOCOL['guidance_scale'],
    'num_inference_steps': _PIXART_SIGMA_PROTOCOL['num_inference_steps'],
    'scheduler': _PIXART_SIGMA_PROTOCOL['scheduler'],
    'protocol_note': _PIXART_SIGMA_PROTOCOL['protocol_note'],
}


def load_pixart_sigma_pipeline(device='cuda'):
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained(
        PIXART_SIGMA_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    apply_generation_scheduler(pipe, PIXART_SIGMA_CONFIG.get('scheduler'))
    freeze_pipeline_modules(pipe)
    pipe = pipe.to(device)
    freeze_pipeline_modules(pipe)
    return pipe


# --- SANA-Sprint 0.6B (matches Noise Hypernetworks paper exactly) ---
_SANA_SPRINT_PROTOCOL = get_generation_protocol('sana_sprint')
SANA_SPRINT_CONFIG = {
    'model_id': 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers',
    'resolution': 1024,
    'latent_shape': (32, 32, 32),     # DC-AE f32c32: 32 channels, 32x spatial compression
    'text_dim': 2304,                 # Gemma-2-2B-IT caption_channels
    'text_seq_len': 300,              # max_sequence_length
    'guidance_scale': _SANA_SPRINT_PROTOCOL['guidance_scale'],  # Embedded CFG (guidance_embeds_scale=0.1)
    'num_inference_steps': _SANA_SPRINT_PROTOCOL['num_inference_steps'],  # Matching HyperNoise paper's SANA-Sprint inference setting
    'scheduler': _SANA_SPRINT_PROTOCOL['scheduler'],
    'protocol_note': _SANA_SPRINT_PROTOCOL['protocol_note'],
}


def load_sana_sprint_pipeline(device='cuda'):
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(
        SANA_SPRINT_CONFIG['model_id'], torch_dtype=torch.bfloat16,
    )
    apply_generation_scheduler(pipe, SANA_SPRINT_CONFIG.get('scheduler'))
    freeze_pipeline_modules(pipe)
    pipe = pipe.to(device)
    freeze_pipeline_modules(pipe)
    return pipe
