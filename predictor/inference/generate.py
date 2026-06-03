"""
Multi-model image generation with predictor-selected noise.

Supports: SDXL, DreamShaper, Hunyuan-DiT, PixArt-Sigma, PixArt-Alpha.

Usage:
    python -m predictor.inference.generate --model_type sdxl --checkpoint experiments/sdxl_best/best_model.pth --prompt "A cat"
    python -m predictor.inference.generate --model_type hunyuan_dit --checkpoint experiments/hunyuan_best/best_model.pth --prompt "A sunset"
"""

import argparse
import sys
from pathlib import Path

import torch

from predictor.inference.loader import load_predictor
from predictor.inference.noise_selection import generate_noise_candidates, select_top_k_noise
from predictor.configs.model_dims import MODEL_DIMS, get_dims

from gen_dataset.datagen.generation_protocol import (
    apply_generation_scheduler,
    get_generation_protocol,
)

JPEG_QUALITY = 90


def save_jpeg(image, path: Path):
    image.convert("RGB").save(path, format="JPEG", quality=JPEG_QUALITY)


# Pipeline loading registry
PIPELINE_CONFIG = {
    'sdxl': {
        'class': 'StableDiffusionXLPipeline',
        'pretrained': 'stabilityai/stable-diffusion-xl-base-1.0',
    },
    'dreamshaper': {
        'class': 'StableDiffusionXLPipeline',
        'pretrained': 'Lykon/dreamshaper-xl-v2-turbo',
    },
    'hunyuan_dit': {
        'class': 'HunyuanDiTPipeline',
        'pretrained': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    },
    'pixart_sigma': {
        'class': 'PixArtSigmaPipeline',
        'pretrained': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
    },
    'sana_sprint': {
        'class': 'SanaSprintPipeline',
        'pretrained': 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers',
        'dtype': torch.bfloat16,
    },
}


def freeze_pipeline_modules(pipe):
    for module_name in (
        'text_encoder',
        'text_encoder_2',
        'tokenizer',
        'tokenizer_2',
        'unet',
        'transformer',
        'vae',
    ):
        module = getattr(pipe, module_name, None)
        if module is None or not hasattr(module, 'parameters'):
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    return pipe


def load_pipeline(
    model_type: str,
    device: str = 'cuda',
    dtype=torch.float16,
    move_to_device: bool = True,
):
    import diffusers

    config = PIPELINE_CONFIG[model_type]
    pipe_class = getattr(diffusers, config['class'])
    pipe_dtype = config.get('dtype', dtype)
    pipe = pipe_class.from_pretrained(
        config['pretrained'],
        torch_dtype=pipe_dtype,
    )
    protocol = get_generation_protocol(model_type)
    apply_generation_scheduler(pipe, protocol.get('scheduler'))
    freeze_pipeline_modules(pipe)
    if move_to_device:
        pipe = pipe.to(device)
    freeze_pipeline_modules(pipe)
    return pipe


def _concat_hunyuan_t5_clip_for_predictor(
    t5_embeds: torch.Tensor,
    t5_mask: torch.Tensor,
    clip_embeds: torch.Tensor,
    clip_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clip_embeds.shape[-1] < t5_embeds.shape[-1]:
        pad = torch.zeros(
            *clip_embeds.shape[:-1],
            t5_embeds.shape[-1] - clip_embeds.shape[-1],
            device=clip_embeds.device,
            dtype=clip_embeds.dtype,
        )
        clip_embeds = torch.cat([clip_embeds, pad], dim=-1)
    elif clip_embeds.shape[-1] > t5_embeds.shape[-1]:
        clip_embeds = clip_embeds[..., :t5_embeds.shape[-1]]

    pred_embeds = torch.cat([t5_embeds, clip_embeds.to(dtype=t5_embeds.dtype)], dim=1)
    pred_mask = torch.cat([t5_mask, clip_mask.to(device=t5_mask.device, dtype=t5_mask.dtype)], dim=1)
    return pred_embeds, pred_mask


def _detach_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach_tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_tensor_tree(v) for v in value)
    return value


def encode_prompt_for_model(
    pipe,
    prompt: str,
    model_type: str,
    device: str = 'cuda',
    hunyuan_pred_embed_type: str = 't5',
):
    if model_type in ('sdxl', 'dreamshaper'):
        # SDXL-family: encode_prompt returns (embeds, neg_embeds, pooled, neg_pooled)
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        # For predictor: use positive embeddings only
        pred_embeds = prompt_embeds  # [1, 77, 2048]
        pred_mask = torch.ones(pred_embeds.shape[:2], device=device, dtype=torch.long)

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
        }

    elif model_type == 'hunyuan_dit':
        # Hunyuan-DiT: encode_prompt return signature varies by diffusers version:
        #   >=0.30 (newer): 8 values (CLIP_emb, neg, T5_emb, neg, mask, neg_mask, mask2, neg_mask2)
        #     or 4 values (CLIP_emb, neg, T5_emb, neg) without masks
        #   0.36.0 (older): 4 values — CLIP only (emb, neg_emb, mask, neg_mask), no T5
        with torch.inference_mode():
            result = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        # Detect version by checking result[2] dtype:
        #   T5 embeds would be float (3D tensor), CLIP mask would be int/long (2D tensor)
        has_t5_from_encode = (len(result) >= 4 and result[2].dtype in (torch.float16, torch.float32, torch.bfloat16))

        if has_t5_from_encode:
            # Newer diffusers: encode_prompt returns both CLIP + T5
            prompt_embeds, negative_prompt_embeds = result[0], result[1]
            prompt_embeds_2, negative_prompt_embeds_2 = result[2], result[3]

            if len(result) >= 8:
                prompt_attention_mask = result[4]
                negative_prompt_attention_mask = result[5]
                prompt_attention_mask_2 = result[6]
                negative_prompt_attention_mask_2 = result[7]
            else:
                prompt_attention_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)
                negative_prompt_attention_mask = torch.ones(negative_prompt_embeds.shape[:2], device=device, dtype=torch.long)
                prompt_attention_mask_2 = torch.ones(prompt_embeds_2.shape[:2], device=device, dtype=torch.long)
                negative_prompt_attention_mask_2 = torch.ones(negative_prompt_embeds_2.shape[:2], device=device, dtype=torch.long)

            if hunyuan_pred_embed_type == 't5+clip':
                pred_embeds, pred_mask = _concat_hunyuan_t5_clip_for_predictor(
                    prompt_embeds_2,
                    prompt_attention_mask_2,
                    prompt_embeds,
                    prompt_attention_mask,
                )
            else:
                # Use T5 embeddings for predictor (prompt_embeds_2)
                pred_embeds = prompt_embeds_2  # [1, 256, 2048]
                pred_mask = prompt_attention_mask_2

            gen_kwargs = {
                'prompt_embeds': prompt_embeds,
                'negative_prompt_embeds': negative_prompt_embeds,
                'prompt_embeds_2': prompt_embeds_2,
                'negative_prompt_embeds_2': negative_prompt_embeds_2,
                'prompt_attention_mask': prompt_attention_mask,
                'negative_prompt_attention_mask': negative_prompt_attention_mask,
                'prompt_attention_mask_2': prompt_attention_mask_2,
                'negative_prompt_attention_mask_2': negative_prompt_attention_mask_2,
            }
        else:
            # Older diffusers (e.g. 0.36.0): encode_prompt returns CLIP only.
            # Manually encode T5 for predictor scoring.
            max_seq_len = get_dims(model_type)['seq_len']  # 256
            tokens = pipe.tokenizer_2(
                prompt,
                max_length=max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).to(device)
            with torch.inference_mode():
                t5_output = pipe.text_encoder_2(
                    tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
            t5_embeds = t5_output[0].to(dtype=torch.float16)  # [1, 256, 2048]
            t5_mask = tokens.attention_mask  # [1, 256]
            if hunyuan_pred_embed_type == 't5+clip':
                clip_embeds = result[0]
                clip_mask = result[2] if len(result) >= 4 else None
                if clip_mask is None:
                    clip_mask = torch.ones(clip_embeds.shape[:2], device=device, dtype=torch.long)
                pred_embeds, pred_mask = _concat_hunyuan_t5_clip_for_predictor(
                    t5_embeds,
                    t5_mask,
                    clip_embeds,
                    clip_mask,
                )
            else:
                pred_embeds = t5_embeds
                pred_mask = t5_mask

            # Let the pipeline handle encoding internally during generation
            gen_kwargs = {}

    elif model_type == 'pixart_sigma':
        # PixArt: encode_prompt returns (embeds, mask, neg_embeds, neg_mask)
        max_seq_len = get_dims(model_type)['seq_len']
        with torch.inference_mode():
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
                device=device,
                clean_caption=False,
                max_sequence_length=max_seq_len,
            )

        pred_embeds = prompt_embeds  # [1, seq_len, embed_dim]
        pred_mask = prompt_attention_mask if prompt_attention_mask is not None else \
            torch.ones(pred_embeds.shape[:2], device=device, dtype=torch.long)

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
            'negative_prompt_embeds': negative_prompt_embeds,
            'negative_prompt_attention_mask': negative_prompt_attention_mask,
        }

    elif model_type == 'sana_sprint':
        # SANA-Sprint: encode_prompt returns (prompt_embeds, prompt_attention_mask)
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                max_sequence_length=get_dims(model_type)['seq_len'],
            )

        pred_embeds = prompt_embeds  # [1, 300, 2304]
        pred_mask = prompt_attention_mask  # [1, 300]

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return (
        pred_embeds.detach(),
        pred_mask.detach() if isinstance(pred_mask, torch.Tensor) else pred_mask,
        _detach_tensor_tree(gen_kwargs),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with multi-model predictor-selected noise"
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=list(MODEL_DIMS.keys()), help="Diffusion model type")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to predictor checkpoint (.pth)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--N", type=int, default=100, help="Number of noise candidates")
    parser.add_argument("--B", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--head", type=int, default=0,
                        help="Prediction head (0=hpsv2, 1=image_reward, 2=clip_score)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps. Defaults to the fixed protocol for --model_type.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale. Defaults to the fixed protocol for --model_type.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate B random baseline images")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--hunyuan-pred-embed-type", default="t5",
                        choices=["t5", "t5+clip"],
                        help="Predictor prompt embeddings for Hunyuan-DiT checkpoints")
    args = parser.parse_args()

    protocol = get_generation_protocol(args.model_type)
    if args.steps is None:
        args.steps = protocol['num_inference_steps']
    if args.guidance_scale is None:
        args.guidance_scale = protocol['guidance_scale']

    dims = get_dims(args.model_type)
    latent_shape = dims['latent_shape']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PNM Multi-Model Image Generation")
    print(f"{'='*60}")
    print(f"  Model type: {args.model_type}")
    print(f"  Latent:     {latent_shape}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  N:          {args.N} noise candidates")
    print(f"  B:          {args.B} images")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {output_dir}")
    print(f"  Steps/CFG:  {args.steps} / {args.guidance_scale}")
    print(f"  Scheduler:  {protocol.get('scheduler') or 'pipeline default'}")
    print(f"{'='*60}")

    # Load pipeline
    print(f"\nLoading {args.model_type} pipeline...")
    pipe = load_pipeline(args.model_type, device=args.device)

    # Load predictor
    print(f"Loading predictor from {args.checkpoint}...")
    predictor, norm_info = load_predictor(args.checkpoint, device=args.device)
    print(f"  num_heads={predictor.num_heads}")

    # Encode prompt
    pred_embeds, pred_mask, gen_kwargs = encode_prompt_for_model(
        pipe,
        args.prompt,
        args.model_type,
        args.device,
        hunyuan_pred_embed_type=args.hunyuan_pred_embed_type,
    )

    # Generate and select noise
    generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=latent_shape,
        device=args.device,
        dtype=pipe.unet.dtype if hasattr(pipe, 'unet') else pipe.transformer.dtype,
        generator=generator,
    )

    selected = select_top_k_noise(
        predictor=predictor,
        noises=noises,
        prompt_embeds=pred_embeds,
        prompt_mask=pred_mask,
        num_select=args.B,
        head_index=args.head,
    )

    # Don't pre-scale — pipeline's prepare_latents applies init_noise_sigma internally
    latents = selected

    # Expand embeddings for B images
    B = args.B
    expanded_kwargs = {}
    for k, v in gen_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            expanded_kwargs[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            expanded_kwargs[k] = v

    # Generate images
    print(f"\nGenerating {B} images from top-{B} of {args.N} candidates...")
    with torch.inference_mode():
        result = pipe(
            prompt=None,
            **expanded_kwargs,
            latents=latents,
            num_images_per_prompt=1,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )

    for i, img in enumerate(result.images):
        path = output_dir / f"{args.model_type}_predictor_{i:02d}.jpg"
        save_jpeg(img, path)
        print(f"  Saved: {path}")

    if args.compare:
        print(f"\nGenerating {B} baseline images (random noise)...")
        gen_random = torch.Generator(device=args.device).manual_seed(args.seed + 999) if args.seed else None
        with torch.inference_mode():
            result_rand = pipe(
                prompt=args.prompt,
                num_images_per_prompt=B,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen_random,
            )
        for i, img in enumerate(result_rand.images):
            path = output_dir / f"{args.model_type}_random_{i:02d}.jpg"
            save_jpeg(img, path)
            print(f"  Saved: {path}")

    print(f"\nDone! Images saved to {output_dir}/")


if __name__ == "__main__":
    main()
