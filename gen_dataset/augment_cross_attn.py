#!/usr/bin/env python3
"""Augment existing generated datasets with first-block cross-attention maps.

Loads existing (noise, prompt_embeds) pairs from a generated dataset, runs
partial DM inference (num_inference_steps=1), captures the first attn2 layer's
cross-attention map, and saves to cross_attn/ subdirectory.

Usage:
  python augment_cross_attn.py --model sdxl --data-dir /path/to/sdxl_1024_v2
  python augment_cross_attn.py --model pixart_sigma --data-dir /path/to/pixart_sigma_1024
  python augment_cross_attn.py --model hunyuan_dit --data-dir /path/to/hunyuan_dit_1024
  python augment_cross_attn.py --model sana_sprint --data-dir /path/to/sana_sprint_1024
  python augment_cross_attn.py --model dreamshaper --data-dir /path/to/dreamshaper_xl_turbo

Output:
  <data-dir>/cross_attn/p{pid:04d}_s{sid:02d}.pt  - attention map + summary stats
  <data-dir>/cross_attn_scalars.jsonl             - (prompt_id, sample_idx, global_entropy)
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from datagen.model_registry import (
    SDXL_CONFIG, load_sdxl_pipeline,
    DREAMSHAPER_CONFIG, load_dreamshaper_pipeline,
    HUNYUAN_CONFIG, load_hunyuan_pipeline,
    PIXART_SIGMA_CONFIG, load_pixart_sigma_pipeline,
    SANA_SPRINT_CONFIG, load_sana_sprint_pipeline,
)
from datagen.partial_ca_extractor import FirstCAExtractor


MODEL_SPECS = {
    'sdxl':          (SDXL_CONFIG,          load_sdxl_pipeline),
    'dreamshaper':   (DREAMSHAPER_CONFIG,   load_dreamshaper_pipeline),
    'hunyuan_dit':   (HUNYUAN_CONFIG,       load_hunyuan_pipeline),
    'pixart_sigma':  (PIXART_SIGMA_CONFIG,  load_pixart_sigma_pipeline),
    'sana_sprint':   (SANA_SPRINT_CONFIG,   load_sana_sprint_pipeline),
}


def load_embeds(embed_path, device):
    d = torch.load(embed_path, map_location='cpu', weights_only=False)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def run_sdxl_like(pipe, embeds, noise, guidance):
    """SDXL / DreamShaper forward with num_inference_steps=1."""
    return pipe(
        prompt_embeds=embeds['prompt_embeds'],
        negative_prompt_embeds=embeds['negative_prompt_embeds'],
        pooled_prompt_embeds=embeds['pooled_prompt_embeds'],
        negative_pooled_prompt_embeds=embeds['negative_pooled_prompt_embeds'],
        latents=noise,
        num_inference_steps=1,
        guidance_scale=guidance,
        output_type='latent',
    )


def run_hunyuan(pipe, embeds, noise, guidance):
    return pipe(
        prompt_embeds=embeds['prompt_embeds'],
        prompt_embeds_2=embeds['prompt_embeds_2'],
        negative_prompt_embeds=embeds['negative_prompt_embeds'],
        negative_prompt_embeds_2=embeds['negative_prompt_embeds_2'],
        prompt_attention_mask=embeds['prompt_attention_mask'],
        prompt_attention_mask_2=embeds['prompt_attention_mask_2'],
        negative_prompt_attention_mask=embeds['negative_prompt_attention_mask'],
        negative_prompt_attention_mask_2=embeds['negative_prompt_attention_mask_2'],
        latents=noise,
        num_inference_steps=1,
        guidance_scale=guidance,
        output_type='latent',
    )


def run_pixart(pipe, embeds, noise, guidance):
    return pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=embeds['prompt_embeds'],
        prompt_attention_mask=embeds.get('prompt_attention_mask'),
        negative_prompt_embeds=embeds.get('negative_prompt_embeds'),
        negative_prompt_attention_mask=embeds.get('negative_prompt_attention_mask'),
        latents=noise,
        num_inference_steps=1,
        guidance_scale=guidance,
        output_type='latent',
    )


def run_sana(pipe, embeds, noise, guidance):
    return pipe(
        prompt=None,
        prompt_embeds=embeds['prompt_embeds'],
        prompt_attention_mask=embeds.get('prompt_attention_mask'),
        latents=noise,
        num_inference_steps=1,
        guidance_scale=guidance,
        output_type='latent',
    )


FORWARD_FNS = {
    'sdxl':          run_sdxl_like,
    'dreamshaper':   run_sdxl_like,
    'hunyuan_dit':   run_hunyuan,
    'pixart_sigma':  run_pixart,
    'sana_sprint':   run_sana,
}


def get_denoiser(pipe):
    """DiT models expose .transformer; UNet models expose .unet."""
    return getattr(pipe, 'transformer', None) or pipe.unet


def is_first_ca_file(path):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    required = {'entropy', 'std', 'max', 'layer_name', 'T_N'}
    return required.issubset(data.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_SPECS.keys()))
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset dir with embeds/, noise/, metadata.jsonl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N samples (for testing)')
    parser.add_argument('--max-samples-per-prompt', type=int, default=None,
                        help='Cap samples per prompt_id (e.g. 20). None = use all.')
    parser.add_argument('--shard', type=int, default=0,
                        help='Shard index for multi-GPU split (by prompt_id mod num_shards)')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total number of shards (one process per GPU)')
    parser.add_argument('--replace-invalid', action='store_true',
                        help='Overwrite existing cross_attn files that are not first-block CA-map tensors.')
    args = parser.parse_args()
    assert 0 <= args.shard < args.num_shards, "shard must satisfy 0 <= shard < num_shards"

    data_dir = Path(args.data_dir)
    embed_dir = data_dir / 'embeds'
    noise_dir = data_dir / 'noise'
    ca_dir = data_dir / 'cross_attn'
    meta_files = sorted(data_dir.glob('metadata*.jsonl'))
    shard_suffix = f'.shard{args.shard}' if args.num_shards > 1 else ''
    scalar_path = data_dir / f'cross_attn_scalars{shard_suffix}.jsonl'

    assert embed_dir.exists(), f"Missing {embed_dir}"
    assert noise_dir.exists(), f"Missing {noise_dir}"
    assert meta_files, f"No metadata*.jsonl found in {data_dir}"
    ca_dir.mkdir(exist_ok=True)

    # Load pipeline
    config, loader = MODEL_SPECS[args.model]
    forward_fn = FORWARD_FNS[args.model]
    print(f"Loading {args.model} pipeline...")
    pipe = loader(device=args.device)
    guidance = config['guidance_scale']

    denoiser = get_denoiser(pipe)
    print(f"Denoiser: {type(denoiser).__name__}")

    extractor = FirstCAExtractor(denoiser)
    print(f"Hooked first attn2 layer: {extractor.layer_name}")

    # Find already-augmented samples. In recovery mode, validate only the shard's
    # candidate files so we do not torch.load every cross-attn tensor up front.
    existing = set()
    if not args.replace_invalid:
        existing = {p.stem for p in ca_dir.glob('*.pt')}
        print(f"Already augmented: {len(existing)} samples")
    else:
        print("replace-invalid enabled: existing files are checked per shard")

    # Load metadata from all metadata*.jsonl files, deduplicate on (pid, sid).
    samples = []
    seen = set()
    for meta_file in meta_files:
        with open(meta_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid, sid = d['prompt_id'], d['sample_idx']
                if args.max_samples_per_prompt is not None and sid >= args.max_samples_per_prompt:
                    continue
                if pid % args.num_shards != args.shard:
                    continue
                name = f"p{pid:04d}_s{sid:02d}"
                if name in seen:
                    continue
                if name in existing:
                    continue
                out_path = ca_dir / f"{name}.pt"
                if args.replace_invalid and out_path.exists() and is_first_ca_file(out_path):
                    continue
                seen.add(name)
                samples.append((pid, sid, name))
    print(f"Metadata files: {len(meta_files)}  (matched {len(samples)} samples after filters)")

    if args.limit:
        samples = samples[:args.limit]

    print(f"Shard {args.shard}/{args.num_shards} | to augment: {len(samples)} samples")
    if not samples:
        print("Nothing to do.")
        return

    scalar_f = open(scalar_path, 'a')

    for pid, sid, name in tqdm(samples, desc=f"Augmenting {args.model}"):
        embed_path = embed_dir / f"p{pid:04d}.pt"
        noise_path = noise_dir / f"{name}.pt"
        out_path = ca_dir / f"{name}.pt"

        if not embed_path.exists() or not noise_path.exists():
            print(f"SKIP {name}: missing embeds or noise")
            continue

        embeds = load_embeds(embed_path, args.device)
        noise = torch.load(noise_path, map_location=args.device, weights_only=True)

        extractor.reset()
        with torch.inference_mode():
            forward_fn(pipe, embeds, noise, guidance)

        if extractor.result is None:
            print(f"SKIP {name}: hook captured nothing")
            continue

        torch.save(extractor.result, out_path)

        scalar_f.write(json.dumps({
            'prompt_id': pid,
            'sample_idx': sid,
            'mean_entropy': round(extractor.result['entropy'].float().mean().item(), 4),
        }) + '\n')
        scalar_f.flush()

        torch.cuda.empty_cache()

    scalar_f.close()
    extractor.remove()
    print(f"Done. Cross-attn saved to {ca_dir}")
    print(f"Scalars appended to {scalar_path}")


if __name__ == '__main__':
    main()
