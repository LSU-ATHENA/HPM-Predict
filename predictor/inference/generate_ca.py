
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from predictor.configs.model_dims import MODEL_DIMS, get_dims
from predictor.inference.generate import encode_prompt_for_model, load_pipeline, save_jpeg
from predictor.inference.noise_selection import generate_noise_candidates
from predictor.models import CAMapPredictor
from gen_dataset.datagen.generation_protocol import get_generation_protocol

from gen_dataset.datagen.partial_ca_extractor import FirstCAExtractor


def _get_denoiser(pipe):
    return getattr(pipe, 'transformer', None) or pipe.unet


def _prune_kwargs_for_partial(model_type: str, gen_kwargs: dict) -> dict:
    """Extract only the pipeline kwargs partial_ca_extractor needs.

    We re-use the full gen_kwargs from encode_prompt_for_model since
    the pipelines accept the same kwargs in 1-step and full mode.
    Hunyuan's older-diffusers fallback (empty gen_kwargs) is handled
    by letting the pipeline re-encode from `prompt` argument downstream.
    """
    return dict(gen_kwargs)


def score_all_candidates_ca(
    pipe,
    predictor: CAMapPredictor,
    noises: torch.Tensor,
    args,
    gen_kwargs: dict,
    fallback_prompt: str,
    device: str,
) -> torch.Tensor:
    predictor.eval()
    denoiser = _get_denoiser(pipe)
    extractor = FirstCAExtractor(denoiser)

    use_std = getattr(predictor, 'use_std', False)
    use_max = getattr(predictor, 'use_max', False)
    partial_kwargs = _prune_kwargs_for_partial(args.model_type, gen_kwargs)

    entropies, stds, maxes = [], [], []

    for i in tqdm(range(noises.shape[0]), desc="1-step CA per candidate"):
        noise_i = noises[i:i + 1]
        extractor.reset()
        with torch.inference_mode():
            # Empty partial_kwargs (Hunyuan older-diffusers) -> let pipeline re-encode.
            if partial_kwargs:
                kw = dict(
                    prompt=None,
                    **partial_kwargs,
                    latents=noise_i,
                    num_inference_steps=1,
                    guidance_scale=args.guidance_scale,
                    output_type='latent',
                )
                # PixArt-Sigma: passing negative_prompt_embeds AND default
                # negative_prompt="" trips check_inputs. Suppress the latter
                # whenever the embeds are present.
                if 'negative_prompt_embeds' in partial_kwargs:
                    kw['negative_prompt'] = None
                pipe(**kw)
            else:
                pipe(
                    prompt=fallback_prompt,
                    latents=noise_i,
                    num_inference_steps=1,
                    guidance_scale=args.guidance_scale,
                    output_type='latent',
                )

        if extractor.result is None:
            raise RuntimeError(
                f"Extractor captured nothing for candidate {i}. "
                "Hook site may not match this pipeline."
            )

        # Move the captured stats to a small, persistent tensor on `device`
        # and immediately drop the extractor's reference so the (B, H, T_p, T_N)
        # attention matrix it captured can be freed. Then call empty_cache so
        # PyTorch's allocator coalesces fragmented blocks before the next
        # transformer forward. Without this, 100 sequential pipe(...) calls
        # accumulate fragmentation that OOMs PixArt-Sigma even on 48 GB cards.
        entropies.append(extractor.result['entropy'].to(device).float())
        if use_std:
            stds.append(extractor.result['std'].to(device).float())
        if use_max:
            maxes.append(extractor.result['max'].to(device).float())
        extractor.result = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extractor.remove()

    entropy = torch.stack(entropies, dim=0)                # (N, H, T_p)
    std = torch.stack(stds, dim=0) if use_std else None
    max_ = torch.stack(maxes, dim=0) if use_max else None

    with torch.inference_mode():
        scores = predictor(entropy, std=std, max_=max_).squeeze(-1)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Baseline-1 (CA-Map from DM) image generation + scoring."
    )
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_DIMS.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps. Defaults to the fixed protocol for --model_type.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale. Defaults to the fixed protocol for --model_type.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="output/ca")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=['hpsv2', 'hpsv3', 'image_reward', 'pick_score'],
                        choices=['hpsv2', 'hpsv3', 'image_reward', 'pick_score'])
    args = parser.parse_args()

    protocol = get_generation_protocol(args.model_type)
    if args.steps is None:
        args.steps = protocol['num_inference_steps']
    if args.guidance_scale is None:
        args.guidance_scale = protocol['guidance_scale']

    dims = get_dims(args.model_type)
    latent_shape = dims['latent_shape']
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Baseline 1 (CA-Map from DM) generation")
    print(f"  model_type : {args.model_type}")
    print(f"  prompt     : {args.prompt}")
    print(f"  N / B      : {args.N} / {args.B}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  output     : {out_dir}")
    print(f"  steps/cfg  : {args.steps} / {args.guidance_scale}")
    print(f"  scheduler  : {protocol.get('scheduler') or 'pipeline default'}")
    print("=" * 60)

    print(f"\nLoading {args.model_type} pipeline...")
    pipe = load_pipeline(args.model_type, device=args.device)

    print(f"Loading CA-Map predictor from {args.checkpoint}...")
    predictor, norm_info = CAMapPredictor.from_checkpoint(args.checkpoint, device=args.device)
    y_mean = norm_info.get('y_mean', 0.0)
    y_std = norm_info.get('y_std', 1.0)
    print(f"  use_std={predictor.use_std} use_max={predictor.use_max} "
          f"H={predictor.num_heads} T_p={predictor.seq_len} "
          f"c={predictor.head_compression_ratio}")
    print(f"  target={norm_info.get('target', 'unknown')}  "
          f"y_mean={y_mean:.4f}  y_std={y_std:.4f}")

    pred_embeds, pred_mask, gen_kwargs = encode_prompt_for_model(
        pipe, args.prompt, args.model_type, args.device
    )

    generator = (torch.Generator(device=args.device).manual_seed(args.seed)
                 if args.seed is not None else None)
    dtype = pipe.unet.dtype if hasattr(pipe, 'unet') else pipe.transformer.dtype
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=latent_shape,
        device=args.device,
        dtype=dtype,
        generator=generator,
    )

    print(f"\nScoring {args.N} candidates (each needs a 1-step DM forward)...")
    scores_norm = score_all_candidates_ca(
        pipe, predictor, noises, args, gen_kwargs,
        fallback_prompt=args.prompt, device=args.device,
    )
    scores_raw = scores_norm * y_std + y_mean
    topk = torch.topk(scores_raw, args.B)
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()
    print(f"  top-{args.B} scores (raw {norm_info.get('target', 'y')}): "
          f"{[round(s, 4) for s in top_scores]}")

    selected = noises[topk.indices]

    B = args.B
    expanded_kwargs = {}
    for k, v in gen_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            expanded_kwargs[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            expanded_kwargs[k] = v

    print(f"\nGenerating {B} images from top-{B} of {args.N} candidates...")
    with torch.inference_mode():
        result = pipe(
            prompt=None,
            **expanded_kwargs,
            latents=selected,
            num_images_per_prompt=1,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    images = result.images

    print(f"\nLoading MultiMetricScorer ({args.metrics})...")
    from gen_dataset.metrics.scorer import MultiMetricScorer
    scorer = MultiMetricScorer(metrics=args.metrics, device=args.device)

    print(f"Scoring {B} predictor-selected images...")
    selected_results = []
    for i, img in enumerate(images):
        img_path = out_dir / f"{args.model_type}_ca_{i:02d}.jpg"
        save_jpeg(img, img_path)
        metrics = scorer.score(img, args.prompt, image_path=str(img_path))
        record = {
            'rank': i,
            'candidate_index': int(top_indices[i]),
            'pred_score_raw': float(top_scores[i]),
            'image': str(img_path),
            **metrics,
        }
        selected_results.append(record)
        metric_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"  [{i}] cand={top_indices[i]:3d}  pred={top_scores[i]:.4f}  {metric_str}")

    out = {
        'mode': 'ca_map',
        'prompt': args.prompt,
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'N': args.N,
        'B': args.B,
        'num_inference_steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'scheduler': protocol.get('scheduler'),
        'predictor_target': norm_info.get('target', 'unknown'),
        'selected': selected_results,
    }

    if args.compare:
        print(f"\nGenerating {B} random-noise baseline images...")
        gen_r = (torch.Generator(device=args.device).manual_seed(args.seed + 999)
                 if args.seed is not None else None)
        with torch.inference_mode():
            result_r = pipe(
                prompt=args.prompt,
                num_images_per_prompt=B,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen_r,
            )
        random_results = []
        for i, img in enumerate(result_r.images):
            img_path = out_dir / f"{args.model_type}_random_{i:02d}.jpg"
            save_jpeg(img, img_path)
            metrics = scorer.score(img, args.prompt, image_path=str(img_path))
            record = {'rank': i, 'image': str(img_path), **metrics}
            random_results.append(record)
            metric_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"  [rand {i}]  {metric_str}")
        out['random_baseline'] = random_results

    json_path = out_dir / f"{args.model_type}_ca_scores.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Wrote {json_path}")


if __name__ == "__main__":
    main()
