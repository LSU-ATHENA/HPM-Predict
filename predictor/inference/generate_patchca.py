
import argparse
import json
from pathlib import Path

import torch

from predictor.configs.model_dims import MODEL_DIMS, get_dims
from predictor.inference.generate import encode_prompt_for_model, load_pipeline, save_jpeg
from predictor.inference.noise_selection import generate_noise_candidates
from predictor.models import PatchPromptCAPredictor
from gen_dataset.datagen.generation_protocol import get_generation_protocol


@torch.inference_mode()
def score_candidates_patchca(
    predictor: PatchPromptCAPredictor,
    noises: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    predictor.eval()
    device = noises.device
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)

    all_scores = []
    N = noises.shape[0]
    for i in range(0, N, batch_size):
        batch = noises[i:i + batch_size].float()
        bs = batch.shape[0]
        pe = prompt_embeds.expand(bs, -1, -1).float()
        pm = prompt_mask.expand(bs, -1).float()
        scores = predictor(batch, pe, pm).squeeze(-1)
        all_scores.append(scores)
    return torch.cat(all_scores, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline-2 (external Patch-CA) image generation + scoring."
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
    parser.add_argument("--output-dir", type=str, default="output/patchca")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate B random-noise baseline images")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=['hpsv2', 'hpsv3', 'image_reward', 'pick_score'],
                        choices=['hpsv2', 'hpsv3', 'image_reward', 'pick_score'])
    parser.add_argument("--score_batch_size", type=int, default=32)
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
    print(f"Baseline 2 (Patch-CA) generation")
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

    print(f"Loading Patch-CA predictor from {args.checkpoint}...")
    predictor, norm_info = PatchPromptCAPredictor.from_checkpoint(
        args.checkpoint, device=args.device
    )
    y_mean = norm_info.get('y_mean', 0.0)
    y_std = norm_info.get('y_std', 1.0)
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

    print(f"\nScoring {args.N} candidates with Patch-CA predictor...")
    scores_norm = score_candidates_patchca(
        predictor, noises, pred_embeds, pred_mask,
        batch_size=args.score_batch_size,
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
        img_path = out_dir / f"{args.model_type}_patchca_{i:02d}.jpg"
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
        'mode': 'patchca',
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

    json_path = out_dir / f"{args.model_type}_patchca_scores.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Wrote {json_path}")


if __name__ == "__main__":
    main()
