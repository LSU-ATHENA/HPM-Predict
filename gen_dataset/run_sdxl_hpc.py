#!/usr/bin/env python3
"""SDXL v2 dataset generation for HPC (SLURM job arrays).

Features over run_sdxl.py:
  - Automatic SLURM chunk splitting via SLURM_ARRAY_TASK_ID
  - Cross-attention entropy extraction (on by default)
  - Configurable metrics, seed, device, extraction steps
  - Resumable: skips already-generated (prompt_id, sample_idx) pairs

Usage:
  # Single GPU (e.g. ATHENA-Eris test):
  python run_sdxl_hpc.py --n-prompts 3 --images-per-prompt 5 --save-dir /tmp/test

  # SLURM array (auto-chunks by SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_COUNT):
  sbatch submit_sdxl_datagen_v2.sh
"""
import argparse
import os

from datagen.prompt_loader import load_train_prompts
from datagen.sdxl_1024 import SDXLGenerator

ALL_METRICS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score']


def compute_chunk(task_id, num_tasks, total):
    """Compute [start, end) range for a SLURM array task."""
    chunk = total // num_tasks
    remainder = total % num_tasks
    if task_id < remainder:
        start = task_id * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (task_id - remainder) * chunk
        end = start + chunk
    return start, end


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-prompts', type=int, default=3500)
    p.add_argument('--images-per-prompt', type=int, default=50)
    p.add_argument('--save-dir', type=str, default='data/generated/sdxl_1024_v2')
    p.add_argument('--prompts-file', type=str, default='data/pickscore_train_prompts.json')
    p.add_argument('--data-dir', type=str, default='data/')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--start-idx', type=int, default=None)
    p.add_argument('--end-idx', type=int, default=None)
    p.add_argument('--task-id', type=int, default=None)
    p.add_argument('--num-tasks', type=int, default=None,
                   help='Total SLURM array tasks (auto-detected from SLURM_ARRAY_TASK_COUNT)')
    p.add_argument('--no-cross-attn', action='store_true',
                   help='Disable cross-attention entropy extraction')
    p.add_argument('--cross-attn-steps', type=int, nargs='+', default=[1, 15, 40])
    p.add_argument('--metrics', type=str, nargs='+', default=ALL_METRICS,
                   choices=ALL_METRICS)
    args = p.parse_args()

    # Resolve task ID: CLI flag > SLURM env
    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        task_id = int(env_id) if env_id is not None else None

    prompts = load_train_prompts(
        path=args.prompts_file, n=args.n_prompts,
        seed=args.seed, data_dir=args.data_dir,
    )

    # Compute start/end from SLURM task if not explicitly given
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(prompts)

    if task_id is not None and args.start_idx is None and args.end_idx is None:
        num_tasks = args.num_tasks
        if num_tasks is None:
            env_count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
            if env_count is not None:
                num_tasks = int(env_count)
            else:
                raise ValueError(
                    "--num-tasks required when using --task-id without --start-idx/--end-idx"
                )
        start_idx, end_idx = compute_chunk(task_id, num_tasks, len(prompts))

    print(f"[config] prompts={len(prompts)} range=[{start_idx}, {end_idx}) "
          f"task_id={task_id} cross_attn={not args.no_cross_attn} "
          f"steps={args.cross_attn_steps} metrics={args.metrics}")

    gen = SDXLGenerator(
        save_dir=args.save_dir, prompts=prompts,
        num_images_per_prompt=args.images_per_prompt,
        master_seed=args.seed, device=args.device,
        metrics=args.metrics, task_id=task_id,
        extract_cross_attn=not args.no_cross_attn,
        cross_attn_steps=tuple(args.cross_attn_steps),
    )
    gen.run(start_idx=start_idx, end_idx=end_idx)


if __name__ == '__main__':
    main()
