#!/usr/bin/env python3
"""SDXL dataset generation with cross-attention entropy extraction.

For HPC parallel runs, set SLURM_ARRAY_TASK_ID env var to split work across jobs.
"""
import argparse
import os

from datagen.prompt_loader import load_train_prompts
from datagen.sdxl_1024 import SDXLGenerator

ALL_METRICS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-prompts', type=int, default=3500)
    p.add_argument('--images-per-prompt', type=int, default=50)
    p.add_argument('--save-dir', type=str, default='data/generated/sdxl_1024')
    p.add_argument('--prompts-file', type=str, default='data/pickscore_train_prompts.json')
    p.add_argument('--data-dir', type=str, default='data/')
    args = p.parse_args()

    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    task_id = int(task_id) if task_id is not None else None

    prompts = load_train_prompts(
        path=args.prompts_file, n=args.n_prompts,
        seed=42, data_dir=args.data_dir,
    )

    gen = SDXLGenerator(
        save_dir=args.save_dir, prompts=prompts,
        num_images_per_prompt=args.images_per_prompt,
        master_seed=42, device='cuda',
        metrics=ALL_METRICS, task_id=task_id,
    )
    gen.run()


if __name__ == '__main__':
    main()
