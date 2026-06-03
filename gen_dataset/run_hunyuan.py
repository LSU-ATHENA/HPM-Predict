#!/usr/bin/env python3
import argparse
from datagen.prompt_loader import load_train_prompts
from datagen.hunyuan_dit import HunyuanDiTGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-prompts', type=int, default=5000)
    p.add_argument('--images-per-prompt', type=int, default=10)
    p.add_argument('--save-dir', type=str, default='data/generated/hunyuan_dit_1024')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--start-idx', type=int, default=0)
    p.add_argument('--end-idx', type=int, default=None)
    p.add_argument('--prompts-file', type=str, default='data/pickscore_train_prompts.json')
    p.add_argument('--data-dir', type=str, default='data/')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed-range', type=int, nargs=2, default=[0, 2**31 - 1],
                   metavar=('LOW', 'HIGH'))
    p.add_argument('--metrics', type=str, nargs='+', default=None)
    p.add_argument('--task-id', type=int, default=None)
    args = p.parse_args()

    prompts = load_train_prompts(
        path=args.prompts_file, n=args.n_prompts,
        seed=args.seed, data_dir=args.data_dir,
    )
    end_idx = args.end_idx or len(prompts)

    gen = HunyuanDiTGenerator(
        save_dir=args.save_dir, prompts=prompts,
        num_images_per_prompt=args.images_per_prompt,
        master_seed=args.seed, seed_range=tuple(args.seed_range),
        device=args.device, metrics=args.metrics,
        task_id=args.task_id,
    )
    gen.run(start_idx=args.start_idx, end_idx=end_idx)


if __name__ == '__main__':
    main()
