#!/usr/bin/env python3
"""Recover Hunyuan-DiT samples from an explicit prompt/sample worklist."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


SEED_RANGE = (0, 2**31 - 1)


def compute_chunk(task_id, num_tasks, total):
    chunk = total // num_tasks
    remainder = total % num_tasks
    if task_id < remainder:
        start = task_id * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (task_id - remainder) * chunk
        end = start + chunk
    return start, end


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def load_worklist(path: Path):
    rows = list(iter_jsonl(path))
    for row in rows:
        row["prompt_id"] = int(row["prompt_id"])
        row["sample_idx"] = int(row["sample_idx"])
        row["prompt"] = str(row["prompt"]).strip()
        if not row["prompt"]:
            raise ValueError(f"Empty prompt for prompt_id={row['prompt_id']}")
    return rows


def load_existing_metadata(data_dir: Path):
    records = {}
    seeds_by_prompt = defaultdict(set)
    paths = []
    main = data_dir / "metadata.jsonl"
    if main.exists():
        paths.append(main)
    paths.extend(sorted(data_dir.glob("metadata_[0-9]*.jsonl")))
    generation_main = data_dir / "metadata_generation.jsonl"
    if generation_main.exists():
        paths.append(generation_main)
    paths.extend(sorted(data_dir.glob("metadata_generation_*.jsonl")))
    paths.extend(sorted(data_dir.glob("metadata_generation_recovery_*.jsonl")))
    for path in paths:
        for rec in iter_jsonl(path):
            pid = int(rec["prompt_id"])
            sid = int(rec["sample_idx"])
            key = (pid, sid)
            if key not in records:
                records[key] = rec
            if rec.get("seed") is not None:
                seeds_by_prompt[pid].add(int(rec["seed"]))
    return records, seeds_by_prompt


def tensor_dict_to_device(d, device):
    import torch

    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def load_image(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def deterministic_seed(master_seed, prompt_id, sample_idx, used_seeds):
    attempt = 0
    while True:
        rng = random.Random(
            master_seed * 1_000_003
            + prompt_id * 10_007
            + sample_idx * 101
            + attempt
        )
        seed = rng.randrange(SEED_RANGE[0], SEED_RANGE[1] + 1)
        if seed not in used_seeds:
            return seed
        attempt += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args()

    try:
        from datagen.hunyuan_dit import HunyuanDiTGenerator
    except ImportError:
        from gen_dataset.datagen.hunyuan_dit import HunyuanDiTGenerator
    import torch
    from tqdm import tqdm

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is None:
        task_id = 0
    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT") is not None:
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    if num_tasks is None:
        num_tasks = 1

    rows = load_worklist(Path(args.worklist))
    start_idx, end_idx = compute_chunk(task_id, num_tasks, len(rows))
    shard_rows = rows[start_idx:end_idx]
    data_dir = Path(args.save_dir)
    for subdir in ("images", "noise", "embeds"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)

    existing_records, seeds_by_prompt = load_existing_metadata(data_dir)
    print(
        f"[hunyuan-sample-worklist] worklist={args.worklist} samples={len(rows)} "
        f"range=[{start_idx}, {end_idx}) task_id={task_id}/{num_tasks} "
        f"shard_samples={len(shard_rows)}"
    )
    if not shard_rows:
        print("Nothing assigned to this shard.")
        return

    generator = HunyuanDiTGenerator(
        save_dir=str(data_dir),
        prompts=[],
        num_images_per_prompt=0,
        master_seed=args.seed,
        device=args.device,
        metrics=[],
        task_id=None,
        extract_cross_attn=False,
    )
    pipe = generator.setup_pipeline()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    meta_path = data_dir / f"metadata_generation_recovery_{task_id}.jsonl"
    written = 0
    regenerated_images = 0
    regenerated_noise = 0

    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for row in tqdm(shard_rows, desc="Hunyuan sample worklist"):
            pid = int(row["prompt_id"])
            sid = int(row["sample_idx"])
            prompt = row["prompt"]
            name = f"p{pid:04d}_s{sid:02d}"
            key = (pid, sid)

            embed_path = data_dir / "embeds" / f"p{pid:04d}.pt"
            if embed_path.exists():
                embeds_dict = torch.load(embed_path, map_location="cpu", weights_only=False)
                embeds_dict = tensor_dict_to_device(embeds_dict, args.device)
            else:
                embeds_dict = generator.encode_and_save_prompt(pipe, prompt, embed_path)

            existing_rec = existing_records.get(key)
            if existing_rec and existing_rec.get("seed") is not None:
                seed = int(existing_rec["seed"])
            else:
                used_seeds = seeds_by_prompt[pid]
                seed = deterministic_seed(args.seed, pid, sid, used_seeds)
                used_seeds.add(seed)

            noise_path = data_dir / "noise" / f"{name}.pt"
            image_path = data_dir / "images" / f"{name}.jpg"

            if noise_path.exists():
                noise = torch.load(noise_path, map_location=args.device, weights_only=True)
            else:
                noise = generator.generate_noise(seed)
                torch.save(noise.cpu().half(), noise_path)
                regenerated_noise += 1

            if image_path.exists():
                image = load_image(image_path)
            else:
                image = generator.generate_image(pipe, embeds_dict, noise, extractor=None)
                image.save(image_path)
                regenerated_images += 1

            if key not in existing_records:
                meta = {
                    "prompt_id": pid,
                    "sample_idx": sid,
                    "seed": int(seed),
                    "prompt": prompt,
                }
                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                meta_file.flush()
                existing_records[key] = meta
                written += 1

            del image, noise, embeds_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"Done. metadata_written={written} regenerated_images={regenerated_images} "
        f"regenerated_noise={regenerated_noise} metadata_path={meta_path}"
    )


if __name__ == "__main__":
    main()
