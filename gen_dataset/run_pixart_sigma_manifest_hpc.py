#!/usr/bin/env python3
"""Generate PixArt-Sigma samples from an explicit prompt_id-ordered manifest."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


SEED_RANGE = (0, 2**31 - 1)
ALL_METRICS = ["hpsv2", "hpsv3", "image_reward", "pick_score"]


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


def load_manifest(path: Path, n_prompts: int | None):
    rows = list(iter_jsonl(path))
    rows.sort(key=lambda r: int(r["prompt_id"]))
    if n_prompts is not None and n_prompts > 0:
        rows = rows[:n_prompts]
    for expected_pid, row in enumerate(rows):
        pid = int(row["prompt_id"])
        if pid != expected_pid:
            raise ValueError(
                "Prompt manifest must be contiguous from prompt_id 0. "
                f"Expected {expected_pid}, found {pid}"
            )
        row["prompt_id"] = pid
        row["prompt"] = str(row["prompt"]).strip()
        if not row["prompt"]:
            raise ValueError(f"Empty prompt for prompt_id={pid}")
    return rows


def load_existing_metadata(data_dir: Path):
    seen_pairs = set()
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
            seen_pairs.add((pid, sid))
            if rec.get("seed") is not None:
                seeds_by_prompt[pid].add(int(rec["seed"]))
    return seen_pairs, seeds_by_prompt


def tensor_dict_to_device(d, device):
    import torch

    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def load_image(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def save_jpg(image, path: Path):
    image.convert("RGB").save(path, format="JPEG")


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


def existing_output_paths(data_dir: Path, name: str):
    paths = [
        data_dir / "noise" / f"{name}.pt",
        data_dir / "images" / f"{name}.jpg",
    ]
    return [p for p in paths if p.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--images-per-prompt", type=int, required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Write generation metadata only and do not load reward models.",
    )
    parser.add_argument(
        "--recover-existing-outputs",
        action="store_true",
        help="Reuse existing noise/image files when metadata was not written before interruption.",
    )
    args = parser.parse_args()

    try:
        from datagen.pixart_sigma import PixartSigmaGenerator
    except ImportError:
        from gen_dataset.datagen.pixart_sigma import PixartSigmaGenerator
    import torch
    from tqdm import tqdm

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    rows = load_manifest(Path(args.prompt_manifest), args.n_prompts)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(rows)
    if task_id is not None and args.start_idx is None and args.end_idx is None:
        num_tasks = args.num_tasks
        if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT") is not None:
            num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        if num_tasks is None:
            raise ValueError("--num-tasks is required with --task-id")
        start_idx, end_idx = compute_chunk(task_id, num_tasks, len(rows))
    shard_rows = rows[start_idx:end_idx]

    data_dir = Path(args.save_dir)
    for subdir in ("images", "noise", "embeds"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)

    existing_pairs, seeds_by_prompt = load_existing_metadata(data_dir)
    planned = []
    for row in shard_rows:
        pid = int(row["prompt_id"])
        for sid in range(args.images_per_prompt):
            if (pid, sid) not in existing_pairs:
                planned.append((pid, sid, row["prompt"]))

    print(
        f"[pixart-sigma-manifest] manifest={args.prompt_manifest} prompts={len(rows)} "
        f"range=[{start_idx}, {end_idx}) task_id={task_id} "
        f"images_per_prompt={args.images_per_prompt} planned_missing={len(planned)} "
        f"skip_scoring={args.skip_scoring}"
    )
    if not planned:
        print("Nothing to generate.")
        return

    generator = PixartSigmaGenerator(
        save_dir=str(data_dir),
        prompts=[],
        num_images_per_prompt=0,
        master_seed=args.seed,
        device=args.device,
        metrics=[] if args.skip_scoring else args.metrics,
        task_id=None,
        extract_cross_attn=False,
    )
    pipe = generator.setup_pipeline()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if args.skip_scoring:
        meta_name = f"metadata_generation_{task_id}.jsonl" if task_id is not None else "metadata_generation.jsonl"
    else:
        meta_name = f"metadata_{task_id}.jsonl" if task_id is not None else "metadata.jsonl"
    meta_path = data_dir / meta_name

    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for row in tqdm(shard_rows, desc="PixArt-Sigma manifest prompts"):
            pid = int(row["prompt_id"])
            prompt = row["prompt"]
            missing_sids = [
                sid
                for sid in range(args.images_per_prompt)
                if (pid, sid) not in existing_pairs
            ]
            if not missing_sids:
                continue

            embed_path = data_dir / "embeds" / f"p{pid:04d}.pt"
            if embed_path.exists():
                embeds_dict = torch.load(embed_path, map_location="cpu", weights_only=False)
                embeds_dict = tensor_dict_to_device(embeds_dict, args.device)
            else:
                embeds_dict = generator.encode_and_save_prompt(pipe, prompt, embed_path)

            used_seeds = seeds_by_prompt[pid]
            for sid in missing_sids:
                name = f"p{pid:04d}_s{sid:02d}"
                existing_outputs = existing_output_paths(data_dir, name)
                if existing_outputs and not args.recover_existing_outputs:
                    raise FileExistsError(
                        "Found existing output without metadata; rerun with "
                        f"--recover-existing-outputs: {existing_outputs}"
                    )

                seed = deterministic_seed(args.seed, pid, sid, used_seeds)
                used_seeds.add(seed)
                noise_path = data_dir / "noise" / f"{name}.pt"
                image_path = data_dir / "images" / f"{name}.jpg"

                if noise_path.exists():
                    noise = torch.load(noise_path, map_location=args.device, weights_only=True)
                else:
                    noise = generator.generate_noise(seed)
                    torch.save(noise.cpu().half(), noise_path)

                if image_path.exists():
                    image = load_image(image_path)
                else:
                    image = generator.generate_image(pipe, embeds_dict, noise, extractor=None)
                    save_jpg(image, image_path)

                meta = {
                    "prompt_id": pid,
                    "sample_idx": sid,
                    "seed": int(seed),
                    "prompt": prompt,
                }
                if not args.skip_scoring:
                    scores = generator.scorer.score(image, prompt, image_path=str(image_path))
                    meta.update(scores)

                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                meta_file.flush()
                existing_pairs.add((pid, sid))

                del image, noise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del embeds_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Done. Metadata: {meta_path}")


if __name__ == "__main__":
    main()
