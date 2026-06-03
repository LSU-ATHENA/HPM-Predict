#!/usr/bin/env python3
"""Extend selected SDXL prompts from sample_idx 20..99 without touching old data."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

try:
    from datagen.base_generator import SEED_RANGE
    from datagen.cross_attn_extractor import CrossAttentionEntropyExtractor
    from datagen.sdxl_1024 import SDXLGenerator
except ImportError:
    from gen_dataset.datagen.base_generator import SEED_RANGE
    from gen_dataset.datagen.cross_attn_extractor import CrossAttentionEntropyExtractor
    from gen_dataset.datagen.sdxl_1024 import SDXLGenerator


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


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_metadata(data_dir: Path):
    by_prompt = defaultdict(list)
    seen_pairs = set()
    for meta_path in sorted(data_dir.glob("metadata*.jsonl")):
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = int(rec["prompt_id"])
                sid = int(rec["sample_idx"])
                key = (pid, sid)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                by_prompt[pid].append(rec)
    return by_prompt, seen_pairs


def deterministic_extension_seeds(master_seed, prompt_id, count, used_seeds, low=SEED_RANGE[0], high=SEED_RANGE[1]):
    rng = random.Random(master_seed * 1_000_003 + prompt_id)
    selected = []
    used = set(used_seeds)
    while len(selected) < count:
        seed = rng.randrange(low, high + 1)
        if seed in used:
            continue
        used.add(seed)
        selected.append(seed)
    return selected


def tensor_dict_to_device(d, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def assert_no_existing_outputs(data_dir: Path, name: str, extract_cross_attn: bool):
    paths = [
        data_dir / "noise" / f"{name}.pt",
        data_dir / "images" / f"{name}.jpg",
    ]
    if extract_cross_attn:
        paths.append(data_dir / "cross_attn" / f"{name}.pt")
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing rank100 extension outputs: "
            + ", ".join(existing)
        )


def load_image(path: Path):
    with Image.open(path) as img:
        return img.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", nargs="+", default=["rank100_train", "rank100_val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output-metadata", default=None)
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
    parser.add_argument("--extract-cross-attn", action="store_true")
    parser.add_argument("--cross-attn-steps", type=int, nargs="+", default=[1, 15, 40])
    parser.add_argument(
        "--recover-existing-outputs",
        action="store_true",
        help=(
            "Recover metadata for partial reruns by reusing existing noise/image "
            "files instead of failing on them. Missing files are still generated."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_rows = [
        r for r in load_jsonl(Path(args.manifest))
        if r.get("split") in set(args.splits)
    ]
    manifest_rows.sort(key=lambda r: (r["split"], int(r["prompt_id"])))

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(manifest_rows)
    if task_id is not None and args.start_idx is None and args.end_idx is None:
        num_tasks = args.num_tasks
        if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT") is not None:
            num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        if num_tasks is None:
            raise ValueError("--num-tasks is required with --task-id unless SLURM_ARRAY_TASK_COUNT is set")
        start_idx, end_idx = compute_chunk(task_id, num_tasks, len(manifest_rows))

    rows = manifest_rows[start_idx:end_idx]
    print(
        f"[rank100] data_dir={data_dir} selected_rows={len(manifest_rows)} "
        f"range=[{start_idx}, {end_idx}) task_id={task_id} "
        f"cross_attn={args.extract_cross_attn}"
    )

    existing_by_prompt, existing_pairs = load_existing_metadata(data_dir)
    missing_base = []
    planned = 0
    skipped_existing = 0
    for row in rows:
        pid = int(row["prompt_id"])
        existing_sample_idxs = {int(r["sample_idx"]) for r in existing_by_prompt.get(pid, [])}
        if not set(range(20)).issubset(existing_sample_idxs):
            missing_base.append(pid)
        for sid in range(int(row["new_sample_idx_start"]), int(row["new_sample_idx_end"]) + 1):
            if (pid, sid) in existing_pairs:
                skipped_existing += 1
            else:
                planned += 1
    if missing_base:
        raise ValueError(f"Selected prompts missing sample_idx 0..19: {missing_base[:20]}")

    print(f"[rank100] planned_new_samples={planned} skipped_existing_metadata={skipped_existing}")
    if args.dry_run:
        return

    if args.output_metadata is None:
        if task_id is None:
            output_metadata = data_dir / "metadata_rank100_extension.jsonl"
        else:
            output_metadata = data_dir / f"metadata_rank100_extension_task_{task_id}.jsonl"
    else:
        output_metadata = Path(args.output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    generator = SDXLGenerator(
        save_dir=str(data_dir),
        prompts=[],
        num_images_per_prompt=0,
        master_seed=args.seed,
        device=args.device,
        metrics=args.metrics,
        task_id=None,
        extract_cross_attn=args.extract_cross_attn,
        cross_attn_steps=tuple(args.cross_attn_steps),
    )
    pipe = generator.setup_pipeline()

    extractor = None
    if args.extract_cross_attn:
        denoiser = getattr(pipe, "transformer", None) or pipe.unet
        extractor = CrossAttentionEntropyExtractor(
            denoiser, extract_steps=tuple(args.cross_attn_steps)
        )
        print(
            f"[cross_attn] Hooked {len(extractor.layer_names)} layers; "
            f"steps={sorted(args.cross_attn_steps)}"
        )

    try:
        with open(output_metadata, "a") as meta_file:
            for row in tqdm(rows, desc="Rank100 prompts"):
                pid = int(row["prompt_id"])
                prompt = row["prompt"]
                sid_start = int(row["new_sample_idx_start"])
                sid_end = int(row["new_sample_idx_end"])

                existing_records = existing_by_prompt.get(pid, [])
                existing_sample_idxs = {int(r["sample_idx"]) for r in existing_records}
                existing_seeds = {int(r["seed"]) for r in existing_records if r.get("seed") is not None}
                new_sids = [sid for sid in range(sid_start, sid_end + 1) if sid not in existing_sample_idxs]
                seeds = deterministic_extension_seeds(args.seed, pid, len(new_sids), existing_seeds)

                embed_path = data_dir / "embeds" / f"p{pid:04d}.pt"
                if embed_path.exists():
                    embeds_dict = torch.load(embed_path, map_location="cpu", weights_only=False)
                    embeds_dict = tensor_dict_to_device(embeds_dict, args.device)
                else:
                    embeds_dict = generator.encode_and_save_prompt(pipe, prompt, embed_path)

                for sid, seed in zip(new_sids, seeds):
                    name = f"p{pid:04d}_s{sid:02d}"
                    noise_path = data_dir / "noise" / f"{name}.pt"
                    image_path = data_dir / "images" / f"{name}.jpg"
                    ca_path = data_dir / "cross_attn" / f"{name}.pt"

                    if not args.recover_existing_outputs:
                        assert_no_existing_outputs(data_dir, name, args.extract_cross_attn)

                    noise = None
                    if noise_path.exists():
                        noise = torch.load(noise_path, map_location=args.device, weights_only=True)
                    else:
                        noise = generator.generate_noise(seed)
                        torch.save(noise.cpu().half(), noise_path)

                    need_forward = not image_path.exists() or (
                        extractor is not None and not ca_path.exists()
                    )
                    if need_forward:
                        image = generator.generate_image(pipe, embeds_dict, noise, extractor=extractor)
                        if not image_path.exists():
                            image.save(image_path)
                    else:
                        image = load_image(image_path)

                    scores = generator.scorer.score(image, prompt, image_path=str(image_path))
                    meta = {
                        "prompt_id": pid,
                        "sample_idx": sid,
                        "seed": int(seed),
                        "prompt": prompt,
                        "rank100_split": row["split"],
                        "rank100_extension": True,
                        **scores,
                    }

                    if extractor is not None and need_forward:
                        ca_results = extractor.get_results()
                        if not ca_path.exists():
                            torch.save(ca_results, ca_path)
                        meta.update(extractor.get_metadata_scalars(ca_results))

                    meta_file.write(json.dumps(meta) + "\n")
                    meta_file.flush()
                    existing_pairs.add((pid, sid))
                    existing_by_prompt[pid].append(meta)

                    del image, noise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if extractor is not None:
            extractor.remove_hooks()

    print(f"Done. Extension metadata: {output_metadata}")


if __name__ == "__main__":
    main()
