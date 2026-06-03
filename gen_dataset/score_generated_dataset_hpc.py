#!/usr/bin/env python3
"""Score generated image datasets after diffusion generation has exited."""

import argparse
import json
import os
from pathlib import Path


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


def load_generation_records(data_dir: Path):
    records = {}
    paths = []
    main = data_dir / "metadata_generation.jsonl"
    if main.exists():
        paths.append(main)
    paths.extend(sorted(data_dir.glob("metadata_generation_*.jsonl")))
    paths.extend(sorted(data_dir.glob("metadata_generation_recovery_*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No metadata_generation*.jsonl files found in {data_dir}")

    for path in paths:
        for rec in iter_jsonl(path):
            pid = int(rec["prompt_id"])
            sid = int(rec["sample_idx"])
            key = (pid, sid)
            if key in records:
                prior = records[key]
                if prior.get("seed") != rec.get("seed") or prior.get("prompt") != rec.get("prompt"):
                    raise ValueError(f"Conflicting generation metadata for {key} in {path}")
                continue
            normalized = dict(rec)
            normalized["prompt_id"] = pid
            normalized["sample_idx"] = sid
            normalized["seed"] = int(rec["seed"])
            normalized["prompt"] = str(rec["prompt"])
            records[key] = normalized
    return [records[key] for key in sorted(records)]


def load_existing_scored_pairs(data_dir: Path):
    pairs = set()
    paths = []
    main = data_dir / "metadata.jsonl"
    if main.exists():
        paths.append(main)
    paths.extend(sorted(data_dir.glob("metadata_[0-9]*.jsonl")))
    paths.extend(sorted(data_dir.glob("metadata_recovery_*.jsonl")))
    for path in paths:
        for rec in iter_jsonl(path):
            pairs.add((int(rec["prompt_id"]), int(rec["sample_idx"])))
    return pairs


def load_image(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
    args = parser.parse_args()

    try:
        from metrics import MultiMetricScorer
    except ImportError:
        from gen_dataset.metrics import MultiMetricScorer
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

    data_dir = Path(args.data_dir)
    try:
        records = load_generation_records(data_dir)
    except FileNotFoundError:
        if (data_dir / "metadata.jsonl").exists():
            print(f"No generation shard metadata found; finalized metadata exists in {data_dir}")
            return
        raise
    start_idx, end_idx = compute_chunk(task_id, num_tasks, len(records))
    shard_records = records[start_idx:end_idx]
    existing_scored = load_existing_scored_pairs(data_dir)
    to_score = [
        rec
        for rec in shard_records
        if (int(rec["prompt_id"]), int(rec["sample_idx"])) not in existing_scored
    ]

    print(
        f"[score-generated] data_dir={data_dir} records={len(records)} "
        f"range=[{start_idx}, {end_idx}) task_id={task_id}/{num_tasks} "
        f"to_score={len(to_score)} metrics={args.metrics}"
    )
    if not to_score:
        print("Nothing to score.")
        return

    scorer = MultiMetricScorer(metrics=args.metrics, device=args.device)
    meta_path = data_dir / f"metadata_{task_id}.jsonl"
    written = 0

    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for rec in tqdm(to_score, desc="Scoring generated images"):
            pid = int(rec["prompt_id"])
            sid = int(rec["sample_idx"])
            name = f"p{pid:04d}_s{sid:02d}"
            image_path = data_dir / "images" / f"{name}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing generated image: {image_path}")

            image = load_image(image_path)
            scores = scorer.score(image, rec["prompt"], image_path=str(image_path))
            out = dict(rec)
            out.update(scores)
            meta_file.write(json.dumps(out, ensure_ascii=False) + "\n")
            meta_file.flush()
            existing_scored.add((pid, sid))
            written += 1

    print(f"Done. metadata_written={written} metadata_path={meta_path}")


if __name__ == "__main__":
    main()
