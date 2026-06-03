#!/usr/bin/env python3
"""Stage-2 rank100 fine-tuning with LambdaLoss@10 + alpha MAE."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from predictor.configs.model_dims import get_dims
from predictor.models import (
    get_ca_map_model,
    get_model,
    get_patchca_model,
    NOISE_ENCODERS,
    TEXT_ENCODERS,
)
from predictor.training.dataloader import AVAILABLE_TARGETS, denormalize
from predictor.training.losses import LambdaLossAtK
from predictor.training.rank100_dataloader import (
    STAGE1_SPLITS,
    build_rank100_loader,
    compute_global_normalization,
    group_records_by_prompt,
    load_metadata_records,
    load_prompt_split_ids,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def true_rank_desc(values: np.ndarray, selected_idx: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return float(ranks[int(selected_idx)])


def ndcg_from_scores(pred: np.ndarray, rewards: np.ndarray, k: int) -> float:
    if len(pred) < 2:
        return float("nan")
    reward_range = float(np.max(rewards) - np.min(rewards))
    if reward_range < 1e-12:
        return 1.0
    rel = (rewards - np.min(rewards)) / reward_range
    gains = np.power(2.0, rel) - 1.0
    order = np.argsort(-pred)
    ideal = np.argsort(-rewards)
    kk = min(k, len(pred))
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float(np.sum(gains[order[:kk]] * discounts))
    idcg = float(np.sum(gains[ideal[:kk]] * discounts))
    return dcg / idcg if idcg > 1e-12 else 1.0


def spearman_np(pred: np.ndarray, rewards: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-12 or np.std(rewards) < 1e-12:
        return float("nan")
    pred_rank = np.argsort(np.argsort(pred)).astype(np.float64)
    reward_rank = np.argsort(np.argsort(rewards)).astype(np.float64)
    pred_rank -= pred_rank.mean()
    reward_rank -= reward_rank.mean()
    denom = np.sqrt(np.sum(pred_rank ** 2) * np.sum(reward_rank ** 2))
    return float(np.sum(pred_rank * reward_rank) / denom) if denom > 1e-12 else float("nan")


def load_checkpoint_normalization(path: str):
    if not path:
        return None, {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("normalization", {}), ckpt.get("model_config", {})


def build_model(args, camap_shape=None):
    stage1_cfg = getattr(args, "stage1_model_config", {}) or {}
    if args.model_kind == "traditional":
        dims = get_dims(args.model_type)
        return get_model(
            noise_enc=stage1_cfg.get("noise_enc", args.noise_enc),
            text_enc=stage1_cfg.get("text_enc", args.text_enc),
            dropout=stage1_cfg.get("dropout", args.dropout),
            num_heads=stage1_cfg.get("num_heads", 1),
            spatial_size=stage1_cfg.get("spatial_size", dims["spatial_size"]),
            in_channels=stage1_cfg.get("in_channels", dims["latent_shape"][0]),
            embed_dim=stage1_cfg.get("embed_dim", dims["embed_dim"]),
            seq_len=stage1_cfg.get("seq_len", args.seq_len_override or dims["seq_len"]),
            pos_encoding=stage1_cfg.get("pos_encoding", "sinusoidal"),
            text_encoder_project_dims=stage1_cfg.get("text_encoder_project_dims"),
        )
    if args.model_kind == "patchca":
        return get_patchca_model(
            model_type=args.model_type,
            patch_size=args.patch_size,
            ca_embed_dim=args.ca_embed_dim,
            ca_num_heads=args.ca_num_heads,
            head_compression_ratio=args.head_compression,
            use_std=args.use_std,
            use_max=args.use_max,
            dropout=args.dropout,
            ffn_ratio=args.ffn_ratio,
        )
    if args.model_kind == "camap":
        if camap_shape is None:
            raise ValueError("camap_shape is required for CA-Map rank100 training")
        num_heads, seq_len = camap_shape
        return get_ca_map_model(
            num_heads=num_heads,
            seq_len=seq_len,
            head_compression_ratio=args.head_compression,
            use_std=args.use_std,
            use_max=args.use_max,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model_kind {args.model_kind}")


def load_model_state(model, checkpoint_path: str):
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)


def model_config_for_checkpoint(model, args, camap_shape=None):
    stage1_cfg = getattr(args, "stage1_model_config", {}) or {}
    if args.model_kind == "traditional":
        dims = get_dims(args.model_type)
        return {
            "noise_enc": stage1_cfg.get("noise_enc", args.noise_enc),
            "text_enc": stage1_cfg.get("text_enc", args.text_enc),
            "dropout": stage1_cfg.get("dropout", args.dropout),
            "num_heads": stage1_cfg.get("num_heads", 1),
            "model_type": args.model_type,
            "spatial_size": stage1_cfg.get("spatial_size", dims["spatial_size"]),
            "in_channels": stage1_cfg.get("in_channels", dims["latent_shape"][0]),
            "embed_dim": stage1_cfg.get("embed_dim", dims["embed_dim"]),
            "seq_len": stage1_cfg.get("seq_len", args.seq_len_override or dims["seq_len"]),
            "pos_encoding": stage1_cfg.get("pos_encoding", "sinusoidal"),
            "text_encoder_project_dims": getattr(model.text_encoder, "project_dims", None),
        }
    if hasattr(model, "_config_dict"):
        return model._config_dict(model_type=args.model_type)
    if args.model_kind == "camap":
        return {
            "model_type": args.model_type,
            "num_heads": int(camap_shape[0]),
            "seq_len": int(camap_shape[1]),
            "head_compression": args.head_compression,
            "use_std": args.use_std,
            "use_max": args.use_max,
            "dropout": args.dropout,
        }
    raise ValueError(f"Cannot build config for {args.model_kind}")


def forward_batch(model, batch, args, device):
    if args.model_kind in {"traditional", "patchca"}:
        noise = batch["noise"].to(device, non_blocking=True)
        prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
        prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)
        bsz, cand = noise.shape[:2]
        noise = noise.view(bsz * cand, *noise.shape[2:])
        prompt_embeds = prompt_embeds.unsqueeze(1).expand(-1, cand, -1, -1).reshape(
            bsz * cand, *prompt_embeds.shape[1:]
        )
        prompt_mask = prompt_mask.unsqueeze(1).expand(-1, cand, -1).reshape(
            bsz * cand, prompt_mask.shape[-1]
        )
        pred = model(noise, prompt_embeds, prompt_mask).float().view(bsz, cand)
        return pred

    entropy = batch["entropy"].to(device, non_blocking=True)
    bsz, cand = entropy.shape[:2]
    entropy = entropy.view(bsz * cand, *entropy.shape[2:])
    std = None
    max_ = None
    if args.use_std:
        std = batch["std"].to(device, non_blocking=True).view(bsz * cand, *entropy.shape[1:])
    if args.use_max:
        max_ = batch["max"].to(device, non_blocking=True).view(bsz * cand, *entropy.shape[1:])
    return model(entropy, std=std, max_=max_).float().view(bsz, cand)


def train_one_epoch(model, loader, optimizer, lambda_loss_fn, args, device):
    model.train()
    totals = {"loss": 0.0, "lambda": 0.0, "mae": 0.0, "groups": 0}
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"

    for step, batch in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds = forward_batch(model, batch, args, device)
        preds = preds.float()
        rel = batch["rel"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        lambda_loss = lambda_loss_fn(preds, rel)
        mae_loss = F.l1_loss(preds, y)
        loss = lambda_loss + args.alpha * mae_loss
        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        groups = int(y.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * groups
        totals["lambda"] += float(lambda_loss.detach().cpu()) * groups
        totals["mae"] += float(mae_loss.detach().cpu()) * groups
        totals["groups"] += groups

    n = max(1, totals["groups"])
    return {
        "loss": totals["loss"] / n,
        "lambda_loss_at_10": totals["lambda"] / n,
        "mae_normalized_train_loss": totals["mae"] / n,
    }


@torch.no_grad()
def evaluate_rank100(model, loader, args, device, y_mean, y_std) -> Dict[str, float]:
    model.eval()
    per_prompt = []
    mae_values = []
    use_amp = device.type == "cuda"

    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds_norm = forward_batch(model, batch, args, device)
        preds_norm = preds_norm.float()
        raw_y = batch["raw_y"].to(device, non_blocking=True).float()
        preds_raw = denormalize(preds_norm, y_mean, y_std)
        mae_values.append((preds_raw - raw_y).abs().detach().cpu().reshape(-1).numpy())

        preds_np = preds_norm.detach().cpu().numpy()
        raw_np = raw_y.detach().cpu().numpy()
        for i in range(preds_np.shape[0]):
            selected = int(np.argmax(preds_np[i]))
            rank = true_rank_desc(raw_np[i], selected)
            oracle = float(np.max(raw_np[i]))
            selected_reward = float(raw_np[i, selected])
            per_prompt.append({
                "selected_reward": selected_reward,
                "oracle_regret": oracle - selected_reward,
                "rank": rank,
                "top1": 1.0 if rank <= 1.0 else 0.0,
                "top5": 1.0 if rank <= 5.0 else 0.0,
                "top10": 1.0 if rank <= 10.0 else 0.0,
                "bottom25": 1.0 if rank > (raw_np.shape[1] - 25) else 0.0,
                "rank_ge_92": 1.0 if rank >= 92.0 else 0.0,
                "ndcg_10": ndcg_from_scores(preds_np[i], raw_np[i], 10),
                "ndcg_20": ndcg_from_scores(preds_np[i], raw_np[i], 20),
                "pool_srcc": spearman_np(preds_np[i], raw_np[i]),
            })

    if not per_prompt:
        raise ValueError("No validation prompts were evaluated")

    def mean_key(key):
        vals = np.asarray([row[key] for row in per_prompt], dtype=np.float64)
        return float(np.nanmean(vals))

    ranks = np.asarray([row["rank"] for row in per_prompt], dtype=np.float64)
    mae = float(np.concatenate(mae_values).mean()) if mae_values else float("nan")
    return {
        "n_prompts": len(per_prompt),
        "selected_reward": mean_key("selected_reward"),
        "oracle_regret": mean_key("oracle_regret"),
        "top1_oracle_hit": mean_key("top1"),
        "top5_hit": mean_key("top5"),
        "top10_hit": mean_key("top10"),
        "ndcg_10": mean_key("ndcg_10"),
        "ndcg_20": mean_key("ndcg_20"),
        "pool_srcc": mean_key("pool_srcc"),
        "mae_raw": mae,
        "selected_rank_mean": float(np.mean(ranks)),
        "selected_rank_median": float(np.median(ranks)),
        "bottom_25_rate": mean_key("bottom25"),
        "rank_ge_92_count": int(np.sum([row["rank_ge_92"] for row in per_prompt])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_kind", required=True, choices=["traditional", "patchca", "camap"])
    parser.add_argument("--model_type", default="sdxl")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--rank100_manifest", required=True)
    parser.add_argument("--target", default="pick_score", choices=AVAILABLE_TARGETS)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--exp_name", required=True)

    parser.add_argument("--train_split", default="rank100_train")
    parser.add_argument("--val_split", default="rank100_val")
    parser.add_argument("--candidates_per_prompt", type=int, default=100, choices=[20, 50, 100])
    parser.add_argument("--batch_size", type=int, default=2, help="Physical prompt groups per batch.")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lambda_k", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--primary_metric", default="selected_reward",
                        choices=["selected_reward", "oracle_regret", "top5_hit", "top10_hit", "ndcg_10"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_prompts", type=int, default=-1)
    parser.add_argument("--max_val_prompts", type=int, default=-1)
    parser.add_argument("--metadata_glob", default="metadata*.jsonl")

    parser.add_argument("--noise_enc", default="spatial_shrink", choices=NOISE_ENCODERS)
    parser.add_argument("--text_enc", default="pertokenscalar", choices=TEXT_ENCODERS)
    parser.add_argument("--text_embed_type", default="default", choices=["default", "t5+clip"])
    parser.add_argument("--seq_len_override", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--patch_size", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--ca_embed_dim", type=int, default=256)
    parser.add_argument("--ca_num_heads", type=int, default=8)
    parser.add_argument("--ffn_ratio", type=int, default=4)
    parser.add_argument("--use_std", action="store_true")
    parser.add_argument("--use_max", action="store_true")
    parser.add_argument("--head_compression", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_norm, stage1_model_config = load_checkpoint_normalization(args.stage1_checkpoint)
    args.stage1_model_config = stage1_model_config
    records = load_metadata_records(args.data_dir, metadata_glob=args.metadata_glob)
    groups = group_records_by_prompt(records)
    split_ids = load_prompt_split_ids(args.split_json)
    if ckpt_norm:
        y_mean = float(ckpt_norm["y_mean"])
        y_std = float(ckpt_norm["y_std"])
        if ckpt_norm.get("target") != args.target:
            raise ValueError(
                f"Stage-1 checkpoint target={ckpt_norm.get('target')} does not match --target {args.target}"
            )
    else:
        y_mean, y_std = compute_global_normalization(
            groups,
            split_ids[STAGE1_SPLITS["stage1_train"]],
            args.target,
            max_sample_idx=20,
        )

    input_kind = "camap" if args.model_kind == "camap" else "noise"
    train_loader, train_ds = build_rank100_loader(
        data_dir=args.data_dir,
        split_name=args.train_split,
        split_json=args.split_json,
        rank100_manifest=args.rank100_manifest,
        model_type=args.model_type,
        target=args.target,
        y_mean=y_mean,
        y_std=y_std,
        candidates_per_prompt=args.candidates_per_prompt,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        input_kind=input_kind,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_train_prompts,
        metadata_glob=args.metadata_glob,
    )
    val_loader, val_ds = build_rank100_loader(
        data_dir=args.data_dir,
        split_name=args.val_split,
        split_json=args.split_json,
        rank100_manifest=args.rank100_manifest,
        model_type=args.model_type,
        target=args.target,
        y_mean=y_mean,
        y_std=y_std,
        candidates_per_prompt=args.candidates_per_prompt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        input_kind=input_kind,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_val_prompts,
        metadata_glob=args.metadata_glob,
    )

    camap_shape = None
    if args.model_kind == "camap":
        camap_shape = (train_ds.num_heads, train_ds.ca_seq_len)
        if args.head_compression == 2:
            print("[warn] CA-Map canonical c=1; current --head_compression=2")

    if args.model_kind == "camap" and args.head_compression == 2:
        print("[warn] For canonical CA-Map use --head_compression 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model = build_model(args, camap_shape=camap_shape).to(device)
    load_model_state(model, args.stage1_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.primary_metric == "oracle_regret" else "max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )
    lambda_loss_fn = LambdaLossAtK(k=args.lambda_k, sigma=1.0, gain_type="exp2")

    config = vars(args).copy()
    config.update({
        "loss_name": f"LambdaLoss@{args.lambda_k}",
        "mae_weight_alpha": args.alpha,
        "normalization": {"target": args.target, "y_mean": y_mean, "y_std": y_std},
        "train_prompt_groups": len(train_ds),
        "val_prompt_groups": len(val_ds),
    })
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"Stage-2 rank100 fine-tuning: {args.model_kind} {args.model_type} "
        f"{args.target} | loss=LambdaLoss@{args.lambda_k} + {args.alpha}*MAE "
        f"| train={len(train_ds)}x{args.candidates_per_prompt} "
        f"val={len(val_ds)}x{args.candidates_per_prompt}"
    )

    best_value = float("inf") if args.primary_metric == "oracle_regret" else float("-inf")
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, lambda_loss_fn, args, device)
        vm = evaluate_rank100(model, val_loader, args, device, y_mean, y_std)
        current = vm[args.primary_metric]
        scheduler.step(current)

        improved = current < best_value if args.primary_metric == "oracle_regret" else current > best_value
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train LambdaLoss@{args.lambda_k}={tr['lambda_loss_at_10']:.4f} "
            f"MAE(norm)={tr['mae_normalized_train_loss']:.4f} "
            f"val selected_reward={vm['selected_reward']:.6f} "
            f"regret={vm['oracle_regret']:.6f} "
            f"top5={vm['top5_hit']:.4f} top10={vm['top10_hit']:.4f} "
            f"NDCG@10={vm['ndcg_10']:.4f} SRCC={vm['pool_srcc']:.4f} "
            f"rank_mean={vm['selected_rank_mean']:.2f} bottom25={vm['bottom_25_rate']:.4f}"
        )

        if improved:
            best_value = current
            patience_left = args.patience
            ckpt = {
                "model_state_dict": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config_for_checkpoint(model, args, camap_shape=camap_shape),
                "normalization": {"target": args.target, "y_mean": y_mean, "y_std": y_std},
                "stage2": {
                    "loss_name": f"LambdaLoss@{args.lambda_k}",
                    "mae_weight_alpha": args.alpha,
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "candidates_per_prompt": args.candidates_per_prompt,
                    "primary_metric": args.primary_metric,
                    "best_epoch": epoch,
                    "best_value": best_value,
                },
            }
            torch.save(ckpt, exp_dir / "best_model.pth")
            with open(exp_dir / "best_val_metrics.json", "w") as f:
                json.dump(vm, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} with no {args.primary_metric} improvement")
                break

        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Best checkpoint: {exp_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
