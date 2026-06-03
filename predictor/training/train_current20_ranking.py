#!/usr/bin/env python3
"""Train current 20-candidate prompt groups with top-heavy ranking losses."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from predictor.configs.model_dims import get_dims
from predictor.models import (
    NOISE_ENCODERS,
    TEXT_ENCODERS,
    get_ca_map_model,
    get_model,
    get_patchca_model,
)
from predictor.training.current20_dataloader import (
    build_current20_loader,
    compute_global_normalization,
    group_records_by_prompt,
    load_metadata_records,
    load_prompt_split_ids,
)
from predictor.training.dataloader import AVAILABLE_TARGETS, denormalize
from predictor.training.losses import LambdaLossAtK, SRCCLoss, TopBottomPairLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_stage1_split_ids(prompt_ids: List[int], seed: int) -> Dict[str, List[int]]:
    shuffled = sorted(int(pid) for pid in prompt_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train_prompt_ids": shuffled[:n_train],
        "val_prompt_ids": shuffled[n_train:n_train + n_val],
        "test_prompt_ids": shuffled[n_train + n_val:],
    }


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
    dims = get_dims(args.model_type)
    seq_len = args.seq_len_override
    if seq_len is None and args.text_embed_type == "t5+clip" and args.model_type == "hunyuan_dit":
        seq_len = 333

    if args.model_kind == "traditional":
        return get_model(
            noise_enc=stage1_cfg.get("noise_enc", args.noise_enc),
            text_enc=stage1_cfg.get("text_enc", args.text_enc),
            dropout=stage1_cfg.get("dropout", args.dropout),
            num_heads=stage1_cfg.get("num_heads", 1),
            spatial_size=stage1_cfg.get("spatial_size", dims["spatial_size"]),
            in_channels=stage1_cfg.get("in_channels", dims["latent_shape"][0]),
            embed_dim=stage1_cfg.get("embed_dim", dims["embed_dim"]),
            seq_len=stage1_cfg.get("seq_len", seq_len or dims["seq_len"]),
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
            seq_len_override=stage1_cfg.get("seq_len", seq_len),
        )
    if args.model_kind == "camap":
        if camap_shape is None:
            raise ValueError("camap_shape is required for CA-Map current20 training")
        num_heads, ca_seq_len = camap_shape
        return get_ca_map_model(
            num_heads=num_heads,
            seq_len=ca_seq_len,
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
    dims = get_dims(args.model_type)
    seq_len = args.seq_len_override
    if seq_len is None and args.text_embed_type == "t5+clip" and args.model_type == "hunyuan_dit":
        seq_len = 333

    if args.model_kind == "traditional":
        return {
            "noise_enc": stage1_cfg.get("noise_enc", args.noise_enc),
            "text_enc": stage1_cfg.get("text_enc", args.text_enc),
            "dropout": stage1_cfg.get("dropout", args.dropout),
            "num_heads": stage1_cfg.get("num_heads", 1),
            "model_type": args.model_type,
            "spatial_size": stage1_cfg.get("spatial_size", dims["spatial_size"]),
            "in_channels": stage1_cfg.get("in_channels", dims["latent_shape"][0]),
            "embed_dim": stage1_cfg.get("embed_dim", dims["embed_dim"]),
            "seq_len": stage1_cfg.get("seq_len", seq_len or dims["seq_len"]),
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
        return model(noise, prompt_embeds, prompt_mask).float().view(bsz, cand)

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


def srcc_mae_loss(preds: torch.Tensor, y: torch.Tensor, srcc_loss_fn: SRCCLoss) -> torch.Tensor:
    bsz, cand = preds.shape
    group_ids = torch.arange(bsz, device=preds.device).repeat_interleave(cand)
    mae = F.l1_loss(preds, y)
    srcc = srcc_loss_fn(preds.reshape(-1), y.reshape(-1), group_ids=group_ids)
    return mae + srcc


def srcc_diagnostic(preds: torch.Tensor, y: torch.Tensor, srcc_loss_fn: SRCCLoss) -> Dict[str, float]:
    bsz, cand = preds.shape
    group_ids = torch.arange(bsz, device=preds.device).repeat_interleave(cand)
    loss = srcc_loss_fn(preds.reshape(-1), y.reshape(-1), group_ids=group_ids)
    value = 1.0 - loss
    return {
        "srcc_loss_diagnostic": float(loss.detach().cpu()),
        "srcc_diagnostic": float(value.detach().cpu()),
    }


def compute_loss(preds, batch, args, loss_fns, device):
    y = batch["y"].to(device, non_blocking=True).float()
    raw_y = batch["raw_y"].to(device, non_blocking=True).float()
    rel = batch["rel"].to(device, non_blocking=True).float()

    if args.loss == "mae+srcc":
        mae_loss = F.l1_loss(preds, y)
        bsz, cand = preds.shape
        group_ids = torch.arange(bsz, device=preds.device).repeat_interleave(cand)
        srcc_loss = loss_fns["srcc"](preds.reshape(-1), y.reshape(-1), group_ids=group_ids)
        srcc_value = 1.0 - srcc_loss
        total = mae_loss + srcc_loss
        return total, {
            "ranking_loss": float(srcc_loss.detach().cpu()),
            "mae_normalized": float(mae_loss.detach().cpu()),
            "weighted_mae": float(mae_loss.detach().cpu()),
            "srcc_loss_diagnostic": float(srcc_loss.detach().cpu()),
            "srcc_diagnostic": float(srcc_value.detach().cpu()),
        }

    mae_loss = F.l1_loss(preds, y)
    if args.loss == "lambdaloss+mae":
        ranking_loss = loss_fns["lambda"](preds, rel)
    elif args.loss == "topbottom+mae":
        ranking_loss = loss_fns["topbottom"](preds, raw_y)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    total = ranking_loss + args.alpha * mae_loss
    parts = {
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "mae_normalized": float(mae_loss.detach().cpu()),
        "weighted_mae": float((args.alpha * mae_loss).detach().cpu()),
    }
    if args.log_loss_components:
        parts.update(srcc_diagnostic(preds, y, loss_fns["srcc"]))
    else:
        parts.update({
            "srcc_loss_diagnostic": float("nan"),
            "srcc_diagnostic": float("nan"),
        })
    return total, parts


def train_one_epoch(model, loader, optimizer, args, loss_fns, device) -> Dict[str, float]:
    model.train()
    use_amp = device.type == "cuda"
    totals = {
        "loss": 0.0,
        "ranking_loss": 0.0,
        "mae_normalized": 0.0,
        "weighted_mae": 0.0,
        "srcc_loss_diagnostic": 0.0,
        "srcc_diagnostic": 0.0,
        "srcc_groups": 0,
        "groups": 0,
    }

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds = forward_batch(model, batch, args, device)
        preds = preds.float()
        loss, parts = compute_loss(preds, batch, args, loss_fns, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        groups = int(preds.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * groups
        if not np.isnan(parts["ranking_loss"]):
            totals["ranking_loss"] += parts["ranking_loss"] * groups
        totals["mae_normalized"] += parts["mae_normalized"] * groups
        totals["weighted_mae"] += parts["weighted_mae"] * groups
        if not np.isnan(parts["srcc_diagnostic"]):
            totals["srcc_loss_diagnostic"] += parts["srcc_loss_diagnostic"] * groups
            totals["srcc_diagnostic"] += parts["srcc_diagnostic"] * groups
            totals["srcc_groups"] += groups
        totals["groups"] += groups

    n = max(1, totals["groups"])
    out = {
        "loss": totals["loss"] / n,
        "mae_normalized_train_loss": totals["mae_normalized"] / n,
        "weighted_mae_train_loss": totals["weighted_mae"] / n,
    }
    out["ranking_loss"] = totals["ranking_loss"] / n
    if totals["srcc_groups"]:
        out["srcc_loss_diagnostic"] = totals["srcc_loss_diagnostic"] / totals["srcc_groups"]
        out["srcc_diagnostic"] = totals["srcc_diagnostic"] / totals["srcc_groups"]
    return out


@torch.no_grad()
def evaluate_current20(model, loader, args, device, y_mean, y_std) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda"
    per_prompt = []
    mae_values = []
    pred_all = []
    raw_all = []

    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds_norm = forward_batch(model, batch, args, device)
        preds_norm = preds_norm.float()
        raw_y = batch["raw_y"].to(device, non_blocking=True).float()
        preds_raw = denormalize(preds_norm, y_mean, y_std)

        mae_values.append((preds_raw - raw_y).abs().detach().cpu().reshape(-1).numpy())
        pred_np = preds_norm.detach().cpu().numpy()
        raw_np = raw_y.detach().cpu().numpy()
        pred_all.append(preds_raw.detach().cpu().reshape(-1).numpy())
        raw_all.append(raw_np.reshape(-1))

        for i in range(pred_np.shape[0]):
            selected = int(np.argmax(pred_np[i]))
            rank = true_rank_desc(raw_np[i], selected)
            oracle = float(np.max(raw_np[i]))
            selected_reward = float(raw_np[i, selected])
            per_prompt.append({
                "selected_reward": selected_reward,
                "oracle_regret": oracle - selected_reward,
                "selected_rank": rank,
                "top1_hit": 1.0 if rank <= 1.0 else 0.0,
                "top3_hit": 1.0 if rank <= 3.0 else 0.0,
                "ndcg_2": ndcg_from_scores(pred_np[i], raw_np[i], 2),
                "ndcg_3": ndcg_from_scores(pred_np[i], raw_np[i], 3),
                "ndcg_5": ndcg_from_scores(pred_np[i], raw_np[i], 5),
                "srcc_per_prompt": spearman_np(pred_np[i], raw_np[i]),
            })

    if not per_prompt:
        raise ValueError("No validation prompts were evaluated")

    def mean_key(key):
        vals = np.asarray([row[key] for row in per_prompt], dtype=np.float64)
        return float(np.nanmean(vals))

    pred_flat = np.concatenate(pred_all)
    raw_flat = np.concatenate(raw_all)
    mae = float(np.concatenate(mae_values).mean()) if mae_values else float("nan")
    ranks = np.asarray([row["selected_rank"] for row in per_prompt], dtype=np.float64)
    return {
        "n_prompts": len(per_prompt),
        "n_candidates_per_prompt": 20,
        "mae_raw": mae,
        "srcc_global": spearman_np(pred_flat, raw_flat),
        "srcc_per_prompt": mean_key("srcc_per_prompt"),
        "ndcg_2": mean_key("ndcg_2"),
        "ndcg_3": mean_key("ndcg_3"),
        "ndcg_5": mean_key("ndcg_5"),
        "selected_reward": mean_key("selected_reward"),
        "oracle_regret": mean_key("oracle_regret"),
        "top1_hit": mean_key("top1_hit"),
        "top3_hit": mean_key("top3_hit"),
        "selected_rank_mean": float(np.mean(ranks)),
        "selected_rank_median": float(np.median(ranks)),
    }


def metric_is_better(metric: str, current: float, best: float) -> bool:
    if metric == "oracle_regret" or metric == "selected_rank_mean":
        return current < best
    return current > best


def write_history_csv(path: Path, rows: List[Dict[str, float]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_kind", default="traditional", choices=["traditional", "patchca", "camap"])
    parser.add_argument("--model_type", default="sdxl")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--metadata_glob", default="metadata.jsonl",
                        help="Current20 metadata filename. Keep as metadata.jsonl for this task.")
    parser.add_argument("--target", default="pick_score", choices=AVAILABLE_TARGETS)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--exp_name", required=True)

    parser.add_argument("--loss", default="lambdaloss+mae",
                        choices=["mae+srcc", "lambdaloss+mae", "topbottom+mae"])
    parser.add_argument("--lambda_k", type=int, default=3, choices=[2, 3, 5])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--primary_metric", default="selected_reward",
                        choices=[
                            "selected_reward", "oracle_regret", "ndcg_3",
                            "ndcg_5", "top3_hit", "srcc_per_prompt",
                            "selected_rank_mean",
                        ])
    parser.add_argument("--batch_size", type=int, default=12, help="Prompt groups per batch.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--log_loss_components",
        action="store_true",
        help="Also compute/print SRCC diagnostics for ranking-loss training. "
             "This is useful for debugging but adds training overhead.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_prompts", type=int, default=-1)
    parser.add_argument("--max_val_prompts", type=int, default=-1)
    parser.add_argument("--max_test_prompts", type=int, default=-1)
    parser.add_argument("--require_num_prompts", type=int, default=5000)

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
    exp_dir.mkdir(parents=True, exist_ok=False)

    if any(ch in args.metadata_glob for ch in "*?["):
        raise ValueError("Current20 training must use one metadata file, normally metadata.jsonl; globs are not allowed")
    records = load_metadata_records(args.data_dir, metadata_name=args.metadata_glob)
    groups = group_records_by_prompt(records)
    all_prompt_ids = sorted(groups)
    if args.require_num_prompts > 0 and len(all_prompt_ids) != args.require_num_prompts:
        raise ValueError(
            f"Expected {args.require_num_prompts} prompts, found {len(all_prompt_ids)} in {args.metadata_glob}"
        )
    if args.split_json:
        split_ids = load_prompt_split_ids(args.split_json)
    else:
        split_ids = derive_stage1_split_ids(all_prompt_ids, args.seed)

    print(
        f"Current20 split: {len(split_ids['train_prompt_ids'])} train / "
        f"{len(split_ids['val_prompt_ids'])} val / {len(split_ids['test_prompt_ids'])} test prompts"
    )

    ckpt_norm, stage1_model_config = load_checkpoint_normalization(args.stage1_checkpoint)
    args.stage1_model_config = stage1_model_config
    if ckpt_norm:
        if ckpt_norm.get("target") != args.target:
            raise ValueError(
                f"Checkpoint target={ckpt_norm.get('target')} does not match --target {args.target}"
            )
        y_mean = float(ckpt_norm["y_mean"])
        y_std = float(ckpt_norm["y_std"])
    else:
        y_mean, y_std = compute_global_normalization(
            groups, split_ids["train_prompt_ids"], args.target, candidates_per_prompt=20,
        )

    input_kind = "camap" if args.model_kind == "camap" else "noise"
    train_loader, train_ds = build_current20_loader(
        data_dir=args.data_dir,
        records_by_prompt=groups,
        prompt_ids=split_ids["train_prompt_ids"],
        model_type=args.model_type,
        target=args.target,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        input_kind=input_kind,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_train_prompts,
    )
    val_loader, val_ds = build_current20_loader(
        data_dir=args.data_dir,
        records_by_prompt=groups,
        prompt_ids=split_ids["val_prompt_ids"],
        model_type=args.model_type,
        target=args.target,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        input_kind=input_kind,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_val_prompts,
    )
    test_loader, test_ds = build_current20_loader(
        data_dir=args.data_dir,
        records_by_prompt=groups,
        prompt_ids=split_ids["test_prompt_ids"],
        model_type=args.model_type,
        target=args.target,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        input_kind=input_kind,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_test_prompts,
    )

    camap_shape = None
    if args.model_kind == "camap":
        camap_shape = (train_ds.num_heads, train_ds.ca_seq_len)
        if args.head_compression != 1:
            print("[warn] CA-Map canonical head_compression is c=1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model = build_model(args, camap_shape=camap_shape).to(device)
    load_model_state(model, args.stage1_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.primary_metric in {"oracle_regret", "selected_rank_mean"} else "max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )
    loss_fns = {
        "srcc": SRCCLoss(regularization_strength=1e-2),
        "lambda": LambdaLossAtK(k=args.lambda_k, sigma=1.0, gain_type="exp2"),
        "topbottom": TopBottomPairLoss(top_frac=0.2, bottom_frac=0.5),
    }

    config = vars(args).copy()
    config.update({
        "dataset_scope": "current_5000x20_metadata_only",
        "candidates_per_prompt_train": 20,
        "normalization": {"target": args.target, "y_mean": y_mean, "y_std": y_std},
        "train_prompt_groups": len(train_ds),
        "val_prompt_groups": len(val_ds),
        "test_prompt_groups": len(test_ds),
        "split_counts": {
            "train_prompts": len(split_ids["train_prompt_ids"]),
            "val_prompts": len(split_ids["val_prompt_ids"]),
            "test_prompts": len(split_ids["test_prompt_ids"]),
        },
    })
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"Current20 training: {args.model_kind} {args.model_type} {args.target} | "
        f"loss={args.loss} lambda_k={args.lambda_k} alpha={args.alpha} | "
        f"train={len(train_ds)}x20 val={len(val_ds)}x20 primary={args.primary_metric}"
    )

    best_value = float("inf") if args.primary_metric in {"oracle_regret", "selected_rank_mean"} else float("-inf")
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, args, loss_fns, device)
        vm = evaluate_current20(model, val_loader, args, device, y_mean, y_std)
        current = vm[args.primary_metric]
        scheduler.step(current)
        improved = metric_is_better(args.primary_metric, current, best_value)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in vm.items()},
        }
        history.append(row)

        if args.loss == "mae+srcc":
            rank_label = "SRCCLoss"
        elif args.loss == "lambdaloss+mae":
            rank_label = f"LambdaLoss@{args.lambda_k}"
        else:
            rank_label = "RankLoss"
        srcc_diag = tr.get("srcc_diagnostic", float("nan"))
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={tr['loss']:.4f} "
            f"{rank_label}={tr['ranking_loss']:.4f} "
            f"MAE(norm)={tr['mae_normalized_train_loss']:.4f} "
            f"MAE_term={tr['weighted_mae_train_loss']:.4f} "
            f"SRCC(train_diag)={srcc_diag:.4f} "
            f"val selected_reward={vm['selected_reward']:.6f} "
            f"val_MAE(raw)={vm['mae_raw']:.4f} "
            f"regret={vm['oracle_regret']:.6f} "
            f"NDCG@2={vm['ndcg_2']:.4f} NDCG@3={vm['ndcg_3']:.4f} "
            f"NDCG@5={vm['ndcg_5']:.4f} top3={vm['top3_hit']:.4f} "
            f"rank_mean={vm['selected_rank_mean']:.2f} SRCC(val_prompt)={vm['srcc_per_prompt']:.4f}"
        )

        if improved:
            best_value = current
            patience_left = args.patience
            ckpt = {
                "model_state_dict": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config_for_checkpoint(model, args, camap_shape=camap_shape),
                "normalization": {"target": args.target, "y_mean": y_mean, "y_std": y_std},
                "current20": {
                    "loss": args.loss,
                    "lambda_k": args.lambda_k if args.loss == "lambdaloss+mae" else None,
                    "mae_weight_alpha": args.alpha if args.loss != "mae+srcc" else None,
                    "candidates_per_prompt": 20,
                    "primary_metric": args.primary_metric,
                    "best_epoch": epoch,
                    "best_value": best_value,
                    "validation_metrics": vm,
                },
            }
            torch.save(ckpt, exp_dir / "checkpoint.pt")
            torch.save(ckpt, exp_dir / "best_model.pth")
            with open(exp_dir / "best_val_metrics.json", "w") as f:
                json.dump(vm, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} with no {args.primary_metric} improvement")
                break

        with open(exp_dir / "validation_metrics.json", "w") as f:
            json.dump(history, f, indent=2)
        write_history_csv(exp_dir / "validation_metrics.csv", history)

    checkpoint = torch.load(exp_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    test_metrics = evaluate_current20(model, test_loader, args, device, y_mean, y_std)
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(
        f"Best checkpoint: {exp_dir / 'checkpoint.pt'} | "
        f"test selected_reward={test_metrics['selected_reward']:.6f} "
        f"regret={test_metrics['oracle_regret']:.6f} "
        f"NDCG@3={test_metrics['ndcg_3']:.4f} NDCG@5={test_metrics['ndcg_5']:.4f}"
    )


if __name__ == "__main__":
    main()
