#!/usr/bin/env python3
"""Single-stage grouped trainer for SDXL fixed-budget predictor ablations."""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from predictor.configs.model_dims import get_dims
from predictor.models import (
    NOISE_ENCODERS,
    TEXT_ENCODERS,
    get_ca_map_model,
    get_model,
    get_patchca_model,
)
from predictor.training.current20_dataloader import (
    group_records_by_prompt,
    load_metadata_records,
    load_prompt_split_ids,
)
from predictor.training.dataloader import AVAILABLE_TARGETS, _extract_embeds, denormalize


SPLIT_KEYS = ("train_prompt_ids", "val_prompt_ids", "test_prompt_ids")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def require_target(records: Iterable[dict], target: str) -> None:
    missing = 0
    for rec in records:
        if target not in rec or rec[target] is None:
            missing += 1
            if missing >= 5:
                break
    if missing:
        raise ValueError(f"Metadata is missing target column/value {target!r}")


def create_prompt_split(
    prompt_ids: List[int],
    split_json: Path,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    dataset_label: str,
) -> Dict[str, List[int]]:
    total = n_train + n_val + n_test
    if len(prompt_ids) != total:
        raise ValueError(
            f"Expected {total} prompts for split "
            f"{n_train}/{n_val}/{n_test}, found {len(prompt_ids)}"
        )

    if split_json.exists():
        split = load_prompt_split_ids(str(split_json))
        counts = tuple(len(split[key]) for key in SPLIT_KEYS)
        if counts != (n_train, n_val, n_test):
            raise ValueError(f"Split {split_json} has counts {counts}, expected {(n_train, n_val, n_test)}")
        observed = sorted({pid for key in SPLIT_KEYS for pid in split[key]})
        if observed != sorted(int(pid) for pid in prompt_ids):
            raise ValueError(f"Split {split_json} does not match prompt IDs in metadata")
        return split

    shuffled = sorted(int(pid) for pid in prompt_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split = {
        "train_prompt_ids": shuffled[:n_train],
        "val_prompt_ids": shuffled[n_train:n_train + n_val],
        "test_prompt_ids": shuffled[n_train + n_val:n_train + n_val + n_test],
    }
    payload = {
        "dataset": dataset_label,
        "split_policy": "prompt_id",
        "split_seed": seed,
        "train_prompts": n_train,
        "val_prompts": n_val,
        "test_prompts": n_test,
        **split,
    }
    split_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = split_json.with_suffix(f".tmp.{os.getpid()}.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        os.replace(tmp, split_json)
    except FileExistsError:
        tmp.unlink(missing_ok=True)
    return load_prompt_split_ids(str(split_json))


def compute_grouped_normalization(
    records_by_prompt: Dict[int, List[dict]],
    train_prompt_ids: List[int],
    target: str,
    candidates_per_prompt: int,
) -> Tuple[float, float]:
    train_ids = set(int(pid) for pid in train_prompt_ids)
    values = []
    wanted = set(range(candidates_per_prompt))
    for pid in train_prompt_ids:
        by_sid = {
            int(rec["sample_idx"]): rec
            for rec in records_by_prompt.get(int(pid), [])
            if int(rec["sample_idx"]) < candidates_per_prompt
        }
        if set(by_sid) != wanted:
            missing = sorted(wanted - set(by_sid))[:10]
            raise ValueError(f"Incomplete train prompt group pid={pid}, missing sample_idx={missing}")
        values.extend(float(by_sid[sid][target]) for sid in range(candidates_per_prompt))
    if not values:
        raise ValueError("No train-split target values available for normalization")
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    if std <= 1e-8:
        std = 1.0
    if len(values) != len(train_ids) * candidates_per_prompt:
        raise ValueError("Normalization did not use all train prompt candidates")
    return float(arr.mean()), std


class SDXLBudgetPromptDataset(Dataset):
    """One item is one prompt with its M candidate noises or CA maps."""

    def __init__(
        self,
        data_dir: str,
        records_by_prompt: Dict[int, List[dict]],
        prompt_ids: List[int],
        model_type: str,
        target: str,
        y_mean: float,
        y_std: float,
        input_kind: str,
        candidates_per_prompt: int,
        text_embed_type: str = "default",
        seq_len_override: Optional[int] = None,
        use_std: bool = False,
        use_max: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.records_by_prompt = records_by_prompt
        self.prompt_ids = [int(pid) for pid in prompt_ids]
        self.model_type = model_type
        self.target = target
        self.y_mean = float(y_mean)
        self.y_std = max(float(y_std), 1e-8)
        self.input_kind = input_kind
        self.candidates_per_prompt = int(candidates_per_prompt)
        self.text_embed_type = text_embed_type
        self.use_std = use_std
        self.use_max = use_max

        if input_kind not in {"noise", "camap"}:
            raise ValueError(f"input_kind must be 'noise' or 'camap', got {input_kind!r}")

        dims = get_dims(model_type)
        self.embed_dim = dims["embed_dim"]
        self.seq_len = seq_len_override if seq_len_override is not None else dims["seq_len"]

        self.group_records: Dict[int, List[dict]] = {}
        wanted = set(range(self.candidates_per_prompt))
        missing = []
        for pid in self.prompt_ids:
            by_sid = {
                int(rec["sample_idx"]): rec
                for rec in records_by_prompt.get(pid, [])
                if int(rec["sample_idx"]) < self.candidates_per_prompt
            }
            if set(by_sid) != wanted:
                missing.append((pid, sorted(wanted - set(by_sid))[:10]))
                continue
            self.group_records[pid] = [by_sid[sid] for sid in range(self.candidates_per_prompt)]
        if missing:
            preview = ", ".join(f"pid={pid} missing={m}" for pid, m in missing[:5])
            raise ValueError(f"Incomplete prompt groups: {preview}")
        if not self.group_records:
            raise ValueError("No complete prompt groups available")
        self.prompt_ids = [pid for pid in self.prompt_ids if pid in self.group_records]

        self._embed_cache = {}
        if self.input_kind == "noise":
            for pid in self.prompt_ids:
                emb_path = self.data_dir / "embeds" / f"p{pid:04d}.pt"
                embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
                embeds, mask = _extract_embeds(
                    embeddings,
                    self.model_type,
                    self.embed_dim,
                    self.seq_len,
                    text_embed_type=self.text_embed_type,
                )
                self._embed_cache[pid] = (embeds, mask)

        self.num_heads = None
        self.ca_seq_len = None
        if self.input_kind == "camap":
            first = self._load_camap(self.prompt_ids[0], 0)["entropy"]
            self.num_heads, self.ca_seq_len = first.shape

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def _load_noise(self, pid: int, sid: int) -> torch.Tensor:
        path = self.data_dir / "noise" / f"p{pid:04d}_s{sid:02d}.pt"
        noise = torch.load(path, map_location="cpu", weights_only=False)
        if noise.dim() == 4:
            noise = noise.squeeze(0)
        return noise.float()

    def _load_camap(self, pid: int, sid: int) -> Dict[str, torch.Tensor]:
        path = self.data_dir / "cross_attn" / f"p{pid:04d}_s{sid:02d}.pt"
        d = torch.load(path, map_location="cpu", weights_only=False)
        out = {"entropy": d["entropy"].float()}
        if self.use_std:
            out["std"] = d["std"].float()
        if self.use_max:
            out["max"] = d["max"].float()
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.prompt_ids[idx]
        records = self.group_records[pid]
        sample_idxs = torch.tensor([int(r["sample_idx"]) for r in records], dtype=torch.long)
        seeds = torch.tensor([int(r.get("seed", -1)) for r in records], dtype=torch.long)
        raw = torch.tensor([float(r[self.target]) for r in records], dtype=torch.float32)
        y = (raw - self.y_mean) / self.y_std
        rel = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

        out = {
            "prompt_id": torch.tensor(pid, dtype=torch.long),
            "sample_idx": sample_idxs,
            "seed": seeds,
            "prompt": records[0].get("prompt", ""),
            "y": y,
            "raw_y": raw,
            "rel": rel,
        }
        if self.input_kind == "noise":
            noises = [self._load_noise(pid, int(sid)) for sid in sample_idxs.tolist()]
            prompt_embeds, prompt_mask = self._embed_cache[pid]
            out.update({
                "noise": torch.stack(noises, dim=0),
                "prompt_embeds": prompt_embeds.float(),
                "prompt_mask": prompt_mask.float(),
            })
        else:
            entries = [self._load_camap(pid, int(sid)) for sid in sample_idxs.tolist()]
            out["entropy"] = torch.stack([e["entropy"] for e in entries], dim=0)
            if self.use_std:
                out["std"] = torch.stack([e["std"] for e in entries], dim=0)
            if self.use_max:
                out["max"] = torch.stack([e["max"] for e in entries], dim=0)
        return out


def build_loader(
    data_dir: str,
    records_by_prompt: Dict[int, List[dict]],
    prompt_ids: List[int],
    model_type: str,
    target: str,
    y_mean: float,
    y_std: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    input_kind: str,
    candidates_per_prompt: int,
    text_embed_type: str,
    seq_len_override: Optional[int],
    use_std: bool,
    use_max: bool,
    max_prompts: int,
) -> Tuple[DataLoader, SDXLBudgetPromptDataset]:
    if max_prompts > 0:
        prompt_ids = prompt_ids[:max_prompts]
    ds = SDXLBudgetPromptDataset(
        data_dir=data_dir,
        records_by_prompt=records_by_prompt,
        prompt_ids=prompt_ids,
        model_type=model_type,
        target=target,
        y_mean=y_mean,
        y_std=y_std,
        input_kind=input_kind,
        candidates_per_prompt=candidates_per_prompt,
        text_embed_type=text_embed_type,
        seq_len_override=seq_len_override,
        use_std=use_std,
        use_max=use_max,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, ds


class LambdaRankPairLoss(torch.nn.Module):
    """Differentiable LambdaRank-style pairwise loss over full prompt groups."""

    def __init__(self, sigma: float = 1.0, gain_type: str = "exp2", eps: float = 1e-8):
        super().__init__()
        self.sigma = float(sigma)
        self.gain_type = gain_type
        self.eps = eps

    def _gains(self, rel: torch.Tensor) -> torch.Tensor:
        if self.gain_type == "exp2":
            return torch.pow(2.0, rel) - 1.0
        if self.gain_type == "identity":
            return rel
        raise ValueError(f"Unknown gain_type: {self.gain_type}")

    def _group_loss(self, pred: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1).float()
        rel = rel.view(-1).float()
        n = pred.numel()
        if n < 2 or torch.max(rel) - torch.min(rel) < self.eps:
            return pred.sum() * 0.0

        gains = self._gains(rel)
        ideal_gains, _ = torch.sort(gains, descending=True)
        positions = torch.arange(1, n + 1, device=pred.device, dtype=torch.float32)
        discounts_ideal = 1.0 / torch.log2(positions + 1.0)
        idcg = torch.sum(ideal_gains * discounts_ideal)
        if idcg <= self.eps:
            return pred.sum() * 0.0

        with torch.no_grad():
            pred_order = torch.argsort(pred, descending=True)
            pred_positions = torch.empty(n, device=pred.device, dtype=torch.float32)
            pred_positions[pred_order] = torch.arange(1, n + 1, device=pred.device, dtype=torch.float32)
            discounts = 1.0 / torch.log2(pred_positions + 1.0)
            delta_ndcg = torch.abs(
                (gains.view(-1, 1) - gains.view(1, -1))
                * (discounts.view(-1, 1) - discounts.view(1, -1))
            ) / idcg.clamp_min(self.eps)
            pair_mask = rel.view(-1, 1) > rel.view(1, -1)

        if not torch.any(pair_mask):
            return pred.sum() * 0.0
        score_diff = pred.view(-1, 1) - pred.view(1, -1)
        pair_losses = F.softplus(-self.sigma * score_diff)
        return (pair_losses[pair_mask] * delta_ndcg[pair_mask]).mean()

    def forward(self, preds: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2 or relevance.dim() != 2:
            raise ValueError(
                f"LambdaRankPairLoss expects [B, M] tensors, got {tuple(preds.shape)} and {tuple(relevance.shape)}"
            )
        losses = [self._group_loss(preds[i], relevance[i]) for i in range(preds.shape[0])]
        return torch.stack(losses).mean()


class LambdaLossAtK(torch.nn.Module):
    """NDCG-oriented pairwise LambdaLoss@K without importing torchsort."""

    def __init__(
        self,
        k: int = 10,
        sigma: float = 1.0,
        gain_type: str = "exp2",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = int(k)
        self.sigma = float(sigma)
        self.gain_type = gain_type
        self.eps = eps

    def _gains(self, rel: torch.Tensor) -> torch.Tensor:
        if self.gain_type == "exp2":
            return torch.pow(2.0, rel) - 1.0
        if self.gain_type == "identity":
            return rel
        raise ValueError(f"Unknown gain_type: {self.gain_type}")

    def _group_loss(self, pred: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1).float()
        rel = rel.view(-1).float()
        n = pred.numel()
        if n < 2 or torch.max(rel) - torch.min(rel) < self.eps:
            return pred.sum() * 0.0

        k = min(self.k, n)
        gains = self._gains(rel)
        ideal_gains, _ = torch.sort(gains, descending=True)
        ideal_positions = torch.arange(1, n + 1, device=pred.device, dtype=torch.float32)
        ideal_discounts = 1.0 / torch.log2(ideal_positions + 1.0)
        idcg = torch.sum(ideal_gains[:k] * ideal_discounts[:k])
        if idcg <= self.eps:
            return pred.sum() * 0.0

        with torch.no_grad():
            pred_order = torch.argsort(pred, descending=True)
            positions = torch.empty(n, device=pred.device, dtype=torch.float32)
            positions[pred_order] = torch.arange(1, n + 1, device=pred.device, dtype=torch.float32)
            discounts = 1.0 / torch.log2(positions + 1.0)
            delta_ndcg = torch.abs(
                (gains.view(-1, 1) - gains.view(1, -1))
                * (discounts.view(-1, 1) - discounts.view(1, -1))
            ) / idcg.clamp_min(self.eps)
            topk_pair = (positions.view(-1, 1) <= k) | (positions.view(1, -1) <= k)
            ordered_pair = rel.view(-1, 1) > rel.view(1, -1)
            pair_mask = ordered_pair & topk_pair

        if not torch.any(pair_mask):
            return pred.sum() * 0.0

        score_diff = pred.view(-1, 1) - pred.view(1, -1)
        pair_losses = F.softplus(-self.sigma * score_diff)
        weighted = pair_losses[pair_mask] * delta_ndcg[pair_mask]
        return weighted.mean()

    def forward(self, preds: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2 or relevance.dim() != 2:
            raise ValueError(
                f"LambdaLossAtK expects [B, M] tensors, got {tuple(preds.shape)} and {tuple(relevance.shape)}"
            )
        losses = [self._group_loss(preds[i], relevance[i]) for i in range(preds.shape[0])]
        return torch.stack(losses).mean()


class PairwiseSoftSRCCLoss(torch.nn.Module):
    """SRCC surrogate that does not depend on the torchsort CUDA extension."""

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = eps

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1).float()
        diff = x.view(1, -1) - x.view(-1, 1)
        return 1.0 + torch.sigmoid(diff / self.temperature).sum(dim=1) - 0.5

    def _group_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() < 2:
            return pred.sum() * 0.0
        pr = self._soft_rank(pred)
        tr = self._soft_rank(target)
        pr = pr - pr.mean()
        tr = tr - tr.mean()
        denom = torch.sqrt(torch.sum(pr * pr) * torch.sum(tr * tr)).clamp_min(self.eps)
        corr = torch.sum(pr * tr) / denom
        return 1.0 - corr

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2 or targets.dim() != 2:
            raise ValueError(
                f"PairwiseSoftSRCCLoss expects [B, M] tensors, got {tuple(preds.shape)} and {tuple(targets.shape)}"
            )
        losses = [self._group_loss(preds[i], targets[i]) for i in range(preds.shape[0])]
        return torch.stack(losses).mean()


def build_model(args, camap_shape=None):
    dims = get_dims(args.model_type)
    seq_len = args.seq_len_override or dims["seq_len"]
    if args.model_kind == "traditional":
        return get_model(
            noise_enc=args.noise_enc,
            text_enc=args.text_enc,
            dropout=args.dropout,
            num_heads=1,
            spatial_size=dims["spatial_size"],
            in_channels=dims["latent_shape"][0],
            embed_dim=dims["embed_dim"],
            seq_len=seq_len,
            pos_encoding=args.pos_encoding,
            text_encoder_project_dims=None,
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
            seq_len_override=args.seq_len_override,
        )
    if args.model_kind == "camap":
        if camap_shape is None:
            raise ValueError("camap_shape is required for CA-Map training")
        return get_ca_map_model(
            num_heads=int(camap_shape[0]),
            seq_len=int(camap_shape[1]),
            head_compression_ratio=args.head_compression,
            use_std=args.use_std,
            use_max=args.use_max,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model_kind {args.model_kind}")


def model_config_for_checkpoint(model, args, camap_shape=None):
    dims = get_dims(args.model_type)
    seq_len = args.seq_len_override or dims["seq_len"]
    if args.model_kind == "traditional":
        return {
            "noise_enc": args.noise_enc,
            "text_enc": args.text_enc,
            "dropout": args.dropout,
            "num_heads": 1,
            "model_type": args.model_type,
            "spatial_size": dims["spatial_size"],
            "in_channels": dims["latent_shape"][0],
            "embed_dim": dims["embed_dim"],
            "seq_len": seq_len,
            "pos_encoding": args.pos_encoding,
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
    raise ValueError(f"Cannot build checkpoint config for {args.model_kind}")


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


def srcc_mae_loss(preds: torch.Tensor, y: torch.Tensor, srcc_loss_fn: PairwiseSoftSRCCLoss) -> torch.Tensor:
    mae = F.l1_loss(preds, y)
    srcc = srcc_loss_fn(preds, y)
    return mae + srcc


def compute_loss(preds, batch, args, loss_fns, device):
    y = batch["y"].to(device, non_blocking=True).float()
    rel = batch["rel"].to(device, non_blocking=True).float()

    if args.loss == "mae+srcc":
        mae_loss = F.l1_loss(preds, y)
        srcc_loss = loss_fns["srcc"](preds, y)
        total = mae_loss + srcc_loss
        return total, {
            "ranking_loss": float(srcc_loss.detach().cpu()),
            "mae_normalized": float(mae_loss.detach().cpu()),
            "weighted_mae": float(mae_loss.detach().cpu()),
            "srcc_loss_diagnostic": float(srcc_loss.detach().cpu()),
            "srcc_diagnostic": float((1.0 - srcc_loss).detach().cpu()),
        }

    mae_loss = F.l1_loss(preds, y)
    if args.loss == "lambdarank+mae":
        ranking_loss = loss_fns["lambdarank"](preds, rel)
    elif args.loss == "lambdaloss+mae":
        ranking_loss = loss_fns["lambda"](preds, rel)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    total = ranking_loss + args.alpha * mae_loss
    parts = {
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "mae_normalized": float(mae_loss.detach().cpu()),
        "weighted_mae": float((args.alpha * mae_loss).detach().cpu()),
    }
    if args.log_loss_components:
        srcc_loss = loss_fns["srcc"](preds, y)
        parts.update({
            "srcc_loss_diagnostic": float(srcc_loss.detach().cpu()),
            "srcc_diagnostic": float((1.0 - srcc_loss).detach().cpu()),
        })
    else:
        parts.update({
            "srcc_loss_diagnostic": float("nan"),
            "srcc_diagnostic": float("nan"),
        })
    return total, parts


def train_one_epoch(model, loader, optimizer, args, loss_fns, device) -> Dict[str, float]:
    model.train()
    use_amp = device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
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
    steps_since_update = 0

    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds = forward_batch(model, batch, args, device)
        preds = preds.float()
        loss, parts = compute_loss(preds, batch, args, loss_fns, device)
        scaled_loss = loss / max(1, args.grad_accum)
        scaled_loss.backward()
        steps_since_update += 1

        if steps_since_update >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_since_update = 0

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

    if steps_since_update:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
def evaluate(model, loader, args, device, y_mean: float, y_std: float) -> Dict[str, float]:
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
            n = raw_np.shape[1]
            per_prompt.append({
                "selected_reward": selected_reward,
                "oracle_regret": oracle - selected_reward,
                "selected_rank": rank,
                "top1_hit": 1.0 if rank <= 1.0 else 0.0,
                "top5_hit": 1.0 if rank <= 5.0 else 0.0,
                "top10_hit": 1.0 if rank <= 10.0 else 0.0,
                "bottom25": 1.0 if rank > 0.75 * n else 0.0,
                "rank_ge_90pct": 1.0 if rank >= 0.9 * n else 0.0,
                "ndcg_5": ndcg_from_scores(pred_np[i], raw_np[i], 5),
                "ndcg_10": ndcg_from_scores(pred_np[i], raw_np[i], 10),
                "ndcg_20": ndcg_from_scores(pred_np[i], raw_np[i], 20),
                "srcc_per_prompt": spearman_np(pred_np[i], raw_np[i]),
            })

    if not per_prompt:
        raise ValueError("No prompts were evaluated")

    def mean_key(key: str) -> float:
        vals = np.asarray([row[key] for row in per_prompt], dtype=np.float64)
        return float(np.nanmean(vals))

    pred_flat = np.concatenate(pred_all)
    raw_flat = np.concatenate(raw_all)
    mae = float(np.concatenate(mae_values).mean()) if mae_values else float("nan")
    ranks = np.asarray([row["selected_rank"] for row in per_prompt], dtype=np.float64)
    return {
        "n_prompts": len(per_prompt),
        "n_candidates_per_prompt": args.candidates_per_prompt,
        "mae_raw": mae,
        "srcc_global": spearman_np(pred_flat, raw_flat),
        "srcc_per_prompt": mean_key("srcc_per_prompt"),
        "ndcg_5": mean_key("ndcg_5"),
        "ndcg_10": mean_key("ndcg_10"),
        "ndcg_20": mean_key("ndcg_20"),
        "selected_reward": mean_key("selected_reward"),
        "oracle_regret": mean_key("oracle_regret"),
        "top1_hit": mean_key("top1_hit"),
        "top5_hit": mean_key("top5_hit"),
        "top10_hit": mean_key("top10_hit"),
        "selected_rank_mean": float(np.mean(ranks)),
        "selected_rank_median": float(np.median(ranks)),
        "bottom25_rate": mean_key("bottom25"),
        "rank_ge_90pct_count": int(round(sum(row["rank_ge_90pct"] for row in per_prompt))),
    }


def selection_tuple(metrics: Dict[str, float], primary_metric: str) -> Tuple[float, float, float, float, float]:
    primary = float(metrics[primary_metric])
    if primary_metric in {"oracle_regret", "selected_rank_mean"}:
        primary = -primary
    return (
        primary,
        -float(metrics["oracle_regret"]),
        float(metrics["top10_hit"]),
        -float(metrics["selected_rank_mean"]),
        float(metrics["selected_reward"]),
    )


def write_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_kind", required=True, choices=["traditional", "patchca", "camap"])
    parser.add_argument("--model_type", default="sdxl")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--metadata_name", default="metadata.jsonl")
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--target", required=True, choices=["pick_score", "hpsv2", "image_reward"])
    parser.add_argument("--output_dir", default="./experiments/sdxl_budget_ablation_pick_score")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--config_label", default="")

    parser.add_argument("--loss", required=True, choices=["mae+srcc", "lambdarank+mae", "lambdaloss+mae"])
    parser.add_argument("--lambda_k", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--srcc_temperature", type=float, default=0.1)
    parser.add_argument("--primary_metric", default="ndcg_10",
                        choices=[
                            "ndcg_5", "ndcg_10", "ndcg_20",
                            "selected_reward", "oracle_regret",
                            "top5_hit", "top10_hit", "selected_rank_mean",
                        ])
    parser.add_argument("--candidates_per_prompt", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
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
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_prompts", type=int, default=-1)
    parser.add_argument("--max_val_prompts", type=int, default=-1)
    parser.add_argument("--max_test_prompts", type=int, default=-1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--noise_enc", default="spatial_shrink", choices=NOISE_ENCODERS)
    parser.add_argument("--text_enc", default="pertokenscalar", choices=TEXT_ENCODERS)
    parser.add_argument("--text_embed_type", default="default", choices=["default", "t5+clip"])
    parser.add_argument("--seq_len_override", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_encoding", default="sinusoidal", choices=["none", "sinusoidal", "learned"])

    parser.add_argument("--patch_size", type=int, default=4, choices=[2, 4, 8, 16])
    parser.add_argument("--ca_embed_dim", type=int, default=256)
    parser.add_argument("--ca_num_heads", type=int, default=8)
    parser.add_argument("--ffn_ratio", type=int, default=4)
    parser.add_argument("--use_std", action="store_true")
    parser.add_argument("--use_max", action="store_true")
    parser.add_argument("--head_compression", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.output_dir) / args.exp_name
    if exp_dir.exists() and any(exp_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    for subdir in ("noise", "embeds"):
        if not (data_dir / subdir).is_dir():
            raise FileNotFoundError(f"Missing required dataset directory: {data_dir / subdir}")
    if args.model_kind == "camap" and not (data_dir / "cross_attn").is_dir():
        raise FileNotFoundError(f"CA-Map training requires cross_attn directory: {data_dir / 'cross_attn'}")

    if args.target not in AVAILABLE_TARGETS:
        raise ValueError(f"Unknown target {args.target!r}; available targets are {AVAILABLE_TARGETS}")
    records = load_metadata_records(str(data_dir), metadata_name=args.metadata_name)
    require_target(records, args.target)
    groups = group_records_by_prompt(records)
    all_prompt_ids = sorted(groups)
    n_prompts = len(all_prompt_ids)
    n_train = int(n_prompts * args.train_ratio)
    n_val = int(n_prompts * args.val_ratio)
    n_test = n_prompts - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split counts for {n_prompts} prompts: "
            f"train={n_train} val={n_val} test={n_test}"
        )

    split_json = (
        Path(args.split_json)
        if args.split_json
        else data_dir / f"splits_prompt_ids_{n_train}_{n_val}_{n_test}.json"
    )
    dataset_label = f"{args.model_type}_{n_prompts}x{args.candidates_per_prompt}"
    split_ids = create_prompt_split(
        all_prompt_ids,
        split_json=split_json,
        seed=args.split_seed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        dataset_label=dataset_label,
    )
    y_mean, y_std = compute_grouped_normalization(
        groups,
        split_ids["train_prompt_ids"],
        args.target,
        candidates_per_prompt=args.candidates_per_prompt,
    )

    input_kind = "camap" if args.model_kind == "camap" else "noise"
    train_loader, train_ds = build_loader(
        data_dir=str(data_dir),
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
        candidates_per_prompt=args.candidates_per_prompt,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_train_prompts,
    )
    val_loader, val_ds = build_loader(
        data_dir=str(data_dir),
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
        candidates_per_prompt=args.candidates_per_prompt,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_val_prompts,
    )
    test_loader, test_ds = build_loader(
        data_dir=str(data_dir),
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
        candidates_per_prompt=args.candidates_per_prompt,
        text_embed_type=args.text_embed_type,
        seq_len_override=args.seq_len_override,
        use_std=args.use_std,
        use_max=args.use_max,
        max_prompts=args.max_test_prompts,
    )

    camap_shape = None
    if args.model_kind == "camap":
        camap_shape = (train_ds.num_heads, train_ds.ca_seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model = build_model(args, camap_shape=camap_shape).to(device)
    param_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.primary_metric in {"oracle_regret", "selected_rank_mean"} else "max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )
    loss_fns = {
        "srcc": PairwiseSoftSRCCLoss(temperature=args.srcc_temperature),
        "lambdarank": LambdaRankPairLoss(sigma=1.0, gain_type="exp2"),
        "lambda": LambdaLossAtK(k=args.lambda_k, sigma=1.0, gain_type="exp2"),
    }

    config = vars(args).copy()
    config.update({
        "dataset_scope": f"{dataset_label}_single_stage",
        "dataset_label": dataset_label,
        "total_prompt_groups": n_prompts,
        "normalization": {
            "target": args.target,
            "y_mean": y_mean,
            "y_std": y_std,
            "source": f"train_prompts_first_{args.candidates_per_prompt}_candidates",
        },
        "split_json": str(split_json),
        "train_prompt_groups": len(train_ds),
        "val_prompt_groups": len(val_ds),
        "test_prompt_groups": len(test_ds),
        "split_counts": {
            "train_prompts": len(split_ids["train_prompt_ids"]),
            "val_prompts": len(split_ids["val_prompt_ids"]),
            "test_prompts": len(split_ids["test_prompt_ids"]),
        },
        "param_count": param_count,
        "params_M": param_count / 1_000_000.0,
        "effective_batch_groups": args.batch_size * args.grad_accum,
        "srcc_surrogate": "pairwise_soft_rank_no_torchsort_cuda",
    })
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"{args.model_type} budget training: {args.exp_name} | model_kind={args.model_kind} "
        f"target={args.target} loss={args.loss} lambda_k={args.lambda_k} alpha={args.alpha} | "
        f"train={len(train_ds)}x{args.candidates_per_prompt} "
        f"val={len(val_ds)}x{args.candidates_per_prompt} primary={args.primary_metric} "
        f"params={param_count / 1_000_000.0:.3f}M"
    )
    print(f"Normalization: target={args.target} mean={y_mean:.8f} std={y_std:.8f}")

    best_tuple = None
    best_epoch = 0
    best_val_metrics = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args, loss_fns, device)
        val_metrics = evaluate(model, val_loader, args, device, y_mean, y_std)
        current_tuple = selection_tuple(val_metrics, args.primary_metric)
        scheduler.step(val_metrics[args.primary_metric])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        write_history_csv(exp_dir / "history.csv", history)

        if args.loss == "mae+srcc":
            rank_label = "SRCCLoss"
        elif args.loss == "lambdaloss+mae":
            rank_label = f"LambdaLoss@{args.lambda_k}"
        else:
            rank_label = "RankLoss"
        srcc_diag = train_metrics.get("srcc_diagnostic", float("nan"))
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_metrics['loss']:.4f} "
            f"{rank_label}={train_metrics['ranking_loss']:.4f} "
            f"MAE(norm)={train_metrics['mae_normalized_train_loss']:.4f} "
            f"MAE_term={train_metrics['weighted_mae_train_loss']:.4f} "
            f"SRCC(train_diag)={srcc_diag:.4f} "
            f"val_MAE(raw)={val_metrics['mae_raw']:.4f} "
            f"val NDCG@5={val_metrics['ndcg_5']:.4f} "
            f"NDCG@10={val_metrics['ndcg_10']:.4f} "
            f"NDCG@20={val_metrics['ndcg_20']:.4f} "
            f"SRCC(val_prompt)={val_metrics['srcc_per_prompt']:.4f} "
            f"regret={val_metrics['oracle_regret']:.6f} "
            f"top10={val_metrics['top10_hit']:.4f} "
            f"rank_mean={val_metrics['selected_rank_mean']:.2f}"
        )

        improved = best_tuple is None or current_tuple > best_tuple
        if improved:
            best_tuple = current_tuple
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_left = args.patience
            ckpt = {
                "model_state_dict": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config_for_checkpoint(model, args, camap_shape=camap_shape),
                "normalization": {"target": args.target, "y_mean": y_mean, "y_std": y_std},
                "grouped_budget": {
                    "loss": args.loss,
                    "lambda_k": args.lambda_k if args.loss == "lambdaloss+mae" else None,
                    "mae_weight_alpha": args.alpha if args.loss != "mae+srcc" else None,
                    "candidates_per_prompt": args.candidates_per_prompt,
                    "primary_metric": args.primary_metric,
                    "selection_tuple": list(best_tuple),
                    "best_epoch": best_epoch,
                    "validation_metrics": val_metrics,
                },
            }
            torch.save(ckpt, exp_dir / "checkpoint.pt")
            torch.save(ckpt, exp_dir / "best_model.pth")
            with open(exp_dir / "best_val_metrics.json", "w") as f:
                json.dump({"best_epoch": best_epoch, **val_metrics}, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} with no {args.primary_metric} improvement")
                break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without a best checkpoint")

    checkpoint = torch.load(exp_dir / "best_model.pth", map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    test_metrics = evaluate(model, test_loader, args, device, y_mean, y_std)
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    final_summary = {
        "exp_name": args.exp_name,
        "model_kind": args.model_kind,
        "target": args.target,
        "loss": args.loss,
        "lambda_k": args.lambda_k if args.loss == "lambdaloss+mae" else None,
        "alpha": args.alpha if args.loss != "mae+srcc" else None,
        "config": args.config_label,
        "param_count": param_count,
        "params_M": param_count / 1_000_000.0,
        "best_epoch": best_epoch,
        "primary_metric": args.primary_metric,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    with open(exp_dir / "run_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    config["best_epoch"] = best_epoch
    config["best_val_metrics"] = best_val_metrics
    config["test_metrics"] = test_metrics
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"Best checkpoint: {exp_dir / 'best_model.pth'} | best_epoch={best_epoch} "
        f"test NDCG@10={test_metrics['ndcg_10']:.4f} "
        f"NDCG@20={test_metrics['ndcg_20']:.4f} "
        f"regret={test_metrics['oracle_regret']:.6f} "
        f"top10={test_metrics['top10_hit']:.4f}"
    )


if __name__ == "__main__":
    main()
