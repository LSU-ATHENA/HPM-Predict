import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from predictor.models import (
    CAMapPredictor,
    get_ca_map_model,
)
from predictor.training.ca_dataloader import prep_ca_dataloaders
from predictor.training.dataloader import AVAILABLE_TARGETS, denormalize
from predictor.training.losses import (
    MAELambdaRankLoss,
    MAESRCCLoss,
    ndcg_at_k_per_prompt,
    pearson_corrcoef,
    spearman_corrcoef,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pack_inputs(batch: dict, device: torch.device, use_std: bool, use_max: bool):
    entropy = batch['entropy'].to(device)
    std = batch['std'].to(device) if use_std else None
    max_ = batch['max'].to(device) if use_max else None
    return entropy, std, max_


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_std: bool,
    use_max: bool,
    use_grouped: bool = False,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    targetlist, predictionlist = [], []

    uses_lambdarank = isinstance(criterion, MAELambdaRankLoss)
    use_amp = (device.type == 'cuda')

    for batch in loader:
        entropy, std, max_ = _pack_inputs(batch, device, use_std, use_max)
        targets = batch['y'].to(device).unsqueeze(1)
        group_ids = batch['prompt_id'].to(device) if use_grouped else None

        optimizer.zero_grad()
        # bf16 autocast: halves activation memory at no accuracy cost for
        # SRCC-based ranking. Cast preds back to fp32 for the loss so torchsort
        # / lambdarank stay numerically stable.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds = model(entropy, std=std, max_=max_)
        preds = preds.float()

        if uses_lambdarank:
            loss = criterion(preds, targets, group_ids=group_ids)
            criterion.backward(preds, targets, loss, group_ids=group_ids)
        else:
            if group_ids is not None:
                loss = criterion(preds, targets, group_ids=group_ids)
            else:
                loss = criterion(preds, targets)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * entropy.size(0)
        targetlist.extend(targets.squeeze(1).cpu().numpy())
        predictionlist.extend(preds.squeeze(1).detach().cpu().numpy())

    n_samples = len(loader.dataset)
    return {
        'loss': running_loss / n_samples,
        'target_mean': float(np.mean(targetlist)),
        'target_std': float(np.std(targetlist)),
        'pred_mean': float(np.mean(predictionlist)),
        'pred_std': float(np.std(predictionlist)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_std: bool,
    use_max: bool,
    ndcg_ks: tuple = (3, 5),
    y_mean: float = 0.0,
    y_std: float = 1.0,
    gain_type: str = 'exp2',
) -> Dict[str, float]:
    model.eval()
    all_preds_raw, all_targets_raw, all_pids = [], [], []
    use_amp = (device.type == 'cuda')

    for batch in loader:
        entropy, std, max_ = _pack_inputs(batch, device, use_std, use_max)
        targets_raw = batch['raw_y'].to(device)
        prompt_ids = batch['prompt_id'].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            preds_norm = model(entropy, std=std, max_=max_).squeeze(1)
        preds_norm = preds_norm.float()
        preds_raw = denormalize(preds_norm, y_mean, y_std)

        all_preds_raw.append(preds_raw)
        all_targets_raw.append(targets_raw)
        all_pids.append(prompt_ids)

    preds = torch.cat(all_preds_raw, 0)
    tgts = torch.cat(all_targets_raw, 0)
    pids = torch.cat(all_pids, 0)

    n = len(preds)
    mae_raw = (preds - tgts).abs().mean().item()
    if n > 1 and preds.std() > 1e-9:
        srcc = spearman_corrcoef(preds, tgts).item()
        pearson = pearson_corrcoef(preds, tgts).item()
        ndcgs = {
            f'ndcg_{k}': ndcg_at_k_per_prompt(preds, tgts, pids, k=k, gain_type=gain_type)
            for k in ndcg_ks
        }
    else:
        srcc = pearson = 0.0
        ndcgs = {f'ndcg_{k}': 0.0 for k in ndcg_ks}

    out = {
        'n_samples': n,
        'mae_raw': mae_raw,
        'srcc': srcc,
        'pearson': pearson,
        'target_mean': tgts.mean().item(),
        'target_std': tgts.std().item(),
        'pred_mean': preds.mean().item(),
        'pred_std': preds.std().item(),
    }
    out.update(ndcgs)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='sdxl',
                        help='Label for the DM this CA map came from (also picks head defaults)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset root containing cross_attn/, metadata.jsonl')

    parser.add_argument('--use_std', action='store_true')
    parser.add_argument('--use_max', action='store_true')
    parser.add_argument('--head_compression', type=int, default=1)

    parser.add_argument('--target', type=str, default='pick_score', choices=AVAILABLE_TARGETS)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', type=str, default='mae+srcc',
                        choices=['mae+srcc', 'mae+lambdarank'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--exp_name', type=str, default='ca_baseline')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_prompts', type=int, default=-1)
    parser.add_argument('--k_prompts', type=int, default=6)
    parser.add_argument('--ndcg_ks', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--primary_ndcg_k', type=int, default=3,
                        help='k used when --primary_metric=ndcg')
    parser.add_argument('--primary_metric', type=str, default='srcc',
                        choices=['ndcg', 'srcc'])
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    use_grouped = args.k_prompts > 0

    train_loader, val_loader, test_loader, stats = prep_ca_dataloaders(
        data_dir=args.data_dir,
        target=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        k_prompts_per_batch=args.k_prompts,
        max_prompts=args.max_prompts,
        use_std=args.use_std,
        use_max=args.use_max,
    )
    y_mean, y_std = stats['y_mean'], stats['y_std']
    num_heads, seq_len = stats['num_heads'], stats['seq_len']

    print(f"CA dims: H={num_heads}, T_p={seq_len}, c={args.head_compression}, "
          f"stats=entropy{'+std' if args.use_std else ''}{'+max' if args.use_max else ''}")

    model = get_ca_map_model(
        num_heads=num_heads,
        seq_len=seq_len,
        head_compression_ratio=args.head_compression,
        use_std=args.use_std,
        use_max=args.use_max,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    feat_dim = model.encoder.output_dim
    print(f"CAMapPredictor params: {n_params/1e6:.2f}M  (encoder feat_dim={feat_dim})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss == 'mae+srcc':
        criterion = MAESRCCLoss(srcc_weight=1.0, regularization_strength=1e-2)
    elif args.loss == 'mae+lambdarank':
        criterion = MAELambdaRankLoss(lambdarank_weight=1.0, sigma=1.0, gain_type='exp2')
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    primary_higher_better = (args.primary_metric != 'mae')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max' if primary_higher_better else 'min', factor=0.5, patience=5
    )

    best_val = float('-inf') if primary_higher_better else float('inf')
    primary_ndcg_key = f'ndcg_{args.primary_ndcg_k}'
    patience_left = args.patience

    for epoch in range(args.epochs):
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_std=args.use_std, use_max=args.use_max, use_grouped=use_grouped,
        )
        vm = evaluate(
            model, val_loader, device,
            use_std=args.use_std, use_max=args.use_max,
            ndcg_ks=tuple(args.ndcg_ks), y_mean=y_mean, y_std=y_std,
        )

        current = vm[primary_ndcg_key] if args.primary_metric == 'ndcg' else vm['srcc']

        ndcg_report = "  ".join(f"NDCG@{k}={vm[f'ndcg_{k}']:.4f}" for k in args.ndcg_ks)
        print(f"Epoch {epoch+1}/{args.epochs} "
              f"train_loss={tr['loss']:.4f} "
              f"val SRCC={vm['srcc']:.4f}  {ndcg_report}  "
              f"MAE={vm['mae_raw']:.4f}")

        scheduler.step(current)

        improved = (primary_higher_better and current > best_val) or \
                   (not primary_higher_better and current < best_val)
        if improved:
            best_val = current
            patience_left = args.patience
            cfg = model._config_dict(model_type=args.model_type)
            ckpt = {
                'model_state_dict': {k: v.half() for k, v in model.state_dict().items()},
                'model_config': cfg,
                'normalization': {
                    'target': args.target,
                    'y_mean': y_mean,
                    'y_std': y_std,
                },
            }
            torch.save(ckpt, exp_dir / 'best_model.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {args.patience})")
                break

    ckpt = torch.load(exp_dir / 'best_model.pth', weights_only=False)
    state = {k: v.float() for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)

    test_m = evaluate(
        model, test_loader, device,
        use_std=args.use_std, use_max=args.use_max,
        ndcg_ks=tuple(args.ndcg_ks), y_mean=y_mean, y_std=y_std,
    )
    test_ndcg_report = "  ".join(f"NDCG@{k}={test_m[f'ndcg_{k}']:.4f}" for k in args.ndcg_ks)
    print(f"\nTest SRCC={test_m['srcc']:.4f}  {test_ndcg_report}  MAE={test_m['mae_raw']:.4f}")

    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_m, f, indent=4)


if __name__ == '__main__':
    main()
