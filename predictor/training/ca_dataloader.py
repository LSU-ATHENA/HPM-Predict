import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from predictor.training.dataloader import PromptGroupedBatchSampler, _load_all_metadata


class CAMapDataset(Dataset):
    """Reads pre-computed CA-map stats from <data_dir>/cross_attn/.

    Each file 'p{pid:04d}_s{sid:02d}.pt' stores (H, T_p) fp16 tensors for
    'entropy', 'std', 'max', produced by
    gen_dataset/datagen/partial_ca_extractor.py.
    """

    def __init__(
        self,
        data_dir: str,
        samples: List[dict],
        target: str,
        y_mean: float,
        y_std: float,
        use_std: bool = False,
        use_max: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.target = target
        self.y_mean = y_mean
        self.y_std = y_std
        self.use_std = use_std
        self.use_max = use_max

        ca_dir = self.data_dir / 'cross_attn'
        self._cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        for rec in samples:
            pid, sid = rec['prompt_id'], rec['sample_idx']
            path = ca_dir / f"p{pid:04d}_s{sid:02d}.pt"
            d = torch.load(path, map_location='cpu', weights_only=False)
            entry = {'entropy': d['entropy'].float()}
            if use_std:
                entry['std'] = d['std'].float()
            if use_max:
                entry['max'] = d['max'].float()
            self._cache[(pid, sid)] = entry

        first = next(iter(self._cache.values()))['entropy']
        self.num_heads, self.seq_len = first.shape

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]
        pid, sid = rec['prompt_id'], rec['sample_idx']
        entry = self._cache[(pid, sid)]

        raw_score = float(rec.get(self.target, 0.0))
        normalized = (raw_score - self.y_mean) / max(self.y_std, 1e-8)

        out = {
            'entropy': entry['entropy'],
            'prompt_id': pid,
            'y': torch.tensor(normalized, dtype=torch.float32),
            'raw_y': torch.tensor(raw_score, dtype=torch.float32),
        }
        if self.use_std:
            out['std'] = entry['std']
        if self.use_max:
            out['max'] = entry['max']
        return out


def prep_ca_dataloaders(
    data_dir: str,
    target: str = 'pick_score',
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    k_prompts_per_batch: int = 0,
    max_prompts: int = -1,
    use_std: bool = False,
    use_max: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    all_records = _load_all_metadata(data_dir)
    print(f"Loaded {len(all_records)} samples from metadata")

    ca_dir = Path(data_dir) / 'cross_attn'
    available = {p.stem for p in ca_dir.glob('*.pt')}
    all_records = [
        r for r in all_records
        if f"p{r['prompt_id']:04d}_s{r['sample_idx']:02d}" in available
    ]
    print(f"With CA map available: {len(all_records)}")
    if not all_records:
        raise ValueError(f"No CA-augmented samples in {ca_dir}")

    records_by_prompt: Dict[int, List[dict]] = {}
    for rec in all_records:
        records_by_prompt.setdefault(rec['prompt_id'], []).append(rec)
    all_prompt_ids = sorted(records_by_prompt.keys())
    print(f"Found {len(all_prompt_ids)} unique prompts with CA maps")

    if max_prompts > 0 and max_prompts < len(all_prompt_ids):
        all_prompt_ids = all_prompt_ids[:max_prompts]
        kept = set(all_prompt_ids)
        all_records = [r for r in all_records if r['prompt_id'] in kept]
        print(f"Using {len(all_prompt_ids)} prompts ({len(all_records)} samples)")

    rng = random.Random(seed)
    shuffled_ids = all_prompt_ids.copy()
    rng.shuffle(shuffled_ids)
    n = len(shuffled_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])

    train_records = [r for r in all_records if r['prompt_id'] in train_ids]
    val_records   = [r for r in all_records if r['prompt_id'] in val_ids]
    test_records  = [r for r in all_records if r['prompt_id'] in test_ids]

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test prompts")
    print(f"Samples: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")

    vals = np.array([float(r.get(target, 0.0)) for r in train_records])
    y_mean = float(vals.mean())
    y_std = float(vals.std())
    print(f"  {target}: mean={y_mean:.6f}, std={y_std:.6f}, n={len(vals)}")

    common = dict(
        data_dir=data_dir, target=target,
        y_mean=y_mean, y_std=y_std,
        use_std=use_std, use_max=use_max,
    )

    train_ds = CAMapDataset(samples=train_records, **common)
    val_ds   = CAMapDataset(samples=val_records,   **common)
    test_ds  = CAMapDataset(samples=test_records,  **common)

    stats = {
        'target': target,
        'y_mean': y_mean,
        'y_std': y_std,
        'num_heads': train_ds.num_heads,
        'seq_len': train_ds.seq_len,
    }
    print(f"  CA shape per sample: (H={stats['num_heads']}, T_p={stats['seq_len']})")

    if k_prompts_per_batch > 0:
        sampler = PromptGroupedBatchSampler(train_ds, k_prompts_per_batch, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, stats
