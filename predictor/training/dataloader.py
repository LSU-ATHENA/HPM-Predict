import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from predictor.configs.model_dims import MODEL_DIMS, get_dims


AVAILABLE_TARGETS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score', 'clip_score']

# Model-specific embedding extraction config
# key: which key in embeds/*.pt to load
# mask_key: which key for attention mask (None = all ones)
EMBEDDING_CONFIG = {
    'sdxl': {
        'key': 'prompt_embeds',
        'mask_key': None,  
    },
    'dreamshaper': {
        'key': 'prompt_embeds',
        'mask_key': None,
    },
    'hunyuan_dit': {
        'key': 'prompt_embeds_2',  
        'mask_key': 'prompt_attention_mask_2',
    },
    'pixart_sigma': {
        'key': 'prompt_embeds',
        'mask_key': 'prompt_attention_mask',
    },
'sana_sprint': {
        'key': 'prompt_embeds',
        'mask_key': 'prompt_attention_mask',
    },
}

def _extract_t5_clip_combined(
    embeddings: dict,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate Hunyuan-DiT T5 (256, 2048) and CLIP (77, 1024 -> 2048) along seq."""
    t5 = embeddings['prompt_embeds_2']
    if t5.dim() == 3:
        t5 = t5.squeeze(0)
    t5_mask = embeddings.get('prompt_attention_mask_2')
    if t5_mask is not None and t5_mask.dim() == 2:
        t5_mask = t5_mask.squeeze(0)
    if t5_mask is None:
        t5_mask = torch.ones(t5.shape[0], dtype=torch.long)

    clip = embeddings['prompt_embeds']
    if clip.dim() == 3:
        clip = clip.squeeze(0)
    clip_mask = embeddings.get('prompt_attention_mask')
    if clip_mask is not None and clip_mask.dim() == 2:
        clip_mask = clip_mask.squeeze(0)
    if clip_mask is None:
        clip_mask = torch.ones(clip.shape[0], dtype=torch.long)

    if clip.shape[1] < t5.shape[1]:
        pad = torch.zeros(clip.shape[0], t5.shape[1] - clip.shape[1])
        clip = torch.cat([clip, pad], dim=1)

    embeds = torch.cat([t5, clip], dim=0)
    mask = torch.cat([t5_mask, clip_mask], dim=0)

    if embeds.shape[0] < seq_len:
        pad_e = torch.zeros(seq_len - embeds.shape[0], embeds.shape[1])
        embeds = torch.cat([embeds, pad_e], dim=0)
        pad_m = torch.zeros(seq_len - mask.shape[0], dtype=mask.dtype)
        mask = torch.cat([mask, pad_m], dim=0)
    elif embeds.shape[0] > seq_len:
        embeds = embeds[:seq_len]
        mask = mask[:seq_len]
    return embeds, mask


def _extract_embeds(
    embeddings: dict,
    model_type: str,
    embed_dim: int,
    seq_len: int,
    text_embed_type: str = 'default',
) -> Tuple[torch.Tensor, torch.Tensor]:
    if text_embed_type == 't5+clip' and model_type == 'hunyuan_dit':
        return _extract_t5_clip_combined(embeddings, seq_len)

    config = EMBEDDING_CONFIG[model_type]

    embeds = embeddings[config['key']]
    if embeds.dim() == 3:
        embeds = embeds.squeeze(0)

    if config['mask_key'] is not None and config['mask_key'] in embeddings:
        mask = embeddings[config['mask_key']]
        if mask.dim() == 2:
            mask = mask.squeeze(0)
    else:
        mask = torch.ones(embeds.shape[0], dtype=torch.long)

    current_len = embeds.shape[0]
    if current_len < seq_len:
        pad_embeds = torch.zeros(seq_len - current_len, embeds.shape[1])
        embeds = torch.cat([embeds, pad_embeds], dim=0)
        pad_mask = torch.zeros(seq_len - current_len, dtype=mask.dtype)
        mask = torch.cat([mask, pad_mask], dim=0)
    elif current_len > seq_len:
        embeds = embeds[:seq_len]
        mask = mask[:seq_len]

    return embeds, mask



def _load_all_metadata(data_dir: str) -> List[dict]:
    data_path = Path(data_dir)
    records = []

    meta_files = sorted(data_path.glob("metadata*.jsonl"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata*.jsonl found in {data_dir}")

    seen = set()
    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record['prompt_id'], record['sample_idx'])
                if key not in seen:
                    seen.add(key)
                    records.append(record)

    return records


class PNMDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        samples: List[dict],
        model_type: str,
        target: str = 'hpsv2',
        y_mean: float = 0.0,
        y_std: float = 1.0,
        text_embed_type: str = 'default',
        seq_len_override: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.target = target
        self.y_mean = y_mean
        self.y_std = y_std
        self.text_embed_type = text_embed_type

        dims = get_dims(model_type)
        self.embed_dim = dims['embed_dim']
        self.seq_len = seq_len_override if seq_len_override is not None else dims['seq_len']

        self.samples = samples

        unique_pids = sorted(set(r['prompt_id'] for r in samples))
        self._embed_cache = {}
        for pid in unique_pids:
            emb_path = self.data_dir / "embeds" / f"p{pid:04d}.pt"
            embeddings = torch.load(emb_path, map_location='cpu', weights_only=False)
            embeds, mask = _extract_embeds(
                embeddings, self.model_type, self.embed_dim, self.seq_len,
                text_embed_type=self.text_embed_type,
            )
            self._embed_cache[pid] = (embeds, mask)

        # Preload all noise tensors into RAM to eliminate disk I/O during training.
        self._noise_cache = {}
        for rec in samples:
            pid, sid = rec['prompt_id'], rec['sample_idx']
            key = (pid, sid)
            noise_path = self.data_dir / "noise" / f"p{pid:04d}_s{sid:02d}.pt"
            noise = torch.load(noise_path, map_location='cpu', weights_only=False)
            if noise.dim() == 4:
                noise = noise.squeeze(0)
            self._noise_cache[key] = noise

    def __len__(self):
        return len(self.samples)

    def _get_embeddings(self, prompt_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._embed_cache[prompt_id]

    def __getitem__(self, idx):
        record = self.samples[idx]
        prompt_id = record['prompt_id']
        sample_idx = record['sample_idx']

        # Load noise from cache
        noise = self._noise_cache[(prompt_id, sample_idx)]

        prompt_embeds, prompt_mask = self._get_embeddings(prompt_id)

        raw_score = float(record.get(self.target, 0.0))
        normalized = (raw_score - self.y_mean) / max(self.y_std, 1e-8)

        return {
            'noise': noise.float(),
            'prompt_embeds': prompt_embeds.float(),
            'prompt_mask': prompt_mask.float(),
            'prompt_id': prompt_id,
            'y': torch.tensor(normalized, dtype=torch.float32),
            'raw_y': torch.tensor(raw_score, dtype=torch.float32),
        }


class PromptGroupedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset: PNMDataset, k_prompts_per_batch: int, shuffle: bool = True):
        self.shuffle = shuffle
        self.k = k_prompts_per_batch

        # Build prompt_id -> [dataset indices] mapping
        self.prompt_to_indices: Dict[int, List[int]] = {}
        for idx, record in enumerate(dataset.samples):
            pid = record['prompt_id']
            if pid not in self.prompt_to_indices:
                self.prompt_to_indices[pid] = []
            self.prompt_to_indices[pid].append(idx)

        self.prompt_ids = list(self.prompt_to_indices.keys())

        samples_per_prompt = [len(v) for v in self.prompt_to_indices.values()]
        print(f"  PromptGroupedBatchSampler: {len(self.prompt_ids)} prompts, "
              f"~{sum(samples_per_prompt) / len(samples_per_prompt):.0f} samples/prompt, "
              f"k={self.k}, batch_size={self.k * samples_per_prompt[0]}")

    def __iter__(self) -> Iterator[List[int]]:
        prompt_ids = self.prompt_ids.copy()
        if self.shuffle:
            random.shuffle(prompt_ids)

        # Yield complete batches only (drop_last behavior)
        for i in range(0, len(prompt_ids) - self.k + 1, self.k):
            batch_prompts = prompt_ids[i:i + self.k]
            batch_indices = []
            for pid in batch_prompts:
                batch_indices.extend(self.prompt_to_indices[pid])
            yield batch_indices

    def __len__(self) -> int:
        return len(self.prompt_ids) // self.k


def prep_dataloaders(
    data_dir: str,
    model_type: str,
    target: str = 'hpsv2',
    split_by: str = 'prompt',
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    k_prompts_per_batch: int = 0,
    max_prompts: int = -1,
    max_samples_per_prompt: int = -1,
    text_embed_type: str = 'default',
    seq_len_override: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    all_records = _load_all_metadata(data_dir)
    print(f"Loaded {len(all_records)} samples from metadata")

    if max_samples_per_prompt > 0:
        all_records = [r for r in all_records if r['sample_idx'] < max_samples_per_prompt]
        print(f"Filtered to first {max_samples_per_prompt} samples/prompt: {len(all_records)} samples")

    records_by_prompt = {}
    for rec in all_records:
        pid = rec['prompt_id']
        if pid not in records_by_prompt:
            records_by_prompt[pid] = []
        records_by_prompt[pid].append(rec)

    all_prompt_ids = sorted(records_by_prompt.keys())

    if not all_prompt_ids:
        raise ValueError(f"No samples found in {data_dir}")

    print(f"Found {len(all_prompt_ids)} unique prompts")

    # Limit number of prompts if requested
    if max_prompts > 0 and max_prompts < len(all_prompt_ids):
        all_prompt_ids = all_prompt_ids[:max_prompts]
        all_records = [r for r in all_records if r['prompt_id'] in set(all_prompt_ids)]
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
    val_records = [r for r in all_records if r['prompt_id'] in val_ids]
    test_records = [r for r in all_records if r['prompt_id'] in test_ids]

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test prompts")
    print(f"Samples: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")

    vals = np.array([float(r.get(target, 0.0)) for r in train_records])
    y_mean = float(vals.mean())
    y_std = float(vals.std())
    print(f"  {target}: mean={vals.mean():.6f}, std={vals.std():.6f}, n={len(vals)}")

    stats = {
        'target': target,
        'y_mean': y_mean,
        'y_std': y_std,
    }

    if seq_len_override is None and text_embed_type == 't5+clip' and model_type == 'hunyuan_dit':
        seq_len_override = 333

    common_kwargs = dict(
        data_dir=data_dir,
        model_type=model_type,
        target=target,
        y_mean=y_mean,
        y_std=y_std,
        text_embed_type=text_embed_type,
        seq_len_override=seq_len_override,
    )

    train_ds = PNMDataset(samples=train_records, **common_kwargs)
    val_ds = PNMDataset(samples=val_records, **common_kwargs)
    test_ds = PNMDataset(samples=test_records, **common_kwargs)

    if k_prompts_per_batch > 0:
        # Grouped batching: K prompts per batch, all their samples together
        grouped_sampler = PromptGroupedBatchSampler(train_ds, k_prompts_per_batch, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=grouped_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Standard global shuffle
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


def denormalize(pred: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return pred * std + mean
