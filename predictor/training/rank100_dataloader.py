import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from predictor.configs.model_dims import get_dims
from predictor.training.dataloader import _extract_embeds


STAGE1_SPLITS = {
    "stage1_train": "train_prompt_ids",
    "stage1_val": "val_prompt_ids",
    "stage1_test": "test_prompt_ids",
}
RANK100_SPLITS = {"rank100_train", "rank100_val", "rank100_test"}


def load_metadata_records(data_dir: str, metadata_glob: str = "metadata*.jsonl") -> List[dict]:
    data_path = Path(data_dir)
    records = []
    seen = set()
    for meta_path in sorted(data_path.glob(metadata_glob)):
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["prompt_id"]), int(rec["sample_idx"]))
                if key in seen:
                    continue
                seen.add(key)
                records.append(rec)
    if not records:
        raise FileNotFoundError(f"No metadata records found in {data_dir} with {metadata_glob}")
    return records


def group_records_by_prompt(records: List[dict]) -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = {}
    for rec in records:
        groups.setdefault(int(rec["prompt_id"]), []).append(rec)
    for recs in groups.values():
        recs.sort(key=lambda r: int(r["sample_idx"]))
    return groups


def load_prompt_split_ids(split_json: str) -> Dict[str, List[int]]:
    with open(split_json) as f:
        split = json.load(f)
    return {key: [int(pid) for pid in split[key]] for key in STAGE1_SPLITS.values()}


def load_rank100_manifest_ids(manifest_path: str, split_name: str) -> List[int]:
    ids = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == split_name:
                ids.append(int(rec["prompt_id"]))
    return ids


def prompt_ids_for_split(split_name: str, split_json: str, rank100_manifest: Optional[str]) -> List[int]:
    if split_name in STAGE1_SPLITS:
        split = load_prompt_split_ids(split_json)
        return split[STAGE1_SPLITS[split_name]]
    if split_name in RANK100_SPLITS:
        if rank100_manifest is None:
            raise ValueError(f"--rank100_manifest is required for split {split_name}")
        return load_rank100_manifest_ids(rank100_manifest, split_name)
    raise ValueError(f"Unknown split {split_name!r}")


def compute_global_normalization(
    records_by_prompt: Dict[int, List[dict]],
    train_prompt_ids: List[int],
    target: str,
    max_sample_idx: int = 20,
) -> Tuple[float, float]:
    values = []
    train_ids = set(train_prompt_ids)
    for pid, recs in records_by_prompt.items():
        if pid not in train_ids:
            continue
        for rec in recs:
            if int(rec["sample_idx"]) < max_sample_idx:
                values.append(float(rec[target]))
    if not values:
        raise ValueError("No values available to compute train-split normalization")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


class Rank100PromptDataset(Dataset):
    """Prompt/query-group dataset.

    Each item is one prompt with M candidate noises or CA-map features. The
    ranking relevance is within-prompt, while y remains global train-normalized
    for MAE.
    """

    def __init__(
        self,
        data_dir: str,
        records_by_prompt: Dict[int, List[dict]],
        prompt_ids: List[int],
        model_type: str,
        target: str,
        y_mean: float,
        y_std: float,
        candidates_per_prompt: int = 100,
        input_kind: str = "noise",
        text_embed_type: str = "default",
        seq_len_override: Optional[int] = None,
        use_std: bool = False,
        use_max: bool = False,
        require_complete: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.records_by_prompt = records_by_prompt
        self.prompt_ids = [int(pid) for pid in prompt_ids]
        self.model_type = model_type
        self.target = target
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.candidates_per_prompt = int(candidates_per_prompt)
        self.input_kind = input_kind
        self.text_embed_type = text_embed_type
        self.use_std = use_std
        self.use_max = use_max

        if input_kind not in {"noise", "camap"}:
            raise ValueError(f"input_kind must be 'noise' or 'camap', got {input_kind!r}")

        dims = get_dims(model_type)
        self.embed_dim = dims["embed_dim"]
        self.seq_len = seq_len_override if seq_len_override is not None else dims["seq_len"]
        if seq_len_override is None and text_embed_type == "t5+clip" and model_type == "hunyuan_dit":
            self.seq_len = 333

        self.group_records: Dict[int, List[dict]] = {}
        missing = []
        for pid in self.prompt_ids:
            by_sid = {
                int(rec["sample_idx"]): rec
                for rec in records_by_prompt.get(pid, [])
                if int(rec["sample_idx"]) < self.candidates_per_prompt
            }
            wanted = set(range(self.candidates_per_prompt))
            if require_complete and set(by_sid) != wanted:
                missing.append((pid, sorted(wanted - set(by_sid))[:10]))
                continue
            ordered = [by_sid[sid] for sid in range(self.candidates_per_prompt) if sid in by_sid]
            if len(ordered) >= 2:
                self.group_records[pid] = ordered

        if missing:
            preview = ", ".join(f"pid={pid} missing={m}" for pid, m in missing[:5])
            raise ValueError(f"Incomplete prompt groups for M={self.candidates_per_prompt}: {preview}")
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

    def __len__(self):
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

    def __getitem__(self, idx):
        pid = self.prompt_ids[idx]
        records = self.group_records[pid]
        sample_idxs = torch.tensor([int(r["sample_idx"]) for r in records], dtype=torch.long)
        seeds = torch.tensor([int(r.get("seed", -1)) for r in records], dtype=torch.long)
        raw = torch.tensor([float(r[self.target]) for r in records], dtype=torch.float32)
        y = (raw - self.y_mean) / max(self.y_std, 1e-8)
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


def build_rank100_loader(
    data_dir: str,
    split_name: str,
    split_json: str,
    rank100_manifest: Optional[str],
    model_type: str,
    target: str,
    y_mean: float,
    y_std: float,
    candidates_per_prompt: int = 100,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    input_kind: str = "noise",
    text_embed_type: str = "default",
    seq_len_override: Optional[int] = None,
    use_std: bool = False,
    use_max: bool = False,
    max_prompts: int = -1,
    metadata_glob: str = "metadata*.jsonl",
) -> Tuple[DataLoader, Rank100PromptDataset]:
    records = load_metadata_records(data_dir, metadata_glob=metadata_glob)
    groups = group_records_by_prompt(records)
    prompt_ids = prompt_ids_for_split(split_name, split_json, rank100_manifest)
    if max_prompts > 0:
        prompt_ids = prompt_ids[:max_prompts]

    ds = Rank100PromptDataset(
        data_dir=data_dir,
        records_by_prompt=groups,
        prompt_ids=prompt_ids,
        model_type=model_type,
        target=target,
        y_mean=y_mean,
        y_std=y_std,
        candidates_per_prompt=candidates_per_prompt,
        input_kind=input_kind,
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
