import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

try:
    from metrics import MultiMetricScorer           # cwd=gen_dataset/ (datagen scripts)
except ImportError:
    from gen_dataset.metrics import MultiMetricScorer  # cwd=repo root (inference scripts)

SEED_RANGE = (0, 2**31 - 1)


def sample_seeds(master_seed, prompt_idx, n, low=SEED_RANGE[0], high=SEED_RANGE[1]):
    rng = random.Random(master_seed * 100000 + prompt_idx)
    return rng.sample(range(low, high + 1), n)


class BaseDataGenerator(ABC):
    def __init__(self, save_dir, prompts, latent_shape, num_images_per_prompt=10,
                 master_seed=42, seed_range=SEED_RANGE, device='cuda', metrics=None,
                 task_id=None, extract_cross_attn=True, cross_attn_steps=(1, 15, 40)):
        self.save_dir = Path(save_dir)
        self.prompts = prompts
        self.latent_shape = latent_shape
        self.num_images_per_prompt = num_images_per_prompt
        self.master_seed = master_seed
        self.seed_range = seed_range
        self.device = device
        self.task_id = task_id
        self.extract_cross_attn = extract_cross_attn
        self.cross_attn_steps = cross_attn_steps

        subdirs = ['images', 'noise', 'embeds']
        if extract_cross_attn:
            subdirs.append('cross_attn')
        for subdir in subdirs:
            (self.save_dir / subdir).mkdir(parents=True, exist_ok=True)

        self.scorer = MultiMetricScorer(metrics=metrics, device=device)

    @abstractmethod
    def setup_pipeline(self):
        ...

    @abstractmethod
    def encode_and_save_prompt(self, pipe, prompt, save_path):
        ...

    @abstractmethod
    def generate_image(self, pipe, embeds_dict, noise, extractor=None):
        ...

    @property
    def _meta_filename(self):
        if self.task_id is not None:
            return f"metadata_{self.task_id}.jsonl"
        return "metadata.jsonl"

    def _load_existing(self):
        existing = set()
        # When using per-task metadata, also read main metadata.jsonl for prior work
        paths = [self.save_dir / self._meta_filename]
        if self.task_id is not None:
            main_meta = self.save_dir / "metadata.jsonl"
            if main_meta.exists():
                paths.insert(0, main_meta)
        for meta_path in paths:
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            existing.add((d['prompt_id'], d['sample_idx']))
        return existing

    def generate_noise(self, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn((1, *self.latent_shape), generator=gen,
                           device=self.device, dtype=torch.float16)

    def run(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(self.prompts)
        end_idx = min(end_idx, len(self.prompts))

        pipe = self.setup_pipeline()
        existing = self._load_existing()
        if existing:
            print(f"Resuming: {len(existing)} samples already done")

        extractor = None
        if self.extract_cross_attn:
            from .cross_attn_extractor import CrossAttentionEntropyExtractor
            # DiT models (Hunyuan, PixArt) use pipe.transformer; UNet models use pipe.unet
            denoiser = getattr(pipe, 'transformer', None) or pipe.unet
            extractor = CrossAttentionEntropyExtractor(
                denoiser, extract_steps=self.cross_attn_steps,
            )
            print(f"[cross_attn] Hooked {len(extractor.layer_names)} layers on "
                  f"{type(denoiser).__name__}, "
                  f"extracting at steps {sorted(self.cross_attn_steps)}")

        meta_path = self.save_dir / self._meta_filename

        with open(meta_path, 'a') as meta_file:
            for p_idx in tqdm(range(start_idx, end_idx), desc="Prompts"):
                prompt_str = self.prompts[p_idx]

                embed_path = self.save_dir / "embeds" / f"p{p_idx:04d}.pt"
                if embed_path.exists():
                    embeds_dict = torch.load(embed_path, map_location='cpu', weights_only=False)
                    embeds_dict = {k: v.to(self.device) for k, v in embeds_dict.items()}
                else:
                    embeds_dict = self.encode_and_save_prompt(pipe, prompt_str, embed_path)

                seeds = sample_seeds(self.master_seed, p_idx, self.num_images_per_prompt,
                                     self.seed_range[0], self.seed_range[1])

                for j in range(self.num_images_per_prompt):
                    if (p_idx, j) in existing:
                        continue

                    seed = seeds[j]
                    name = f"p{p_idx:04d}_s{j:02d}"

                    noise = self.generate_noise(seed)
                    torch.save(noise.cpu().half(), self.save_dir / "noise" / f"{name}.pt")

                    image = self.generate_image(pipe, embeds_dict, noise, extractor=extractor)
                    image_path = self.save_dir / "images" / f"{name}.jpg"
                    image.save(image_path)

                    scores = self.scorer.score(image, prompt_str, image_path=str(image_path))
                    meta = {'prompt_id': p_idx, 'sample_idx': j, 'seed': seed,
                            'prompt': prompt_str, **scores}

                    if extractor is not None:
                        ca_results = extractor.get_results()
                        torch.save(ca_results, self.save_dir / "cross_attn" / f"{name}.pt")
                        meta.update(extractor.get_metadata_scalars(ca_results))

                    meta_file.write(json.dumps(meta) + '\n')
                    meta_file.flush()

                    del image, noise
                    torch.cuda.empty_cache()

        if extractor is not None:
            extractor.remove_hooks()

        print(f"Done. Output: {self.save_dir}")
