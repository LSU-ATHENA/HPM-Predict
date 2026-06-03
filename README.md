# Can We Predict The Human Preference For Text-to-Image Content Prior To Generation And Is It Even Useful To Do So?

## Overview

![fig1](assets/predictor.pdf)
Diffusion Models (DM) have revolutionized text-driven generation by enabling the synthesis of high-quality, photorealistic visual content from user prompts. Whereas prior advances in visual generation such as VAEs and GANs were primarily evaluated on perceptual or visual similarity metrics such as FID PSNR, DM advances have fostered the development of more advanced Human Preference Metrics (HPM) that model and quantify human judgment as scalar values. However, DMs synthesize content using an inherently stochastic process where random noise seeds generation. The initial random noise directly affects the quality of generated outputs, both qualitatively and quantitatively. This influence is pronounced in smaller models for local deployment scenarios. Given this phenomenon, we first investigate to what extent we can predict scalar HPM scores prior to committing compute resources for generation. Further, we then investigate to what extent we can leverage such prediction to improve the quality of generated images, and also study which HPMs are best suited for this task. Our investigation reveals that not only is this possible, but that it is feasible to achieve negligible hardware overhead.



### Prerequisites

- Python 3.10+



### Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/LSU-ATHENA/HPM-Predict
cd paine
conda create -n paine python=3.10
conda activate paine
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install diffusers transformers accelerate sentencepiece safetensors beautifulsoup4
pip install -r requirements.txt
```


This repository provides three predictor families across four diffusion models:

- **CA-I** — Cross-Attention Internal: reads the diffusion model's first cross-attention map.
- **CA-E** — Cross-Attention External: computes its own patch/prompt cross-attention.
- **EnCat** — Encode-and-Concatenate: encodes noise and prompt separately, then fuses.

## Supported Models

| Model | `--model_type` | Latent | Text Encoder | Embed dim / Seq len |
|-------|----------------|--------|--------------|---------------------|
| Stable Diffusion XL | `sdxl` | 4×128×128 | CLIP | 2048 / 77 |
| DreamShaper-XL | `dreamshaper` | 4×128×128 | CLIP | 2048 / 77 |
| Hunyuan-DiT | `hunyuan_dit` | 4×128×128 | T5 | 2048 / 333 |
| PixArt-Σ | `pixart_sigma` | 4×128×128 | T5 | 4096 / 300 |

## Repository Structure

```
predictor/
  models/                  # CA-I, CA-E, EnCat predictors + noise/text encoders
  training/                # Training scripts, dataset loaders, loss functions
  inference/               # Checkpoint loading, noise selection, generation
  configs/                 # Model dimension registry + per-DM training configs
gen_dataset/               # Dataset generation: per-DM runners + CA-map extractor
```

The workflow is three stages: **(1)** generate a per-DM dataset with `gen_dataset/`, **(2)** train one or more predictors on it, **(3)** run inference to select noise and generate images. Datasets and trained checkpoints are produced by these steps and are not committed to the repository.

## Dataset Generation

Each diffusion model has a generation runner under `gen_dataset/` that, for a set of prompts, samples initial noises, generates images, scores every image with all four Human Preference Metrics (HPSv2, HPSv3, ImageReward, PickScore), and caches the prompt embeddings and noise tensors. Run the runners from inside `gen_dataset/`:

```bash
cd gen_dataset

# SDXL
python run_sdxl.py          --n-prompts 5000 --images-per-prompt 20 --save-dir data/generated/sdxl_1024
# DreamShaper-XL
python run_dreamshaper.py   --n-prompts 5000 --images-per-prompt 20 --save-dir data/generated/dreamshaper_xl_turbo
# Hunyuan-DiT
python run_hunyuan.py       --n-prompts 5000 --images-per-prompt 20 --save-dir data/generated/hunyuan_dit_1024
# PixArt-Σ
python run_pixart_sigma.py  --n-prompts 5000 --images-per-prompt 20 --save-dir data/generated/pixart_sigma_1024
```

Prompts are read from `--prompts-file` (default `data/pickscore_train_prompts.json`); provide your own prompt list or the PickScore training prompts. Each runner writes a dataset directory:

```
<save-dir>/
  metadata.jsonl           # one row per (prompt_id, sample_idx): prompt + hpsv2, hpsv3, image_reward, pick_score
  embeds/                  # pre-encoded prompt embeddings (p0000.pt, p0001.pt, ...)
  noise/                   # initial noise tensors (p0000_s00.pt, p0000_s01.pt, ...)
```

**CA-I only** additionally needs the diffusion model's first cross-attention statistics. After generation, augment the dataset in place to add a `cross_attn/` directory (per-head entropy/std/max of the first cross-attention map):

```bash
python augment_cross_attn.py --model <sdxl|dreamshaper|hunyuan_dit|pixart_sigma> --data-dir data/generated/sdxl_1024
```

EnCat and CA-E train from `embeds/` + `noise/`; CA-I trains from `cross_attn/`.

## Training

Pick a dataset directory produced above as `--data_dir`. Targets (`--target`) are `pick_score`, `hpsv2`, `hpsv3`, or `image_reward`. The data is split 80/10/10 by prompt id (seed 42), and predictors are optimized with AdamW (lr 1e-4, weight decay 1e-8) under the ranking + MAE objective (`--loss mae+lambdarank`). Checkpoints are written to `experiments/{exp_name}/best_model.pth`.

The diffusion model is selected with `--model_type {sdxl,dreamshaper,hunyuan_dit,pixart_sigma}`; the predictor reads the matching latent/text dimensions automatically (T5 vs CLIP embeddings are handled per model).

### EnCat — `predictor/training/train.py`

```bash
python -m predictor.training.train \
    --model_type <sdxl|dreamshaper|hunyuan_dit|pixart_sigma> \
    --data_dir gen_dataset/data/generated/<dataset> \
    --target pick_score \
    --loss mae+lambdarank \
    --k_prompts 12 \
    --epochs 80 \
    --exp_name encat_<model_type>
```

### CA-E — `predictor/training/train_patchca.py`

External patch cross-attention. The paper's configuration is entropy-only statistics with patch size 4 and head compression 2:

```bash
python -m predictor.training.train_patchca \
    --model_type <sdxl|dreamshaper|hunyuan_dit|pixart_sigma> \
    --data_dir gen_dataset/data/generated/<dataset> \
    --target pick_score \
    --loss mae+lambdarank \
    --patch_size 4 \
    --head_compression 2 \
    --k_prompts 12 \
    --epochs 80 \
    --exp_name cae_<model_type>
```

### CA-I — `predictor/training/train_ca.py`

Internal cross-attention statistics; requires the `cross_attn/` directory created by `augment_cross_attn.py`. The paper's configuration uses entropy+std+max with head compression 1:

```bash
python -m predictor.training.train_ca \
    --model_type <sdxl|dreamshaper|hunyuan_dit|pixart_sigma> \
    --data_dir gen_dataset/data/generated/<dataset> \
    --target pick_score \
    --loss mae+lambdarank \
    --use_std --use_max \
    --head_compression 1 \
    --k_prompts 12 \
    --epochs 80 \
    --exp_name cai_<model_type>
```

To reproduce all baselines, run the three commands for each `--model_type`: swap `sdxl` for `dreamshaper`, `hunyuan_dit`, or `pixart_sigma`, pointing `--data_dir` at the matching generated dataset.

## Inference

Each predictor has its own inference script. All three sample `N` candidate noises, score them with the trained predictor, select the best `B`, and generate with the chosen diffusion model. Point `--checkpoint` at a `best_model.pth` produced by training:

```bash
# EnCat
python -m predictor.inference.generate \
    --model_type sdxl \
    --checkpoint experiments/encat_sdxl/best_model.pth \
    --prompt "two dogs and one cat" \
    --N 100 --B 1

# CA-E
python -m predictor.inference.generate_patchca \
    --model_type sdxl \
    --checkpoint experiments/cae_sdxl/best_model.pth \
    --prompt "two dogs and one cat" \
    --N 100 --B 1

# CA-I  (runs the diffusion model once per candidate to read its cross-attention map)
python -m predictor.inference.generate_ca \
    --model_type sdxl \
    --checkpoint experiments/cai_sdxl/best_model.pth \
    --prompt "two dogs and one cat" \
    --N 100 --B 1
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | required | Diffusion model (`sdxl`, `dreamshaper`, `hunyuan_dit`, `pixart_sigma`) |
| `--checkpoint` | required | Path to a trained predictor checkpoint |
| `--prompt` | required | Text prompt for generation |
| `--N` | `100` | Number of candidate noises to sample |
| `--B` | `1` | Number of top noises to generate from |
| `--seed` | — | Random seed |
| `--compare` | off | Also generate a random-noise baseline |

Generation steps, guidance scale, and scheduler follow a fixed per-model protocol (`gen_dataset/datagen/generation_protocol.py`). Each checkpoint stores its architecture config and target-score normalization, so the loader reconstructs the correct model and denormalizes predictions to native score units.

## Architecture

All three predictors output a single scalar preference score and share a small MLP score head (512 → 256 → 64 → 1, SiLU, dropout 0.1).

- **CA-I**: reads the diffusion model's first cross-attention map, reduces each attention row to per-head, per-token entropy/std/max statistics, applies head compression, and scores the flattened statistics. No separate noise or text encoder.
- **CA-E**: patchifies the latent, projects prompt tokens, runs one cross-attention block (patches attend to prompt), reduces the attention map to entropy statistics, and scores them — without invoking the diffusion model.
- **EnCat**: a convolutional noise encoder downsamples the latent to a 1024-d vector while a text encoder summarizes the prompt embedding; the two are concatenated and scored.
