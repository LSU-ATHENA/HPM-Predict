from typing import Optional, Tuple, Union

import torch


def generate_noise_candidates(
    num_candidates: int,
    latent_shape: Tuple[int, ...],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    generator: Optional[Union[torch.Generator, list]] = None,
) -> torch.Tensor:

    if isinstance(generator, list):
        generator = generator[0] if generator else None

    return torch.randn(
        (num_candidates, *latent_shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def select_top_k_noise(
    predictor: torch.nn.Module,
    noises: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_select: int = 1,
    head_index: int = 0,
    batch_size: int = 64,
    return_indices: bool = False,
) -> torch.Tensor:

    predictor.eval()
    num_candidates = noises.shape[0]
    device = noises.device
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)

    all_scores = []

    with torch.inference_mode():
        for i in range(0, num_candidates, batch_size):
            batch = noises[i:i + batch_size]
            bs = batch.shape[0]

            pe = prompt_embeds.expand(bs, -1, -1).float()
            pm = prompt_mask.expand(bs, -1)

            scores = predictor.predict(pe, batch.float(), pm)

            if scores.dim() == 2 and scores.shape[1] > 1:
                scores = scores[:, head_index]
            else:
                scores = scores.squeeze(-1)

            all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)
    topk = torch.topk(all_scores, min(num_select, num_candidates))
    if return_indices:
        return noises[topk.indices], topk.values, topk.indices
    return noises[topk.indices], topk.values
