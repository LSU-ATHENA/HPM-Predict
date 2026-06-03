import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PerTokenScalarTextEncoder(nn.Module):
    """Per-token scalar projection: maps each (B, T, D) embedding to a scalar.

    Architecture is conditional:
      - D > 2048: D -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 1
      - D <= 2048: D -> 1024 -> 512 -> 256 -> 128 -> 1
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        seq_len: int = 120,
        legacy: bool = False,
        project_dims: list[int] | None = None,
        pos_encoding: str = 'none',
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding
        self.project_dims = list(project_dims or default_pertokenscalar_project_dims(embed_dim, legacy))

        layers = []
        in_dim = embed_dim
        for out_dim in self.project_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.project = nn.Sequential(*layers)
        self._output_dim = seq_len

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # [B, seq_len, embed_dim] → [B, seq_len, 1] → [B, seq_len]
        return self.project(prompt_embeds).squeeze(-1)


def get_text_encoder(name: str, embed_dim: int = 4096, seq_len: int = 120, **kwargs) -> nn.Module:
    encoders = {
        'pertokenscalar': PerTokenScalarTextEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown text encoder: {name}. Available: {list(encoders.keys())}")
    return encoders[name](embed_dim=embed_dim, seq_len=seq_len, **kwargs)
