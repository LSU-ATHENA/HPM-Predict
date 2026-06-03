from typing import Optional

import torch
import torch.nn as nn


class CAStatsEncoder(nn.Module):
    """Packs per-head per-token CA-map stats into a feature representation.

    Inputs are tensors of shape (B, H, T_p), precomputed offline by
    gen_dataset/datagen/partial_ca_extractor.py.

    With `flatten=True` (default, legacy small-MLP head): returns
        (B, H_out * T_p * n_stats)
    With `flatten=False` (stats_transformer head): returns
        (B, H_out, T_p, n_stats)

    Either way, head_compression sums adjacent head pairs (when c > 1).
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        head_compression_ratio: int = 1,
        use_std: bool = False,
        use_max: bool = False,
    ):
        super().__init__()
        assert head_compression_ratio >= 1
        assert num_heads % head_compression_ratio == 0, \
            f"num_heads={num_heads} must be divisible by c={head_compression_ratio}"

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.c = head_compression_ratio
        self.use_std = use_std
        self.use_max = use_max

        self.H_out = num_heads // head_compression_ratio
        self.n_stats = 1 + int(use_std) + int(use_max)
        self.output_dim = self.H_out * seq_len * self.n_stats

    def forward(
        self,
        entropy: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        max_: Optional[torch.Tensor] = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        parts = [entropy]
        if self.use_std:
            assert std is not None, "use_std=True but std tensor not provided"
            parts.append(std)
        if self.use_max:
            assert max_ is not None, "use_max=True but max tensor not provided"
            parts.append(max_)

        x = torch.stack(parts, dim=-1)              # (B, H, T_p, S)
        B, H, Tp, S = x.shape

        if self.c > 1:
            x = x.reshape(B, self.H_out, self.c, Tp, S).sum(dim=2)

        if flatten:
            return x.reshape(B, -1)                 # (B, H_out * T_p * S)
        return x                                    # (B, H_out, T_p, S)


class CAMapPredictor(nn.Module):
    """Baseline 1: score <- small MLP head over CA-map stats from a frozen DM.

    No noise encoder, no text encoder, no fusion -- the diffusion model's
    first cross-attention layer has already fused (noise, prompt) into the
    attention map, and we read structural properties of that map.
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        head_compression_ratio: int = 1,
        use_std: bool = False,
        use_max: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_compression_ratio = head_compression_ratio
        self.use_std = use_std
        self.use_max = use_max
        self.dropout = dropout

        self.encoder = CAStatsEncoder(
            num_heads=num_heads,
            seq_len=seq_len,
            head_compression_ratio=head_compression_ratio,
            use_std=use_std,
            use_max=use_max,
        )

        feat_dim = self.encoder.output_dim
        self.fusion_backbone = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.SiLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(
        self,
        entropy: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        max_: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.encoder(entropy, std, max_, flatten=True)
        return self.head(self.fusion_backbone(x))

    @torch.inference_mode()
    def predict(
        self,
        entropy: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        max_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(entropy, std, max_)
        if was_training:
            self.train()
        return out

    def _config_dict(self, model_type: Optional[str] = None) -> dict:
        return {
            'predictor': 'ca_map',
            'num_heads': self.num_heads,
            'seq_len': self.seq_len,
            'head_compression_ratio': self.head_compression_ratio,
            'use_std': self.use_std,
            'use_max': self.use_max,
            'dropout': self.dropout,
            'model_type': model_type,
        }

    def save(self, path: str, normalization: Optional[dict] = None,
             model_type: Optional[str] = None) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self._config_dict(model_type),
            'normalization': normalization or {},
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['model_config']
        model = cls(
            num_heads=cfg['num_heads'],
            seq_len=cfg['seq_len'],
            head_compression_ratio=cfg.get('head_compression_ratio', 1),
            use_std=cfg.get('use_std', False),
            use_max=cfg.get('use_max', False),
            dropout=cfg.get('dropout', 0.1),
        )
        state_dict = {k: v.float() for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, ckpt.get('normalization', {})
