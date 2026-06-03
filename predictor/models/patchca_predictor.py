from typing import Optional

import torch
import torch.nn as nn

from .ca_map_predictor import CAStatsEncoder


class PatchCABlock(nn.Module):
    """One residual cross-attention block: prompt attends to noise patches.

    Q = prompt, K = V = noise patches. With residual + FFN around the
    attention, multiple blocks can be stacked. Returning the attention map
    from the final block gives the (B, H, T_p, T_N) tensor used by Patch-CA.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.SiLU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        prompt: torch.Tensor,
        patches: torch.Tensor,
        return_attn: bool = False,
    ):
        q = self.norm_q(prompt)
        kv = self.norm_kv(patches)
        out, attn = self.attn(
            q, kv, kv,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        prompt = prompt + self.drop1(out)
        prompt = prompt + self.drop2(self.ffn(self.norm_ffn(prompt)))
        return prompt, attn

    def forward_no_attn(self, prompt: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Single-tensor wrapper for torch.utils.checkpoint (which needs a
        plain Tensor return, not a tuple)."""
        out, _ = self.forward(prompt, patches, return_attn=False)
        return out


class PatchPromptCAPredictor(nn.Module):
    """Baseline 2: 1 learnable cross-attention block + small MLP head.

    Patchify the noise, project the prompt, run a single nn.MultiheadAttention
    (Q=prompt, K=V=noise patches), compute entropy/std/max along the noise
    axis, flatten, and feed a small Linear stack to a scalar score.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        prompt_dim: int,
        seq_len: int,
        head_compression_ratio: int = 1,
        use_std: bool = False,
        use_max: bool = False,
        dropout: float = 0.1,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.prompt_dim = prompt_dim
        self.seq_len = seq_len
        self.head_compression_ratio = head_compression_ratio
        self.use_std = use_std
        self.use_max = use_max
        self.dropout = dropout
        self.ffn_ratio = ffn_ratio

        self.patchify = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                      stride=patch_size, bias=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1,
                      padding=1, groups=embed_dim, bias=True),
        )

        if prompt_dim == embed_dim:
            self.prompt_proj = nn.Identity()
        else:
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True,
        )

        self.stats_encoder = CAStatsEncoder(
            num_heads=num_heads,
            seq_len=seq_len,
            head_compression_ratio=head_compression_ratio,
            use_std=use_std,
            use_max=use_max,
        )

        feat_dim = self.stats_encoder.output_dim
        self.fusion_backbone = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.SiLU(),
        )
        self.head = nn.Linear(64, 1)

    def _compute_stats(self, attn_map: torch.Tensor):
        # attn_map: (B, H, T_p, T_N) -- already softmaxed across T_N by MHA.
        eps = torch.finfo(attn_map.dtype).eps
        probs = attn_map.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1)
        std = attn_map.std(dim=-1) if self.use_std else None
        max_ = attn_map.max(dim=-1)[0] if self.use_max else None
        return entropy, std, max_

    def forward(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        patch_feats = self.patchify(noise)
        patches = patch_feats.flatten(2).transpose(1, 2)            # (B, T_N, D)
        prompt_q = self.prompt_proj(prompt_embeds)                  # (B, T_p, D)

        _, attn_map = self.cross_attn(
            query=prompt_q, key=patches, value=patches,
            need_weights=True, average_attn_weights=False,
        )                                                           # (B, H, T_p, T_N)

        entropy, std, max_ = self._compute_stats(attn_map)

        if prompt_mask is not None:
            mask = prompt_mask.to(dtype=entropy.dtype).unsqueeze(1)  # (B, 1, T_p)
            entropy = entropy * mask
            if std is not None:
                std = std * mask
            if max_ is not None:
                max_ = max_ * mask

        feat = self.stats_encoder(entropy, std=std, max_=max_, flatten=True)
        return self.head(self.fusion_backbone(feat))

    @torch.inference_mode()
    def predict(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(noise, prompt_embeds, prompt_mask)
        if was_training:
            self.train()
        return out

    def _config_dict(self, model_type: Optional[str] = None) -> dict:
        return {
            'predictor': 'patch_ca',
            'in_channels': self.in_channels,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'prompt_dim': self.prompt_dim,
            'seq_len': self.seq_len,
            'head_compression_ratio': self.head_compression_ratio,
            'use_std': self.use_std,
            'use_max': self.use_max,
            'dropout': self.dropout,
            'ffn_ratio': self.ffn_ratio,
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
            in_channels=cfg['in_channels'],
            patch_size=cfg['patch_size'],
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            prompt_dim=cfg['prompt_dim'],
            seq_len=cfg['seq_len'],
            head_compression_ratio=cfg.get('head_compression_ratio', 1),
            use_std=cfg.get('use_std', False),
            use_max=cfg.get('use_max', False),
            dropout=cfg.get('dropout', 0.1),
            ffn_ratio=cfg.get('ffn_ratio', 4),
        )
        state_dict = {k: v.float() for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, ckpt.get('normalization', {})
