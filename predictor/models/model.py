import torch
import torch.nn as nn
import yaml
from typing import Optional, Dict, Any

from .noise_encoders import get_noise_encoder, ResidualConv


class ScorePredictor(nn.Module):
    def __init__(
        self,
        noise_encoder: nn.Module,
        text_encoder: nn.Module,
        dropout: float = 0.1,
        num_heads: int = 1,
    ):
        super().__init__()

        self.noise_encoder = noise_encoder
        self.text_encoder = text_encoder
        self.num_heads = num_heads
        self.dropout = dropout

        noise_dim = noise_encoder.output_dim
        text_dim = text_encoder.output_dim

        fusion_dim = noise_dim + text_dim

        self.fusion_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.SiLU(),
        )

        if num_heads == 1:
            self.head = nn.Linear(64, 1)
        else:
            self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        mask_noise: bool = False,
    ) -> torch.Tensor:
        text_feat = self.text_encoder(prompt_embeds, prompt_mask)

        noise_out = self.noise_encoder(noise)
        if mask_noise:
            noise_out = torch.zeros_like(noise_out)

        combined = torch.cat([noise_out, text_feat], dim=1)

        backbone_out = self.fusion_backbone(combined)

        if self.num_heads == 1:
            return self.head(backbone_out)
        else:
            outputs = [head(backbone_out) for head in self.heads]
            return torch.cat(outputs, dim=1)

    @torch.inference_mode()
    def predict(self, prompt_embeds, noise, prompt_mask=None, mask_noise=False):
        was_training = self.training
        self.eval()

        if prompt_mask is None:
            prompt_mask = torch.ones(
                prompt_embeds.shape[:2],
                device=prompt_embeds.device
            )

        scores = self.forward(noise, prompt_embeds, prompt_mask, mask_noise=mask_noise)

        if was_training:
            self.train()

        return scores

    def save(
        self,
        path: str,
        normalization: Optional[Dict[str, Any]] = None,
        model_type: str = None,
        spatial_size: int = None,
        in_channels: int = 4,
        embed_dim: int = None,
        seq_len: int = None,
    ) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'text_enc': self.text_encoder.__class__.__name__.replace('TextEncoder', '').lower(),
                'noise_enc': self._get_noise_enc_name(),
                'dropout': self.dropout,
                'num_heads': self.num_heads,
                'model_type': model_type,
                'spatial_size': spatial_size,
                'in_channels': in_channels,
                'embed_dim': embed_dim,
                'seq_len': seq_len,
                'pos_encoding': getattr(self.text_encoder, 'pos_encoding_type', 'none'),
                'text_encoder_project_dims': getattr(self.text_encoder, 'project_dims', None),
            },
            'normalization': normalization or {},
        }
        torch.save(checkpoint, path)

    def _get_noise_enc_name(self) -> str:
        return 'residualconv'

    @classmethod
    def from_config(cls, config_path: str, device: str = None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        pred_config = config['predictor']
        arch = pred_config['architecture']
        weights = pred_config['weights']

        device = device or weights.get('device', 'cuda')

        checkpoint_path = weights['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_config = checkpoint.get('model_config', {})

        from .text_encoders import get_text_encoder
        from .noise_encoders import get_noise_encoder

        text_enc_name = model_config.get('text_enc', arch['text_encoder'])
        noise_enc_name = model_config.get('noise_enc', arch['noise_encoder'])
        dropout = model_config.get('dropout', arch.get('dropout', 0.1))
        num_heads = model_config.get('num_heads', arch.get('num_heads', 1))

        # Get dimension info from checkpoint (with backward-compatible defaults)
        spatial_size = model_config.get('spatial_size', 64)
        in_channels = model_config.get('in_channels', 4)
        embed_dim = model_config.get('embed_dim', 4096)
        seq_len = model_config.get('seq_len', 120)

        pos_encoding = model_config.get('pos_encoding', 'none')
        text_kwargs = dict(embed_dim=embed_dim, seq_len=seq_len, pos_encoding=pos_encoding)
        if text_enc_name == 'pertokenscalar':
            text_kwargs['project_dims'] = model_config.get('text_encoder_project_dims')
        text_encoder = get_text_encoder(text_enc_name, **text_kwargs)
        noise_encoder = get_noise_encoder(spatial_size=spatial_size, in_channels=in_channels)

        model = cls(
            noise_encoder=noise_encoder,
            text_encoder=text_encoder,
            dropout=dropout,
            num_heads=num_heads,
        )

        # Handle float16 checkpoints: cast to float32 to match model dtype
        state_dict = {k: v.float() for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        normalization_info = checkpoint.get('normalization', {})
        return model, normalization_info
