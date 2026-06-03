from typing import Optional

from .noise_encoders import get_noise_encoder
from .text_encoders import get_text_encoder
from .model import ScorePredictor
from .ca_map_predictor import CAMapPredictor, CAStatsEncoder
from .patchca_predictor import PatchPromptCAPredictor, PatchCABlock
from predictor.configs.model_dims import get_dims


NOISE_ENCODERS = ['custom', 'spatial_shrink']
TEXT_ENCODERS = ['summarytoken', 'lightsummary', 'pertokenscalar']

_NOISE_ALIASES = {'residualconv': 'custom'}
_TEXT_ALIASES = {'attnpool': 'summarytoken'}


def get_model(
    noise_enc: str = 'custom',
    text_enc: str = 'summarytoken',
    dropout: float = 0.1,
    num_heads: int = 1,
    spatial_size: int = 128,
    in_channels: int = 4,
    embed_dim: int = 2048,
    seq_len: int = 77,
    pos_encoding: str = 'none',
    text_enc_legacy: bool = False,
    text_encoder_project_dims: Optional[list[int]] = None,
) -> ScorePredictor:
    noise_enc = _NOISE_ALIASES.get(noise_enc, noise_enc)
    text_enc = _TEXT_ALIASES.get(text_enc, text_enc)

    if noise_enc not in NOISE_ENCODERS:
        raise ValueError(f"Unknown noise encoder: {noise_enc}. Available: {NOISE_ENCODERS}")
    if text_enc not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder: {text_enc}. Available: {TEXT_ENCODERS}")

    text_kwargs = dict(embed_dim=embed_dim, seq_len=seq_len, pos_encoding=pos_encoding)
    if text_enc == 'pertokenscalar':
        text_kwargs['legacy'] = text_enc_legacy
        text_kwargs['project_dims'] = text_encoder_project_dims
    text_encoder = get_text_encoder(text_enc, **text_kwargs)
    noise_encoder = get_noise_encoder(
        name=noise_enc, spatial_size=spatial_size, in_channels=in_channels,
    )

    return ScorePredictor(
        noise_encoder=noise_encoder,
        text_encoder=text_encoder,
        dropout=dropout,
        num_heads=num_heads,
    )


def get_ca_map_model(
    num_heads: int,
    seq_len: int,
    head_compression_ratio: int = 1,
    use_std: bool = False,
    use_max: bool = False,
    dropout: float = 0.1,
) -> CAMapPredictor:
    return CAMapPredictor(
        num_heads=num_heads,
        seq_len=seq_len,
        head_compression_ratio=head_compression_ratio,
        use_std=use_std,
        use_max=use_max,
        dropout=dropout,
    )


def get_patchca_model(
    model_type: str,
    patch_size: int,
    ca_embed_dim: int = 256,
    ca_num_heads: int = 8,
    head_compression_ratio: int = 1,
    use_std: bool = False,
    use_max: bool = False,
    dropout: float = 0.1,
    ffn_ratio: int = 4,
    seq_len_override: Optional[int] = None,
) -> PatchPromptCAPredictor:
    dims = get_dims(model_type)
    seq_len = seq_len_override if seq_len_override is not None else dims['seq_len']
    return PatchPromptCAPredictor(
        in_channels=dims['latent_shape'][0],
        patch_size=patch_size,
        embed_dim=ca_embed_dim,
        num_heads=ca_num_heads,
        prompt_dim=dims['embed_dim'],
        seq_len=seq_len,
        head_compression_ratio=head_compression_ratio,
        use_std=use_std,
        use_max=use_max,
        dropout=dropout,
        ffn_ratio=ffn_ratio,
    )


__all__ = [
    'get_model',
    'get_ca_map_model',
    'get_patchca_model',
    'ScorePredictor',
    'CAMapPredictor',
    'CAStatsEncoder',
    'PatchPromptCAPredictor',
    'PatchCABlock',
    'get_text_encoder',
    'get_noise_encoder',
    'NOISE_ENCODERS',
    'TEXT_ENCODERS',
]
