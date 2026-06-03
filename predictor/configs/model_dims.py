MODEL_DIMS = {
    'sdxl': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 77,
    },
    'dreamshaper': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 77,
    },
    'hunyuan_dit': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 256,
    },
    'pixart_sigma': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 4096,
        'seq_len': 300,
    },
    'sana_sprint': {
        'latent_shape': (32, 32, 32),
        'spatial_size': 32,
        'embed_dim': 2304,
        'seq_len': 300,
    },
}


def get_dims(model_type: str) -> dict:
    if model_type not in MODEL_DIMS:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(MODEL_DIMS.keys())}"
        )
    return MODEL_DIMS[model_type].copy()
