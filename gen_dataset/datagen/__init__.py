"""Dataset generator package.

Generator classes are imported lazily so lightweight config modules can be
imported on machines without local Torch/diffusers installs.
"""

from importlib import import_module


_EXPORTS = {
    'SDXLGenerator': 'sdxl_1024',
    'HunyuanDiTGenerator': 'hunyuan_dit',
    'DreamShaperGenerator': 'dreamshaper',
    'PixartSigmaGenerator': 'pixart_sigma',
    'SanaSprintGenerator': 'sana_sprint',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
