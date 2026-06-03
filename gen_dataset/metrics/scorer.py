import torch
from PIL import Image
from typing import Dict


class _InputDtypeCastModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def _target_dtype(self):
        for param in self.module.parameters():
            if param.is_floating_point():
                return param.dtype
        return None

    def forward(self, x, *args, **kwargs):
        target_dtype = self._target_dtype()
        if target_dtype is not None and torch.is_tensor(x) and x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        return self.module(x, *args, **kwargs)


class MultiMetricScorer:
    METRICS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score']

    def __init__(self, metrics=None, device='cuda'):
        self.device = device
        self.metrics = self.METRICS if metrics is None else metrics
        self._models = {}

        if 'hpsv2' in self.metrics:
            import hpsv2 as _hpsv2
            self._models['hpsv2'] = _hpsv2

        if 'hpsv3' in self.metrics:
            from hpsv3 import HPSv3RewardInferencer
            self._models['hpsv3'] = HPSv3RewardInferencer(device=device)
            self._set_eval(self._models['hpsv3'])
            self._wrap_hpsv3_reward_head_dtype(self._models['hpsv3'])

        if 'image_reward' in self.metrics:
            import ImageReward as RM
            self._models['image_reward'] = RM.load("ImageReward-v1.0")
            self._prepare_image_reward_model(self._models['image_reward'], device)

        if 'pick_score' in self.metrics:
            from .pick_score import PickScorer
            self._models['pick_score'] = PickScorer(device=device)

    @staticmethod
    def _set_eval(model):
        if hasattr(model, "eval"):
            model.eval()
        nested = getattr(model, "model", None)
        if nested is not None and hasattr(nested, "eval"):
            nested.eval()

    @staticmethod
    def _move_to_device_if_supported(model, device):
        if hasattr(model, "to"):
            try:
                model.to(device)
            except TypeError:
                pass

    @classmethod
    def _prepare_image_reward_model(cls, model, device):
        cls._set_eval(model)
        cls._move_to_device_if_supported(model, device)
        device_obj = torch.device(device)
        for owner in (model, getattr(model, "blip", None)):
            if owner is None:
                continue
            if hasattr(owner, "device"):
                owner.device = device_obj
            if hasattr(owner, "_device"):
                owner._device = device_obj

    @staticmethod
    def _wrap_hpsv3_reward_head_dtype(inferencer):
        model = getattr(inferencer, "model", None)
        rm_head = getattr(model, "rm_head", None)
        if rm_head is not None and not isinstance(rm_head, _InputDtypeCastModule):
            model.rm_head = _InputDtypeCastModule(rm_head)

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str, image_path: str = None) -> Dict[str, float]:
        prompt = prompt.strip()
        image_rgb = image.convert("RGB")
        results = {}

        if 'hpsv2' in self.metrics:
            val = float(self._models['hpsv2'].score([image_rgb], prompt, hps_version="v2.1")[0])
            results['hpsv2'] = round(val, 4)

        if 'hpsv3' in self.metrics:
            rewards = self._models['hpsv3'].reward([image_path], [prompt])
            val = float(rewards[0][0].item())
            results['hpsv3'] = round(val, 4)

        if 'image_reward' in self.metrics:
            val = float(self._models['image_reward'].score(prompt, image_rgb))
            results['image_reward'] = round(val, 4)

        if 'pick_score' in self.metrics:
            val = self._models['pick_score'].score(image_rgb, prompt)
            results['pick_score'] = round(val, 4)

        return results
