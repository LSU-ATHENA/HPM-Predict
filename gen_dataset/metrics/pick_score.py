import torch
from PIL import Image
from typing import List


class PickScorer:
    MODEL_ID = "yuvalkirstain/PickScore_v1"
    PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(self, device='cuda'):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).eval().to(device)

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> float:
        inputs = self.processor(
            images=image, text=prompt,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        ).to(self.device)
        return self.model(**inputs).logits_per_image.item()
