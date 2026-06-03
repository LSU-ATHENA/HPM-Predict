import torch

from .base_generator import BaseDataGenerator
from .model_registry import DREAMSHAPER_CONFIG, load_dreamshaper_pipeline


class DreamShaperGenerator(BaseDataGenerator):
    def __init__(self, save_dir, prompts, **kwargs):
        super().__init__(
            save_dir=save_dir, prompts=prompts,
            latent_shape=DREAMSHAPER_CONFIG['latent_shape'], **kwargs,
        )
        self.guidance_scale = DREAMSHAPER_CONFIG['guidance_scale']
        self.num_inference_steps = DREAMSHAPER_CONFIG['num_inference_steps']

    def setup_pipeline(self):
        return load_dreamshaper_pipeline(device=self.device)

    def encode_and_save_prompt(self, pipe, prompt, save_path):
        with torch.inference_mode():
            prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, device=self.device,
            )
        torch.save({
            'prompt_embeds': prompt_embeds.cpu().half(),
            'pooled_prompt_embeds': pooled.cpu().half(),
            'negative_prompt_embeds': neg_embeds.cpu().half(),
            'negative_pooled_prompt_embeds': neg_pooled.cpu().half(),
        }, save_path)

        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled,
            'negative_prompt_embeds': neg_embeds,
            'negative_pooled_prompt_embeds': neg_pooled,
        }

    @torch.inference_mode()
    def generate_image(self, pipe, embeds_dict, noise, extractor=None):
        result = pipe(
            prompt_embeds=embeds_dict['prompt_embeds'],
            negative_prompt_embeds=embeds_dict['negative_prompt_embeds'],
            pooled_prompt_embeds=embeds_dict['pooled_prompt_embeds'],
            negative_pooled_prompt_embeds=embeds_dict['negative_pooled_prompt_embeds'],
            latents=noise,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        return result.images[0]
