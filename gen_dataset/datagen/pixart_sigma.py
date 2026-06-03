import torch

from .base_generator import BaseDataGenerator
from .model_registry import PIXART_SIGMA_CONFIG, load_pixart_sigma_pipeline


class PixartSigmaGenerator(BaseDataGenerator):
    def __init__(self, save_dir, prompts, **kwargs):
        super().__init__(
            save_dir=save_dir, prompts=prompts,
            latent_shape=PIXART_SIGMA_CONFIG['latent_shape'], **kwargs,
        )
        self.guidance_scale = PIXART_SIGMA_CONFIG['guidance_scale']
        self.num_inference_steps = PIXART_SIGMA_CONFIG['num_inference_steps']

    def setup_pipeline(self):
        return load_pixart_sigma_pipeline(device=self.device)

    def encode_and_save_prompt(self, pipe, prompt, save_path):
        with torch.inference_mode():
            # Returns 4-tuple: (prompt_embeds, prompt_mask, neg_embeds, neg_mask)
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
                prompt=prompt, device=self.device,
                max_sequence_length=PIXART_SIGMA_CONFIG['t5_seq_len'],
            )

        embeds_dict = {
            'prompt_embeds': prompt_embeds.cpu().half(),
            'prompt_attention_mask': prompt_mask.cpu(),
            'negative_prompt_embeds': neg_embeds.cpu().half(),
            'negative_prompt_attention_mask': neg_mask.cpu(),
        }
        torch.save(embeds_dict, save_path)

        return {k: v.to(self.device) for k, v in embeds_dict.items()}

    @torch.inference_mode()
    def generate_image(self, pipe, embeds_dict, noise, extractor=None):
        # Pipeline handles latent scaling (noise * init_noise_sigma) internally
        result = pipe(
            negative_prompt=None,
            prompt_embeds=embeds_dict['prompt_embeds'],
            prompt_attention_mask=embeds_dict['prompt_attention_mask'],
            negative_prompt_embeds=embeds_dict['negative_prompt_embeds'],
            negative_prompt_attention_mask=embeds_dict['negative_prompt_attention_mask'],
            latents=noise,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        return result.images[0]
