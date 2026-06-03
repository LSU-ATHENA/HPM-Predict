import torch

from .base_generator import BaseDataGenerator
from .model_registry import SANA_SPRINT_CONFIG, load_sana_sprint_pipeline


class SanaSprintGenerator(BaseDataGenerator):
    def __init__(self, save_dir, prompts, **kwargs):
        super().__init__(
            save_dir=save_dir, prompts=prompts,
            latent_shape=SANA_SPRINT_CONFIG['latent_shape'], **kwargs,
        )
        self.guidance_scale = SANA_SPRINT_CONFIG['guidance_scale']
        self.num_inference_steps = SANA_SPRINT_CONFIG['num_inference_steps']

    def setup_pipeline(self):
        return load_sana_sprint_pipeline(device=self.device)

    def generate_noise(self, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn((1, *self.latent_shape), generator=gen,
                           device=self.device, dtype=torch.bfloat16)

    def encode_and_save_prompt(self, pipe, prompt, save_path):
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                max_sequence_length=SANA_SPRINT_CONFIG['text_seq_len'],
            )

        embeds_dict = {
            'prompt_embeds': prompt_embeds.cpu().to(torch.bfloat16),
            'prompt_attention_mask': prompt_attention_mask.cpu(),
        }
        torch.save(embeds_dict, save_path)

        return {k: v.to(self.device) for k, v in embeds_dict.items()}

    @torch.inference_mode()
    def generate_image(self, pipe, embeds_dict, noise, extractor=None):
        # Match HyperNoise repo's sana_inference.py base model settings exactly
        result = pipe(
            latents=noise,
            prompt_embeds=embeds_dict['prompt_embeds'],
            prompt_attention_mask=embeds_dict['prompt_attention_mask'],
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            intermediate_timesteps=None,
        )
        return result.images[0]
