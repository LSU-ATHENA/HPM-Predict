import torch

from .base_generator import BaseDataGenerator
from .model_registry import HUNYUAN_CONFIG, load_hunyuan_pipeline


class HunyuanDiTGenerator(BaseDataGenerator):
    def __init__(self, save_dir, prompts, **kwargs):
        super().__init__(
            save_dir=save_dir, prompts=prompts,
            latent_shape=HUNYUAN_CONFIG['latent_shape'], **kwargs,
        )
        self.guidance_scale = HUNYUAN_CONFIG['guidance_scale']
        self.num_inference_steps = HUNYUAN_CONFIG['num_inference_steps']

    def setup_pipeline(self):
        return load_hunyuan_pipeline(device=self.device)

    def encode_and_save_prompt(self, pipe, prompt, save_path):
        with torch.inference_mode():
            # CLIP encoder (index 0): [1, 77, 1024] + mask [1, 77]
            clip_embeds, clip_neg, clip_mask, clip_neg_mask = pipe.encode_prompt(
                prompt=prompt, device=self.device,
                dtype=torch.float16, text_encoder_index=0,
            )
            # T5 encoder (index 1): [1, 256, 2048] + mask [1, 256]
            t5_embeds, t5_neg, t5_mask, t5_neg_mask = pipe.encode_prompt(
                prompt=prompt, device=self.device,
                dtype=torch.float16, text_encoder_index=1,
            )

        embeds_dict = {
            'prompt_embeds': clip_embeds.cpu().half(),
            'prompt_attention_mask': clip_mask.cpu(),
            'prompt_embeds_2': t5_embeds.cpu().half(),
            'prompt_attention_mask_2': t5_mask.cpu(),
            'negative_prompt_embeds': clip_neg.cpu().half(),
            'negative_prompt_attention_mask': clip_neg_mask.cpu(),
            'negative_prompt_embeds_2': t5_neg.cpu().half(),
            'negative_prompt_attention_mask_2': t5_neg_mask.cpu(),
        }
        torch.save(embeds_dict, save_path)

        return {k: v.to(self.device) for k, v in embeds_dict.items()}

    @torch.inference_mode()
    def generate_image(self, pipe, embeds_dict, noise, extractor=None):
        kwargs = dict(
            prompt_embeds=embeds_dict['prompt_embeds'],
            prompt_embeds_2=embeds_dict['prompt_embeds_2'],
            negative_prompt_embeds=embeds_dict['negative_prompt_embeds'],
            negative_prompt_embeds_2=embeds_dict['negative_prompt_embeds_2'],
            prompt_attention_mask=embeds_dict['prompt_attention_mask'],
            prompt_attention_mask_2=embeds_dict['prompt_attention_mask_2'],
            negative_prompt_attention_mask=embeds_dict['negative_prompt_attention_mask'],
            negative_prompt_attention_mask_2=embeds_dict['negative_prompt_attention_mask_2'],
            latents=noise,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        if extractor is not None:
            kwargs['callback_on_step_end'] = extractor.step_callback
        result = pipe(**kwargs)
        return result.images[0]
