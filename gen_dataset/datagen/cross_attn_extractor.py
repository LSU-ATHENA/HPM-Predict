import math
from collections import defaultdict

import torch
import torch.nn.functional as F


class CrossAttentionEntropyExtractor:
    """Hooks into UNet/DiT attn2 (cross-attention) modules to extract attention entropy.

    Captures Q and K at specified denoising steps, recomputes attention probabilities
    (since F.scaled_dot_product_attention doesn't return them), and computes per-token entropy.
    """

    def __init__(self, denoiser, extract_steps=(1, 5, 10)):
        # callback_on_step_end fires AFTER the step, so when hooks run at step N,
        # current_step is still N-1 from the previous callback.
        self.extract_steps = set(extract_steps)
        self._hook_steps = {s - 1 for s in extract_steps}
        self.current_step = -2
        self.hooks = []
        self.layer_names = []
        self.entropy_data = defaultdict(list)
        self._register_hooks(denoiser)

    def _register_hooks(self, denoiser):
        for name, module in denoiser.named_modules():
            if not (hasattr(module, 'to_q') and hasattr(module, 'to_k')):
                continue
            if 'attn2' not in name:
                continue
            self.layer_names.append(name)
            hook = module.register_forward_hook(
                self._make_hook(name), with_kwargs=True,
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_name):
        def hook_fn(module, args, kwargs, output):
            if self.current_step not in self._hook_steps:
                return

            hidden_states = args[0] if args else kwargs.get('hidden_states')
            encoder_hidden_states = (
                args[1] if len(args) > 1
                else kwargs.get('encoder_hidden_states')
            )
            if hidden_states is None or encoder_hidden_states is None:
                return

            with torch.inference_mode():
                # Compute everything in float32 to avoid NaN from fp16 underflow
                q = module.to_q(hidden_states).float()
                k = module.to_k(encoder_hidden_states).float()

                heads = module.heads
                head_dim = q.shape[-1] // heads
                batch = q.shape[0]

                q = q.view(batch, -1, heads, head_dim).transpose(1, 2)
                k = k.view(batch, -1, heads, head_dim).transpose(1, 2)

                scale = 1.0 / math.sqrt(head_dim)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_probs = F.softmax(attn_scores, dim=-1)

                # Per-image-patch entropy (how spread across text tokens)
                # attn_probs: (batch, heads, img_patches, text_tokens)
                entropy = -(attn_probs * torch.log2(attn_probs + 1e-9)).sum(dim=-1)
                patch_entropy_mean = entropy.mean().item()

                # Per-token attention weight: mean over img_patches, batch, heads
                per_token = attn_probs.mean(dim=2).mean(dim=0).mean(dim=0)  # (text_tokens,)

                # Per-token entropy of attention distribution across image patches
                attn_to_token = attn_probs.permute(0, 1, 3, 2)
                attn_to_token_norm = attn_to_token / (attn_to_token.sum(dim=-1, keepdim=True) + 1e-9)
                token_entropy = -(attn_to_token_norm * torch.log2(attn_to_token_norm + 1e-9)).sum(dim=-1)
                token_entropy = token_entropy.mean(dim=0).mean(dim=0)  # (text_tokens,)

                # Store under actual target step (current_step + 1), cast to fp16 for storage
                self.entropy_data[self.current_step + 1].append({
                    'layer': layer_name,
                    'patch_entropy_mean': patch_entropy_mean,
                    'per_token_attn_weight': per_token.cpu().half(),
                    'per_token_entropy': token_entropy.cpu().half(),
                })

        return hook_fn

    def step_callback(self, pipe, step_index, timestep, callback_kwargs):
        self.current_step = step_index
        return callback_kwargs

    def get_results(self):
        """Return collected entropy data structured for saving, then reset."""
        results = {}

        for step, layers in self.entropy_data.items():
            key = f't{step:02d}'
            n_layers = len(layers)

            per_layer_entropy = torch.stack([l['per_token_entropy'] for l in layers])
            per_layer_attn = torch.stack([l['per_token_attn_weight'] for l in layers])
            global_entropy = sum(l['patch_entropy_mean'] for l in layers) / n_layers

            results[key] = {
                'per_layer_entropy': per_layer_entropy,
                'per_layer_attn_weight': per_layer_attn,
                'mean_entropy': per_layer_entropy.mean(dim=0),
                'global_entropy': global_entropy,
            }

        self.entropy_data.clear()
        return results

    def get_metadata_scalars(self, results):
        """Extract scalar entropy values for metadata.jsonl."""
        meta = {}
        for key, data in results.items():
            meta[f'cross_attn_entropy_{key}'] = round(data['global_entropy'], 4)
        if results:
            meta['cross_attn_entropy_mean'] = round(
                sum(d['global_entropy'] for d in results.values()) / len(results), 4
            )
        return meta

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
