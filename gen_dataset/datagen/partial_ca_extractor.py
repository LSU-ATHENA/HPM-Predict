"""Partial cross-attention extractor for analysis paper baselines.

Captures the first attn2 (cross-attention) layer's map at the first denoising
step, re-normalizes it as a per-token distribution over image patches, then
stores per-head per-token summary stats (entropy, std, max) along T_N.
These stats are the Baseline-1 input features.
"""
import math

import torch
import torch.nn.functional as F


class FirstCAExtractor:
    """Hooks ONLY the first cross-attention module, captures at step 0.

    Stored per sample (shape per tensor is (H, T_p), fp16):
      - entropy: H(A[h, i, :]) over the T_N (image-patch) axis
      - std:    std of A[h, i, :]
      - max:    max of A[h, i, :]
    where A is attn_probs permuted to (H, T_p, T_N) and re-normalized so each
    row sums to 1 along T_N.
    """

    def __init__(self, denoiser):
        self.hook = None
        self.layer_name = None
        self.result = None
        self._register(denoiser)

    def _register(self, denoiser):
        for name, module in denoiser.named_modules():
            if not (hasattr(module, 'to_q') and hasattr(module, 'to_k')):
                continue
            if 'attn2' not in name:
                continue
            self.layer_name = name
            self.hook = module.register_forward_hook(self._hook_fn, with_kwargs=True)
            return
        raise RuntimeError("No attn2 module found in denoiser")

    def _hook_fn(self, module, args, kwargs, output):
        if self.result is not None:
            return  # only capture first call

        hidden_states = args[0] if args else kwargs.get('hidden_states')
        encoder_hidden_states = (
            args[1] if len(args) > 1 else kwargs.get('encoder_hidden_states')
        )
        if hidden_states is None or encoder_hidden_states is None:
            return

        with torch.inference_mode():
            q = module.to_q(hidden_states).float()
            k = module.to_k(encoder_hidden_states).float()

            heads = module.heads
            head_dim = q.shape[-1] // heads
            batch = q.shape[0]

            q = q.view(batch, -1, heads, head_dim).transpose(1, 2)  # (B, H, T_N, d)
            k = k.view(batch, -1, heads, head_dim).transpose(1, 2)  # (B, H, T_p, d)

            scale = 1.0 / math.sqrt(head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_probs = F.softmax(attn_scores, dim=-1)  # (B, H, T_N, T_p), softmax over T_p

            # CFG: diffusers concatenates as [uncond, cond]; keep conditional only.
            if batch == 2:
                attn_probs = attn_probs[1:]

            # "For each prompt token, distribution over image patches"
            a = attn_probs.permute(0, 1, 3, 2)  # (1, H, T_p, T_N)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)

            eps = torch.finfo(a.dtype).eps
            a_log = a.clamp_min(eps).log()
            entropy = -(a * a_log).sum(dim=-1)        # (1, H, T_p)
            std = a.std(dim=-1)                       # (1, H, T_p)
            max_vals = a.max(dim=-1).values           # (1, H, T_p)

            self.result = {
                'entropy': entropy.squeeze(0).cpu().half(),  # (H, T_p) fp16
                'std':     std.squeeze(0).cpu().half(),      # (H, T_p) fp16
                'max':     max_vals.squeeze(0).cpu().half(), # (H, T_p) fp16
                'layer_name': self.layer_name,
                'T_N': int(a.shape[-1]),
            }

    def reset(self):
        self.result = None

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
