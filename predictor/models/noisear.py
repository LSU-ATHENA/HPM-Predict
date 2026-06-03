"""
NoiseAR: Autoregressive noise generation model.

Extracted from the NoiseAR codebase (try.py / eval.py) for use as an
evaluation baseline. Only the model class is included — no pipeline or
dataset dependencies.

Architecture: patch-based AR transformer that generates latent noise
conditioned on text embeddings. Uses causal self-attention + cross-attention
to text, predicting each noise patch autoregressively.

Supported pipelines: SDXL, DreamShaper (shared weights), DiT (separate weights + dim_adapter).
"""

import math
import torch
import torch.nn as nn


class NoiseARNet(nn.Module):
    def __init__(
        self,
        patch_size: int = 32,
        d_model: int = 2048,
        d_ff: int = 4096,
        n_heads: int = 1,
        n_layers: int = 1,
        dropout: float = 0.15,
        pretrained_path: str = "",
        pipeline: str = "SDXL",
        n_prediction_head: int = 0,
    ):
        super(NoiseARNet, self).__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.pipeline = pipeline

        if self.pipeline == "DiT":
            self.dim_adapter = nn.Linear(1024, 2048)

        self.latent_patch_encoder = nn.Sequential(
            nn.Linear(4 * patch_size * patch_size, 2 * patch_size * patch_size),
            nn.SiLU(),
            nn.Linear(2 * patch_size * patch_size, d_model),
        )
        self.register_buffer("learnable_start_token_emb", torch.randn(d_model) * 0.1)
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.self_attn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.cross_attn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.prediction_head_mu = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 4 * patch_size * patch_size)
        )
        self.prediction_head_logvar = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 4 * patch_size * patch_size)
        )

        if n_prediction_head != 0:
            ffn_layers_mu = []
            for _ in range(n_prediction_head):
                ffn_layers_mu.append(nn.Linear(d_model, d_model * 2))
                ffn_layers_mu.append(nn.GELU())
                ffn_layers_mu.append(nn.Linear(d_model * 2, d_model))
            self.prediction_head_mu = nn.Sequential(
                *ffn_layers_mu,
                nn.Linear(d_model, 4 * patch_size * patch_size),
            )

            ffn_layers_logvar = []
            for _ in range(n_prediction_head):
                ffn_layers_logvar.append(nn.Linear(d_model, d_model * 2))
                ffn_layers_logvar.append(nn.GELU())
                ffn_layers_logvar.append(nn.Linear(d_model * 2, d_model))
            self.prediction_head_logvar = nn.Sequential(
                *ffn_layers_logvar,
                nn.Linear(d_model, 4 * patch_size * patch_size),
            )

        nr_patches = 128 * 128 // self.patch_size // self.patch_size
        self.seq_len = nr_patches + 1
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(1, self.seq_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe)
        self._init_weights()

        if pretrained_path and ".pth" in pretrained_path:
            state = torch.load(pretrained_path, weights_only=True)
            self.load_state_dict(state, strict=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.zeros_(m.out_proj.bias)
        with torch.no_grad():
            if hasattr(self, "prediction_head_mu"):
                final_layer = self.prediction_head_mu[-1]
                nn.init.uniform_(final_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_layer.bias)
            if hasattr(self, "prediction_head_logvar"):
                final_layer = self.prediction_head_logvar[-1]
                nn.init.uniform_(final_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_layer.bias)

    @staticmethod
    def patchfy_to_origin(x, patch_size):
        shp = x.shape
        x = x.reshape([shp[0], 128 // patch_size, 128 // patch_size, -1, patch_size, patch_size])
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(shp[0], -1, 128, 128)
        return x

    @staticmethod
    def origin_to_patchfy(x, patch_size):
        b, c, h, w = x.shape
        x = x.reshape([b, c, h // patch_size, patch_size, w // patch_size, patch_size])
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * patch_size * patch_size)
        return x

    @staticmethod
    def sample_noise(mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_network(self, x, causal_mask, y):
        for layer_id in range(self.n_layers):
            x = self.self_attn_norm[layer_id](x)
            x = self.self_attn_layers[layer_id](x, x, x, attn_mask=causal_mask)[0] + x
            x = self.cross_attn_norm[layer_id](x)
            x = self.cross_attn_layers[layer_id](x, y, y)[0] + x
            x = self.ffn_norm[layer_id](x)
            x = self.ffn_layers[layer_id](x) + x
        mu = self.prediction_head_mu(x)
        logvar = self.prediction_head_logvar(x)
        return mu, logvar

    def forward(self, text_emb, x=None):
        if self.pipeline == "DiT":
            text_emb = self.dim_adapter(text_emb)

        if x is None:
            return self.inference(text_emb)

        x = self.origin_to_patchfy(x, self.patch_size)
        x = self.latent_patch_encoder(x)
        start_token_emb = self.learnable_start_token_emb[None, None, :].expand(x.shape[0], -1, -1)
        patch_sequence = torch.cat([start_token_emb, x], dim=1) + self.pos_encoding

        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device) * float("-inf"), diagonal=1)
        mu, logvar = self.forward_network(patch_sequence, causal_mask, text_emb)
        mu = self.patchfy_to_origin(mu[:, :-1, ...], self.patch_size)
        logvar = self.patchfy_to_origin(logvar[:, :-1, ...], self.patch_size)
        return mu, logvar

    def inference(self, text_emb=None):
        bs = text_emb.shape[0]
        start_token_emb = self.learnable_start_token_emb[None, None, :].expand(bs, -1, -1)
        sequence = start_token_emb

        sampled_noise_list = []
        for patch_id in range(128**2 // self.patch_size // self.patch_size):
            pos_enc = self.pos_encoding[:, : sequence.size(1), :]
            sequence_with_pos = sequence + pos_enc
            causal_mask = torch.triu(
                torch.ones(sequence.size(1), sequence.size(1), device=sequence.device) * float("-inf"), diagonal=1
            )
            mu, logvar = self.forward_network(sequence_with_pos, causal_mask, text_emb)
            std = torch.exp(0.5 * torch.clamp(logvar[:, -1, ...], -20, 10))
            sampled_noise = self.sample_noise(mu[:, -1, ...], std * 0.5)
            sampled_noise_list.append(sampled_noise)

            patch_emb = self.latent_patch_encoder(sampled_noise.reshape(bs, 1, -1))
            sequence = torch.cat([sequence, patch_emb], dim=1)

        return self.patchfy_to_origin(torch.stack(sampled_noise_list, dim=1), self.patch_size)
