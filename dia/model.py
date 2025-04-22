# DiaModel with diffusion-style decoder (StyleTTS2 inspired)

import torch
import torch.nn as nn
import torch.nn.functional as FCan you

class DiaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_dim = config.model.decoder.d_model
        vocab_size = config.model.tgt_vocab_size
        self.spk_proj = nn.Linear(192, model_dim)

        # Encoder: text + lang embedding
        self.encoder_embed = nn.Embedding(config.model.encoder_vocab_size, model_dim)
        self.encoder_proj = nn.Linear(model_dim, model_dim)

        # Diffusion decoder setup
        self.token_proj = nn.Linear(config.model.input_dim, model_dim)
        self.diffusion_steps = config.model.diffusion_steps
        self.time_embed = nn.Embedding(self.diffusion_steps, model_dim)

        # Conditional U-Net style block
        self.diffusion_layers = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        self.to_logits = nn.Linear(model_dim, vocab_size)

    def encoder(self, input_ids, lang_ids):
        x = self.encoder_embed(input_ids)
        return self.encoder_proj(x)

    def diffusion_step(self, x, t_embed, spk_embed=None, encoder_out=None):
        cond = x + t_embed
        if spk_embed is not None:
            spk = self.spk_proj(spk_embed).unsqueeze(1).expand_as(cond)
            cond = cond + spk
        if encoder_out is not None:
            cond = cond + encoder_out  # simple cross-attn via addition
        return self.diffusion_layers(cond)

    def decoder_forward(self, tgt_ids_BxTxC, spk_embed=None, encoder_out=None, **kwargs):
        B, T, _ = tgt_ids_BxTxC.shape
        x = self.token_proj(tgt_ids_BxTxC)

        for t in range(self.diffusion_steps):
            t_embed = self.time_embed(torch.tensor([t], device=x.device)).unsqueeze(0).expand(B, T, -1)
            x = self.diffusion_step(x, t_embed, spk_embed, encoder_out)

        return x

    def decode_step(self, tgt_ids_Bx1xC, tgt_pos_Bx1, encoder_out, spk_embed=None, **kwargs):
        B, T, _ = tgt_ids_Bx1xC.shape
        x = self.token_proj(tgt_ids_Bx1xC)
        t_embed = self.time_embed(torch.tensor([0], device=x.device)).unsqueeze(0).expand(B, T, -1)
        x = self.diffusion_step(x, t_embed, spk_embed, encoder_out)
        return self.to_logits(x), {}

    def compute_loss(self, output, targets):
        return F.mse_loss(output, targets)

    def extract_kv_cache(self):
        return {}

    def forward(self, batch):
        input_ids = batch["input_ids"]
        audio = batch["audio"]
        lang_ids = batch["lang_token_ids"]
        spk_embed = batch.get("spk_embed", None)

        enc_out = self.encoder(input_ids, lang_ids)
        output = self.decoder_forward(audio, spk_embed=spk_embed, encoder_out=enc_out)
        loss = self.compute_loss(output, audio)
        return loss
