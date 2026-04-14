"""Conditioning Encoder (Transformer) + 1D U-Net Denoising Network."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    feature_dim: int = 5
    context_len: int = 60
    target_len: int = 15
    num_coins: int = 4
    d_model: int = 256
    n_heads: int = 8
    encoder_layers: int = 4
    encoder_dropout: float = 0.1
    coin_embed_dim: int = 32
    unet_channels: list = field(default_factory=lambda: [128, 256])
    time_embed_dim: int = 256
    num_res_blocks: int = 2
    attn_heads: int = 8
    dropout: float = 0.1
    num_train_steps: int = 1000


# ── Building Blocks ──────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    """Timestep -> sinusoidal embedding -> MLP."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device).float() * -(math.log(10000) / (half - 1)))
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1))


class ConditioningEncoder(nn.Module):
    """Context (B,60,5) + coin_id (B,) -> cond_seq (B,60,d_model) via Transformer."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(cfg.feature_dim, cfg.d_model), nn.LayerNorm(cfg.d_model), nn.SiLU())
        self.coin_emb = nn.Embedding(cfg.num_coins, cfg.coin_embed_dim)
        self.coin_proj = nn.Linear(cfg.coin_embed_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.context_len, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_model * 4,
                                           cfg.encoder_dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.encoder_layers, nn.LayerNorm(cfg.d_model))
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, context, coin_id):
        x = self.proj(context) + self.coin_proj(self.coin_emb(coin_id)).unsqueeze(1) + self.pos[:, :context.size(1)]
        return self.out_proj(self.transformer(x))


class ResBlock1D(nn.Module):
    """Conv1D residual block with time embedding injection."""
    def __init__(self, ch_in, ch_out, t_dim, dropout=0.1):
        super().__init__()
        ng = lambda c: min(32, c)
        self.norm1, self.conv1 = nn.GroupNorm(ng(ch_in), ch_in), nn.Conv1d(ch_in, ch_out, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, ch_out))
        self.norm2, self.conv2 = nn.GroupNorm(ng(ch_out), ch_out), nn.Conv1d(ch_out, ch_out, 3, padding=1)
        self.drop, self.act = nn.Dropout(dropout), nn.SiLU()
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x))) + self.t_proj(t_emb).unsqueeze(-1)
        return self.conv2(self.drop(self.act(self.norm2(h)))) + self.skip(x)


class CrossAttention1D(nn.Module):
    """Q from U-Net features, K/V from conditioning sequence."""
    def __init__(self, ch, cond_dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads, self.hd = heads, ch // heads
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.to_q, self.to_k, self.to_v = nn.Linear(ch, ch), nn.Linear(cond_dim, ch), nn.Linear(cond_dim, ch)
        self.to_out = nn.Sequential(nn.Linear(ch, ch), nn.Dropout(dropout))
        self.scale = self.hd ** -0.5

    def forward(self, x, cond):
        B, C, L = x.shape
        q = self.to_q(self.norm(x).permute(0, 2, 1))
        k, v = self.to_k(self.cond_norm(cond)), self.to_v(self.cond_norm(cond))
        reshape = lambda t, l: t.view(B, l, self.heads, self.hd).permute(0, 2, 1, 3)
        q, k, v = reshape(q, L), reshape(k, -1), reshape(v, -1)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, L, C)
        return x + self.to_out(out).permute(0, 2, 1)


class SelfAttention1D(nn.Module):
    """Self-attention for bottleneck."""
    def __init__(self, ch, heads=8):
        super().__init__()
        self.heads, self.hd = heads, ch // heads
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv, self.to_out = nn.Linear(ch, ch * 3), nn.Linear(ch, ch)
        self.scale = self.hd ** -0.5

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x).permute(0, 2, 1)).view(B, L, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = torch.matmul(F.softmax(q @ k.transpose(-2, -1) * self.scale, -1), v)
        return x + self.to_out(out.permute(0, 2, 1, 3).contiguous().view(B, L, C)).permute(0, 2, 1)


class UNetBlock(nn.Module):
    """[ResBlock + CrossAttention] x N."""
    def __init__(self, ch_in, ch_out, cond_dim, t_dim, n_blocks=2, heads=8, dropout=0.1):
        super().__init__()
        self.res = nn.ModuleList()
        self.att = nn.ModuleList()
        for i in range(n_blocks):
            self.res.append(ResBlock1D(ch_in if i == 0 else ch_out, ch_out, t_dim, dropout))
            self.att.append(CrossAttention1D(ch_out, cond_dim, heads, dropout))

    def forward(self, x, t, cond):
        for r, a in zip(self.res, self.att):
            x = a(r(x, t), cond)
        return x


# ── U-Net ────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """2-level 1D U-Net with cross-attention conditioning."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ch = cfg.unet_channels  # [128, 256]
        self.time_emb = SinusoidalTimeEmb(cfg.time_embed_dim)
        self.in_conv = nn.Conv1d(cfg.feature_dim, ch[0], 3, padding=1)

        # Encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.enc.append(UNetBlock(c, co, cfg.d_model, cfg.time_embed_dim, cfg.num_res_blocks, cfg.attn_heads, cfg.dropout))
            if i < len(ch) - 1:
                self.downs.append(nn.Conv1d(co, co, 3, stride=2, padding=1))
            c = co

        # Bottleneck
        m = ch[-1]
        self.mid_r1, self.mid_sa, self.mid_ca, self.mid_r2 = (
            ResBlock1D(m, m, cfg.time_embed_dim), SelfAttention1D(m, cfg.attn_heads),
            CrossAttention1D(m, cfg.d_model, cfg.attn_heads), ResBlock1D(m, m, cfg.time_embed_dim))

        # Decoder
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        rev = list(reversed(ch))
        for i, co in enumerate(rev):
            ci = (rev[0] * 2) if i == 0 else (rev[i-1] + rev[i])
            self.dec.append(UNetBlock(ci, co, cfg.d_model, cfg.time_embed_dim, cfg.num_res_blocks, cfg.attn_heads, cfg.dropout))
            if i < len(rev) - 1:
                self.ups.append(nn.Conv1d(co, co, 3, padding=1))

        self.out_norm = nn.GroupNorm(min(32, ch[0]), ch[0])
        self.out_conv = nn.Conv1d(ch[0], cfg.feature_dim, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, noisy, t, cond):
        x = self.in_conv(noisy.permute(0, 2, 1))
        te = self.time_emb(t)

        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x, te, cond); skips.append(x)
            if i < len(self.downs): x = self.downs[i](x)

        x = self.mid_r1(x, te); x = self.mid_sa(x); x = self.mid_ca(x, cond); x = self.mid_r2(x, te)

        for i, dec in enumerate(self.dec):
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]: x = F.interpolate(x, skip.shape[-1], mode="nearest")
            x = dec(torch.cat([x, skip], 1), te, cond)
            if i < len(self.ups):
                tl = skips[-1].shape[-1] if skips else noisy.shape[1]
                x = self.ups[i](F.interpolate(x, tl, mode="nearest"))

        return F.silu(self.out_norm(x)).permute(0, 2, 1)  # -> (B, L, feat) but need out_conv
        # Oops, let me fix: out_conv should be before permute


class UNet1D(nn.Module):
    """2-level 1D U-Net with cross-attention conditioning. (corrected)"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ch = cfg.unet_channels
        self.time_emb = SinusoidalTimeEmb(cfg.time_embed_dim)
        self.in_conv = nn.Conv1d(cfg.feature_dim, ch[0], 3, padding=1)
        self.enc, self.downs = nn.ModuleList(), nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.enc.append(UNetBlock(c, co, cfg.d_model, cfg.time_embed_dim, cfg.num_res_blocks, cfg.attn_heads, cfg.dropout))
            if i < len(ch) - 1: self.downs.append(nn.Conv1d(co, co, 3, stride=2, padding=1))
            c = co
        m = ch[-1]
        self.mid_r1 = ResBlock1D(m, m, cfg.time_embed_dim)
        self.mid_sa = SelfAttention1D(m, cfg.attn_heads)
        self.mid_ca = CrossAttention1D(m, cfg.d_model, cfg.attn_heads)
        self.mid_r2 = ResBlock1D(m, m, cfg.time_embed_dim)
        self.dec, self.ups = nn.ModuleList(), nn.ModuleList()
        rev = list(reversed(ch))
        for i, co in enumerate(rev):
            ci = (rev[0] * 2) if i == 0 else (rev[i-1] + rev[i])
            self.dec.append(UNetBlock(ci, co, cfg.d_model, cfg.time_embed_dim, cfg.num_res_blocks, cfg.attn_heads, cfg.dropout))
            if i < len(rev) - 1: self.ups.append(nn.Conv1d(co, co, 3, padding=1))
        self.out_norm = nn.GroupNorm(min(32, ch[0]), ch[0])
        self.out_conv = nn.Conv1d(ch[0], cfg.feature_dim, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, noisy, t, cond):
        x, te = self.in_conv(noisy.permute(0, 2, 1)), self.time_emb(t)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x, te, cond); skips.append(x)
            if i < len(self.downs): x = self.downs[i](x)
        x = self.mid_r2(self.mid_ca(self.mid_sa(self.mid_r1(x, te)), cond), te)
        for i, dec in enumerate(self.dec):
            s = skips.pop()
            if x.shape[-1] != s.shape[-1]: x = F.interpolate(x, s.shape[-1], mode="nearest")
            x = dec(torch.cat([x, s], 1), te, cond)
            if i < len(self.ups):
                tl = skips[-1].shape[-1] if skips else noisy.shape[1]
                x = self.ups[i](F.interpolate(x, tl, mode="nearest"))
        return self.out_conv(F.silu(self.out_norm(x))).permute(0, 2, 1)


# ── Final Model ──────────────────────────────────────────────────────

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = ConditioningEncoder(self.config)
        self.unet = UNet1D(self.config)

    def forward(self, noisy_target, timestep, context, coin_id):
        cond = self.encoder(context, coin_id)
        return self.unet(noisy_target, timestep, cond)

    def get_num_params(self):
        c = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"encoder": f"{c(self.encoder):,}", "unet": f"{c(self.unet):,}", "total": f"{c(self):,}"}


if __name__ == "__main__":
    cfg = ModelConfig()
    model = ConditionalDiffusionModel(cfg)
    print(model.get_num_params())
    x = torch.randn(4, cfg.target_len, cfg.feature_dim)
    t = torch.randint(0, 1000, (4,))
    ctx = torch.randn(4, cfg.context_len, cfg.feature_dim)
    cid = torch.randint(0, 4, (4,))
    out = model(x, t, ctx, cid)
    print(f"Output: {out.shape}")  # (4, 15, 5)
