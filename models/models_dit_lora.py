"""
LoRA-enabled DiT model for MeanFlow fine-tuning.

This creates modified Attention and DiT classes that add LoRA adapters
to the QKV and output projections in each attention block.

LoRA A matrices are initialized with small random values.
LoRA B matrices are initialized to zero.
This means at initialization, LoRA has no effect (base model behavior preserved).
"""

import math
from functools import partial

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

from models.timm_models import Mlp, PatchEmbed
from models.torch_models import TorchEmbedding, TorchLayerNorm, TorchLinear
from models.models_dit import (
    TimestepEmbedder, LabelEmbedder, FinalLayer,
    DiTLinear, DiTMlp,
    unsqueeze, modulate,
    get_2d_sincos_pos_embed,
)

# ============================================================
# LoRA Attention
# ============================================================

class LoRAAttention(nn.Module):
    """Multi-head attention with LoRA adapters on QKV and output projections."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0

    def setup(self):
        dim = self.dim
        num_heads = self.num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Base layers (frozen during LoRA fine-tuning)
        self.qkv = DiTLinear(dim, dim * 3, bias=self.qkv_bias)
        self.proj = DiTLinear(dim, dim, bias=True)

        # LoRA adapters for QKV
        self.qkv_lora_A = nn.Dense(
            features=self.lora_rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=1.0 / self.lora_rank),
            name="qkv_lora_A",
        )
        self.qkv_lora_B = nn.Dense(
            features=dim * 3,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="qkv_lora_B",
        )

        # LoRA adapters for output projection
        self.proj_lora_A = nn.Dense(
            features=self.lora_rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=1.0 / self.lora_rank),
            name="proj_lora_A",
        )
        self.proj_lora_B = nn.Dense(
            features=dim,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="proj_lora_B",
        )

        self.lora_scaling = self.lora_alpha / self.lora_rank

    def __call__(self, x):
        B, N, C = x.shape

        # QKV with LoRA
        qkv_base = self.qkv(x)
        qkv_lora = self.qkv_lora_B(self.qkv_lora_A(x)) * self.lora_scaling
        qkv = qkv_base + qkv_lora

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = jnp.split(qkv, 3, axis=0)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 2, 4, 3)
        attn = nn.softmax(attn, axis=-1)
        x = attn @ v

        x = x[0].transpose(0, 2, 1, 3).reshape(B, N, C)

        # Output projection with LoRA
        proj_base = self.proj(x)
        proj_lora = self.proj_lora_B(self.proj_lora_A(x)) * self.lora_scaling
        x = proj_base + proj_lora

        return x


# ============================================================
# LoRA DiT Block
# ============================================================

class LoRADiTBlock(nn.Module):
    """DiT block with LoRA attention."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    lora_rank: int = 8
    lora_alpha: float = 16.0

    def setup(self):
        hidden_size = self.hidden_size
        num_heads = self.num_heads
        mlp_ratio = self.mlp_ratio

        self.norm1 = TorchLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = LoRAAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
        )
        self.norm2 = TorchLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: partial(nn.gelu, approximate=True)
        self.mlp = DiTMlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu, drop=0,
        )
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            TorchLinear(hidden_size, 6 * hidden_size, bias=True, weight_init='zeros', bias_init='zeros'),
        ])

    def __call__(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(c), 6, axis=1
        )
        x = x + unsqueeze(gate_msa, 1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + unsqueeze(gate_mlp, 1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================
# LoRA DiT Model
# ============================================================

class LoRADiT(nn.Module):
    """DiT with LoRA adapters in all attention layers."""
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0

    def setup(self):
        hidden_size = self.hidden_size
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, hidden_size, self.class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(
            get_2d_sincos_pos_embed(hidden_size, int(num_patches ** 0.5))
        ).astype(jnp.float32)

        self.blocks = nn.Sequential([
            LoRADiTBlock(
                hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio,
                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def __call__(self, x, t, h, y, train=False, key=None):
        x = self.x_embedder(x) + self.pos_embed_func()
        t = self.t_embedder(t)
        h = self.h_embedder(h)
        y = self.y_embedder(y, train=train, rng=key)
        c = t + h + y
        for block in self.blocks.layers:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


# ============================================================
# LoRA DiT Configs (matching base DiT configs)
# ============================================================

def make_lora_dit(base_config, lora_rank=8, lora_alpha=16.0):
    """Create a LoRA DiT config from a base DiT config."""
    return partial(
        LoRADiT,
        depth=base_config.keywords.get('depth', 12),
        hidden_size=base_config.keywords.get('hidden_size', 768),
        patch_size=base_config.keywords.get('patch_size', 4),
        num_heads=base_config.keywords.get('num_heads', 12),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


# Pre-configured LoRA variants
LoRA_DiT_B_4 = partial(LoRADiT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
LoRA_DiT_B_2 = partial(LoRADiT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
LoRA_DiT_L_4 = partial(LoRADiT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
LoRA_DiT_XL_2 = partial(LoRADiT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
