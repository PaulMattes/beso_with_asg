from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from inspect import isfunction

from torch.distributions import Categorical

from typing import Optional, Tuple

import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from .position_embedding import *
from .transformer_layers import *
from .moe_layers import *


class TransformerMoE(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            causal: bool = False,
            bias: bool = False,
            use_rms_norm: bool = False,
            use_rot_embed: bool = False,
            use_relative_pos: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[BlockMoE(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=causal, 
            use_rms_norm=use_rms_norm,
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            use_relative_pos=use_relative_pos,
            rotary_xpos=rotary_xpos,
            bias=bias,
            num_experts=num_experts,
            top_k=top_k,
            router_normalize=router_normalize
            ) 
            for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim) if use_rms_norm else LayerNorm(embed_dim, bias) 

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerFiLMMoE(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            causal: bool = False,
            bias: bool = False,
            cond_router: bool = False,
            use_rms_norm: bool = False,
            use_rot_embed: bool = False,
            use_relative_pos: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            num_experts: int = 4,
            top_k: int = 2,
            noise_in_cross_attention: bool = False,
            router_normalize: bool = True,
            router_context_cond_only: bool = False,
            use_router_cond_only: bool = False,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder and not use_router_cond_only:
            self.blocks = nn.Sequential(
                *[NoiseBlockMoE(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=causal, 
                use_rms_norm=use_rms_norm,
                use_cross_attention=use_cross_attention,
                noise_in_cross_attention=noise_in_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                num_experts=num_experts,
                top_k=top_k,
                router_normalize=router_normalize,
                cond_router=cond_router,
                router_context_cond_only=router_context_cond_only,
                ) 
                for _ in range(n_layers)]
            )
        elif not use_router_cond_only:
            self.blocks = nn.Sequential(
                *[CondBlockMoE(
                embed_dim, 
                film_cond_dim,
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=causal, 
                cond_router=cond_router,
                use_rms_norm=use_rms_norm,
                top_k=top_k,
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                router_context_cond_only=router_context_cond_only,
                ) 
                for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.Sequential(
                *[RounterCondBlockMoE(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=causal, 
                cond_router=cond_router,
                use_rms_norm=use_rms_norm,
                top_k=top_k,
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                router_context_cond_only=router_context_cond_only,
                ) 
                for _ in range(n_layers)]
            )
        self.ln = RMSNorm(embed_dim) if use_rms_norm else LayerNorm(embed_dim, bias) 

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, c, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


