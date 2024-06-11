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



class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rms_norm: bool = False,
            use_rot_embed: bool = False,
            use_relative_pos: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=True, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            use_relative_pos=use_relative_pos,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = RMSNorm(embed_dim) if use_rms_norm else LayerNorm(embed_dim, bias) 

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x



class TransformerFiLMDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rms_norm: bool = False,
            use_rot_embed: bool = False,
            use_relative_pos: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder:
            self.blocks = nn.Sequential(
                *[NoiseBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_rms_norm=use_rms_norm,
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                ) 
                for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConditionedBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_rms_norm=use_rms_norm,
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                film_cond_dim=film_cond_dim,
                ) 
                for _ in range(n_layers)]
            )
        self.ln = RMSNorm(embed_dim) if use_rms_norm else LayerNorm(embed_dim, bias) 

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, c, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerFiLMDecoderInterleaved(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rms_norm: bool = False,
            use_rot_embed: bool = False,
            use_relative_pos: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder:
            self.blocks = nn.Sequential(
                *[NoiseBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_rms_norm=use_rms_norm,
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                ) 
                for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConditionedBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                use_relative_pos=use_relative_pos,
                rotary_xpos=rotary_xpos,
                bias=bias,
                film_cond_dim=film_cond_dim,
                ) 
                for _ in range(n_layers)]
            )
        self.ln = RMSNorm(embed_dim) if use_rms_norm else LayerNorm(embed_dim, bias) 

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for idx, layer in enumerate(self.blocks):
            cond_tokens =cond[idx]
            x = layer(x, c, cond_tokens, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
