import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from .position_embedding import *
from .utils import RMSNorm, LayerNorm, SwishGLU
from .transformer_layers import Attention, MLP, modulate, AdaLNZero


class CondRouterMLP(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            num_experts: int,
            bias: bool,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0
        ):
        super().__init__()
        layers = []        
        if use_swish:
            layers.append(SwishGLU(n_embd, 2 * n_embd))
        else:
            layers.append(nn.Linear(n_embd, 2 * n_embd, bias=bias))
            if use_relus:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(2 * n_embd, num_experts, bias=bias))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

class BlockMoE(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            block_size: int, 
            causal: bool,
            use_rms_norm: bool = False,
            use_cross_attention: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            use_relative_pos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize = True,
        ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.attn = Attention(
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            block_size, 
            causal, 
            bias, 
            use_rot_embed, 
            use_relative_pos,
            rotary_xpos
        )
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
            self.ln_3 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.ln_2 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)

        # MoE 
        self.router = Router(n_embd, num_experts, top_k, normalize=router_normalize)
        '''
        normalize = True -> [0.25, 0.25, 0.25, 0.25] -> [0, 0.25, 0, 0.25] -> [0, 0.5, 0, 0.5]
        normalize = False -> [0.25, 0.25, 0.25, 0.25] -> [0, 0.25, 0, 0.25]
        '''
        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": MLP(n_embd, bias, mlp_pdrop)
                for i in range(num_experts)
            }
        )

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = self.ln_2(x)

        router_mask, top_k_indices, router_probs, true_probs = self.router(x)
        next_states = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            probs = router_probs[:, :, idx][token_indices].unsqueeze(-1)
            next_states[token_indices] += probs * expert(x[token_indices]).to(
                next_states.dtype
            )

        return x + next_states


class Router(nn.Module):
    def __init__(
        self,
        hidden_states,
        num_experts,
        top_k,
        bias=False,
        normalize=True
    ):
        super().__init__()
        self.router = nn.Linear(hidden_states, num_experts, bias=bias)
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize

    def forward(self, inputs):
        input_shape = inputs.size()

        inputs = inputs.reshape(-1, inputs.size(-1))
        logits = self.router(inputs)
        probs = torch.softmax(logits, dim=-1)

        if self.training:
            top_k_indices = torch.multinomial(probs, self.top_k)
        else:
            top_k_indices = probs.topk(self.top_k).indices

        router_mask = torch.zeros_like(probs).scatter_(1, top_k_indices, 1)
        router_probs = torch.zeros_like(probs).scatter_(
            1, top_k_indices, probs.gather(1, top_k_indices)
        )

        top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
        router_mask = router_mask.view(*input_shape[:-1], -1)
        router_probs = router_probs.view(*input_shape[:-1], -1)

        if self.normalize:
            if self.top_k == 1:  # allow gradients to flow
                router_probs = (
                    router_probs / router_probs.sum(dim=-1, keepdim=True).detach()
                )
            else:
                router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        return (
            router_mask,
            top_k_indices,
            router_probs,
            probs.view(*input_shape[:-1], -1),
        )


class RouterCond(nn.Module):
    def __init__(
        self,
        hidden_states: int,
        cond_dim: int,
        num_experts: int,
        top_k: int,
        bias: bool = False,
        normalize: bool = True,
        cond_router: bool = False,
        router_context_cond_only: bool = False,
        temperature: float = 1.0,  # Add this line
    ):
        super().__init__()
        self.cond_router = cond_router
        self.router_context_cond_only = router_context_cond_only
        if self.cond_router:
            if self.router_context_cond_only:
                input_dim = cond_dim
            else:
                input_dim = hidden_states + cond_dim
            self.router = CondRouterMLP(
                input_dim, 
                num_experts, 
                bias=bias,
                use_swish=False,
                dropout=0
            )
        else:
            self.router = nn.Linear(hidden_states, num_experts, bias=bias)

        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.temperature = temperature  # Add this line

    def forward(self, inputs, cond):
        epsilon = 1e-9  # Define epsilon for numerical stability
        input_shape = inputs.size()

        if self.cond_router:
            if cond.shape[-2] != inputs.shape[-2]:
                cond = einops.repeat(cond, 'b t d -> b (t n) d', n=int(inputs.shape[-2] / cond.shape[-2]))

            if self.router_context_cond_only:
                router_inputs = cond.reshape(-1, cond.size(-1))
                logits = self.router(router_inputs)
            else:
                router_inputs = torch.concat([inputs, cond], dim=-1)
                router_inputs = router_inputs.reshape(-1, router_inputs.size(-1))
                logits = self.router(router_inputs)
        else:
            inputs = inputs.reshape(-1, inputs.size(-1))
            logits = self.router(inputs)

        # Check for inf or NaN in logits
        if not torch.isfinite(logits).all():
            # raise RuntimeError("Logits contain inf or NaN values")
            print("Logits contain inf or NaN values")

        logits = logits - logits.max(dim=-1, keepdim=True).values
        # add temperature
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        
        # Adding epsilon to avoid zeros and ensure non-negativity
        probs = probs + epsilon
        
        # Clipping probabilities to ensure they stay within [epsilon, 1-epsilon]
        probs = torch.clamp(probs, min=epsilon, max=1-epsilon)

        # Check for inf, NaN, or negative values in probs
        if (probs < 0).any() or not torch.isfinite(probs).all():
            print("Softmax probabilities contain inf, NaN, or negative values")
        #    raise RuntimeError("Softmax probabilities contain inf, NaN, or negative values")
        if not torch.allclose(probs.sum(dim=-1), torch.tensor(1.0), atol=1e-5):
        #    raise RuntimeError("Probabilities do not sum up to 1")
            print('Probabilities do not sum up to 1')
        if self.training:
            top_k_indices = torch.multinomial(probs, self.top_k, replacement=False)
        else:
            top_k_indices = probs.topk(self.top_k, dim=-1).indices

        router_mask = torch.zeros_like(probs).scatter_(1, top_k_indices, 1)
        router_probs = torch.zeros_like(probs).scatter_(
            1, top_k_indices, probs.gather(1, top_k_indices)
        )

        top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
        router_mask = router_mask.view(*input_shape[:-1], -1)
        router_probs = router_probs.view(*input_shape[:-1], -1)

        if self.normalize:
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        return (
            router_mask,
            top_k_indices,
            router_probs,
            probs.view(*input_shape[:-1], -1),
        )


class CondBlockMoE(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            film_cond_dim: int,
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            block_size: int, 
            causal: bool,
            cond_router: bool = False,
            use_rms_norm: bool = False,
            use_cross_attention: bool = False,
            use_relative_pos: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize = True,
            router_context_cond_only: bool = False,
        ):
        super().__init__()
        self.cond_router = cond_router
        self.ln_1 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.attn = Attention(
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            block_size, 
            causal, 
            bias, 
            use_rot_embed, 
            use_relative_pos,
            rotary_xpos
        )
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(
                n_embd, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                block_size, 
                causal, 
                bias, 
                use_rot_embed, 
                use_relative_pos,
                rotary_xpos
            )
            self.ln_3 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.ln_2 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)

        # MoE 
        self.router = RouterCond(
            n_embd, 
            film_cond_dim,
            num_experts, 
            top_k, 
            normalize=router_normalize,
            cond_router=cond_router,
            router_context_cond_only=router_context_cond_only,
        )
        '''
        normalize = True -> [0.25, 0.25, 0.25, 0.25] -> [0, 0.25, 0, 0.25] -> [0, 0.5, 0, 0.5]
        normalize = False -> [0.25, 0.25, 0.25, 0.25] -> [0, 0.25, 0, 0.25]
        '''
        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": MLP(n_embd, bias, mlp_pdrop)
                for i in range(num_experts)
            }
        )
        self.adaLN_zero = AdaLNZero(film_cond_dim)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        # get the shift and scale from the film layer
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # attention with additional modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x_mlp = self.ln_2(x)

        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        # add the conditional routing
        router_mask, top_k_indices, router_probs, true_probs = self.router(x_mlp, c)
        next_states = torch.zeros_like(x)
        
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            probs = router_probs[:, :, idx][token_indices].unsqueeze(-1) 
            next_states[token_indices] += probs * expert(x[token_indices]).to(
                next_states.dtype
            )

        return x + gate_mlp * next_states
    
    

class NoiseBlockMoE(BlockMoE):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            use_rms_norm=False,
            noise_in_cross_attention=False,
            cond_router: bool = False,
            use_cross_attention=False, 
            use_relative_pos=False,
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False, # and any other arguments from the Block class
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize = True,
            router_context_cond_only: bool = False,
        ):
        super().__init__(n_embd, 
                         n_heads, 
                         attn_pdrop, 
                         resid_pdrop, 
                         mlp_pdrop, 
                         block_size, 
                         causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         use_relative_pos=use_relative_pos,
                         rotary_xpos=rotary_xpos, 
                         bias=bias,
                         use_rms_norm=use_rms_norm,
                         num_experts=num_experts,
                         top_k=top_k,
                         router_normalize=router_normalize
        )
        self.cond_router = cond_router
        
        # if self.cond_router:
        self.router = RouterCond(
            n_embd, 
            n_embd,
            num_experts, 
            top_k, 
            normalize=router_normalize,
            cond_router=cond_router,
            router_context_cond_only=router_context_cond_only,
        )
        self.noise_in_cross_attention =noise_in_cross_attention
        
    def forward(self, x, c, context=None, custom_attn_mask=None):
        
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            if self.noise_in_cross_attention:
                x = x + self.cross_att(self.ln_3(x) + c, context, custom_attn_mask=custom_attn_mask)
            else:
                x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = self.ln_2(x)

        # next use the router to chose the expert
        if self.cond_router:
            # router_input = torch.cat([x, c], dim=-1)
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, c)
        else:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x)
        next_states = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            probs = router_probs[:, :, idx][token_indices].unsqueeze(-1)
            next_states[token_indices] += probs * expert(x[token_indices]).to(
                next_states.dtype
            )

        return x + next_states
    


class RounterCondBlockMoE(BlockMoE):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            use_rms_norm=False,
            cond_router: bool = False,
            use_cross_attention=False, 
            use_relative_pos=False,
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False, # and any other arguments from the Block class
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize = True,
            router_context_cond_only: bool = False,
        ):
        super().__init__(n_embd, 
                         n_heads, 
                         attn_pdrop, 
                         resid_pdrop, 
                         mlp_pdrop, 
                         block_size, 
                         causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         use_relative_pos=use_relative_pos,
                         rotary_xpos=rotary_xpos, 
                         bias=bias,
                         use_rms_norm=use_rms_norm,
                         num_experts=num_experts,
                         top_k=top_k,
                         router_normalize=router_normalize
        )
        self.cond_router = cond_router
        
        # if self.cond_router:
        self.router = RouterCond(
            n_embd, 
            n_embd,
            num_experts, 
            top_k, 
            normalize=router_normalize,
            cond_router=cond_router,
            router_context_cond_only=router_context_cond_only,
        )
        
    def forward(self, x, c, context=None, custom_attn_mask=None):
        
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = self.ln_2(x)

        # next use the router to chose the expert
        if self.cond_router:
            # router_input = torch.cat([x, c], dim=-1)
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, c)
        else:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x)
        next_states = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            probs = router_probs[:, :, idx][token_indices].unsqueeze(-1)
            next_states[token_indices] += probs * expert(x[token_indices]).to(
                next_states.dtype
            )

        return x + next_states