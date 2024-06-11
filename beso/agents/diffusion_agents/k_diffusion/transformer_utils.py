import logging
import math 
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale **einops.rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def rotate_half(x):
    x =einops.rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

# norms

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = einops.rearrange(values, 'i j h -> h i j')
        return bias * self.scale


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))


# code adapted from miniGPT 
class CausalSelfAttention(nn.Module):
    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        use_rel_pos_embed: bool = False,
        use_rotary_pos_embed: bool = False,
        rotary_dim: int = None,
        rotary_scale_base: int = 512,
        rotary_interpolation_factor: float = 1.,
        rotary_base: int = 10000,
        rotary_base_rescale_factor: float = 1.
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        self.use_rel_pos_embed = use_rel_pos_embed
        self.use_rotary_pos_embed = use_rotary_pos_embed
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # relative positional encoding
        if self.use_rel_pos_embed:
            self.relative_position_bias = RelativePositionBias(scale=1.0, causal=True, num_buckets=32, max_distance=128, heads=n_heads)
        # rotary positional encoding
        if self.use_rotary_pos_embed:
            self.rotary_embedding = RotaryEmbedding(
                dim=rotary_dim,
                scale_base=rotary_scale_base,
                interpolation_factor=rotary_interpolation_factor,
                base=rotary_base,
                base_rescale_factor=rotary_base_rescale_factor
            )
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply rotary positional embeddings
        if self.use_rotary_pos_embed:
            freqs, scale = self.rotary_embedding(T, x.device)
            q = apply_rotary_pos_emb(q, freqs, scale)
            k = apply_rotary_pos_emb(k, freqs, scale)

        # Calculate relative positional bias
        if self.use_rel_pos_embed:
            i, j = map(lambda t: t.shape[-2], (q, k))
            rel_pos_bias = self.relative_position_bias(i, j)

        # Adjust the attention score calculation
        att_scores = q @ k.transpose(-2, -1)  # Shape: [B, nh, T, T]
        if self.use_rel_pos_embed:
            att_scores += rel_pos_bias.unsqueeze(0)

        # Apply the mask, softmax, and dropout
        att = att_scores * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Compute the output tensor
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

    

class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, use_rel_embed: bool = False):
        super().__init__()
        assert n_embd % n_heads == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_heads
        self.use_rel_embed = use_rel_embed
        # New addition: relative positional encoding
        if self.use_rel_embed:
            self.relative_position_bias = RelativePositionBias(scale=1.0, causal=True, num_buckets=32, max_distance=128, heads=n_heads)
        
    def forward(self, x, context):
        B, T, C = x.size()
        k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Adjust the attention score calculation
        att_scores = q @ k.transpose(-2, -1)  # Shape: [B, nh, T, T]

        if self.use_rel_embed:
            
            # Calculate relative positional bias
            i, j = map(lambda t: t.shape[-2], (q, k))
            rel_pos_bias = self.relative_position_bias(i, j)

            # Add relative positional bias to attention scores
            att_scores += rel_pos_bias.unsqueeze(0)  # Add relative positional bias

        # Apply softmax and dropout
        att = att_scores * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Compute the output tensor
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_drop(self.proj(y))
        return y
    


class Block(nn.Module):
    """An unassuming Transformer block with cross-attention."""

    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            block_size, 
            use_rel_pos_emb=False
            ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        if use_rel_pos_emb:
            self.attn = CausalSelfAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, use_rel_pos_emb)
            self.cross_attn = CrossAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, use_rel_pos_emb)
        else:
            self.attn = CausalSelfAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, use_rel_pos_emb)
            self.cross_attn = CrossAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, use_rel_pos_emb)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, context=None):
        x = x + self.attn(self.ln1(x))
        if context is not None:
            x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.mlp(self.ln3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, attn_pdrop, resid_pdrop, n_layers, use_rel_pos_emb=False):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_heads, attn_pdrop, resid_pdrop, embed_dim, use_rel_pos_emb) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln(x)
        return x
    


