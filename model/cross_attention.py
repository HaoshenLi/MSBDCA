import torch
from torch import nn
from einops import rearrange
from torch import einsum
import torch.nn.functional as F


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_size, 
        heads, 
        query_embed_size, 
        key_embed_size,
        dropout=0.0
    ):
        super().__init__()
        assert embed_size % heads == 0

        self.heads = heads
        self.scale = (embed_size // heads) ** -0.5

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(query_embed_size, embed_size, bias = False)
        self.to_k = nn.Linear(key_embed_size, embed_size, bias = False)
        self.to_v = nn.Linear(key_embed_size, embed_size, bias = False)

        self.to_out = nn.Linear(embed_size, query_embed_size)

    def forward(
        self,
        query,
        key,
        value,
        query_mask = None,
        key_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = query.shape[0], query.shape[-2], key.shape[-2], self.heads, query.device
        q, k, v = self.to_q(query), self.to_k(key), self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(query_mask) or exists(key_mask):
            query_mask = default(query_mask, torch.ones((b, i), device = device, dtype = torch.bool))
            key_mask = default(key_mask, torch.ones((b, j), device = device, dtype = torch.bool))
            attn_mask = rearrange(query_mask, 'b i -> b 1 i 1') * rearrange(key_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(attn_mask==0, float("-1e20"))
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
    
        if return_attn:
            return out, attn
        return out
