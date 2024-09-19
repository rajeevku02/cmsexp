import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, sequence_len, input_dim, embed_dim, n_heads, dropout=0, bias=False):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim must be multiple of n_heads'
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.tril(torch.ones(sequence_len, sequence_len)))

    def scaled_dot_product(self, q, k, v, mask):
        head_dim = q.size()[-1]
        attn_logits = q @ k.transpose(-2, -1)
        attn_logits = attn_logits / math.sqrt(head_dim)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -torch.inf)
        attention = F.softmax(attn_logits, dim=-1)
        values = attention @ v
        return values

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, self.n_heads, -1)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values = self.scaled_dot_product(q, k, v, self.mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(B, T, -1)
        output = self.o_proj(values)
        return output

