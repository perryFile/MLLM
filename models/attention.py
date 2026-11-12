import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for relative token positions."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RotaryEmbedding")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=dtype),
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Fused Multi-Head Attention with RoPE and FlashAttention (SDPA)."""

    def __init__(self, embedding_dim: int, num_heads: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


# Legacy classes preserved so existing GPT-style checkpoints continue to load
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = Q @ K.transpose(-2, -1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attention = attention / (self.head_size ** 0.5)
        attention = attention.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(attention, dim=-1)
        weights = self.attn_dropout(weights)
        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttention(embedding_dim, head_size, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        output = torch.cat(outputs, dim=-1)
        return self.linear(output)


if __name__ == "__main__":
    attention = CausalSelfAttention(embedding_dim=192, num_heads=4)
    x = torch.randn(16, 128, 192)
    output = attention(x)
    print(output.shape)