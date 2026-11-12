import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CausalSelfAttention, MultiHeadAttention


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Llama / Qwen style)."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        # ~8/3 * embedding_dim rounded to a multiple of 32 keeps parameter count
        # comparable to a standard 4x GELU MLP while giving better quality per param.
        hidden_dim = max(32, ((int(8 * embedding_dim / 3) + 31) // 32) * 32)
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(activated))


class ModernTransformerBlock(nn.Module):
    """Pre-RMSNorm + RoPE FlashAttention + SwiGLU block."""

    def __init__(self, embedding_dim: int, num_heads: int, context_length: int = 512, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.RMSNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            max_seq_len=context_length,
            dropout=dropout,
        )
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.mlp = SwiGLU(embedding_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerBlock(nn.Module):
    """Legacy GPT-2 block preserved for backward compatibility with older checkpoints."""

    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.mlp_act_dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x
        x = self.ln1(x)
        x = self.attention(x)
        output = output + self.dropout(x)
        x = output
        x = self.ln2(x)
        x = self.MLP[0](x)
        x = self.MLP[1](x)
        x = self.mlp_act_dropout(x)
        x = self.MLP[2](x)
        output = output + self.dropout(x)
        return output


if __name__ == "__main__":
    block = ModernTransformerBlock(embedding_dim=192, num_heads=4, context_length=128)
    x = torch.randn(16, 128, 192)
    output = block(x)
    print("Output shape:", output.shape)
    loss = output.sum()
    loss.backward()
    print("Backpropagation successful!")