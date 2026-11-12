import torch
import torch.nn as nn


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
        B, T, C = x.shape #B er batch size, T er sequence length, og C er embedding dimension.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = Q @ K.transpose(-2, -1) #Matrix multiplication of Q and K, where K is transposed to match dimensions for multiplication. This computes the attention scores.
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attention = attention / (self.head_size ** 0.5)
        attention = attention.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(attention, dim=-1)
        weights = self.attn_dropout(weights)
        output = weights @ V #Matrix multiplication of the attention weights and the value matrix V. This computes the final output of the self-attention mechanism.
        return output


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
        self.linear = nn.Linear(embedding_dim,embedding_dim)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        output = torch.cat(outputs, dim=-1)
        output = self.linear(output)
        return output


if __name__ == "__main__":

    attention = MultiHeadAttention(
    embedding_dim=128,
    num_heads=4
    )

    x = torch.randn(16, 128, 128)

    output = attention(x)

    print(output.shape)