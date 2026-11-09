import torch
import torch.nn as nn
from models.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.num_heads = num_heads

        #Attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        #Normalisering
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        #MLP
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )


    def forward(self, x):
        # 1. Attention + residual connection
        output = x
        x = self.ln1(x)
        x = self.attention(x)
        output = output + x
        x = output
        # 2. MLP + residual connection
        x = self.ln2(x)
        x = self.MLP(x)
        output = output + x
        return output

if __name__ == "__main__":

    block = TransformerBlock(
        embedding_dim=128,
        num_heads=4
    )

    x = torch.randn(16, 128, 128)

    output = block(x)

    print("Output shape:", output.shape)

    loss = output.sum()
    loss.backward()

    print("Backpropagation successful!")