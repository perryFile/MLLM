import torch
import torch.nn as nn
from models.transformer import TransformerBlock



class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int = 100, embedding_dim: int = 128, context_length: int = 128, num_heads: int = 4, num_layers: int = 4) -> None:
        super().__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.num_embeddings = self.vocab_size
        self.embedding_dim = embedding_dim
        self.tokens_embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.context_length, self.embedding_dim)

        #Laver transformer blocks.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(self.embedding_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.outputLayer = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        B, T = x.shape #T vil her være længden af inputtet, og B vil være batch size.
        if T > self.context_length:
            raise ValueError("Input exceeds context length")

        positions = torch.arange(T, device=x.device)  #Device sørger bare for at positionerne ligger på samme device.

        token_embeddings = self.tokens_embedding(x)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        #Gennem transformers
        x = self.blocks(x)

        #Final normalisering
        x = self.norm(x)

        x = self.outputLayer(x)

        return x


if __name__ == "__main__":

    model = MiniGPT(
        vocab_size=100,
        context_length=128,
        embedding_dim=128,
        num_heads=4,
        num_layers=4
    )

    x = torch.randint(0, 100, (16, 128))

    output = model(x)

    print("Output shape:", output.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )