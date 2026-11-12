import torch
import torch.nn as nn
from models.transformer import ModernTransformerBlock, TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 192,
        context_length: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.0,
        tie_weights: bool = True,
        architecture: str = "modern",
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.num_embeddings = self.vocab_size
        self.embedding_dim = embedding_dim
        self.tie_weights = tie_weights
        self.architecture = architecture

        self.tokens_embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        if architecture == "modern":
            self.position_embedding = None
            self.blocks = nn.Sequential(
                *[
                    ModernTransformerBlock(
                        self.embedding_dim,
                        num_heads,
                        context_length=self.context_length,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.RMSNorm(self.embedding_dim)
        elif architecture == "legacy":
            self.position_embedding = nn.Embedding(self.context_length, self.embedding_dim)
            self.blocks = nn.Sequential(
                *[
                    TransformerBlock(self.embedding_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )
            self.norm = nn.LayerNorm(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.outputLayer = nn.Linear(
            self.embedding_dim, self.vocab_size, bias=not tie_weights
        )

        self.apply(self._init_weights)
        if self.tie_weights:
            self.outputLayer.weight = self.tokens_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        _, T = x.shape
        if self.position_embedding is not None and T > self.context_length:
            raise ValueError("Input exceeds context length")

        x = self.tokens_embedding(x)
        if self.position_embedding is not None:
            positions = torch.arange(T, device=x.device)
            x = x + self.position_embedding(positions)

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.outputLayer(x)


if __name__ == "__main__":
    model = MiniGPT(
        vocab_size=4000,
        context_length=128,
        embedding_dim=192,
        num_heads=4,
        num_layers=4,
    )
    x = torch.randint(0, 4000, (16, 128))
    output = model(x)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))