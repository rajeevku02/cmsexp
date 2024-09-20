import torch
import torch.nn as nn
from attention import MultiheadAttention

class Feedforward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ma = MultiheadAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.fc = Feedforward(config.embed_dim)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.ma(self.ln1(x)))
        x = x + self.dropout2(self.fc(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeding = nn.Embedding(config.n_vocab, config.embed_dim)
        self.pos_embeding = nn.Embedding(config.max_sequence_len, config.embed_dim)
        self.ln = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.transformers = nn.Sequential(*[Block(config) for _ in range(0, config.n_layers)])
        self.final = nn.Linear(config.embed_dim, config.n_vocab)

    def forward(self, x):
        B, T = x.shape
        embed = self.embeding(x)
        pos = self.pos_embeding(torch.arange(T))
        x = embed + pos
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.ln(x)
        x = self.final(x)
        return x

