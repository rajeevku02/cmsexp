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
        embed_dim = config['embed_dim']
        dropout = config['dropout']

        self.ln1 = nn.LayerNorm()
        self.ma = MultiheadAttention(config)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm()
        self.fc = Feedforward(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self):
        x = x + self.dropout(self.ma(self.ln1(x)))
        x = x+ self.dropout2(self.fc(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_vocab = config['n_vocab']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        sequence_len = config['sequence_len']
        self.embeding = nn.Embedding(n_vocab, embed_dim)
        self.pos_embeding = nn.Embedding(sequence_len, embed_dim)
        self.ln = nn.LayerNorm()
        self.dropout = nn.Dropout(config['dropout'])
        self.transformers = nn.Sequential(*[Block(config) for _ in range(0, n_layers)])
        self.final = nn.Linear(embed_dim, n_vocab)

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
