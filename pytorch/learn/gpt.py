import torch
from torch import nn
from torch.nn import functional as F
import math

block_size = 256
batch_size = 64
n_embd = 384
n_head = 6
dropout = 0.2
n_layer = 2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 5
eval_iters = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

class Data:
    def __init__(self):
        with open('../data/gpt/input.txt', encoding='utf-8') as f:
            text = f.read()
        self.chars = list(set(text))
        self.stoi = {s:i for i,s in enumerate(self.chars)}
        self.itos = {i:s for i,s in enumerate(self.chars)}
        self.n_vocab = len(self.chars)
        self.data = torch.tensor(self.encode(text))
        n = int(0.9*len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, lst):
        return ''.join([self.itos[i] for i in lst])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

class Multihead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query(x).reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, T, self.num_heads, -1).permute(0, 2, 1, 3)

        wei = q @ k.transpose(-2, -1)
        wei = wei / math.sqrt(self.head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        wei = self.dropout(wei)
        out = wei @ v
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        out = self.dropout2(self.proj(out))

        return out

class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = Multihead(n_head, head_size)
        self.ffwd = Feedforward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.embedding_table = nn.Embedding(n_vocab, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.embedding_table(idx)
        pos_embd = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

class Runner:
    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            x, y = self.data.get_batch('train')
            logits, loss = self.model(x, y)
            print(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.data.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def test(self):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(self.data.decode(self.model.generate(context, max_new_tokens=500)[0].tolist()))

    def run(self):
        self.data = Data()
        self.model = GPT(self.data.n_vocab).to(device)
        m = self.model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
        self.train()
        self.test()

if __name__ == '__main__':
    Runner().run()