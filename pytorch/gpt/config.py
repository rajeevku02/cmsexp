import tiktoken

class Config:
    def __init__(self):
        self.sequence_len = 4
        self.n_heads = 2
        self.embed_dim = 8
        self.n_layers = 2
        self.dropout = 0
        self.ma_bias = False
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.n_vocab = self.tokenizer.n_vocab

config = Config()