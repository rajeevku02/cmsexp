import tiktoken

class Config:
    def __init__(self):
        self.lr = 0.001
        self.n_epoch = 2
        self.max_sequence_len = 512
        self.stride = 128
        self.batch_size = 8
        self.n_heads = 12
        self.embed_dim = 768
        self.n_layers = 2
        self.dropout = 0.1
        self.ma_bias = False
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.n_vocab = self.tokenizer.n_vocab

config = Config()