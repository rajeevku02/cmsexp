from dataloader import create_verdict_dataloader
from train import train, generate
from model import GPT

config = {
    'sequence_len': 4,
    'n_head': 2,
    'n_vocab': 4,
    'embed_dim': 8,
    'n_layers': 2,
    'dropout': 0,
    'ma_bias': True
}

def main():
    model = GPT(config)
    generate(model, config, 'hello')
    train(model, config)

if __name__ == 'main':
    main()