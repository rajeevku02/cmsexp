from train import train
from eval import generate
from model import GPT
from config import config

import torch

torch.manual_seed(427)

def main():
    model = GPT(config)
    generate(model, 'hello world', 40)
    #train(model, config)

if __name__ == '__main__':
    main()