from train import train
from eval import generate
from model import GPT
from config import config
import sys

import torch

torch.manual_seed(427)

def load_and_infer():
    model = GPT(config)
    model = model.to(config.device)
    model.load_state_dict(torch.load('./data/savedm/model_weights.pth', weights_only=True))
    model.eval()
    while True:
        text = input("prompt: ")
        if text == 'exit':
            return
        generate(model, text, 40)

def main():
    model = GPT(config)
    model = model.to(config.device)
    print('compiling')
    torch.compile(model)
    print('compiling done') 
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        load_and_infer()
        return
    train(model, config)
    torch.save(model.state_dict(), './data/savedm/model_weights.pth')

if __name__ == '__main__':
    main()
