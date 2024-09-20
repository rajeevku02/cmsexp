from train import train
from eval import generate
from model import GPT
from config import config

def main():
    model = GPT(config)
    generate(model, 'hello', 4)
    #train(model, config)

if __name__ == '__main__':
    main()