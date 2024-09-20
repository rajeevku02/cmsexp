from train import train, generate
from model import GPT
from config import config

def main():
    model = GPT(config)
    #generate(model, config, 'hello')
    #train(model, config)

if __name__ == 'main':
    main()