import torch

def generate(model, start_text, max_words):
    tokens = torch.tensor([model.config.tokenizer.encode(start_text)])
    output = model(tokens)