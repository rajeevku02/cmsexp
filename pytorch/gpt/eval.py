import torch

def generate(model, start_text, max_words):
    tokens = torch.tensor([model.config.tokenizer.encode(start_text)])
    count = 0
    while count < max_words:
        tks = tokens[:, -model.config.sequence_len:]
        output = model(tks)
        newt = torch.argmax(output[-1, -1, :]).unsqueeze(dim=0).unsqueeze(dim=0)
        tokens = torch.cat((tokens, newt), dim=-1)
        count += 1
    tks = tokens.tolist()[0]
    out = model.config.tokenizer.decode(tks)
    print(out)