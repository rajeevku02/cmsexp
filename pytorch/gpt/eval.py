import torch

def generate(model, start_text, max_words):
    tokens = torch.tensor([model.config.tokenizer.encode(start_text)]).to(model.config.device)
    count = 0
    while count < max_words:
        tks = tokens[:, -model.config.max_sequence_len:]
        output = model(tks)
        newt = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, newt), dim=-1)
        count += 1
    tks = tokens.tolist()[0]
    out = model.config.tokenizer.decode(tks)
    print(out)
