import torch
from dataloader import create_verdict_dataloader

def train(model, config):
    print('training')
    loader = create_verdict_dataloader(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    #breakpoint()
    for ep in range(0, config.n_epoch):
        count = 0
        for d, l in loader:
            output = model(d)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), l.flatten())
            loss.backward()
            print(f'epoch: {ep}, count: {count}, loss: {loss.item()}')
            optimizer.step()
            count += 1
