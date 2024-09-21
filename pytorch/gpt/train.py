import torch
from dataloader import create_verdict_dataloader

def train(model, config):
    print('training')
    model.train()
    loader = create_verdict_dataloader(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    #breakpoint()
    for ep in range(0, config.n_epoch):
        count = 0
        for d, l in loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=model.config.device, dtype=torch.bfloat16):
                output = model(d)
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), l.flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f'epoch: {ep}, count: {count}, loss: {loss.item()} norm:{norm}')
            optimizer.step()
            count += 1
