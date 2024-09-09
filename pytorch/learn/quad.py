import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

A = 10
B = 5
C = 8
EPOCH = 500
BS = 100

class QuadDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.arange(-1, 1, 0.0001, dtype=torch.float)
        self.label = A * (self.data * self.data) + B * self.data + C
        self.data = self.data.unsqueeze(dim=1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

''''
class QuadNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return self.net(x)
'''    

class Runner:
    def test(self):
        x = self.ds.data.squeeze(dim=1).numpy().tolist()
        y = self.ds.label.numpy().tolist()
        plt.plot(x, y)
        res = self.model(self.ds.data).squeeze(dim=1)
        plt.plot(x, res.detach().numpy().tolist())

        plt.show()

    def run(self):
        torch.manual_seed(42)

        self.ds = QuadDataset()
        dl = DataLoader(self.ds, BS, shuffle=True)
        
        self.model = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Linear(20, 20),
            #nn.ReLU(),
            #nn.Linear(20, 20),
            #nn.Linear(20, 20),
            #nn.ReLU(),

            nn.Linear(20, 1),
        )
        print(self.model)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        lossf = nn.L1Loss()

        costs = []
        for ei in range(0, EPOCH):
            for ds, l in dl:
                res = self.model(ds)
                res = res.squeeze(dim=1)
                loss = lossf(res, l)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch ', ei)
            print(loss.item())
            costs.append(loss.item())
        #plt.plot(costs)
        #plt.show()
        self.test()

Runner().run()