import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class CircleData(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.cartesian_prod(
            torch.arange(0, 1, 0.005, dtype=torch.float),
            torch.arange(0, 1, 0.005, dtype=torch.float))
        tmp = self.data - torch.tensor([0.5, 0.5])
        tmp = tmp * tmp
        tmp = torch.sum(tmp, dim=1)
        self.label = (tmp <= (0.2 * 0.2)).float()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class Runner:
    def __init__(self):
        self.ds = CircleData()
        self.loader = DataLoader(self.ds, 50, shuffle=True)
        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            #nn.Linear(50, 50),
            #nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def train(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lossf = nn.BCELoss()
        print('Training')
        for iep in range(0, 50):
            for d, l in self.loader:
                res = self.model(d)
                res = res.squeeze(dim=1)
                loss = lossf(res, l)
                optim.zero_grad()
                loss.backward()
                optim.step()
            print('epoc ', iep, ' loss:', loss.item())
            if (loss.item() < 0.0005):
                break

    def test(self):
        res = self.model(self.ds.data)
        res = res.squeeze(dim=1)
        res = torch.where(res > 0.5, 1, 0)
        self.plot(res)

    def plot(self, result):
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        for i in range(0, len(result)):
            if result[i].item() == 1:
                l1.append(self.ds.data[i][0].item())
                l2.append(self.ds.data[i][1].item())
            else:
                l3.append(self.ds.data[i][0].item())
                l4.append(self.ds.data[i][1].item())
        plt.scatter(l1, l2, marker='.')
        plt.scatter(l3, l4, marker='.')
        plt.gca().set_aspect('equal')
        plt.show()

    def run(self):
        self.train()
        self.test()

Runner().run()