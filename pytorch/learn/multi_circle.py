import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class CircleData(Dataset):
    def __init__(self):
        super().__init__()
        numbers = torch.arange(0, 1, 0.005, dtype=torch.float)
        self.data = torch.cartesian_prod(numbers, numbers)
        rad = 0.2
        l1 = self.check(self.data, 0.25, 0.25, rad, 1)
        l2 = self.check(self.data, 0.75, 0.25, rad, 2)
        l3 = self.check(self.data, 0.25, 0.75, rad, 3)
        l4 = self.check(self.data, 0.75, 0.75, rad, 4)
        self.label = (l1 + l2 + l3 + l4).long()

    def check(self, data, cx, cy, r, val):
        tmp = data - torch.tensor([cx, cy])
        tmp = tmp * tmp
        tmp = torch.sum(tmp, dim=1)
        result = ((tmp <= (r * r)).float() * val)
        return result
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class Runner:
    def __init__(self):
        self.ds = CircleData()
        self.loader = DataLoader(self.ds, 100, shuffle=True)
        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
        )
        #print(self.model)
    
    def train(self):
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print('Training')
        for ep in range(0, 20):
            for d, l in self.loader:
                res = self.model(d)
                #res = res.squeeze(dim = 1)
                #breakpoint()
                loss = criterion(res, l)
                optim.zero_grad()
                loss.backward()
                optim.step()
            print('epoch: ', ep, ' loss:', loss.item())

    def test(self):
        self.model.eval()
        res = self.model(self.ds.data)
        res = torch.argmax(res, dim=1)
        self.plot(res)

    def plot(self, result):
        tmp = self.ds.data[result == 1]
        plt.scatter(tmp[:, 0], tmp[:, 1], marker='.')
        
        tmp = self.ds.data[result == 2]
        plt.scatter(tmp[:, 0], tmp[:, 1], marker='.')

        tmp = self.ds.data[result == 3]
        plt.scatter(tmp[:, 0], tmp[:, 1], marker='.')

        tmp = self.ds.data[result == 4]
        plt.scatter(tmp[:, 0], tmp[:, 1], marker='.')

        plt.scatter(0, 0, marker='.')
        plt.scatter(1, 1, marker='.')
        plt.gca().set_aspect('equal')
        plt.show()

    def run(self):
        #self.plot(self.ds.label)
        self.train()
        self.test()


Runner().run()