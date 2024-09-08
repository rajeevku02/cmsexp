import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

TRAIN_SIZE = 200
BATCH_SIZE = 8
EPOCH = 100

class XorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 4),
            nn.ReLU(),
            #nn.Tanh(),
            nn.Linear(4, 1),
            #nn.Sigmoid()
        ])

    def forward(self, x):
        for _,l in enumerate(self.layers):
            x = l(x) 
        return x

class XorDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        self.data = torch.randint(low=0, high=2, size=(size, 2), dtype=torch.float)
        self.label = (self.data.sum(dim=1) == 1).to(torch.long)
        #print(self.data)
        #print(self.lables)

    def visualize_data(self):
        plt.figure(figsize=(4,4))
        data_0 = self.data[self.label == 0]
        data_1 = self.data[self.label == 1]

        plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
        plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
        plt.title("Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.xlabel(r"$x_1$")
        plt.legend()
        plt.show()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class XNN:
    def load_data(self):
        ds = XorDataset(TRAIN_SIZE)
        #ds.visualize_data()
        self.loader = DataLoader(ds, BATCH_SIZE, shuffle=True)

    def train(self):
        self.load_data()
        self.model.train(True)
        lossModule = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        tds = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float()
        exp = torch.Tensor([0, 1, 1, 0]).float()

        losVal = 100
        while losVal > 0.005:
            res = self.model(tds)
            res = res.squeeze(dim=1)
            loss = lossModule(res, exp)
            losVal = loss.item()
            print(losVal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #for _ in range(0, 5):
        #    for ds, l in self.loader:
        #        res = self.model(ds)
        #        res = res.squeeze(dim=1)
        #        loss = lossModule(res, l.float())
        #        optimizer.zero_grad()
        #        loss.backward()
        #        print(loss.item())
        #        losVal = loss.item()
        #        optimizer.step()

    def test(self):
        self.model.train(False)
        testDS = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        exp = torch.Tensor([0, 1, 1, 0]).long()
        res = (torch.sigmoid(self.model(testDS)) > 0.5).long()
        res = res.squeeze(dim=1)
        print (res)
        print(exp)
        '''
        err = 0
        count = 0
        for ds, l in self.loader:
            res = (torch.sigmoid(self.model(ds)) > 0.5).long()
            res = res.squeeze(dim=1)
            err += torch.abs(res - l).sum().item()
            count += res.shape[0]
        print('err = ', err)
        print('count = ', count)
        '''

    def main(self):
        torch.manual_seed(42) # Setting the seed
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        self.model = XorNetwork()
        self.model.to(device)
        print(self.model)
        self.train()
        self.test()

XNN().main()