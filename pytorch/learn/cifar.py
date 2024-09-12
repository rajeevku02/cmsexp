import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Runner:
    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        batch_size = 4
        num_workers = 2

        self.trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def show_imgs(self):
        dataiter = iter(self.trainloader)
        images, labels = next(dataiter)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def test(self):
        print('Testing')
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


    def train(self):
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        print('Start training')
        self.model.train()
        for epoch in range(4):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, label = data
                ouputs = self.model(inputs)
                loss = criterion(ouputs, label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch}, {i+1}] loss: {running_loss / 2000}')
                    running_loss = 0.0
        print('Training done')

    def run(self):
        self.get_data()
        self.model = Net()
        self.train()
        self.test()

if __name__ == '__main__':
    Runner().run()