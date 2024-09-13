from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

#print(findFiles('../data/names/names/*.txt'))
#print(unicodeToAscii('Ślusàrski'))

class NameDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.all_categories = []
        self.data = []
        for filename in findFiles('../data/names/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            for line in lines:
                self.data.append([line, len(self.all_categories) - 1])
        random.shuffle(self.data)
        self.n_categories = len(self.all_categories)

    def category_from_output(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class Runner:
    def init(self):
        self.n_hidden = 128
        self.batch_size = 10
        self.ds = NameDataset()
        self.model = RNN(n_letters, self.n_hidden, self.ds.n_categories)

    def train(self):
        optim = torch.optim.SGD(self.model.parameters(), lr = 0.005)
        criterion = nn.NLLLoss()
        count = 0
        print('Totoa data: ', len(self.ds.data))
        total_loss = 0
        self.model.train()
        for ep in range(0, 2):
            for name, cdx in self.ds.data:
                inp = lineToTensor(name)
                expected = torch.tensor([cdx], dtype=torch.long)
                hidden = self.model.init_hidden()
                for i in range (0, inp.shape[0]):
                    output, hidden = self.model(inp[i], hidden)
                loss = criterion(output, expected)
                total_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                count += 1
                if count % 1000 == 0:
                    print('epoch:', ep, ' count:', count, ' loss:', total_loss / 1000)
                    total_loss = 0
    def test(self):
        print('testing')
        self.model.eval()
        total = 0
        correct = 0
        for name, cdx in self.ds.data:
                total += 1
                inp = lineToTensor(name)
                hidden = self.model.init_hidden()
                for i in range (0, inp.shape[0]):
                    output, hidden = self.model(inp[i], hidden)
                mx = output.argmax().item()
                if mx == cdx:
                    correct += 1
        print('accuracy = ', (correct / total)*100)

    def run(self):
        self.init()
        self.train()
        self.test()
        #input = lineToTensor('Albert')
        #hidden = torch.zeros(1, self.n_hidden)
        #output, next_hidden = self.model(input[0], hidden)
        #print(self.ds.category_from_output(output))

if __name__ == '__main__':
    Runner().run()