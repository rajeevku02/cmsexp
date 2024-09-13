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

def find_files(path): return glob.glob(path)

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

class NameDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.all_categories = []
        self.data = []
        for filename in find_files('../data/names/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = read_lines(filename)
            for line in lines:
                self.data.append([line, len(self.all_categories) - 1])
        random.shuffle(self.data)
        self.n_categories = len(self.all_categories)

    def category_tensor(self, cdx):
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][cdx] = 1
        return tensor

    def category_from_output(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super().__init__()
        self.n_categories = n_categories
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class Runner:
    def init(self):
        self.n_hidden = 128
        self.batch_size = 10
        self.ds = NameDataset()
        self.model = RNN(self.ds.n_categories, n_letters, self.n_hidden, n_letters)

    def train(self):
        optim = torch.optim.SGD(self.model.parameters(), lr = 0.0005)
        criterion = nn.NLLLoss()
        count = 0
        print('Totoa data: ', len(self.ds.data))
        total_loss = 0
        self.model.train()
        for ep in range(0, 2):
            for name, cdx in self.ds.data:
                hidden = self.model.init_hidden()
                inp = line_to_tensor(name)
                categ = self.ds.category_tensor(cdx)
                target = target_tensor(name)
                loss = torch.Tensor([0])
                for i in range (0, inp.shape[0]):
                    output, hidden = self.model(categ, inp[i], hidden)
                    l = criterion(output.squeeze(), target[i])
                    loss += l
                total_loss += loss.item() / inp.shape[0]
                optim.zero_grad()
                loss.backward()
                optim.step()
                count += 1
                if count % 1000 == 0:
                    print('epoch:', ep, ' count:', count, ' loss:', total_loss / 1000)
                    total_loss = 0

    def sample(self, category, start_letter='A'):
        with torch.no_grad():  # no need to track history in sampling

            category_tensor = self.ds.category_tensor(self.ds.all_categories.index(category))
            input = line_to_tensor(start_letter)
            hidden = self.model.init_hidden()

            output_name = start_letter

            max_length = 20
            for i in range(max_length):
                output, hidden = self.model(category_tensor, input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == n_letters - 1:
                    break
                else:
                    letter = all_letters[topi]
                    output_name += letter
                input = line_to_tensor(letter)

            return output_name

    # Get multiple samples from one category and multiple starting letters
    def samples(self, category, start_letters='ABC'):
        for start_letter in start_letters:
            print(self.sample(category, start_letter))

    def test(self):
        print('testing')
        self.samples("English", "Jo")

    def run(self):
        self.init()
        self.train()
        self.test()

if __name__ == '__main__':
    Runner().run()