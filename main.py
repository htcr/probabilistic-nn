import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class ToyDataset(Dataset):
    def __init__(self):
        self.mean_func = lambda x: np.sin(x * np.pi)
        self.var_func = lambda x: 0.2* np.abs(x)
        self.radius = 1.0

    def __getitem__(self, index):
        x = np.random.uniform(low=-self.radius, high=self.radius)
        mean_x = self.mean_func(x)
        var_x = self.var_func(x)
        observe = np.random.normal(loc=mean_x, scale=var_x)
        return x, observe

    def __len__(self):
        return 99999999


def draw_true_distribution():
    dataset = ToyDataset()
    data_loader = DataLoader(dataset, batch_size=1024)

    batch = iter(data_loader).next()
    x, obs = batch
    plt.scatter(x, obs)
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 8)
        self.mean_head = nn.Linear(8, 1)
        self.var_head = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mean_x = self.mean_head(x)
        var_x = self.var_head(x)
        return mean_x, var_x

net = Net()
print(net)