import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


mean_func = lambda x: np.sin(x * np.pi)
var_func = lambda x: 0.2* np.abs(x)
radius = 1.0

class ToyDataset(Dataset):
    def __init__(self):
        self.mean_func = mean_func
        self.var_func = var_func
        self.radius = radius

    def __getitem__(self, index):
        x = np.random.uniform(low=-self.radius, high=self.radius)
        mean_x = self.mean_func(x)
        var_x = self.var_func(x)
        observe = np.random.normal(loc=mean_x, scale=var_x)
        x_ary = np.array([x, ], dtype=np.float32)
        obs_ary = np.array([observe, ], dtype=np.float32)
        return x_ary, obs_ary

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
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, 16)
        self.mean_head = nn.Linear(16, 1)
        self.var_head = nn.Linear(16, 1)
        # self.var_head.bias[0] = 0.1
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        mean_x = self.mean_head(x)
        var_x = F.elu(self.var_head(x)) + 1.0
        return mean_x, var_x


def neg_log_likelihood_loss(y_x, mean_x, var_x):
    var2x = var_x**2
    return 0.5 * torch.sum( (mean_x - y_x)**2 / var2x + torch.log(var2x) ) / y_x.shape[0]


def test_loss():
    dataset = ToyDataset()
    data_loader = DataLoader(dataset, batch_size=16)
    net = Net()
    batch = iter(data_loader).next()
    x, y_x = batch
    print(x.shape)
    mean_x, var_x = net.forward(x)
    print(neg_log_likelihood_loss(y_x, mean_x, var_x))


def train(max_iter=1000):
    dataset = ToyDataset()
    data_loader_iter = iter(DataLoader(dataset, batch_size=128))
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for i in range(max_iter):
        optimizer.zero_grad()
        x, y_x = data_loader_iter.next()
        mean_x, var_x = net.forward(x)
        loss = neg_log_likelihood_loss(y_x, mean_x, var_x)
        loss.backward()
        optimizer.step()
        print(loss.item())
    torch.save(net.state_dict(), 'ckpt-{}.pth'.format(max_iter))


def evaluate(model_path):
    net = Net()
    net.load_state_dict(torch.load(model_path))
    sample_x = np.arange(start=-radius, stop=radius, step=0.001).astype(np.float32)
    sample_x = torch.Tensor(sample_x.reshape((-1, 1)))
    mean_x, var_x = net.forward(sample_x)

    sample_x, mean_x, var_x = sample_x.detach().numpy(), mean_x.detach().numpy(), var_x.detach().numpy()
    sample_x, mean_x, var_x = sample_x.reshape(-1), mean_x.reshape(-1), var_x.reshape(-1)

    gt_mean_x, gt_var_x = mean_func(sample_x), var_func(sample_x)

    fig, axs = plt.subplots(2, 2)
    ax00 = axs[0, 0]
    ax00.set_title('True Distribution')
    ax00.plot(sample_x, gt_mean_x, color='black')
    ax00.fill_between(sample_x, gt_mean_x - gt_var_x, gt_mean_x + gt_var_x, color='green', alpha=0.5)

    ax01 = axs[0, 1]
    ax01.set_title('Estimated Distribution')
    ax01.plot(sample_x, mean_x, color='black')
    ax01.fill_between(sample_x, mean_x-var_x, mean_x+var_x, color='green', alpha=0.5)

    ax10 = axs[1, 0]
    ax10.set_title('Mean Comparison')
    ax10.plot(sample_x, mean_x, color='blue')
    ax10.plot(sample_x, gt_mean_x, color='green')

    ax11 = axs[1, 1]
    ax11.set_title('Var Comparison')
    ax11.plot(sample_x, var_x, color='blue')
    ax11.plot(sample_x, gt_var_x, color='green')

    fig.tight_layout()
    plt.show()


train(1000)
evaluate('ckpt-1000.pth')