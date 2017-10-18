import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        l1_1 = F.relu(self.conv1_1(x))
        l1_2 = F.relu(self.conv1_2(l1_1))
        l1_3 = F.relu(self.conv1_3(l1_2))
        l1 = F.max_pool2d(l1_3, 2)

        l2_1 = F.relu(self.conv2_1(l1))
        l2_2 = F.relu(self.conv2_2(l2_1))
        l3_2 = F.relu(self.conv2_3(l2_2))
        l3 = F.max_pool2d(l3_2, 2)

        flat = l3.view(l3.size()[0], -1)

        fc1 = F.relu(self.fc1(flat))
        fc2 = F.relu(self.fc2(fc1))

        return fc2